import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn import CrossEntropyLoss

from transformers import AutoTokenizer, PreTrainedTokenizer

from tqdm.auto import tqdm

import csv
import numpy as np

from model import ContrastiveWSD
from dataset import load_dataset, TrainDataCollator
from ukc import UKC
from utils import GlossSampler, PolysemySampler

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        validation_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        world_size: int,
        rank: int,
        validate_every: int,
        ukc: UKC,
        tokenizer: PreTrainedTokenizer,
        batch_size: int,
        resume_from: int,
        gloss_sampler: GlossSampler,
        polysemy_sampler: PolysemySampler,
        lemma_sense_mapping: dict
    ) -> None:
        self.model = model.to(rank)
        self.train_data = train_data
        self.validation_data = validation_data
        self.optimizer = optimizer
        self.world_size = world_size
        self.rank = rank
        self.validate_every = validate_every
        self.ukc = ukc
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.resume_from = resume_from
        self.gloss_sampler = gloss_sampler
        self.polysemy_sampler = polysemy_sampler
        self.lemma_sense_mapping = lemma_sense_mapping
        self.loss_fn = CrossEntropyLoss()

        DEFAULT_MAX_LENGTH = 512
        self.max_length = min(self.tokenizer.model_max_length, DEFAULT_MAX_LENGTH)

    def _log_loss_distributed(self, loss: torch.Tensor, world_size, epoch, batch_number):
        loss_tensor = loss.clone().detach().to(torch.float32)

        if self.rank == 0:
            avg_loss = loss_tensor.item() / world_size
            with open("loss.log", "a") as f:
                f.write(f"Epoch {epoch:2d} | Batch {batch_number:4d} | Average Loss: {avg_loss:.3f}\n")

    def _calculate_loss(self, input_embeddings, gnn_vector):
        logits = torch.matmul(input_embeddings, gnn_vector.T)

        labels_len = min(logits.shape[0], self.batch_size)
        labels = torch.arange(labels_len).to(self.rank)

        loss_i = self.loss_fn(logits, labels)
        logits_t = logits.T[:self.batch_size]
        loss_t = self.loss_fn(logits_t, labels)
        loss = (loss_i + loss_t)/2
        return loss

    def _run_batch(self, batch, train=True):
        text_input_ids = batch["input_ids"].to(self.rank)
        text_attention_mask = batch["attention_mask"].to(self.rank)
        labels = batch["labels"].to(self.rank)

        negative_from_polysemy = []
        for lemma, pos, label in zip(batch["lemmas"], batch["pos"], batch["labels"]):
            lemma_pos = f"{lemma}_{pos}"
            possible_senses = set(self.lemma_sense_mapping[lemma_pos])
            possible_senses.discard(int(label))

            if len(possible_senses) > 0:   # possible polysemy
                k = 1
                polysemy_samples = self.polysemy_sampler.generate_samples(k, only=possible_senses)
                negative_from_polysemy.append(polysemy_samples["gnn_id"].iloc[0])
        
        exclude_from_sampling = batch["labels"].tolist() + negative_from_polysemy
        gloss_samples = self.gloss_sampler.generate_samples(self.batch_size, exclude=exclude_from_sampling)
        all_samples = exclude_from_sampling + list(gloss_samples.index)
        all_samples_tensor = torch.tensor(all_samples, dtype=torch.long)

        glosses, edges = self.ukc.sample(all_samples_tensor)
        tokenized_glosses = self.tokenizer(glosses, padding=True, truncation=True, return_tensors="pt", max_length=self.max_length).to(self.rank)
        edges = edges.to(self.rank)

        if train:
            self.optimizer.zero_grad()

        input_embeddings, gnn_vector = self.model(text_input_ids, text_attention_mask, tokenized_glosses, edges, len(labels))
        loss = self._calculate_loss(input_embeddings, gnn_vector)

        if train:
            loss.backward()
            self.optimizer.step()

        return loss

    def _run_epoch(self, epoch):
        b_sz = len(next(iter(self.train_data)).input_ids)
        print(f"[GPU{self.rank}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}", flush=True)

        for batch_number, batch in enumerate(tqdm(self.train_data)):
            loss = self._run_batch(batch)
            if (batch_number % 16 == 0):
                self._log_loss_distributed(loss, world_size=self.world_size, epoch=epoch, batch_number=batch_number)

    def _validate(self, epoch):
        total_loss = 0.0
        num_batches = 0

        for batch in self.validation_data:
            loss = self._run_batch(batch, train=False)
            loss_tensor = loss.clone().detach().to(torch.float32)

            if self.rank == 0:
                total_loss += loss_tensor.item()
                num_batches += 1

        if self.rank == 0:
            avg_loss = total_loss / num_batches
            with open("validation_loss.log", "a") as f:
                f.write(f"Epoch {epoch:2d} | Validation Loss: {avg_loss:.3f}\n")

    def _save_checkpoint(self, epoch):
        ckp = self.model.state_dict()
        PATH = f"checkpoint_{epoch:02d}.pt"
        torch.save(ckp, PATH)
        print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")

    def train(self, max_epochs: int):
        if self.resume_from == 0:
            start = 1
            end = max_epochs + 1
        else:
            start = self.resume_from + 1
            end = start + max_epochs

        for epoch in range(start, end):
            self.model.train()
            self._run_epoch(epoch)
            if epoch % self.validate_every == 0:
                self.model.eval()
                with torch.no_grad():
                    self._validate(epoch)

                if self.rank == 0:
                    self._save_checkpoint(epoch)

                torch.cuda.empty_cache()

def prepare_dataloader(dataset: Dataset, batch_size: int, tokenizer: PreTrainedTokenizer, pin_memory: bool):
    data_collator = TrainDataCollator(tokenizer=tokenizer)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=pin_memory,
        shuffle=False,
        collate_fn=data_collator
    )

def generate_lemma_sense_mapping(lemma_sense_mapping_csv):
    lemma_sense_mapping = {}
    with open(lemma_sense_mapping_csv, "r") as f:
        csv_reader = csv.reader(f, delimiter=",")
        for row in csv_reader:
            lemma = row[0]
            pos = row[1]
            senses = [int(sense) for sense in row[2:]]
            lemma_sense_mapping[f"{lemma}_{pos}"] = senses
    return lemma_sense_mapping

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Script for training Contrastive WSD model')
    parser.add_argument('--seed', default=42, type=int, help='Seed to be used for random number generators')
    parser.add_argument('--total_epochs', default=8, type=int, help='Total epochs to train the model (default: 8)')
    parser.add_argument('--validate_every', default=4, type=int, help='Validates the model every n epochs.')
    parser.add_argument('--batch_size', default=32, type=int, help='Input batch size on each device (default: 32)')
    parser.add_argument('--base_model', default="google-bert/bert-base-uncased", type=str, help='Base transformers model to use (default: bert-base-uncased)')
    parser.add_argument('--small', default=False, type=bool, help='For debugging purposes, only process small amounts of data')
    parser.add_argument('--freeze_concept_encoder', default=True, type=bool, help='Freeze concept encoder')
    parser.add_argument('--resume_from', default=0, type=int, help='Resume training from which batch')
    parser.add_argument('--ukc_num_neighbors', type=int, nargs='+', default=[8, 8], help='Number of neighbors to be sampled during training or inference (default: 8 8)')
    parser.add_argument('--lemma_sense_mapping', type=str, default="lemma_gnn_mapping.csv", help="lemma to id mapping file")
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    train_dataset, validation_dataset, _, ukc = load_dataset(tokenizer, args.small, args.ukc_num_neighbors)
    train_data = prepare_dataloader(train_dataset, args.batch_size, tokenizer, pin_memory=True)
    validation_data = prepare_dataloader(validation_dataset, args.batch_size, tokenizer, pin_memory=False)

    gloss_sampler = GlossSampler("SemCor_Train_New.csv", "ukc.csv", seed=args.seed)
    polysemy_sampler = PolysemySampler("SemCor_Train_New.csv", "ukc.csv", seed=args.seed)
    lemma_sense_mapping = generate_lemma_sense_mapping(args.lemma_sense_mapping_csv)

    model = ContrastiveWSD(args.base_model, device=device, freeze_concept_encoder=args.freeze_concept_encoder)
    if (args.resume_from != 0):
        model_name = f"checkpoint_{args.resume_from:02d}.pt"
        model.load_state_dict(torch.load(model_name, weights_only=True, map_location=torch.device(device)))

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    trainer = Trainer(
        model=model,
        train_data=train_data,
        validation_data=validation_data,
        optimizer=optimizer,
        world_size=args.world_size,
        rank=args.rank,
        validate_every=args.validate_every,
        ukc=ukc,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        resume_from=args.resume_from,
        gloss_sampler=args.gloss_sampler,
        polysemy_sampler=polysemy_sampler,
        lemma_sense_mapping=lemma_sense_mapping)
    trainer.train(args.total_epochs)
