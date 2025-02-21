import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from transformers import AutoTokenizer, PreTrainedTokenizer

from tqdm.auto import tqdm

import numpy as np

import os, csv, subprocess, re

from model import ContrastiveWSD
from dataset import build_ukc, build_dataframes, build_dataset, TrainDataCollator, TestDataCollator
from ukc import UKC
from utils import GlossSampler, PolysemySampler

import lightning as L
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.callbacks.early_stopping import EarlyStopping


class LitContrastiveWSD(L.LightningModule):
    def __init__(
            self,
            model: torch.nn.Module,
            device: str,
            tokenizer: PreTrainedTokenizer,
            batch_size: int,
            learning_rate: float,
            scheduler_patience: int,
            ukc: UKC,
            gloss_sampler: GlossSampler,
            polysemy_sampler: PolysemySampler,
            lemma_sense_mapping: dict,
            gnn_ukc_mapping: dict,
            eval_dir: str,
            test_dir: str
        ):
        super().__init__()

        self.model = model.to(device)
        self.tokenizer = tokenizer

        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.scheduler_patience = scheduler_patience

        self.ukc = ukc
        self.gloss_sampler = gloss_sampler
        self.polysemy_sampler = polysemy_sampler
        self.lemma_sense_mapping = lemma_sense_mapping
        self.gnn_ukc_mapping = gnn_ukc_mapping

        self.eval_dir = eval_dir
        self.test_dir = test_dir

        self.validation_step_outputs = []
        self.validation_step_top1 = []
        self.validation_step_scores = []

        self.test_step_top1 = []
        self.test_step_scores = []

        DEFAULT_MAX_LENGTH = 512
        self.max_length = min(self.tokenizer.model_max_length, DEFAULT_MAX_LENGTH)

    def _calculate_loss(self, input_embeddings, gnn_vector):
        logits = torch.matmul(input_embeddings, gnn_vector.T)

        labels_len = min(logits.shape[0], self.batch_size)
        labels = torch.arange(labels_len).to(self.device)

        loss_i = F.cross_entropy(logits, labels)
        logits_t = logits.T[:labels_len]
        loss_t = F.cross_entropy(logits_t, labels)
        loss = (loss_i + loss_t)/2
        return loss

    def training_step(self, batch, batch_idx):
        ids = batch["id"]
        lemmas = batch["lemma"]
        pos = batch["pos"]
        loc = batch["loc"]
        sentence = batch["sentence"]
        answers_ukc = batch["answers_ukc"]

        negative_from_polysemy = []
        for _lemma, _pos, _labels in zip(lemmas, pos, answers_ukc):
            lemma_pos = f"{_lemma}_{_pos}"
            if lemma_pos not in lemma_sense_mapping:
                continue

            possible_senses = set(lemma_sense_mapping[lemma_pos]) - set([_labels])

            if len(possible_senses) > 0:   # possible polysemy
                k = 1
                polysemy_samples = self.polysemy_sampler.generate_samples(k, only=possible_senses)
                negative_from_polysemy.append(polysemy_samples["gnn_id"].iloc[0])
        
        exclude_from_sampling = np.array(answers_ukc.tolist() + negative_from_polysemy)
        gloss_samples = self.gloss_sampler.generate_samples(self.batch_size, exclude=exclude_from_sampling)
        all_samples = np.concat((exclude_from_sampling, gloss_samples["gnn_id"].to_numpy()))
        all_samples_tensor = torch.tensor(all_samples, dtype=torch.long)

        glosses, edges = self.ukc.sample(all_samples_tensor)
        tokenized_glosses = self.tokenizer(glosses, padding=True, truncation=True, return_tensors="pt", max_length=self.max_length).to(self.device)
        edges = edges.to(self.device)

        tokenized_sentences = self.tokenizer(sentence,
                                             is_split_into_words=True,
                                             padding=True,
                                             truncation=True,
                                             return_tensors="pt",
                                             max_length=self.max_length).to(self.device)
        
        input_embeddings, gnn_vector = self.model(tokenized_sentences, loc, tokenized_glosses, edges, len(all_samples))
        loss = self._calculate_loss(input_embeddings, gnn_vector)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        sentence_ids = batch["id"]
        loc = batch["loc"]
        sentence = batch["sentence"]
        candidates_ukc = batch["candidates_ukc"]
        candidate_id_ranges = batch["candidate_id_ranges"]

        candidates_ukc = torch.tensor(candidates_ukc)

        glosses, edges = self.ukc.sample(candidates_ukc)
        tokenized_glosses = self.tokenizer(glosses, padding=True, truncation=True, return_tensors="pt", max_length=self.max_length).to(self.device)
        edges = edges.to(self.device)

        tokenized_sentences = self.tokenizer(sentence,
                                             is_split_into_words=True,
                                             padding=True,
                                             truncation=True,
                                             return_tensors="pt",
                                             max_length=self.max_length).to(self.device)

        input_embeddings, gnn_vector = self.model(tokenized_sentences, loc, tokenized_glosses, edges, len(candidates_ukc))

        for i, (start, end) in enumerate(candidate_id_ranges):
            sentence_id = sentence_ids[i]
            sub_matrix = gnn_vector[start:end].T
            pairwise_similarity = torch.matmul(input_embeddings[i], sub_matrix)
            candidate_ids = candidates_ukc[start:end].tolist()
            sorted_pairwise_similarity, sorted_candidate_ids = zip(*sorted(zip(pairwise_similarity, candidate_ids), reverse=True))

            for score, candidate_id in zip(sorted_pairwise_similarity, sorted_candidate_ids):
                self.validation_step_scores.append([sentence_id, candidate_id, score])

            top1 = sorted_candidate_ids[0]
            top1 = self.gnn_ukc_mapping[top1]
            self.validation_step_top1.append([sentence_id, top1])

    def on_validation_epoch_end(self):
        # probably do something with the scores here?
        eval_tempfile = f"validation.temp.txt"
        with open(eval_tempfile, 'w') as file:
            writer = csv.writer(file, delimiter=' ')
            writer.writerows(self.validation_step_top1)
        
        def calculate_scores(directory, result_filename):
            result_rows = []
            gold_standards = [
                ('ALL', 'UKC.gold.key.txt'),
                ('Seen Only', 'UKC.in.test.gold.key.txt'),
                ('Unseen Only', 'UKC.out.test.gold.key.txt'),
                ('Single Candidate Only', 'UKC.single.test.gold.key.txt'),
                ('Multiple Candidates Only', 'UKC.multi.test.gold.key.txt'),
                ('Multiple Candidates Seen Only', 'UKC.multi.in.test.gold.key.txt'),
                ('Multiple Candidates Unseen Only', 'UKC.multi.out.test.gold.key.txt')
            ]

            for title, gold_standard in gold_standards:
                result = subprocess.run(
                    ['java', "Scorer", f"{directory}/{gold_standard}", result_filename],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
                )

                output = result.stdout
                key_value_pairs = re.findall('(\\w+)=\\s*([\\d.]+%)', output)
                result = {"Criteria": title}
                for key, value in key_value_pairs:
                    result[key] = value
                result_rows.append(result)
            return result_rows

        eval_scores = calculate_scores(self.eval_dir, eval_tempfile)
        f1_score = float(eval_scores[0]['F1'][:-1])
        self.log('eval_f1_score', f1_score, prog_bar=True)

        self.validation_step_top1 = []
        self.validation_step_scores = []

    def test_step(self, batch, batch_idx):
        sentence_ids = batch["id"]
        loc = batch["loc"]
        sentence = batch["sentence"]
        candidates_ukc = batch["candidates_ukc"]
        candidate_id_ranges = batch["candidate_id_ranges"]

        candidates_ukc = torch.tensor(candidates_ukc)

        glosses, edges = self.ukc.sample(candidates_ukc)
        tokenized_glosses = self.tokenizer(glosses, padding=True, truncation=True, return_tensors="pt", max_length=self.max_length).to(self.device)
        edges = edges.to(self.device)

        tokenized_sentences = self.tokenizer(sentence,
                                             is_split_into_words=True,
                                             padding=True,
                                             truncation=True,
                                             return_tensors="pt",
                                             max_length=self.max_length).to(self.device)

        input_embeddings, gnn_vector = self.model(tokenized_sentences, loc, tokenized_glosses, edges, len(candidates_ukc))

        for i, (start, end) in enumerate(candidate_id_ranges):
            sentence_id = sentence_ids[i]
            sub_matrix = gnn_vector[start:end].T
            pairwise_similarity = torch.matmul(input_embeddings[i], sub_matrix)
            candidate_ids = candidates_ukc[start:end].tolist()
            sorted_pairwise_similarity, sorted_candidate_ids = zip(*sorted(zip(pairwise_similarity, candidate_ids), reverse=True))

            for score, candidate_id in zip(sorted_pairwise_similarity, sorted_candidate_ids):
                self.test_step_scores.append([sentence_id, candidate_id, score])

            top1 = sorted_candidate_ids[0]
            top1 = self.gnn_ukc_mapping[top1]
            self.test_step_top1.append([sentence_id, top1])

    def on_test_epoch_end(self):
        # probably do something with the scores here?
        eval_tempfile = f"test.temp.txt"
        with open(eval_tempfile, 'w') as file:
            writer = csv.writer(file, delimiter=' ')
            writer.writerows(self.test_step_top1)

        def calculate_scores(directory, result_filename):
            result_rows = []
            gold_standards = [
                ('ALL', 'UKC.gold.key.txt'),
                ('Seen Only', 'UKC.in.test.gold.key.txt'),
                ('Unseen Only', 'UKC.out.test.gold.key.txt'),
                ('Single Candidate Only', 'UKC.single.test.gold.key.txt'),
                ('Multiple Candidates Only', 'UKC.multi.test.gold.key.txt'),
                ('Multiple Candidates Seen Only', 'UKC.multi.in.test.gold.key.txt'),
                ('Multiple Candidates Unseen Only', 'UKC.multi.out.test.gold.key.txt')
            ]

            for title, gold_standard in gold_standards:
                result = subprocess.run(
                    ['java', "Scorer", f"{directory}/{gold_standard}", result_filename],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
                )

                output = result.stdout
                key_value_pairs = re.findall('(\\w+)=\\s*([\\d.]+%)', output)
                result = {"Criteria": title}
                for key, value in key_value_pairs:
                    result[key] = value
                result_rows.append(result)
            return result_rows

        eval_scores = calculate_scores(self.test_dir, eval_tempfile)
        f1_score = float(eval_scores[0]['F1'][:-1])
        self.log('test_f1_score', f1_score, prog_bar=True)

        self.test_step_top1 = []
        self.test_step_scores = []

    def configure_optimizers(self):
        optimizer = torch.optim.RAdam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=self.scheduler_patience)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'eval_f1_score'
            }
        }

def prepare_dataloader(dataset: Dataset, batch_size: int, pin_memory: bool, shuffle: bool, data_collator: callable):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=pin_memory,
        shuffle=shuffle,
        collate_fn=data_collator,
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
    parser.add_argument('--project_name', type=str, help='Project name for logging')

    parser.add_argument('--seed', default=42, type=int, help='Seed to be used for random number generators')
    parser.add_argument('--total_epochs', default=8, type=int, help='Total epochs to train the model (default: 8)')
    parser.add_argument('--batch_size', default=32, type=int, help='Input batch size on each device (default: 32)')
    parser.add_argument('--accumulate_grad_batches', default=1, type=int, help='Accumulates gradients over k batches before stepping the optimizer (default: 1)')
    parser.add_argument('--learning_rate', default=1e-5, type=float, help="learning rate, default=1e-5")
    parser.add_argument('--scheduler_patience', default=1, type='int')
    parser.add_argument('--base_model', default="google-bert/bert-base-uncased", type=str, help='Base transformers model to use (default: bert-base-uncased)')
    parser.add_argument('--small', default=False, type=bool, help='For debugging purposes, only process small amounts of data')

    parser.add_argument('--ukc_num_neighbors', type=int, nargs='+', default=[8, 8], help='Number of neighbors to be sampled during training or inference (default: 8 8)')
    parser.add_argument('--lemma_sense_mapping', type=str, default="lemma_gnn_mapping.csv", help="lemma to id mapping file")

    parser.add_argument('--hidden_size', type=int, default=256, help="hidden size for the model")
    parser.add_argument('--gat_heads', type=int, default=1, help="number of multi-head attentions, default=1")
    parser.add_argument('--gat_self_loops', type=bool, default=True, help="enable attention mechanism to see its own features, default=True")
    parser.add_argument('--gat_residual', type=bool, default=False, help="enable residual [f(x) = x + g(x)] to graph attention network, default=False")

    parser.add_argument('--train_filename', type=str, default='train.complete.data.json')
    parser.add_argument('--eval_filename', type=str, default='eval.complete.data.json')
    parser.add_argument('--test_filename', type=str, default='test.complete.data.json')
    parser.add_argument('--ukc_filename', type=str, default='ukc.csv')
    parser.add_argument('--edges_filename', type=str, default='edges.csv')

    parser.add_argument('--eval_dir', type=str, default='eval-gold-standard', help='directory where eval gold standard files are located')
    parser.add_argument('--test_dir', type=str, default='test-gold-standard', help='directory where test gold standard files are located')

    args = parser.parse_args()

    seed_everything(args.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    base_model = args.base_model

    tokenizer_args = {}
    if "roberta" or "bart" in base_model:
        tokenizer_args["add_prefix_space"] = True
    if "ModernBERT" in base_model:
        import torch._dynamo
        torch._dynamo.config.suppress_errors = True

    tokenizer = AutoTokenizer.from_pretrained(base_model, **tokenizer_args)

    ukc, ukc_df, ukc_gnn_mapping, gnn_ukc_mapping = build_ukc(args.ukc_filename, args.edges_filename, args.ukc_num_neighbors)
    train_df, eval_df, test_df = build_dataframes(args.train_filename, args.eval_filename, args.test_filename, ukc_gnn_mapping, args.small)
    train_dataset, eval_dataset, test_dataset = build_dataset(train_df, eval_df, test_df, tokenizer)

    train_data = prepare_dataloader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        pin_memory=True,
        shuffle=True,
        data_collator=TrainDataCollator())

    eval_data = prepare_dataloader(
        dataset=eval_dataset,
        batch_size=args.batch_size,
        pin_memory=False,
        shuffle=False,
        data_collator=TestDataCollator())

    test_data = prepare_dataloader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        pin_memory=False,
        shuffle=False,
        data_collator=TestDataCollator())

    gloss_sampler = GlossSampler(train_df, ukc_df, ukc_gnn_mapping, args.seed)
    polysemy_sampler = PolysemySampler(train_df, ukc_df, ukc_gnn_mapping, args.seed)
    lemma_sense_mapping = generate_lemma_sense_mapping(args.lemma_sense_mapping)

    model = ContrastiveWSD(
        base_model,
        hidden_size=args.hidden_size,
        gat_heads=args.gat_heads,
        gat_self_loops=args.gat_self_loops,
        gat_residual=args.gat_residual
    ).to(device)

    pl_model = LitContrastiveWSD(
        model=model,
        device=device,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        scheduler_patience=args.scheduler_patience,
        ukc=ukc,
        gloss_sampler=gloss_sampler,
        polysemy_sampler=polysemy_sampler,
        lemma_sense_mapping=lemma_sense_mapping,
        gnn_ukc_mapping=gnn_ukc_mapping,
        eval_dir=args.eval_dir,
        test_dir=args.test_dir)

    wandb_logger = WandbLogger(
        project=args.project_name
    )

    wandb_logger.experiment.config.update({
        "base_model": args.base_model,
        "learning_rate": args.learning_rate,
        "attention_multihead": args.gat_heads,
        "scheduler": "ReduceLROnPlateau",
        "scheduler_patience": args.scheduler_patience
    })
    
    early_stop_callback = EarlyStopping(
        monitor="eval_f1_score",
        min_delta=0.00,
        patience=8,
        verbose=False,
        mode="max")

    lr_monitor = LearningRateMonitor(logging_interval='step')

    trainer = L.Trainer(
        accelerator="auto",
        callbacks=[lr_monitor, early_stop_callback],
        default_root_dir="checkpoints/",
        logger=wandb_logger,
        max_epochs=args.total_epochs,
        accumulate_grad_batches=args.accumulate_grad_batches,
        val_check_interval=0.125
    )
    trainer.fit(model=pl_model,
                train_dataloaders=train_data,
                val_dataloaders=eval_data)
    
    trainer.test(model=pl_model,
                 dataloaders=test_data)

