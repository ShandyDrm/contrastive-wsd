import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import LRScheduler, CosineAnnealingLR

from transformers import AutoTokenizer, PreTrainedTokenizer

from tqdm.auto import tqdm

import os, csv, subprocess, re

from model import ContrastiveWSD
from dataset import build_ukc, build_dataframes, build_dataset, TrainDataCollator, TestDataCollator
from ukc import UKC
from utils import GlossSampler, PolysemySampler

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        device: str,
        train_data: DataLoader,
        eval_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        use_scheduler: bool,
        scheduler: LRScheduler,
        scheduler_step: int,
        validate_every: int,
        ukc: UKC,
        tokenizer: PreTrainedTokenizer,
        batch_size: int,
        resume_from: int,
        gloss_sampler: GlossSampler,
        polysemy_sampler: PolysemySampler,
        lemma_sense_mapping: dict,
        gnn_ukc_mapping: dict,
        eval_dir: str
    ) -> None:
        self.model = model.to(device)
        self.device = device
        self.train_data = train_data
        self.eval_data = eval_data
        self.optimizer = optimizer
        self.use_scheduler = use_scheduler
        self.scheduler = scheduler
        self.scheduler_step = scheduler_step
        self.scheduler_counter = 0
        self.validate_every = validate_every
        self.ukc = ukc
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.resume_from = resume_from
        self.gloss_sampler = gloss_sampler
        self.polysemy_sampler = polysemy_sampler
        self.lemma_sense_mapping = lemma_sense_mapping
        self.gnn_ukc_mapping = gnn_ukc_mapping
        self.eval_dir = eval_dir
        self.all_eval_scores = []
        self.loss_fn = CrossEntropyLoss()

        DEFAULT_MAX_LENGTH = 512
        self.max_length = min(self.tokenizer.model_max_length, DEFAULT_MAX_LENGTH)

    def _log_loss_distributed(self, loss: torch.Tensor, epoch, batch_number):
        loss_tensor = loss.clone().detach().to(torch.float32)

        with open("loss.log", "a") as f:
            f.write(f"Epoch {epoch:02d} | Batch {batch_number:5d} | Average Loss: {loss_tensor:.3f} | LR: {self.scheduler.get_last_lr()[0]:.10f} | SchStep: {self.scheduler._step_count:5d}\n")

    def _log_gradient_norm(self, epoch: int, batch_number: int):
        total_gradient_norm = 0.0
        for param in self.model.parameters():
            if param.grad is not None:
                total_gradient_norm += param.grad.data.norm(2).item() ** 2
            total_gradient_norm = total_gradient_norm ** 0.5

        with open("gradient_norm.log", "a") as f:
            f.write(f"Epoch {epoch:02d} | Batch {batch_number:4d} | Gradient Norm: {total_gradient_norm:.10f}\n")

    def _calculate_loss(self, input_embeddings, gnn_vector):
        logits = torch.matmul(input_embeddings, gnn_vector.T)

        labels_len = min(logits.shape[0], self.batch_size)
        labels = torch.arange(labels_len).to(self.device)

        loss_i = self.loss_fn(logits, labels)
        logits_t = logits.T[:labels_len]
        loss_t = self.loss_fn(logits_t, labels)
        loss = (loss_i + loss_t)/2
        return loss

    def _run_batch(self, batch, train=True):
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

        exclude_from_sampling = list(answers_ukc) + negative_from_polysemy
        gloss_samples = self.gloss_sampler.generate_samples(self.batch_size, exclude=exclude_from_sampling)
        all_samples = exclude_from_sampling + list(gloss_samples["gnn_id"])
        all_samples_tensor = torch.tensor(all_samples, dtype=torch.long)

        glosses, edges = self.ukc.sample(all_samples_tensor)
        tokenized_glosses = self.tokenizer(glosses, padding=True, truncation=True, return_tensors="pt", max_length=self.max_length).to(self.device)
        edges = edges.to(self.device)

        tokenized_sentences = self.tokenizer(sentence,
                                             is_split_into_words=True,
                                             padding=True,
                                             truncation=True,
                                             return_tensors="pt",
                                             max_length=self.max_length
                                             ).to(self.device)

        if train:
            self.optimizer.zero_grad()

        input_embeddings, gnn_vector = self.model(tokenized_sentences, loc, tokenized_glosses, edges, len(all_samples))
        loss = self._calculate_loss(input_embeddings, gnn_vector)

        if train:
            loss.backward()
            self.optimizer.step()

            if self.use_scheduler:
                if self.scheduler_counter == self.scheduler_step:
                    self.scheduler.step()
                    self.scheduler_counter = 0
                else:
                    self.scheduler_counter += 1

        return loss

    def _run_epoch(self, epoch):
        b_sz = len(next(iter(self.train_data))["id"])
        print(f"[GPU] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}", flush=True)

        for batch_number, batch in enumerate(tqdm(self.train_data)):
            loss = self._run_batch(batch)
            if (batch_number % 16 == 0):
                self._log_loss_distributed(loss, epoch=epoch, batch_number=batch_number)
                self._log_gradient_norm(epoch, batch_number)

    def _validate(self, epoch):
        all_top1, all_scores = [], []
        for batch in tqdm(self.eval_data):
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
                                                 max_length=self.max_length
                                                 ).to(self.device)

            input_embeddings, gnn_vector = self.model(tokenized_sentences, loc, tokenized_glosses, edges, len(candidates_ukc))

            for i, (start, end) in enumerate(candidate_id_ranges):
                sentence_id = sentence_ids[i]
                sub_matrix = gnn_vector[start:end].T
                pairwise_similarity = torch.matmul(input_embeddings[i], sub_matrix)
                candidate_ids = candidates_ukc[start:end].tolist()
                sorted_pairwise_similarity, sorted_candidate_ids = zip(*sorted(zip(pairwise_similarity, candidate_ids), reverse=True))

                for score, candidate_id in zip(sorted_pairwise_similarity, sorted_candidate_ids):
                    all_scores.append([sentence_id, candidate_id, score])

                top1 = sorted_candidate_ids[0]
                top1 = self.gnn_ukc_mapping[top1]
                all_top1.append([sentence_id, top1])
            
        eval_tempfile = f"validation.epoch{epoch:02d}.txt"
        with open(eval_tempfile, 'w') as file:
            writer = csv.writer(file, delimiter=' ')
            writer.writerows(all_top1)

        def calculate_scores(directory, epoch, result_filename):
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

                result = {
                    "Epoch": epoch,
                    "Criteria": title
                }

                for key, value in key_value_pairs:
                    result[key] = value

                result_rows.append(result)
            return result_rows

        eval_scores = calculate_scores(self.eval_dir, epoch, eval_tempfile)

        with open("validation.log", "a") as f:
            f.write(f"Epoch {epoch:02d}\n")
            for row in eval_scores:
                criteria = row['Criteria']
                precision = row['P']
                recall = row['R']
                f1 = row['F1']
                f.write(f"  {criteria}\n")
                f.write(f"    Precision = {precision}\n")
                f.write(f"    Recall = {recall}\n")
                f.write(f"    F1 = {f1}\n")
        self.all_eval_scores.extend(eval_scores)

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

                self._save_checkpoint(epoch)

                torch.cuda.empty_cache()

def prepare_dataloader(dataset: Dataset, batch_size: int, pin_memory: bool, data_collator: callable):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=pin_memory,
        shuffle=True,
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
    parser.add_argument('--resume_from', default=0, type=int, help='Resume training from which batch')
    parser.add_argument('--ukc_num_neighbors', type=int, nargs='+', default=[8, 8], help='Number of neighbors to be sampled during training or inference (default: 8 8)')
    parser.add_argument('--lemma_sense_mapping', type=str, default="lemma_gnn_mapping.csv", help="lemma to id mapping file")
    parser.add_argument('--use_scheduler', type=bool, default=False, help="use scheduler during training")
    parser.add_argument('--scheduler_step', type=int, default=16, help="update scheduler every n steps, default=16")
    parser.add_argument('--learning_rate', type=float, default=1e-5, help="learning rate, default=1e-5")
    parser.add_argument('--hidden_size', type=int, default=256, help="hidden size for the model")

    parser.add_argument('--train_filename', type=str, default='train.complete.data.json')
    parser.add_argument('--eval_filename', type=str, default='eval.complete.data.json')
    parser.add_argument('--test_filename', type=str, default='test.complete.data.json')
    parser.add_argument('--ukc_filename', type=str, default='ukc.csv')
    parser.add_argument('--edges_filename', type=str, default='edges.csv')

    parser.add_argument('--eval_dir', type=str, default='eval-gold-standard', help='directory where eval gold standard files are located')

    parser.add_argument('--gat_heads', type=int, default=1, help="number of multi-head attentions, default=1")
    parser.add_argument('--gat_self_loops', type=bool, default=True, help="enable attention mechanism to see its own features, default=True")
    parser.add_argument('--gat_residual', type=bool, default=False, help="enable residual [f(x) = x + g(x)] to graph attention network, default=False")
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    ukc, ukc_df, ukc_gnn_mapping, gnn_ukc_mapping = build_ukc(args.ukc_filename, args.edges_filename, args.ukc_num_neighbors)
    train_df, eval_df, test_df = build_dataframes(args.train_filename, args.eval_filename, args.test_filename, ukc_gnn_mapping, args.small)
    train_dataset, eval_dataset, test_dataset = build_dataset(train_df, eval_df, test_df, tokenizer)

    train_data = prepare_dataloader(train_dataset, args.batch_size, True, TrainDataCollator())
    eval_data = prepare_dataloader(eval_dataset, args.batch_size, False, TestDataCollator())

    gloss_sampler = GlossSampler(train_df, ukc_df, ukc_gnn_mapping, args.seed)
    polysemy_sampler = PolysemySampler(train_df, ukc_df, ukc_gnn_mapping, args.seed)
    lemma_sense_mapping = generate_lemma_sense_mapping(args.lemma_sense_mapping)

    model = ContrastiveWSD(
        args.base_model,
        hidden_size=args.hidden_size,
        gat_heads=args.gat_heads,
        gat_self_loops=args.gat_self_loops,
        gat_residual=args.gat_residual
    ).to(device)

    if (args.resume_from != 0):
        model_name = f"checkpoint_{args.resume_from:02d}.pt"
        model.load_state_dict(torch.load(model_name, weights_only=True, map_location=torch.device(device)))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    T_max = (len(train_dataset) * args.total_epochs) / (args.batch_size * args.scheduler_step)
    scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=T_max)

    trainer = Trainer(
        model=model,
        device=device,
        train_data=train_data,
        eval_data=eval_data,
        optimizer=optimizer,
        use_scheduler=args.use_scheduler,
        scheduler=scheduler,
        scheduler_step=args.scheduler_step,
        validate_every=args.validate_every,
        ukc=ukc,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        resume_from=args.resume_from,
        gloss_sampler=gloss_sampler,
        polysemy_sampler=polysemy_sampler,
        lemma_sense_mapping=lemma_sense_mapping,
        gnn_ukc_mapping=gnn_ukc_mapping,
        eval_dir=args.eval_dir)
    trainer.train(args.total_epochs)
