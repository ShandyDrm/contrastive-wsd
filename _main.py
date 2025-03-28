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
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

from pprint import pprint

class LitContrastiveWSD(L.LightningModule):
    def __init__(
            self,
            model: torch.nn.Module,
            device: str,
            tokenizer: PreTrainedTokenizer,
            batch_size: int,
            learning_rate: float,
            scheduler_patience: int,
            scheduler_frequency: int,
            ukc: UKC,
            gloss_sampler: GlossSampler,
            polysemy_sampler: PolysemySampler,
            lemma_sense_mapping: dict,
            gnn_ukc_mapping: dict,
            eval_dir: str,
            test_dir: str,
            random_sample_gloss: bool,
            polysemy_gloss: bool,

        ):
        super().__init__()

        self.model = model.to(device)
        self.tokenizer = tokenizer

        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.scheduler_patience = scheduler_patience
        self.scheduler_frequency = scheduler_frequency

        self.ukc = ukc
        self.gloss_sampler = gloss_sampler
        self.polysemy_sampler = polysemy_sampler
        self.lemma_sense_mapping = lemma_sense_mapping
        self.gnn_ukc_mapping = gnn_ukc_mapping

        self.eval_dir = eval_dir
        self.test_dir = test_dir

        self.random_sample_gloss = random_sample_gloss
        self.polysemy_gloss = polysemy_gloss

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
        answers_ukc = np.array(batch["answers_ukc"].tolist())

        all_samples = answers_ukc

        if self.polysemy_gloss:
            negative_from_polysemy = []
            for _lemma, _pos, _labels in zip(lemmas, pos, answers_ukc):
                lemma_pos = f"{_lemma}_{_pos}"
                if lemma_pos not in lemma_sense_mapping:
                    continue

                possible_senses = set(lemma_sense_mapping[lemma_pos]) - set([_labels])

                if len(possible_senses) > 0:   # possible polysemy
                    k = 1
                    polysemy_samples = self.polysemy_sampler.generate_samples(k, only=possible_senses)
                    polysemy_sample = polysemy_samples["gnn_id"].iloc[0]
                    all_samples = np.append(all_samples, polysemy_sample)

        if self.random_sample_gloss:
            exclude_from_sampling = np.array(answers_ukc.tolist() + negative_from_polysemy)
            gloss_samples = self.gloss_sampler.generate_samples(self.batch_size, exclude=exclude_from_sampling)
            all_samples = np.concat(all_samples, gloss_samples["gnn_id"].to_numpy())

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
        self.log('train_loss', loss, on_step=True, prog_bar=True)
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
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=self.scheduler_patience, threshold=0)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'eval_f1_score',
                'interval': 'step',
                'frequency': self.scheduler_frequency
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
    parser.add_argument('--batch_size', default=32, type=int, help='Input batch size on each device (default: 32)')
    parser.add_argument('--max_steps', default=300_000, type=int, help='Maximum number of steps (default: 300_000)')
    parser.add_argument('--val_check_interval', default=2_000, type=int, help='How often to check the validation set, also used for scheduler frequency (default: 2_000)')
    parser.add_argument('--gradient_clip_val', default=10, type=int, help='The value at which to clip gradients (default: 10)')
    parser.add_argument('--accumulate_grad_batches', default=20, type=int, help='Accumulates gradients over k batches before stepping the optimizer (default: 20)')
    parser.add_argument('--learning_rate', default=1e-5, type=float, help="learning rate, default=1e-5")
    parser.add_argument('--scheduler_patience', default=3, type=int)
    parser.add_argument('--precision', default='16-mixed', type=str)
    parser.add_argument('--log_every_n_steps', default=1, type=int)
    parser.add_argument('--base_model', default="google-bert/bert-base-uncased", type=str, help='Base transformers model to use (default: bert-base-uncased)')
    parser.add_argument('--small', default=False, type=bool, help='For debugging purposes, only process small amounts of data')
    parser.add_argument('--save_topk', default=5, type=int, help='Save top k based on validation metrics (default: 5)')

    parser.add_argument('--ukc_num_neighbors', type=int, nargs='+', default=[8, 8], help='Number of neighbors to be sampled during training or inference (default: 8 8)')
    parser.add_argument('--lemma_sense_mapping', type=str, default="lemma_gnn_mapping.csv", help="lemma to id mapping file")

    parser.add_argument('--hidden_size', type=int, default=256, help="hidden size for the model")
    parser.add_argument('--gat_heads', type=int, default=4, help="number of multi-head attentions, default=4")
    parser.add_argument('--gat_self_loops', type=bool, default=False, help="enable attention mechanism to see its own features, default=False")
    parser.add_argument('--gat_residual', type=bool, default=True, help="enable residual [f(x) = x + g(x)] to graph attention network, default=True")
    parser.add_argument('--random_sample_gloss', type=bool, default=False, help="random sample gloss during training (default: False)")
    parser.add_argument('--polysemy_gloss', type=bool, default=False, help='sample polysemy during training (default: False)')

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
        scheduler_frequency=args.val_check_interval,
        ukc=ukc,
        gloss_sampler=gloss_sampler,
        polysemy_sampler=polysemy_sampler,
        lemma_sense_mapping=lemma_sense_mapping,
        gnn_ukc_mapping=gnn_ukc_mapping,
        eval_dir=args.eval_dir,
        test_dir=args.test_dir, 
        random_sample_gloss=args.random_sample_gloss,
        polysemy_gloss=args.polysemy_gloss,
    )

    wandb_logger = WandbLogger(
        project=args.project_name
    )

    wandb_logger.experiment.config.update({
        "base_model": args.base_model,
        "learning_rate": args.learning_rate,
        "attention_multihead": args.gat_heads,
        "optimizer": "AdamW",
        "scheduler": "ReduceLROnPlateau",
        "scheduler_patience": args.scheduler_patience,
        "scheduler_threshold": 0,
        "precision": args.precision,
        "random_sample_gloss": args.random_sample_gloss,
        "polysemy_gloss":args.polysemy_gloss,
    })
    
    early_stop_callback = EarlyStopping(
        monitor="eval_f1_score",
        min_delta=0.00,
        patience=15,
        verbose=False,
        mode="max")
    
    checkpoint_callback = ModelCheckpoint(
        save_top_k=args.save_topk,
        monitor="eval_f1_score",
        mode="max"
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')

    trainer = L.Trainer(
        accelerator="auto",
        callbacks=[lr_monitor, early_stop_callback, checkpoint_callback],
        default_root_dir="checkpoints/",
        logger=wandb_logger,
        max_steps=args.max_steps,
        val_check_interval=args.val_check_interval,
        check_val_every_n_epoch=None,
        gradient_clip_val=args.gradient_clip_val,
        accumulate_grad_batches=args.accumulate_grad_batches,
        precision=args.precision,
        log_every_n_steps=args.log_every_n_steps,
    )
    trainer.fit(model=pl_model,
                train_dataloaders=train_data,
                val_dataloaders=eval_data)
    
    test_results = []
    for path in checkpoint_callback.best_k_models:
        value = checkpoint_callback.best_k_models[path]
        best_model = LitContrastiveWSD.load_from_checkpoint(
            path,
            model=model,
            device=device,
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            scheduler_patience=args.scheduler_patience,
            scheduler_frequency=args.val_check_interval,
            ukc=ukc,
            gloss_sampler=gloss_sampler,
            polysemy_sampler=polysemy_sampler,
            lemma_sense_mapping=lemma_sense_mapping,
            gnn_ukc_mapping=gnn_ukc_mapping,
            eval_dir=args.eval_dir,
            test_dir=args.test_dir, 
            random_sample_gloss=args.random_sample_gloss,
            polysemy_gloss=args.polysemy_gloss,
        )

        test_result = trainer.test(model=best_model, dataloaders=test_data)

        test_results.append({
            "path": path,
            "val_score": value.item(),
            "test_score": test_result[0]['test_f1_score']
        })
    
    pprint(test_results)
