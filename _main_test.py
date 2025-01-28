import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from transformers import AutoTokenizer, PreTrainedTokenizer

from tqdm.auto import tqdm

import csv
import numpy as np

from model import ContrastiveWSD
from dataset import load_dataset, TestDataCollator
from ukc import UKC

class Evaluator:
    def __init__(
        self,
        id: int,
        model: torch.nn.Module,
        device: str,
        validation_data: DataLoader,
        ukc: UKC,
        tokenizer: PreTrainedTokenizer,
        batch_size: int
    ) -> None:
        self.id = id
        self.model = model.to(device)
        self.device = device
        self.validation_data = validation_data
        self.ukc = ukc
        self.tokenizer = tokenizer
        self.batch_size = batch_size

        self.temperature = 0.07

        DEFAULT_MAX_LENGTH = 512
        self.max_length = min(self.tokenizer.model_max_length, DEFAULT_MAX_LENGTH)

    def validate(self):
        all_top1, all_scores = [], []
        for batch in tqdm(self.validation_data):
            sentence_ids = batch["ids"]
            text_input_ids = batch["input_ids"].to(self.device)
            text_attention_mask = batch["attention_mask"].to(self.device)
            all_candidate_ids = batch["all_candidate_ids"]
            candidate_id_ranges = batch["candidate_id_ranges"]

            glosses, edges = self.ukc.sample(all_candidate_ids)
            tokenized_glosses = self.tokenizer(glosses, padding=True, truncation=True, return_tensors="pt", max_length=self.max_length).to(self.device)
            edges = edges.to(self.device)

            input_embeddings, gnn_vector = self.model(text_input_ids, text_attention_mask, tokenized_glosses, edges, len(all_candidate_ids))

            input_embeddings = F.normalize(input_embeddings, p=2, dim=1)
            gnn_vector = F.normalize(gnn_vector, p=2, dim=1)

            for i, (start, end) in enumerate(candidate_id_ranges):
                sentence_id = sentence_ids[i]
                sub_matrix = gnn_vector[start:end].T
                pairwise_similarity = (torch.matmul(input_embeddings[i], sub_matrix) * np.exp(self.temperature)).tolist()
                candidate_ids = all_candidate_ids[start:end].tolist()

                sorted_pairwise_similarity, sorted_candidate_ids = zip(*sorted(zip(pairwise_similarity, candidate_ids), reverse=True))

                for score, candidate_id in zip(sorted_pairwise_similarity, sorted_candidate_ids):
                    all_scores.append([sentence_id, candidate_id, score])

                top1 = [sentence_id] + list(sorted_candidate_ids[:1])
                all_top1.append(top1)

        with open(f"08-GlossNoise-Norm-{int(self.id):02d}.txt", 'w') as file:
            writer = csv.writer(file, delimiter=' ')
            writer.writerows(all_top1)

        with open(f"08-GlossNoise-Norm-{int(self.id):02d}-Scores.csv", 'w') as file:
            writer = csv.writer(file)
            writer.writerows(all_scores)

def load_model(base_model: str, load_file: str, device: str):
    model = ContrastiveWSD(base_model, device=device, freeze_concept_encoder=True)
    if load_file:
        model.load_state_dict(torch.load(load_file, weights_only=True, map_location=torch.device(device)))
    return model

def prepare_dataloader(dataset: Dataset, batch_size: int, tokenizer: PreTrainedTokenizer):
    data_collator = TestDataCollator(tokenizer=tokenizer)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        collate_fn=data_collator
    )

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument("--ids", nargs='+', help='For logging purposes')
    parser.add_argument('--base_model', default="google-bert/bert-base-uncased", type=str, help='Base transformers model to use (default: bert-base-uncased)')
    parser.add_argument("--load_files", nargs='+', help='Which file to load')
    parser.add_argument('--batch_size', default=32, type=int, help='Input batch size on each device (default: 32)')
    parser.add_argument('--small', default=False, type=bool, help='For debugging purposes, only process small amounts of data')
    parser.add_argument('--ukc_num_neighbors', type=int, nargs='+', default=[8, 8], help='Number of neighbors to be sampled during training or inference (default: 8 8)')

    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    _, _, test_dataset, ukc = load_dataset(tokenizer, args.small, args.ukc_num_neighbors)
    test_data = prepare_dataloader(test_dataset, args.batch_size, tokenizer)

    for id, load_file in zip(args.ids, args.load_files):
        model = load_model(args.base_model, load_file, device)
        evaluator = Evaluator(
            id=id,
            model=model,
            device=device,
            validation_data=test_data,
            ukc=ukc,
            tokenizer=tokenizer,
            batch_size=args.batch_size
        )
        evaluator.validate()
