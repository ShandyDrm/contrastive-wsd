import torch
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, PreTrainedTokenizer

from tqdm.auto import tqdm

import csv
from typing import List

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

            hypernym_glosses, hypernym_edges, hyponym_glosses, hyponym_edges = self.ukc.sample(all_candidate_ids)

            hypernym_tokens = self.tokenizer(hypernym_glosses, padding=True, truncation=True, return_tensors="pt", max_length=self.max_length).to(self.device)
            hypernym_edges = hypernym_edges.to(self.device)

            hyponym_tokens = self.tokenizer(hyponym_glosses, padding=True, truncation=True, return_tensors="pt", max_length=self.max_length).to(self.device)
            hyponym_edges = hyponym_edges.to(self.device)

            input_embeddings, gnn_vector = self.model(text_input_ids, text_attention_mask, hypernym_tokens, hypernym_edges, hyponym_tokens, hyponym_edges, len(all_candidate_ids))

            for i, (start, end) in enumerate(candidate_id_ranges):
                sentence_id = sentence_ids[i]
                sub_matrix = gnn_vector[start:end].T
                pairwise_similarity = torch.matmul(input_embeddings[i], sub_matrix).tolist()
                candidate_ids = all_candidate_ids[start:end].tolist()

                sorted_pairwise_similarity, sorted_candidate_ids = zip(*sorted(zip(pairwise_similarity, candidate_ids), reverse=True))

                for score, candidate_id in zip(sorted_pairwise_similarity, sorted_candidate_ids):
                    all_scores.append([sentence_id, candidate_id, score])

                top1 = [sentence_id] + list(sorted_candidate_ids[:1])
                all_top1.append(top1)

        with open(f"07-bilinear-epoch_{int(self.id):02d}.txt", 'w') as file:
            writer = csv.writer(file, delimiter=' ')
            writer.writerows(all_top1)

        with open(f"07-bilinear-epoch_{int(self.id):02d}_scores.csv", 'w') as file:
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

def main(ids: List[int], base_model: str, load_files: List[str], batch_size: int, small: bool, ukc_num_neighbors: list[int]):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    _, _, test_dataset, ukc = load_dataset(tokenizer, small, ukc_num_neighbors)
    test_data = prepare_dataloader(test_dataset, batch_size, tokenizer)

    for id, load_file in zip(ids, load_files):
        model = load_model(base_model, load_file, device)
        evaluator = Evaluator(
            id=id,
            model=model,
            device=device,
            validation_data=test_data,
            ukc=ukc,
            tokenizer=tokenizer,
            batch_size=batch_size
        )
        evaluator.validate()

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

    main(args.ids, args.base_model, args.load_files, args.batch_size, args.small, args.ukc_num_neighbors)
