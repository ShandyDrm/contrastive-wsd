import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from transformers import AutoTokenizer, PreTrainedTokenizer

from tqdm.auto import tqdm

import csv, subprocess, re
import numpy as np
import pandas as pd

from model import ContrastiveWSD
from dataset import load_dataset, build_ukc, build_dataframes, build_dataset, TestDataCollator
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
        batch_size: int,
        cosine_similarity: bool,
        gnn_ukc_mapping: dict
    ) -> None:
        self.id = id
        self.model = model.to(device)
        self.device = device
        self.validation_data = validation_data
        self.ukc = ukc
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.cosine_similarity = cosine_similarity
        self.gnn_ukc_mapping = gnn_ukc_mapping

        self.temperature = 0.07

        DEFAULT_MAX_LENGTH = 512
        self.max_length = min(self.tokenizer.model_max_length, DEFAULT_MAX_LENGTH)

    def validate(self):
        all_top1, all_scores = [], []
        for batch in tqdm(self.validation_data):
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

            with torch.no_grad():
                input_embeddings, gnn_vector = self.model(tokenized_sentences, loc, tokenized_glosses, edges, len(candidates_ukc))

            if self.cosine_similarity:
                input_embeddings = F.normalize(input_embeddings, p=2, dim=1)
                gnn_vector = F.normalize(gnn_vector, p=2, dim=1)

            for i, (start, end) in enumerate(candidate_id_ranges):
                sentence_id = sentence_ids[i]
                sub_matrix = gnn_vector[start:end].T
                pairwise_similarity = (torch.matmul(input_embeddings[i], sub_matrix) * np.exp(self.temperature)).tolist()
                candidate_ids = candidates_ukc[start:end].tolist()
                sorted_pairwise_similarity, sorted_candidate_ids = zip(*sorted(zip(pairwise_similarity, candidate_ids), reverse=True))

                for score, candidate_id in zip(sorted_pairwise_similarity, sorted_candidate_ids):
                    ukc_id = self.gnn_ukc_mapping[candidate_id]
                    all_scores.append([sentence_id, ukc_id, score])
                
                best_gnn_id = sorted_candidate_ids[0]
                best_ukc_id = self.gnn_ukc_mapping[best_gnn_id]

                top1 = [sentence_id, best_ukc_id]
                all_top1.append(top1)

        return all_top1, all_scores

def prepare_dataloader(dataset: Dataset, batch_size: int, tokenizer: PreTrainedTokenizer):
    data_collator = TestDataCollator()
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
    parser.add_argument('--seed', default=42, type=int, help='Seed to be used for random number generators')
    parser.add_argument("--epochs", nargs='+', help='Which model epoch to pick')
    parser.add_argument('--base_model', default="google-bert/bert-base-uncased", type=str, help='Base transformers model to use (default: bert-base-uncased)')
    parser.add_argument("--load_files", nargs='+', help='Which file to load')
    parser.add_argument('--batch_size', default=32, type=int, help='Input batch size on each device (default: 32)')
    parser.add_argument('--small', default=False, type=bool, help='For debugging purposes, only process small amounts of data')
    parser.add_argument('--ukc_num_neighbors', type=int, nargs='+', default=[8, 8], help='Number of neighbors to be sampled during training or inference (default: 8 8)')
    parser.add_argument('--hidden_size', type=int, default=256, help="hidden size for the model")

    parser.add_argument('--train_filename', type=str, default='train.complete.data.json')
    parser.add_argument('--eval_filename', type=str, default='eval.complete.data.json')
    parser.add_argument('--test_filename', type=str, default='test.complete.data.json')
    parser.add_argument('--ukc_filename', type=str, default='ukc.csv')
    parser.add_argument('--edges_filename', type=str, default='edges.csv')

    parser.add_argument('--test_dir', type=str, default='test-gold-standard', help='directory where test gold standard files are located')

    parser.add_argument('--gat_heads', type=int, default=1, help="number of multi-head attentions, default=1")
    parser.add_argument('--gat_self_loops', type=bool, default=True, help="enable attention mechanism to see its own features, default=True")
    parser.add_argument('--gat_residual', type=bool, default=False, help="enable residual [f(x) = x + g(x)] to graph attention network, default=False")

    args = parser.parse_args()

    torch.manual_seed(args.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    base_model = args.base_model

    tokenizer_args = {}
    if "roberta" in base_model:
        tokenizer_args["add_prefix_space"] = True

    tokenizer = AutoTokenizer.from_pretrained(base_model, **tokenizer_args)

    ukc, ukc_df, ukc_gnn_mapping, gnn_ukc_mapping = build_ukc(args.ukc_filename, args.edges_filename, args.ukc_num_neighbors)
    train_df, eval_df, test_df = build_dataframes(args.train_filename, args.eval_filename, args.test_filename, ukc_gnn_mapping, args.small)
    train_dataset, eval_dataset, test_dataset = build_dataset(train_df, eval_df, test_df, tokenizer)
    test_data = prepare_dataloader(test_dataset, args.batch_size, tokenizer)

    result_rows = []
    for epoch in args.epochs:
        epoch = int(epoch)
        filename = f"checkpoint_{epoch:02d}.pt"
        for cosine_similarity in [False, True]:
            model = ContrastiveWSD(
                base_model,
                hidden_size=args.hidden_size,
                gat_heads=args.gat_heads,
                gat_self_loops=args.gat_self_loops,
                gat_residual=args.gat_residual
            ).to(device)
            model.eval()

            if filename:
                model.load_state_dict(torch.load(filename, weights_only=True, map_location=torch.device(device)))

            evaluator = Evaluator(
                id=id,
                model=model,
                device=device,
                validation_data=test_data,
                ukc=ukc,
                tokenizer=tokenizer,
                batch_size=args.batch_size,
                cosine_similarity=cosine_similarity,
                gnn_ukc_mapping=gnn_ukc_mapping
            )
            all_top1, all_scores = evaluator.validate()

            result_filename = f"Result-Epoch_{epoch:02d}-CosineSim_{cosine_similarity}.txt"
            with open(result_filename, 'w') as file:
                writer = csv.writer(file, delimiter=' ')
                writer.writerows(all_top1)

            def calculate_scores(result_filename):
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
                        ['java', "Scorer", f"{args.test_dir}/{gold_standard}", result_filename],
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
                    )

                    output = result.stdout
                    key_value_pairs = re.findall('(\\w+)=\\s*([\\d.]+%)', output)

                    result = {}
                    result['Similarity Metric'] = "Cosine Similarity" if cosine_similarity else "Dot Product"
                    result['Criteria'] = title
                    for key, value in key_value_pairs:
                        result[key] = value
                    print(result)

                    result_rows.append(result)
                return result_rows

            result_rows.extend(calculate_scores(result_filename))

            with open(f"Result-Epoch_{epoch:02d}-CosineSim_{cosine_similarity}-Scores.csv", 'w') as file:
                writer = csv.writer(file)
                writer.writerows(all_scores)

    result_df = pd.DataFrame(result_rows)
    result_df.to_csv("result.csv")
