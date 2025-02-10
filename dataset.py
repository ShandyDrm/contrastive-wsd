from typing import Tuple

import pandas as pd
import csv

import torch
from torch.utils.data import Dataset

from ukc import UKC

from transformers import DataCollatorWithPadding, PreTrainedTokenizer

def process_edges(filename, mapping):
    print("start processing edges", flush=True)
    with open(filename, mode='r') as file:
        reader = csv.reader(file)
        edges = [row for row in reader]

    edges = edges[1:] # clear header
    edges = [[int(x), int(y)] for x, y in edges]
    print("end processing edges\n", flush=True)

    print("start mapping edges", flush=True)
    # map edges data from ukc to gnn
    mapped_edges = []
    for source, target in edges:
        source = int(source)
        target = int(target)
        if source not in mapping or target not in mapping:
            # skip if not in mapping
            # i hope this does not bite me back later
            continue
        mapped_edges.append([mapping[source], mapping[target]])
    print("end mapping edges\n", flush=True)
    return mapped_edges

class TrainDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.id = list(df["id"])
        self.lemma = list(df["lemma"])
        self.pos = list(df["pos"])
        self.loc = list(df["loc"])
        self.sentence = list(df["sentence"])
        self.answers_ukc = list(df["answers_ukc"])

    def __len__(self):
        return len(self.id)

    def __getitem__(self, idx):
        return {
            "id": self.id[idx],
            "lemma": self.lemma[idx],
            "pos": self.pos[idx],
            "loc": self.loc[idx],
            "sentence": self.sentence[idx],
            "answers_ukc": self.answers_ukc[idx],
        }

class TestDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.id = list(df["id"])
        self.lemma = list(df["lemma"])
        self.pos = list(df["pos"])
        self.loc = list(df["loc"])
        self.sentence = list(df["sentence"])
        self.candidates_ukc = list(df["candidates_ukc"])

        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.id)

    def __getitem__(self, idx):
        return {
            "id": self.id[idx],
            "lemma": self.lemma[idx],
            "pos": self.pos[idx],
            "loc": self.loc[idx],
            "sentence": self.sentence[idx],
            "candidates_ukc": self.candidates_ukc[idx],
        }
    
class TrainDataCollator():
    def __call__(self, features):
        id, lemma, pos, loc, sentence, answers_ukc, answers_ukc_flat, answer_indices = [], [], [], [], [], [], [], []

        for feature in features:
            id.append(feature["id"])
            lemma.append(feature["lemma"])
            pos.append(feature["pos"])
            loc.append(feature["loc"])
            sentence.append(feature["sentence"])
            answers_ukc.append(feature["answers_ukc"])

        return {
            "id": id,
            "lemma": lemma,
            "pos": pos,
            "loc": torch.tensor(loc, dtype=int),
            "sentence": sentence,
            "answers_ukc": torch.tensor(answers_ukc, dtype=int),
        }

class TestDataCollator():
    def __call__(self, features):
        id, lemma, pos, loc, sentence, candidates_ukc, candidate_id_ranges = [], [], [], [], [], [], []

        current_index = 0
        for feature in features:
            id.append(feature["id"])
            lemma.append(feature["lemma"])
            pos.append(feature["pos"])
            loc.append(feature["loc"])
            sentence.append(feature["sentence"])

            candidates_ukc.extend(feature["candidates_ukc"])

            candidate_id_ranges.append((current_index, current_index + len(feature["candidates_ukc"])))
            current_index += len(feature["candidates_ukc"])

        return {
            "id": id,
            "lemma": lemma,
            "pos": pos,
            "loc": loc,
            "sentence": sentence,
            "candidates_ukc": candidates_ukc,
            "candidate_id_ranges": candidate_id_ranges
        }

def convert_ukc_val_to_gnn(row, key, ukc_gnn_mapping):
    ukc_id = row[key]
    if ukc_id in ukc_gnn_mapping:
        return ukc_gnn_mapping[ukc_id]
    else:
        return None

def convert_ukc_to_gnn(row, key, ukc_gnn_mapping):
    lst = row[key]
    gnn_ids = []
    for i in lst:
        if i in ukc_gnn_mapping:
            gnn_ids.append(ukc_gnn_mapping[i])
        else:
            return None

    return gnn_ids

def parse_train_file(train_filename: str, ukc_gnn_mapping: dict, small: bool=False):
    train_df = pd.read_json(train_filename)
    train_df["answers_ukc"] = train_df.apply(lambda row: convert_ukc_val_to_gnn(row, "answers", ukc_gnn_mapping), axis=1)
    train_df = train_df[train_df["answers_ukc"].notnull()]

    if small:
        return train_df[:100]
    return train_df

def parse_test_file(test_filename: str, ukc_gnn_mapping: dict, small: bool=False):
    test_df = pd.read_json(test_filename)
    test_df["candidates_ukc"] = test_df.apply(lambda row: convert_ukc_to_gnn(row, "candidates", ukc_gnn_mapping), axis=1)
    test_df = test_df[test_df["candidates_ukc"].notnull()]

    if small:
        return test_df[:100]
    return test_df

def build_ukc(ukc_gloss_filename: str,
              ukc_lemmas_filename: str,
              edges_file: str,
              ukc_num_neighbors: list=[8, 8],
              no_gloss: bool=False) -> Tuple[UKC, pd.DataFrame, dict, dict]:
    
    if no_gloss:
        ukc_df = build_ukc_df_lemmas_only(ukc_lemmas_filename)
    else:
        ukc_df = pd.read_csv(ukc_gloss_filename)

    ukc_gnn_mapping = dict(zip(ukc_df['ukc_id'], ukc_df['gnn_id']))
    gnn_ukc_mapping = dict(zip(ukc_df['gnn_id'], ukc_df['ukc_id']))
    edges = process_edges(edges_file, ukc_gnn_mapping)
    ukc = UKC(ukc_df, edges, ukc_num_neighbors, no_gloss)

    return ukc, ukc_df, ukc_gnn_mapping, gnn_ukc_mapping

def build_ukc_df_lemmas_only(ukc_lemmas_filename: str) -> pd.DataFrame:
    # build mapping ukc_id -> [gnn_id, List[lemmas]]
    mapping = {}
    with open(ukc_lemmas_filename, "r") as f:
        next(f)   # skip header
        for row in f:
            ukc_id, gnn_id, lemma = row.strip().split(",")
            ukc_id = int(ukc_id)
            gnn_id = int(gnn_id)
            if ukc_id in mapping:
                mapping[ukc_id][1].append(lemma)
            else:
                mapping[ukc_id] = [gnn_id, [lemma]]
    
    # Dict(ukc_id -> [gnn_id, list of lemmas]) -> List(ukc_id, gnn_id, List[lemmas])
    all_rows = []
    for _ukc_id in mapping:
        _gnn_id, _lemmas = mapping[_ukc_id]
        all_rows.append([_ukc_id, _gnn_id, _lemmas])

    return pd.DataFrame(all_rows, columns=["ukc_id", "gnn_id", "lemmas"])

def build_dataframes(train_filename: str,
                     eval_filename: str,
                     test_filename: str,
                     ukc_gnn_mapping: dict,
                     small: bool=False
                     ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = parse_train_file(train_filename, ukc_gnn_mapping, small)
    eval_df = parse_test_file(eval_filename, ukc_gnn_mapping) # eval dataset is already small
    test_df = parse_test_file(test_filename, ukc_gnn_mapping, small)
    return train_df, eval_df, test_df

def build_dataset(train_df: pd.DataFrame,
                  eval_df: pd.DataFrame,
                  test_df: pd.DataFrame,
                  tokenizer: PreTrainedTokenizer
                  ) -> Tuple[Dataset, Dataset, Dataset]:
    train_dataset = TrainDataset(train_df, tokenizer)
    eval_dataset = TestDataset(eval_df, tokenizer)
    test_dataset = TestDataset(test_df, tokenizer)
    return train_dataset, eval_dataset, test_dataset

def load_dataset(
        tokenizer: PreTrainedTokenizer,
        train_filename: str,
        eval_filename: str,
        test_filename: str,
        small: bool=False,
        ukc_num_neighbors: list=[8, 8]
    ) -> Tuple[TrainDataset, TestDataset, TestDataset, UKC]:

    ukc_df = pd.read_csv("ukc.csv")
    ukc_gnn_mapping = dict(zip(ukc_df['ukc_id'], ukc_df['gnn_id']))

    train_df = parse_train_file(train_filename, ukc_gnn_mapping, small)
    eval_df = parse_test_file(eval_filename, ukc_gnn_mapping, small)
    test_df = parse_test_file(test_filename, ukc_gnn_mapping, small)

    train_dataset = TrainDataset(train_df, tokenizer)
    eval_dataset = TestDataset(eval_df, tokenizer)
    test_dataset = TestDataset(test_df, tokenizer)

    edges = process_edges("edges.csv", ukc_gnn_mapping)
    ukc = UKC(ukc_df, edges, ukc_num_neighbors)

    print("End data preprocessing", flush=True)

    return train_dataset, eval_dataset, test_dataset, ukc
