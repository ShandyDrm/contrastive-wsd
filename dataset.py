import pandas as pd
import csv

import torch
from torch.utils.data import Dataset

from ukc import UKC

from transformers import DataCollatorWithPadding

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

def convert_candidate_ids(mapping, candidate_ids):
    return [mapping[c] for c in candidate_ids]

def parse_candidate_ids(candidate_ids_string):
    ids = candidate_ids_string.replace("'", "").split(" ")
    ids = list(map(int, ids))
    return ids

def read_train_csv_to_dataframe(csv_filename, ukc_gnn_mapping, small=False):
    df = pd.read_csv(csv_filename)
    df["text"] = df.apply(lambda row: f"{row['word']} [SEP] {row['sentence']}", axis=1)
    df["candidate_ids"] = df.apply(lambda row: parse_candidate_ids(row["candidate_ids"]), axis=1)
    df['gnn_id'] = df.apply(lambda row: ukc_gnn_mapping[row['ukc_id']], axis=1)
    df['gnn_candidate_ids'] = df.apply(lambda row: convert_candidate_ids(ukc_gnn_mapping, row["candidate_ids"]), axis=1)
    df = df[["id", "lemma", "pos", "text", "gnn_id", "gnn_candidate_ids", "target_index"]]

    if small:
        df = df.loc[:100]

    return df

def read_test_csv_to_dataframe(csv_filename, ukc_gnn_mapping, small=False):
    df = pd.read_csv(csv_filename)
    df["text"] = df.apply(lambda row: f"{row['word']} [SEP] {row['sentence']}", axis=1)
    df["candidate_ids"] = df.apply(lambda row: parse_candidate_ids(row["candidate_ids"]), axis=1)
    df['gnn_candidate_ids'] = df.apply(lambda row: convert_candidate_ids(ukc_gnn_mapping, row["candidate_ids"]), axis=1)
    df = df[["id", "text", "gnn_candidate_ids"]]

    if small:
        df = df.loc[:100]

    return df

class TrainDataset(Dataset):
    def __init__(self, lemmas, pos, texts, labels, tokenizer, max_length):
        self.lemmas = lemmas
        self.pos = pos
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        # Tokenize the text
        inputs = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        return {
            "lemmas": self.lemmas[idx],
            "pos": self.pos[idx],
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }
    
class TestDataset(Dataset):
    def __init__(self, ids, texts, candidate_ids, tokenizer, max_length):
        self.ids = ids
        self.texts = texts
        self.candidate_ids = candidate_ids
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        # Tokenize the text
        inputs = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        return {
            "ids": self.ids[idx],
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "candidate_ids": self.candidate_ids[idx],
        }

class TrainDataCollator(DataCollatorWithPadding):
    def __call__(self, features):
        filtered_features = [
            {key: value for key, value in feature.items() if key in ["input_ids", "attention_mask", "token_type_ids", "labels"]}
            for feature in features
        ]
        
        batch = super().__call__(filtered_features)

        lemmas = []
        pos = []
        
        for feature in features:
            lemmas.append(feature["lemmas"])
            pos.append(feature["pos"])

        batch["lemmas"] = lemmas
        batch["pos"] = pos
        return batch

class TestDataCollator(DataCollatorWithPadding):
    def __call__(self, features):
        filtered_features = [
            {key: value for key, value in feature.items() if key in ["input_ids", "attention_mask", "token_type_ids", "labels"]}
            for feature in features
        ]
        
        batch = super().__call__(filtered_features)

        sentence_ids = []
        all_candidate_ids = []
        candidate_id_ranges = []
        current_index = 0
        
        for feature in features:
            sentence_ids.append(feature["ids"])

            candidates = feature["candidate_ids"]        
            all_candidate_ids.extend(candidates)

            candidate_id_ranges.append((current_index, current_index + len(candidates)))
            current_index += len(candidates)

        batch["ids"] = sentence_ids
        batch["all_candidate_ids"] = torch.tensor(all_candidate_ids)
        batch["candidate_id_ranges"] = candidate_id_ranges
        return batch

def load_dataset(tokenizer, small=False, ukc_num_neighbors=[8, 8]):
    ukc_df = pd.read_csv("ukc.csv")
    ukc_gnn_mapping = dict(zip(ukc_df['ukc_id'], ukc_df['gnn_id']))    

    train_df = read_train_csv_to_dataframe("SemCor_Train_New.csv", ukc_gnn_mapping, small)
    validate_df = read_train_csv_to_dataframe("SemCor_Validate_New.csv", ukc_gnn_mapping, small)
    test_df = read_test_csv_to_dataframe("TestALL_Converted.csv", ukc_gnn_mapping, small)

    DEFAULT_MAX_LENGTH = 512
    max_length = min(tokenizer.model_max_length, DEFAULT_MAX_LENGTH)

    train_dataset = TrainDataset(train_df["lemma"], train_df["pos"], list(train_df["text"]), train_df["gnn_id"], tokenizer, max_length)
    validation_dataset = TrainDataset(train_df["lemma"], train_df["pos"], list(validate_df["text"]), validate_df["gnn_id"], tokenizer, max_length)
    test_dataset = TestDataset(test_df["id"], list(test_df["text"]), test_df["gnn_candidate_ids"], tokenizer, max_length)

    edges = process_edges("edges.csv", ukc_gnn_mapping)
    ukc = UKC(ukc_df, edges, ukc_num_neighbors)

    print("End data preprocessing", flush=True)

    return train_dataset, validation_dataset, test_dataset, ukc
