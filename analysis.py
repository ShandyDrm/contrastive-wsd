import marimo

__generated_with = "0.10.17"
app = marimo.App(width="medium")


@app.cell
def _():
    import torch
    from torch.utils.data import Dataset, DataLoader
    import torch.nn.functional as F

    from transformers import AutoTokenizer, PreTrainedTokenizer

    from tqdm.auto import tqdm

    import csv
    from typing import List
    import numpy as np

    from model import ContrastiveWSD
    from dataset import load_dataset, TestDataCollator
    from ukc import UKC
    return (
        AutoTokenizer,
        ContrastiveWSD,
        DataLoader,
        Dataset,
        F,
        List,
        PreTrainedTokenizer,
        TestDataCollator,
        UKC,
        csv,
        load_dataset,
        np,
        torch,
        tqdm,
    )


@app.cell
def _(torch):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return (device,)


@app.cell
def _(DataLoader, Dataset, PreTrainedTokenizer, TestDataCollator):
    def prepare_dataloader(dataset: Dataset, batch_size: int, tokenizer: PreTrainedTokenizer):
        data_collator = TestDataCollator(tokenizer=tokenizer)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            pin_memory=True,
            shuffle=False,
            collate_fn=data_collator
        )
    return (prepare_dataloader,)


@app.cell
def _(AutoTokenizer, load_dataset, prepare_dataloader):
    base_model = "bert-base-uncased"
    small = False
    ukc_num_neighbors = [4,4,4,4]
    batch_size = 8

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    train_dataset, _, test_dataset, ukc = load_dataset(tokenizer, small, ukc_num_neighbors)
    test_data = prepare_dataloader(test_dataset, batch_size, tokenizer)
    return (
        base_model,
        batch_size,
        small,
        test_data,
        test_dataset,
        tokenizer,
        train_dataset,
        ukc,
        ukc_num_neighbors,
    )


@app.cell
def _(train_dataset):
    len(train_dataset)
    return


@app.cell
def _(ContrastiveWSD, base_model, device, torch):
    def load_model(base_model: str, load_file: str, device: str):
        model = ContrastiveWSD(base_model, device=device, freeze_concept_encoder=True)
        if load_file:
            model.load_state_dict(torch.load(load_file, weights_only=True, map_location=torch.device(device)))
        return model

    load_file = "checkpoint_01.pt"
    model = load_model(base_model, load_file, device)
    return load_file, load_model, model


@app.cell
def _(test_data):
    test_iter = iter(test_data)
    next(test_iter)
    batch = next(test_iter)
    return batch, test_iter


@app.cell
def _(batch, device):
    sentence_ids = batch["ids"]
    text_input_ids = batch["input_ids"].to(device)
    text_attention_mask = batch["attention_mask"].to(device)
    all_candidate_ids = batch["all_candidate_ids"]
    candidate_id_ranges = batch["candidate_id_ranges"]
    return (
        all_candidate_ids,
        candidate_id_ranges,
        sentence_ids,
        text_attention_mask,
        text_input_ids,
    )


@app.cell
def _(sentence_ids):
    sentence_ids
    return


@app.cell
def _(all_candidate_ids, device, tokenizer, ukc):
    DEFAULT_MAX_LENGTH = 512
    max_length = min(tokenizer.model_max_length, DEFAULT_MAX_LENGTH)

    glosses, edges = ukc.sample(all_candidate_ids)
    tokenized_glosses = tokenizer(glosses, padding=True, truncation=True, return_tensors="pt", max_length=max_length).to(device)
    edges = edges.to(device)
    return DEFAULT_MAX_LENGTH, edges, glosses, max_length, tokenized_glosses


@app.cell
def _(
    F,
    all_candidate_ids,
    edges,
    model,
    text_attention_mask,
    text_input_ids,
    tokenized_glosses,
):
    input_embeddings, gnn_vector = model(text_input_ids, text_attention_mask, tokenized_glosses, edges, len(all_candidate_ids))

    input_embeddings = F.normalize(input_embeddings, p=2, dim=1)
    gnn_vector = F.normalize(gnn_vector, p=2, dim=1)
    return gnn_vector, input_embeddings


@app.cell
def _(
    all_candidate_ids,
    candidate_id_ranges,
    gnn_vector,
    input_embeddings,
    np,
    sentence_ids,
    torch,
):
    temperature = 0.07
    for i, (start, end) in enumerate(candidate_id_ranges):
        sentence_id = sentence_ids[i]
        sub_matrix = gnn_vector[start:end].T
        pairwise_similarity = (torch.matmul(input_embeddings[i], sub_matrix) * np.exp(temperature)).tolist()
        candidate_ids = all_candidate_ids[start:end].tolist()

        sorted_pairwise_similarity, sorted_candidate_ids = zip(*sorted(zip(pairwise_similarity, candidate_ids), reverse=True))
        print(sorted_pairwise_similarity, sorted_candidate_ids)
    return (
        candidate_ids,
        end,
        i,
        pairwise_similarity,
        sentence_id,
        sorted_candidate_ids,
        sorted_pairwise_similarity,
        start,
        sub_matrix,
        temperature,
    )


if __name__ == "__main__":
    app.run()
