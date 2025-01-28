from transformers import AutoModel

import torch
from torch.nn import Dropout, BatchNorm1d, GELU, Bilinear, Linear

from torch_geometric.nn import Sequential, GATv2Conv


class ContrastiveWSD(torch.nn.Module):
    def __init__(self, base_model: str, device="cpu", freeze_concept_encoder=True, dropout_p=0.1):
        super().__init__()

        self.word_encoder = AutoModel.from_pretrained(base_model).to(device)
        self.concept_encoder = AutoModel.from_pretrained(base_model).to(device)
        self.gnn_hidden_size = self.concept_encoder.config.hidden_size

        if freeze_concept_encoder:
            for param in self.concept_encoder.parameters():
                param.requires_grad = False

        def build_gnn(hidden_size: int, dropout_p: float):
            return Sequential('x, edge_index', [
                (BatchNorm1d(hidden_size), 'x -> x'),
                GELU(),
                Dropout(p=dropout_p),
            
                (GATv2Conv(hidden_size, hidden_size), 'x, edge_index -> x'),
                BatchNorm1d(hidden_size),
                GELU(),
                Dropout(p=dropout_p),
            
                (GATv2Conv(hidden_size, hidden_size), 'x, edge_index -> x'),
                BatchNorm1d(hidden_size),
                GELU(),
                Dropout(p=dropout_p),    
            ])
        
        self.hypernym_gnn = build_gnn(self.gnn_hidden_size, dropout_p).to(device)
        self.hyponym_gnn = build_gnn(self.gnn_hidden_size, dropout_p).to(device)

        self.bilinear = Sequential('x, y', [
            (Bilinear(self.gnn_hidden_size, self.gnn_hidden_size, self.gnn_hidden_size), 'x, y -> z'),
            BatchNorm1d(self.gnn_hidden_size),
            GELU(),
            Dropout(p=dropout_p),

            Linear(self.gnn_hidden_size, self.gnn_hidden_size),
            BatchNorm1d(self.gnn_hidden_size),
            GELU(),
            Dropout(p=dropout_p),
        ])
        
    def forward(self, text_input_ids, text_attention_mask, hypernym_tokens, hypernym_edges, hyponym_tokens, hyponym_edges, labels_size):
        input_embeddings = self.word_encoder(text_input_ids, text_attention_mask).last_hidden_state[:, 0, :]
        # expected shape: [n_sentences, self.gnn_hidden_size]

        hypernym_embeddings = self.concept_encoder(**hypernym_tokens).last_hidden_state[:, 0, :]
        hypernym_gnn_output = self.hypernym_gnn(hypernym_embeddings, hypernym_edges)

        hyponym_embeddings = self.concept_encoder(**hyponym_tokens).last_hidden_state[:, 0, :]
        hyponym_gnn_output = self.hyponym_gnn(hyponym_embeddings, hyponym_edges)

        bilinear_vectors = self.bilinear(hypernym_gnn_output[:labels_size], hyponym_gnn_output[:labels_size])
        return input_embeddings, bilinear_vectors
