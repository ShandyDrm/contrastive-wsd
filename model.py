from transformers import AutoModel

import torch
from torch.nn import LayerNorm, GELU, Dropout

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

        self.concept_gnn = Sequential('x, edge_index', [
                (LayerNorm(self.gnn_hidden_size), 'x -> x'),
                GELU(),
                Dropout(p=dropout_p),

                (GATv2Conv(self.gnn_hidden_size, self.gnn_hidden_size), 'x, edge_index -> x'),
                LayerNorm(self.gnn_hidden_size),
                GELU(),
                Dropout(p=dropout_p),

                (GATv2Conv(self.gnn_hidden_size, self.gnn_hidden_size), 'x, edge_index -> x'),
                LayerNorm(self.gnn_hidden_size),
                GELU(),
                Dropout(p=dropout_p),
            ]).to(device)

    def forward(self, text_input_ids, text_attention_mask, tokenized_glosses, edges, labels_size):
        input_embeddings = self.word_encoder(text_input_ids, text_attention_mask).last_hidden_state[:, 0, :]
        # expected shape: [n_sentences, self.gnn_hidden_size]

        glosses_embeddings = self.concept_encoder(**tokenized_glosses).last_hidden_state[:, 0, :]
        gnn_vector = self.concept_gnn(glosses_embeddings, edges) # expected shape: [n_glosses, self.gnn_hidden_size]

        gnn_vector = gnn_vector[:labels_size] # because the subgraphs also include surrounding nodes

        return input_embeddings, gnn_vector
