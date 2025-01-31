from transformers import AutoModel

import torch
from torch.nn import Linear, LayerNorm, GELU, Dropout, Sequential
from torch_geometric.nn import GATv2Conv

class ContrastiveWSD(torch.nn.Module):
    def __init__(self, base_model: str, hidden_size: int=256, device="cpu", freeze_concept_encoder=True, dropout_p=0.1):
        super().__init__()

        self.word_encoder = AutoModel.from_pretrained(base_model).to(device)
        self.concept_encoder = AutoModel.from_pretrained(base_model).to(device)

        self.encoder_size = self.concept_encoder.config.hidden_size
        self.hidden_size = hidden_size

        if freeze_concept_encoder:
            for param in self.concept_encoder.parameters():
                param.requires_grad = False

        def get_norm_gelu_dropout(hidden_size, dropout_p, device):
            return Sequential(
                LayerNorm(hidden_size),
                GELU(),
                Dropout(p=dropout_p)
            ).to(device)
        
        self.word_linear = Linear(self.encoder_size, self.hidden_size)
        self.word_norm_gelu_dropout1 = get_norm_gelu_dropout(self.hidden_size, dropout_p, device)

        self.concept_linear = Linear(self.encoder_size, self.hidden_size)
        self.concept_norm_gelu_dropout1 = get_norm_gelu_dropout(self.hidden_size, dropout_p, device)

        self.gat2 = GATv2Conv(self.hidden_size, self.hidden_size).to(device)
        self.concept_norm_gelu_dropout2 = get_norm_gelu_dropout(self.hidden_size, dropout_p, device)

        self.gat3 = GATv2Conv(self.hidden_size, self.hidden_size).to(device)
        self.concept_norm_gelu_dropout3 = get_norm_gelu_dropout(self.hidden_size, dropout_p, device)

    def forward(self, text_input_ids, text_attention_mask, tokenized_glosses, edges, labels_size, return_attention_weights=False):
        def gat_forward(gat_layer: GATv2Conv, embeddings, edges, return_attention_weights):
            if return_attention_weights:
                return gat_layer.forward(embeddings, edges, return_attention_weights=True)
            else:
                gat_embeddings = gat_layer(embeddings, edges)
                return gat_embeddings, None

        input_embeddings = self.word_encoder(text_input_ids, text_attention_mask).last_hidden_state[:, 0, :]
        input_embeddings = self.word_linear(input_embeddings)
        input_embeddings = self.word_norm_gelu_dropout1(input_embeddings)

        glosses_embeddings = self.concept_encoder(**tokenized_glosses).last_hidden_state[:, 0, :]
        glosses_embeddings = self.concept_linear(glosses_embeddings)
        glosses_embeddings = self.concept_norm_gelu_dropout1(glosses_embeddings)

        glosses_embeddings, attention_weights_gat2 = gat_forward(self.gat2, glosses_embeddings, edges, return_attention_weights)
        glosses_embeddings = self.concept_norm_gelu_dropout2(glosses_embeddings)

        glosses_embeddings, attention_weights_gat3 = gat_forward(self.gat3, glosses_embeddings, edges, return_attention_weights)            
        glosses_embeddings = self.concept_norm_gelu_dropout3(glosses_embeddings)

        gnn_vector = glosses_embeddings[:labels_size] # because the subgraphs also include surrounding nodes

        if return_attention_weights:
            return input_embeddings, gnn_vector, attention_weights_gat2, attention_weights_gat3
        else:
            return input_embeddings, gnn_vector
