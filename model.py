from transformers import AutoModel

import torch
from torch.nn import Linear, LayerNorm, GELU, Sigmoid, Dropout, Sequential
from torch_geometric.nn import GATv2Conv

class ContrastiveWSD(torch.nn.Module):
    def __init__(self,
                 base_model: str,
                 dropout_p: float = 0.1,
                 gat_heads: int = 1,
                 eps: float = 0.015):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(base_model, output_hidden_states=True)
        for param in self.encoder.parameters():
            param.requires_grad = False

        self.encoder_size = self.encoder.config.hidden_size

        self.word_layers = Sequential(
            Linear(self.encoder_size, self.encoder_size),
            LayerNorm(self.encoder_size),
            GELU(),
            Dropout(p=dropout_p),

            Linear(self.encoder_size, self.encoder_size),
            LayerNorm(self.encoder_size),
            Sigmoid(),
            Dropout(p=dropout_p)
        )

        self.gat1 = GATv2Conv(self.encoder_size, self.encoder_size, heads=gat_heads, concat=False, add_self_loops=False)
        self.post_gat1 = Sequential(
            LayerNorm(self.encoder_size),
            GELU(),
            Dropout(p=dropout_p),
        )

        self.gat2 = GATv2Conv(self.encoder_size, self.encoder_size, heads=gat_heads, concat=False, add_self_loops=False)
        self.post_gat2 = Sequential(
            LayerNorm(self.encoder_size),
            Sigmoid(),
            Dropout(p=dropout_p),
        )

        self.eps = eps

    def find_relevant_subwords_indices(self, word_ids, loc):
        relevant_subwords_idx = []
        word_ids_idx = 1   # skip [CLS]
        while word_ids[word_ids_idx] < loc:
            word_ids_idx += 1

        while word_ids[word_ids_idx] == loc:
            relevant_subwords_idx.append(word_ids_idx)
            word_ids_idx += 1

        return relevant_subwords_idx

    def forward(self, tokenized_sentences, text_positions, tokenized_glosses, edges, labels_size, return_attention_weights=False):
        encoded_inputs = self.encoder(**tokenized_sentences)
        aggregated_hidden_states = sum(encoded_inputs.hidden_states[layer] for layer in [-4, -3, -2, -1])
        refined_embeddings = []
        for idx in range(aggregated_hidden_states.shape[0]):
            token_ids = tokenized_sentences.word_ids(idx)
            position = text_positions[idx]
            selected_subwords = self.find_relevant_subwords_indices(token_ids, position)

            summed_vectors = sum(aggregated_hidden_states[idx, sub_idx, :] for sub_idx in selected_subwords)
            refined_embeddings.append(summed_vectors)

        refined_embeddings = torch.stack(refined_embeddings)
        processed_embeddings = 2 * self.word_layers(refined_embeddings) - 1
        processed_embeddings = refined_embeddings + self.eps * torch.norm(refined_embeddings, p=2, dim=1).unsqueeze(1) * processed_embeddings

        glosses_x0 = self.encoder(**tokenized_glosses).last_hidden_state[:, 0, :]
        glosses_x1 = glosses_x0 + self.post_gat1(self.gat1(glosses_x0, edges))
        glosses_x2 = glosses_x1 + self.post_gat2(self.gat2(glosses_x1, edges))
        glosses_x2 = 2 * glosses_x2 - 1
        glosses_embeddings = glosses_x0 + self.eps * torch.norm(glosses_x0, p=2, dim=1).unsqueeze(1) * glosses_x2

        gnn_vector = glosses_embeddings[:labels_size] # because the subgraphs also include surrounding nodes

        return processed_embeddings, gnn_vector
