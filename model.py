from transformers import AutoModel, AutoConfig

import torch
from torch.nn import Linear, LayerNorm, GELU, Dropout, Sequential
from torch_geometric.nn import GATv2Conv

class ContrastiveWSD(torch.nn.Module):
    def __init__(self,
                 base_model: str,
                 hidden_size: int = 256,
                 dropout_p: float = 0.1,
                 gat_heads: int = 1,
                 gat_self_loops: bool = True,
                 gat_residual: bool = False):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(base_model, output_hidden_states=True)

        config = AutoConfig.from_pretrained(base_model)
        self.is_encoder_decoder = config.is_encoder_decoder

        self.encoder_size = self.encoder.config.hidden_size
        self.hidden_size = hidden_size

        def get_norm_gelu_dropout(hidden_size, dropout_p):
            return Sequential(
                LayerNorm(hidden_size),
                GELU(),
                Dropout(p=dropout_p)
            )

        self.word_linear = Linear(self.encoder_size, self.hidden_size)
        self.word_norm_gelu_dropout1 = get_norm_gelu_dropout(self.hidden_size, dropout_p)

        self.concept_linear = Linear(self.encoder_size, self.hidden_size)
        self.concept_norm_gelu_dropout1 = get_norm_gelu_dropout(self.hidden_size, dropout_p)

        self.gat2 = GATv2Conv(self.hidden_size, self.hidden_size, heads=gat_heads, concat=False, add_self_loops=gat_self_loops, residual=gat_residual)
        self.concept_norm_gelu_dropout2 = get_norm_gelu_dropout(self.hidden_size, dropout_p)

        self.gat3 = GATv2Conv(self.hidden_size, self.hidden_size, heads=gat_heads, concat=False, add_self_loops=gat_self_loops, residual=gat_residual)
        self.concept_norm_gelu_dropout3 = get_norm_gelu_dropout(self.hidden_size, dropout_p)

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
        def gat_forward(gat_layer: GATv2Conv, embeddings, edges, return_attention_weights):
            if return_attention_weights:
                return gat_layer.forward(embeddings, edges, return_attention_weights=True)
            else:
                gat_embeddings = gat_layer(embeddings, edges)
                return gat_embeddings, None

        encoded_inputs = self.encoder(**tokenized_sentences)

        if self.is_encoder_decoder:
            aggregated_hidden_states = sum(encoded_inputs.decoder_hidden_states[layer] for layer in [-4, -3, -2, -1])
        else:
            aggregated_hidden_states = sum(encoded_inputs.hidden_states[layer] for layer in [-4, -3, -2, -1])

        refined_embeddings = []
        for idx in range(aggregated_hidden_states.shape[0]):
            token_ids = tokenized_sentences.word_ids(idx)
            position = text_positions[idx]
            selected_subwords = self.find_relevant_subwords_indices(token_ids, position)

            summed_vectors = sum(aggregated_hidden_states[idx, sub_idx, :] for sub_idx in selected_subwords)
            refined_embeddings.append(summed_vectors)

        refined_embeddings = torch.stack(refined_embeddings)
        refined_embeddings = self.word_linear(refined_embeddings)
        refined_embeddings = self.word_norm_gelu_dropout1(refined_embeddings)

        glosses_embeddings = self.encoder(**tokenized_glosses).last_hidden_state[:, 0, :]
        glosses_embeddings = self.concept_linear(glosses_embeddings)
        glosses_embeddings = self.concept_norm_gelu_dropout1(glosses_embeddings)

        glosses_embeddings, attention_weights_gat2 = gat_forward(self.gat2, glosses_embeddings, edges, return_attention_weights)
        glosses_embeddings = self.concept_norm_gelu_dropout2(glosses_embeddings)

        glosses_embeddings, attention_weights_gat3 = gat_forward(self.gat3, glosses_embeddings, edges, return_attention_weights)
        glosses_embeddings = self.concept_norm_gelu_dropout3(glosses_embeddings)

        gnn_vector = glosses_embeddings[:labels_size] # because the subgraphs also include surrounding nodes

        if return_attention_weights:
            return refined_embeddings, gnn_vector, attention_weights_gat2, attention_weights_gat3
        else:
            return refined_embeddings, gnn_vector
