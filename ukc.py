import torch

from torch_geometric.data import Data
from torch_geometric.sampler import NeighborSampler, NodeSamplerInput

from typing import List

class UKC():
    def __init__(self, ukc_df, edges, num_neighbors=[8, 8], no_gloss: bool=False):
        self.ukc_df = ukc_df

        data_len = ukc_df["gnn_id"].max()
        x = torch.arange(data_len, dtype=torch.long)

        edge_index = torch.tensor(edges, dtype=torch.long).T

        pyg_data = Data(x=x, edge_index=edge_index).contiguous()
        self.sampler = NeighborSampler(data=pyg_data, num_neighbors=num_neighbors)

        self.no_gloss = no_gloss

    def sample(self, node_ids: torch.Tensor) -> tuple[List[str], torch.Tensor, List[List[int]]|None] :
        node_sampler_input = NodeSamplerInput(input_id=None, node=node_ids)

        samples = self.sampler.sample_from_nodes(node_sampler_input)
        sample_nodes = samples.node
        sample_edges = torch.stack((samples.row, samples.col), dim=0).to(torch.long)

        if self.no_gloss:
            sample_lemmas = self.ukc_df.iloc[sample_nodes]["lemmas"]

            all_lemmas = []
            counter = []
            counter_init = 0
            for lemmas in sample_lemmas:
                all_lemmas.extend(lemmas)
                counter.append([counter_init, counter_init + len(lemmas)])
                counter_init += len(lemmas)
            
            return all_lemmas, sample_edges, counter
        else:
            sample_glosses = list(self.ukc_df.iloc[sample_nodes]["gloss"])
            return sample_glosses, sample_edges
