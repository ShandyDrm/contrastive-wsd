import torch

from torch_geometric.data import Data
from torch_geometric.sampler import NeighborSampler, NodeSamplerInput

from typing import List

class UKC():
    def __init__(self, ukc_df, edges, num_neighbors=[8, 8]):
        self.ukc_df = ukc_df

        hypernym_edge_index = torch.Tensor(edges).T.to(torch.long)
        hypernym_pyg_data = Data(x=torch.arange(len(ukc_df)).to(torch.long), edge_index=hypernym_edge_index).contiguous()
        self.hypernym_sampler = NeighborSampler(data=hypernym_pyg_data, num_neighbors=num_neighbors)

        hyponym_edges = [[y, x] for x, y in edges]
        hyponym_edge_index = torch.Tensor(hyponym_edges).T.to(torch.long)
        hyponym_pyg_data = Data(x=torch.arange(len(ukc_df)).to(torch.long), edge_index=hyponym_edge_index).contiguous()
        self.hyponym_sampler = NeighborSampler(data=hyponym_pyg_data, num_neighbors=num_neighbors)

    def sample(self, node_ids: torch.Tensor) -> tuple[List[str], torch.Tensor, List[str], torch.Tensor] :
        node_sampler_input = NodeSamplerInput(input_id=None, node=node_ids)

        hypernym_samples = self.hypernym_sampler.sample_from_nodes(node_sampler_input)
        hypernym_nodes = hypernym_samples.node
        hypernym_glosses = list(self.ukc_df.iloc[hypernym_nodes]["gloss"])
        hypernym_edges = torch.stack((hypernym_samples.row, hypernym_samples.col), dim=0).to(torch.long)

        hyponym_samples = self.hyponym_sampler.sample_from_nodes(node_sampler_input)
        hyponym_nodes = hyponym_samples.node
        hyponym_glosses = list(self.ukc_df.iloc[hyponym_nodes]["gloss"])
        hyponym_edges = torch.stack((hyponym_samples.row, hyponym_samples.col), dim=0).to(torch.long)

        return hypernym_glosses, hypernym_edges, hyponym_glosses, hyponym_edges
