import torch

from torch_geometric.data import Data
from torch_geometric.sampler import NeighborSampler, NodeSamplerInput

from typing import List

class UKC():
    def __init__(self, ukc_df, edges, num_neighbors=[8, 8]):
        self.ukc_df = ukc_df

        data_len = ukc_df["gnn_id"].max()
        x = torch.arange(data_len, dtype=torch.long)

        edge_index = torch.tensor(edges, dtype=torch.long).T

        pyg_data = Data(x=x, edge_index=edge_index).contiguous()
        self.sampler = NeighborSampler(data=pyg_data, num_neighbors=num_neighbors)

    def sample(self, node_ids: torch.Tensor) -> tuple[List[str], torch.Tensor, List[str], torch.Tensor] :
        node_sampler_input = NodeSamplerInput(input_id=None, node=node_ids)

        samples = self.sampler.sample_from_nodes(node_sampler_input)
        sample_nodes = samples.node
        sample_glosses = list(self.ukc_df.iloc[sample_nodes]["gloss"])
        sample_edges = torch.stack((samples.row, samples.col), dim=0).to(torch.long)

        return sample_glosses, sample_edges
