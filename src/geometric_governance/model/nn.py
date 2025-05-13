import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing


class MessagePassingLayer(MessagePassing):
    def __init__(
        self,
        edge_dim: int = 1,
        node_dim: int = 32,
        aggr: str = "sum",
    ):
        super().__init__(aggr=aggr)

        self.message_mlp = nn.Sequential(
            nn.Linear(2 * node_dim + edge_dim, node_dim),
            nn.LayerNorm(node_dim),
            nn.ReLU(),
            nn.Linear(node_dim, node_dim),
        )

        self.update_mlp = nn.Sequential(
            nn.Linear(2 * node_dim, node_dim),
            nn.LayerNorm(node_dim),
            nn.ReLU(),
            nn.Linear(node_dim, node_dim),
        )

        self.edge_utility_mlp = nn.Sequential(
            nn.Linear(2 * node_dim + edge_dim, edge_dim),
            nn.LayerNorm(edge_dim),
            nn.ReLU(),
            nn.Linear(edge_dim, edge_dim),
        )

    def forward(self, x, edge_index, edge_attr):
        new_x = self.propagate(edge_index, x=x, edge_attr=edge_attr)

        from_node, to_node = edge_index
        edge_features = torch.cat([x[from_node], x[to_node], edge_attr], dim=1)
        new_edge_attr = self.edge_utility_mlp(edge_features)

        return new_x, new_edge_attr

    def message(self, x_i, x_j, edge_attr):
        msg_features = torch.cat([x_i, x_j, edge_attr], dim=1)
        return self.message_mlp(msg_features)

    def update(self, aggr_out, x, edge_attr):
        node_update_features = torch.cat([x, aggr_out], dim=-1)
        return self.update_mlp(node_update_features)

