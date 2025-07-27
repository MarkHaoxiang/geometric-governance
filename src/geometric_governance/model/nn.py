import copy

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


def make_detached_clone(module: nn.Module) -> nn.Module:
    detached_model = copy.deepcopy(module)
    for param in detached_model.parameters():
        param.requires_grad = False
    return detached_model


class OpinionPassingLayer(MessagePassing):
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

        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * node_dim + edge_dim, edge_dim),
            nn.LayerNorm(edge_dim),
            nn.ReLU(),
            nn.Linear(edge_dim, edge_dim),
        )

    def forward(
        self,
        x,
        edge_index,
        edge_attr,
        candidate_mask: torch.Tensor | None = None,
        train_agent_idxs: torch.Tensor | None = None,
        train_mask: torch.Tensor | None = None,
    ):
        if self.training:
            assert candidate_mask is not None
            assert train_agent_idxs is not None
            assert train_mask is not None
            self.candidate_mask = candidate_mask
            self.train_agent_idxs = train_agent_idxs
            self.train_mask = train_mask
            self.frozen_message_mlp = make_detached_clone(self.message_mlp)
            self.frozen_update_mlp = make_detached_clone(self.update_mlp)
            self.frozen_edge_mlp = make_detached_clone(self.edge_mlp)

        new_x = self.propagate(edge_index, x=x, edge_attr=edge_attr)

        from_node, to_node = edge_index
        edge_features = torch.cat([x[from_node], x[to_node], edge_attr], dim=1)

        new_edge_attr = self.edge_mlp(edge_features)
        if self.training:
            frozen_new_edge_attr = self.frozen_edge_mlp(edge_features)
            new_edge_attr[self.train_mask] = frozen_new_edge_attr[self.train_mask]

        return new_x, new_edge_attr

    def message(self, x_i, x_j, edge_attr):
        # Outward message passing
        msg_features = torch.cat([x_i, x_j, edge_attr], dim=1)
        msg = self.message_mlp(msg_features)
        if self.training:
            frozen_msg = self.frozen_message_mlp(msg_features)
            msg[self.train_mask] = frozen_msg[self.train_mask]

        return msg

    def update(self, aggr_out, x, edge_attr):
        node_update_features = torch.cat([x, aggr_out], dim=-1)

        new_x = self.update_mlp(node_update_features)
        if self.training:
            frozen_new_x = self.frozen_update_mlp(node_update_features)
            out_x = frozen_new_x
            # Accumulate from training nodes
            out_x[self.train_agent_idxs] = new_x[self.train_agent_idxs]
            # Accumulate from candidate nodes
            out_x = torch.where(self.candidate_mask, out_x, new_x)

            new_x[~self.train_agent_idxs] = frozen_new_x[~self.train_agent_idxs]

        return self.update_mlp(node_update_features)
