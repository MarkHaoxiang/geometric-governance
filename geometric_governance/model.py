import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data
from torch_scatter import scatter_log_softmax


class MessagePassingElectionLayer(MessagePassing):
    def __init__(
        self,
        edge_dim: int = 1,
        node_dim: int = 32,
    ):
        super().__init__(aggr="add")

        self.message_mlp = nn.Sequential(
            nn.Linear(2 * node_dim + edge_dim, node_dim),
            nn.BatchNorm1d(node_dim),
            nn.ReLU(),
            nn.Linear(node_dim, node_dim),
        )

        self.edge_utility_mlp = nn.Sequential(
            nn.Linear(2 * node_dim + edge_dim, edge_dim),
            nn.BatchNorm1d(edge_dim),
            nn.ReLU(),
            nn.Linear(edge_dim, edge_dim),
        )

    def forward(self, x, edge_index, edge_attr):
        new_x = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        new_x = x + new_x

        from_node, to_node = edge_index
        edge_features = torch.cat([x[from_node], x[to_node], edge_attr], dim=1)
        new_edge_attr = self.edge_utility_mlp(edge_features)
        new_edge_attr = edge_attr + new_edge_attr

        return new_x, new_edge_attr

    def message(self, x_i, x_j, edge_attr):
        msg_features = torch.cat([x_i, x_j, edge_attr], dim=1)
        return self.message_mlp(msg_features)


class MessagePassingStrategyLayer(MessagePassing):
    def __init__(self, in_channels, edge_channels, out_channels):
        super(MessagePassingStrategyLayer, self).__init__(aggr="add")

        self.message_mlp = nn.Sequential(
            nn.Linear(2 * in_channels + edge_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
        )

        self.edge_utility_mlp = nn.Sequential(
            nn.Linear(2 * in_channels + edge_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, edge_channels),
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


class DeepSetStrategyModel(nn.Module):
    def __init__(
        self,
        edge_dim: int = 1,
        emb_dim: int = 32,
        num_layers: int = 1,
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_layers = num_layers
        self.vote_to_hidden = nn.Linear(edge_dim, emb_dim)
        self.transform = nn.Linear(emb_dim, emb_dim)
        self.update = nn.Linear(emb_dim, emb_dim)
        self.hidden_to_vote = nn.Linear(emb_dim, edge_dim)

    def forward(self, edge_attr, edge_index, candidate_idxs, selected_voter_nodes):
        new_edge_attr = self.vote_to_hidden(edge_attr)
        for _ in range(self.num_layers):
            new_edge_attr = self.transform(new_edge_attr)
            print(new_edge_attr)
            index = edge_index[0].unsqueeze(-1).expand(-1, new_edge_attr.size(-1))
            print(index)
            sum_voter_scores = torch.zeros(
                len(candidate_idxs), self.emb_dim
            ).scatter_add_(
                src=new_edge_attr, index=index, dim=0
            )  # [num_nodes including candidates, emb_dim]
            print(sum_voter_scores)


class MessagePassingElectionModel(nn.Module):
    def __init__(
        self,
        edge_dim: int = 1,
        node_emb_dim: int = 32,
        edge_emb_dim: int = 8,
        num_layers: int = 4,
    ):
        super().__init__()
        in_dim = 2

        # Initial embedding
        self.lin_in_node = nn.Linear(in_dim, node_emb_dim)
        self.lin_out_node = nn.Linear(node_emb_dim, 1)
        self.lin_in_edge = nn.Linear(edge_dim, edge_emb_dim)

        # Convolution layers
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(
                MessagePassingElectionLayer(
                    edge_dim=edge_emb_dim, node_dim=node_emb_dim
                )
            )

    def forward(self, data: Data):
        x = self.lin_in_node(data.x)
        edge_attr = self.lin_in_edge(data.edge_attr)
        for conv in self.convs:
            new_x, new_edge = conv(x, data.edge_index, edge_attr)
            x = x + new_x
            edge_attr = edge_attr + new_edge

        logits = self.lin_out_node(x[data.candidate_idxs]).squeeze(dim=-1)

        out = scatter_log_softmax(logits, data.batch[data.candidate_idxs])
        return out
