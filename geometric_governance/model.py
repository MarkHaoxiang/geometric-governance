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
        emb_dim: int = 32,
        monotonicity_constraint: bool = False,
    ):
        super().__init__(aggr="add")
        self.monotonicity_constraint = monotonicity_constraint
        self.positive_parameters = nn.ParameterDict(
            {
                "w_1": nn.Parameter(torch.zeros(emb_dim, 2 * emb_dim + edge_dim)),
                "w_2": nn.Parameter(torch.zeros(emb_dim, emb_dim)),
            }
        )

        self.other_parameters = nn.ParameterDict(
            {
                "b_1": nn.Parameter(torch.zeros(emb_dim)),
                "b_2": nn.Parameter(torch.zeros(emb_dim)),
            }
        )

        for param in self.positive_parameters.values():
            nn.init.xavier_uniform_(param)

        self.batch_norm_1 = nn.BatchNorm1d(emb_dim)
        self.batch_norm_2 = nn.BatchNorm1d(emb_dim)

    def forward(self, x, edge_index, edge_attr):
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        return out

    def message(self, x_i, x_j, edge_attr):
        msg = torch.cat([x_i, x_j, edge_attr], dim=-1)

        def h(x):
            return F.softplus(x) if self.monotonicity_constraint else x

        msg = F.linear(
            msg, h(self.positive_parameters["w_1"]), self.other_parameters["b_1"]
        )
        msg = self.batch_norm_1(msg)
        msg = F.relu(msg)
        msg = F.linear(
            msg, h(self.positive_parameters["w_2"]), self.other_parameters["b_2"]
        )
        msg = self.batch_norm_2(msg)
        msg = F.relu(msg)

        return msg


class MessagePassingStrategyLayer(MessagePassing):
    def __init__(self, in_channels, edge_channels, out_channels):
        super(MessagePassingStrategyLayer, self).__init__(aggr='add') 

        self.message_mlp = nn.Sequential(
            nn.Linear(2 * in_channels + edge_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )

        self.edge_utility_mlp = nn.Sequential(
            nn.Linear(2 * in_channels + edge_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, edge_channels)
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
        # It is important that self.transform has a non-linearity
        # Since all the votes made by a voter sum to 1, a single linear layer
        # would assign the same transformed embedding after the scatter_add operation for all voters
        self.transform = nn.Sequential(nn.Linear(emb_dim, emb_dim), nn.ReLU(), nn.Linear(emb_dim, emb_dim))
        self.update = nn.Sequential(nn.Linear(2 * emb_dim, emb_dim), nn.ReLU(), nn.Linear(emb_dim, emb_dim))
        self.hidden_to_vote = nn.Linear(emb_dim, edge_dim)

    def forward(self, edge_attr, edge_index, candidate_idxs):
        # Updates the votes (each row in edge_attr) according to a DeepSet model,
        # aggregating votes made by the same voter
        # Note that there is no inter-voter communication
        # This module can be alternatively viewed as each voter voting,
        # but not truthfully (i.e. according to their true utility profile)
        new_edge_attr = self.vote_to_hidden(edge_attr)
        for _ in range(self.num_layers):
            transformed_edge_attr = self.transform(new_edge_attr)
            index = edge_index[0].unsqueeze(-1).expand(-1, transformed_edge_attr.size(-1))
            sum_voter_scores = torch.zeros(len(candidate_idxs), self.emb_dim).scatter_add_(src=transformed_edge_attr, index=index, dim=0) # [num_nodes including candidates, emb_dim]
            aggr_edge_attr = torch.cat([new_edge_attr, sum_voter_scores[edge_index[0]]], dim=-1)
            new_edge_attr = self.update(aggr_edge_attr)
        return new_edge_attr


class MessagePassingElectionModel(nn.Module):
    def __init__(
        self,
        edge_dim: int = 1,
        emb_dim: int = 32,
        num_layers: int = 4,
        monotonicity_constraint: bool = False,
    ):
        super().__init__()
        in_dim = 2

        # Initial embedding
        self.lin_in_weights = nn.Parameter(torch.zeros(emb_dim, in_dim))
        self.lin_in_bias = nn.Parameter(torch.zeros(emb_dim))

        # Convolution layers
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(
                MessagePassingElectionLayer(
                    edge_dim, emb_dim, monotonicity_constraint=monotonicity_constraint
                )
            )
        # Readout
        self.lin_out_weights = nn.Parameter(torch.zeros(1, emb_dim))
        self.lin_out_bias = nn.Parameter(torch.zeros(()))

        nn.init.xavier_uniform_(self.lin_in_weights)
        nn.init.xavier_uniform_(self.lin_out_weights)

        self.monotonicity_constraint = monotonicity_constraint

    def forward(self, data: Data):
        def h(x):
            return F.softplus(x) if self.monotonicity_constraint else x

        x = F.linear(data.x, h(self.lin_in_weights), self.lin_in_bias)
        for conv in self.convs:
            x = x + conv(x, data.edge_index, data.edge_attr)
        logits = F.linear(
            x[data.candidate_idxs], h(self.lin_out_weights), self.lin_out_bias
        ).squeeze(dim=-1)
        out = scatter_log_softmax(logits, data.batch[data.candidate_idxs])
        return out
