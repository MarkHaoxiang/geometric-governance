from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import MLP, DeepSetsAggregation, MessagePassing
from torch_scatter import scatter_add, scatter_log_softmax, scatter_softmax, scatter_max


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

        self.edge_utility_mlp = nn.Sequential(
            nn.Linear(2 * node_dim + edge_dim, edge_dim),
            nn.LayerNorm(edge_dim),
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


class DeepSetStrategyModel(nn.Module):
    def __init__(
        self,
        edge_dim: int = 1,
        emb_dim: int = 32,
        num_layers: int = 2,
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_layers = num_layers
        self.vote_to_hidden = nn.Linear(edge_dim, emb_dim)
        # It is important that self.transform has a non-linearity
        # Since all the votes made by a voter sum to 1, a single linear layer
        # would assign the same transformed embedding after the scatter_add operation for all voters
        self.transforms = nn.ModuleList()
        self.updates = nn.ModuleList()
        for _ in range(num_layers):
            transform = nn.Sequential(
                nn.Linear(emb_dim, emb_dim),
                nn.LeakyReLU(),
                nn.Linear(emb_dim, emb_dim),
            )
            update = nn.Sequential(
                # nn.Linear(emb_dim, emb_dim),
                nn.Linear(2 * emb_dim, emb_dim),
                nn.LeakyReLU(),
                nn.Linear(emb_dim, emb_dim),
            )
            self.transforms.append(transform)
            self.updates.append(update)

        self.hidden_to_vote = nn.Sequential(
            nn.Linear(emb_dim * 2, emb_dim),
            nn.LeakyReLU(),
            nn.Linear(emb_dim, emb_dim),
            nn.LeakyReLU(),
            nn.Linear(emb_dim, 1),
        )

    def forward(self, edge_attr, edge_index):
        # Updates the votes (each row in edge_attr) according to a DeepSet model,
        # aggregating votes made by the same voter
        # Note that there is no inter-voter co)mmunication
        # This module can be alternatively viewed as each voter voting,
        # but not truthfully (i.e. according to their true utility profile)
        # vote_sum = scatter_add(edge_attr, index=edge_index[0], dim=0)
        # vote_sum = vote_sum[edge_index[0]]
        # normalised_votes = edge_attr / vote_sum
        # hidden_edge_attr = self.vote_to_hidden(normalised_votes)

        edge_index = edge_index[0]
        hidden_edge_attr = self.vote_to_hidden(edge_attr)

        new_edge_attr = hidden_edge_attr
        for transform, update in zip(self.transforms, self.updates):
            transformed_edge_attr = transform(new_edge_attr)
            index = edge_index.unsqueeze(-1).expand(-1, transformed_edge_attr.size(-1))
            sum_voter_scores = scatter_add(transformed_edge_attr, index, dim=0)
            aggr_edge_attr = torch.cat(
                [new_edge_attr, sum_voter_scores[edge_index]], dim=-1
            )
            # aggr_edge_attr = new_edge_attr + sum_voter_scores[edge_index[0]]
            new_edge_attr = update(aggr_edge_attr)

        new_edge_attr = torch.cat([hidden_edge_attr, new_edge_attr], dim=-1)

        votes = self.hidden_to_vote(new_edge_attr)
        votes = scatter_softmax(votes, edge_index.unsqueeze(-1), dim=0)
        return votes


class MLPStrategyModel(nn.Module):
    def __init__(self, num_candidates: int, emb_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_candidates, emb_dim),
            nn.LeakyReLU(),
            nn.Linear(emb_dim, emb_dim),
            nn.LeakyReLU(),
            nn.Linear(emb_dim, num_candidates),
            nn.Softmax(),
        )
        self.num_candidates = num_candidates

    def forward(self, edge_attr: torch.Tensor, edge_index, candidate_idxs):
        # Reshape edge attr to be of shape (num_voters, num_candidates)
        votes = edge_attr.reshape(-1, self.num_candidates)
        votes = self.net(votes)
        return votes.reshape(edge_attr.shape)


@dataclass
class ElectionResult:
    log_probs: torch.Tensor
    winners: torch.Tensor


class ElectionModel(nn.Module, ABC):
    @abstractmethod
    def election(self, data) -> ElectionResult:
        raise NotImplementedError()


class MessagePassingElectionModel(ElectionModel):
    def __init__(
        self,
        edge_dim: int = 1,
        node_emb_dim: int = 32,
        edge_emb_dim: int = 8,
        num_layers: int = 4,
        aggr: str = "add",
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
                MessagePassingLayer(
                    edge_dim=edge_emb_dim, node_dim=node_emb_dim, aggr=aggr
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

    def election(self, data):
        log_probs = self(data)
        winner_idxs = scatter_max(log_probs, data.batch[data.candidate_idxs])[1]
        winners = torch.zeros_like(log_probs)
        winners[winner_idxs] = 1
        return ElectionResult(log_probs, winners)


class DeepSetElectionModel(ElectionModel):
    def __init__(
        self,
        num_candidates: int,
        embedding_size: int = 128,
        num_layers: int = 3,
        one_hot: bool = False,
    ):
        super().__init__()

        embedding_layers = [embedding_size for _ in range(num_layers)]

        if one_hot:
            self.local_nn = MLP([num_candidates**2] + embedding_layers)
        else:
            self.local_nn = MLP([num_candidates] + embedding_layers)
        self.global_nn = MLP(embedding_layers + [num_candidates])

        self.deepset = DeepSetsAggregation(
            local_nn=self.local_nn, global_nn=self.global_nn
        )

    def forward(self, x, index):
        scores = self.deepset(x, index=index)
        return nn.functional.log_softmax(scores, dim=-1)

    def election(self, data):
        log_probs = self(data.X, data.index)
        winner_idxs = torch.max(log_probs, dim=-1)[1]
        winners = torch.zeros_like(log_probs)
        winners[torch.arange(winners.shape[0]), winner_idxs] = 1
        return ElectionResult(log_probs, winners)


def create_election_model(
    representation: str,
    num_candidates: int | None = None,
    model_size: Literal["small", "medium"] = "small",
    aggr: str = "sum",
) -> ElectionModel:
    model: ElectionModel
    if representation == "set_one_hot":
        representation = "set"
        one_hot = True
    else:
        one_hot = False
    if representation == "graph_unnormalised":
        representation = "graph"
    match representation, model_size:
        case "graph", "small":
            model = MessagePassingElectionModel(
                node_emb_dim=80, edge_emb_dim=20, num_layers=4, edge_dim=1, aggr=aggr
            )
        case "graph", "medium":
            model = MessagePassingElectionModel(
                node_emb_dim=256, edge_emb_dim=64, num_layers=4, edge_dim=1, aggr=aggr
            )
        case "set", "small":
            assert num_candidates is not None, ""
            model = DeepSetElectionModel(
                num_candidates=num_candidates,
                embedding_size=155,
                num_layers=3,
                one_hot=one_hot,
            )
        case "set", "medium":
            assert num_candidates is not None, ""
            model = DeepSetElectionModel(
                num_candidates=num_candidates,
                embedding_size=352,
                num_layers=5,
                one_hot=one_hot,
            )
        case _:
            raise ValueError(f"Unknown model for {representation} {model_size}")
    return model
