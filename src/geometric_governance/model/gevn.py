from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import MLP, DeepSetsAggregation
from torch_scatter import scatter_log_softmax, scatter_max, scatter_min, scatter_add

from geometric_governance.model.nn import MessagePassingLayer


@dataclass
class ElectionResult:
    log_probs: torch.Tensor
    winners: torch.Tensor


class ElectionModel(nn.Module, ABC):
    @abstractmethod
    def election(self, data) -> ElectionResult:
        raise NotImplementedError()


class GraphElectionModel(ElectionModel, ABC):
    def election(self, data):
        log_probs = self(data)
        winner_idxs = scatter_max(log_probs, data.batch[data.candidate_idxs])[1]
        winners = torch.zeros_like(log_probs)
        winners[winner_idxs] = 1
        return ElectionResult(log_probs, winners)


class MessagePassingElectionModel(GraphElectionModel):
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


class ManualElectionModel(ElectionModel):
    def forward(self, data: Data):
        vote_sum = scatter_add(data.edge_attr, index=data.edge_index[0], dim=0)
        vote_sum = vote_sum[data.edge_index[0]]
        normalised_votes = data.edge_attr / torch.maximum(
            vote_sum, torch.ones_like(vote_sum)
        )

        logits = scatter_add(
            src=normalised_votes, index=data.edge_index[1].unsqueeze(-1), dim=0
        )[data.candidate_idxs.nonzero()].squeeze()

        out = scatter_log_softmax(logits, data.batch[data.candidate_idxs])
        return out

    def election(self, data: Data):
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
                node_emb_dim=58, edge_emb_dim=19, num_layers=4, edge_dim=1, aggr=aggr
            )
        case "graph", "medium":
            model = MessagePassingElectionModel(
                node_emb_dim=185, edge_emb_dim=60, num_layers=4, edge_dim=1, aggr=aggr
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
