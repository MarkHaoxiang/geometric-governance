from abc import abstractmethod
from typing import Any, Literal

import torch
import torch.nn as nn
from torch_scatter import scatter_add, scatter_softmax
from torch_geometric.data import Data

from geometric_governance.third_party.fast_soft_sort.pytorch_ops import soft_rank
from geometric_governance.model.gevn import ElectionModel
from geometric_governance.model.nn import MessagePassingLayer, OpinionPassingLayer


class StrategyModel(nn.Module):
    def __init__(
        self,
        constraint: tuple[Literal["sum", "range", "ordinal", "none"], Any] = (
            "none",
            None,
        ),
    ):
        super().__init__()
        self.constraint = constraint[0]
        self.constraint_args = constraint[1]

    @abstractmethod
    def _strategise(self, data: Data, train_agent_idxs):
        raise NotImplementedError()

    def forward(self, data: Data, train_agent_idxs: torch.Tensor | None = None):
        edge_attr, edge_index = data.edge_attr, data.edge_index
        values = self._strategise(data, train_agent_idxs)
        assert values.shape == (edge_attr.shape[0], 1), (
            f"Values must be of shape ({edge_attr.shape[0]}, 1). Found {values.shape} instead."
        )

        match self.constraint:
            case "sum":
                votes = (
                    scatter_softmax(values, edge_index[0].unsqueeze(-1), dim=0)
                    * self.constraint_args
                )
            case "range":
                votes = torch.nn.functional.sigmoid(values)
                minimum_value = self.constraint_args[0]
                maximum_value = self.constraint_args[1]
                votes = minimum_value + (maximum_value - minimum_value) * votes
            case "none":
                votes = values
            case "ordinal":
                # Squeeze votes to (0, 1)
                values = torch.nn.functional.sigmoid(values)
                epsilon = self.constraint_args[0]
                votes = soft_rank(
                    values, regularization_strength=epsilon, direction="DESCENDING"
                )
            case _:
                raise ValueError(
                    f"Unknown constraint {self.constraint}. Must be one of ['sum', 'range']"
                )

        return votes


class NoStrategy(StrategyModel):
    def __init__(self):
        super().__init__(constraint=("none", None))

    def _strategise(self, data: Data, train_agent_idxs: torch.Tensor | None = None):
        return data.edge_attr


class DeepSetStrategyModel(StrategyModel):
    def __init__(
        self,
        emb_dim: int = 32,
        num_layers: int = 2,
        constraint: tuple[Literal["sum", "range"], Any] = ("sum", 1.0),
    ):
        super().__init__(constraint=constraint)
        self.emb_dim = emb_dim
        self.num_layers = num_layers
        self.vote_to_hidden = nn.Linear(1, emb_dim)
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

    def _strategise(self, data: Data, train_agent_idxs: torch.Tensor | None = None):
        # Updates the votes (each row in edge_attr) according to a DeepSet model,
        # aggregating votes made by the same voter
        # Note that there is no inter-voter co)mmunication
        # This module can be alternatively viewed as each voter voting,
        # but not truthfully (i.e. according to their true utility profile)
        # vote_sum = scatter_add(edge_attr, index=edge_index[0], dim=0)
        # vote_sum = vote_sum[edge_index[0]]
        # normalised_votes = edge_attr / vote_sum
        # hidden_edge_attr = self.vote_to_hidden(normalised_votes)
        edge_attr, edge_index = data.edge_attr, data.edge_index
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
        return votes


class DeepSetStrategyModelWithResults(DeepSetStrategyModel):
    def __init__(
        self,
        election_model: ElectionModel,
        emb_dim: int = 32,
        num_layers: int = 2,
        constraint: tuple[Literal["sum", "range"], Any] = ("sum", 1),
    ):
        super().__init__(emb_dim=emb_dim, num_layers=num_layers, constraint=constraint)
        self.vote_to_hidden = nn.Linear(2, emb_dim)  # Increase edge dim by 1
        self.election_model = election_model

    def _strategise(self, data: Data, train_agent_idxs: torch.Tensor | None = None):
        # Compute true election results
        with torch.no_grad():
            truthful_election_result = self.election_model.election(data)
        # Append the election results to the edge attributes
        candidate_idxs_nonzero = data.candidate_idxs.nonzero()
        edge_attr_candidates = data.edge_index[1]
        lookup = torch.full(
            size=(edge_attr_candidates.max() + 1,),
            fill_value=-torch.inf,
            device=data.edge_attr.device,
        )
        lookup[candidate_idxs_nonzero.squeeze()] = truthful_election_result.log_probs
        edge_attr_log_probs = lookup[edge_attr_candidates]
        edge_attr_with_results = torch.cat(
            [data.edge_attr, edge_attr_log_probs.unsqueeze(-1)],
            dim=-1,
        )
        data_placeholder = data.clone()
        data_placeholder.edge_attr = edge_attr_with_results
        # Strategise
        return super()._strategise(data_placeholder)


class MessagePassingStrategyModel(StrategyModel):
    def __init__(
        self,
        edge_dim: int = 1,
        node_emb_dim: int = 32,
        edge_emb_dim: int = 8,
        num_layers: int = 3,
        aggr: str = "add",
    ):
        super().__init__()
        in_dim = 2

        # Initial embedding
        self.lin_in_node = nn.Linear(in_dim, node_emb_dim)
        self.lin_in_edge = nn.Linear(edge_dim, edge_emb_dim)
        self.lin_out_edge = nn.Linear(edge_emb_dim, edge_dim)

        # Convolution layers
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(
                MessagePassingLayer(
                    edge_dim=edge_emb_dim, node_dim=node_emb_dim, aggr=aggr
                )
            )

    def _strategise(self, data: Data, train_agent_idxs: torch.Tensor | None = None):
        x = self.lin_in_node(data.x)
        edge_attr = self.lin_in_edge(data.edge_attr)
        for conv in self.convs:
            new_x, new_edge = conv(x, data.edge_index, edge_attr)
            x = x + new_x
            edge_attr = edge_attr + new_edge

        return self.lin_out_edge(edge_attr)


class OpinionPassingStrategyModel(StrategyModel):
    def __init__(
        self,
        edge_dim: int = 1,
        node_emb_dim: int = 32,
        edge_emb_dim: int = 8,
        num_layers: int = 3,
        aggr: str = "add",
    ):
        super().__init__()
        in_dim = 2

        # Initial embedding
        self.lin_in_node = nn.Linear(in_dim, node_emb_dim)
        self.lin_in_edge = nn.Linear(edge_dim, edge_emb_dim)
        self.lin_out_edge = nn.Linear(edge_emb_dim, edge_dim)

        # Convolution layers
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(
                OpinionPassingLayer(
                    edge_dim=edge_emb_dim, node_dim=node_emb_dim, aggr=aggr
                )
            )

    def _strategise(self, data: Data, train_agent_idxs: torch.Tensor | None = None):
        x = self.lin_in_node(data.x)
        edge_attr = self.lin_in_edge(data.edge_attr)
        strategic_mask = (
            torch.isin(data.edge_index[0], train_agent_idxs)
            if train_agent_idxs is not None
            else None
        )
        for conv in self.convs:
            new_x, new_edge = conv(
                x,
                data.edge_index,
                edge_attr,
                data.candidate_idxs,
                train_agent_idxs,
                strategic_mask,
            )
            x = x + new_x
            edge_attr = edge_attr + new_edge

        return self.lin_out_edge(edge_attr)


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

    def forward(self, data):
        # Reshape edge attr to be of shape (num_voters, num_candidates)
        edge_attr = data.edge_attr
        votes = edge_attr.reshape(-1, self.num_candidates)
        votes = self.net(votes)
        return votes.reshape(edge_attr.shape)
