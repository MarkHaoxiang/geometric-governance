import torch
import torch.nn as nn
from torch_scatter import scatter_add, scatter_softmax


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
