import os
from typing import Literal

import numpy as np
import torch
from torch_geometric.data import Data

DATA_DIR = os.path.join(os.path.dirname(__file__), "../data")
SUSHI = os.path.join(DATA_DIR, "sushi")
NETFLIX = os.path.join(DATA_DIR, "netflix")
PUZZLE = os.path.join(DATA_DIR, "puzzle")


class ElectionData:
    def __init__(
        self,
        num_voters: int,
        num_candidates: int,
        voter_utilities: np.ndarray | torch.Tensor,
    ):
        super().__init__()
        self.num_voters = num_voters
        self.num_candidates = num_candidates

        assert voter_utilities.shape == (self.num_voters, self.num_candidates)
        self.voter_utilities = (
            torch.from_numpy(voter_utilities).to(torch.float32)
            if not torch.is_tensor(voter_utilities)
            else voter_utilities
        )
        self.voter_utilities = self.voter_utilities / torch.sum(self.voter_utilities, dim=-1, keepdim=True) # Normalize so max is 1
        self.voter_preferences = torch.from_numpy(
            np.flip(np.argsort(voter_utilities, axis=-1), axis=-1).copy()
        )

    def to_bipartite_graph(
        self,
        top_k_candidates: int | None = None,
        vote_data: Literal["ranking", "utility"] = "utility",
    ) -> Data:
        """An election can be represented as bipartite graph between voters and candidates.

        Edges between voters and candidates represent votes.

        Args:
            top_k_candidates (int | None, optional): Limit the number of candidates each voter votes for.
                Defaults to None, i.e. a fully-connected graph (k=num_candidates).

        Returns:
            Data: A torch geometric homogenous graph object.
        """
        # Node feature matrix
        # num_voters + num_candidate nodes
        # One-hot encoding of type
        x = torch.zeros(size=(self.num_voters + self.num_candidates, 2))
        x[0 : self.num_voters, 0] = 1  # Voters
        x[self.num_voters :, 1] = 1  # Candidates
        # Graph connectivity
        k = (
            self.num_candidates
            if top_k_candidates is None
            else min(self.num_candidates, top_k_candidates)
        )
        edge_index = []
        edge_attr = []
        for voter in range(self.num_voters):
            if vote_data == "ranking":
                votes = [
                    (
                        self.voter_preferences[voter, i],
                        1 - (i / k),
                    )
                    for i in range(k)
                ]
            else:
                votes = [
                    (
                        self.voter_preferences[voter, i],
                        self.voter_utilities[voter, self.voter_preferences[voter, i]],
                    )
                    for i in range(k)
                ]
            for candidate, score in votes:
                edge_index.append([voter, candidate + self.num_voters])
                edge_attr.append(score)
        edge_index = torch.tensor(edge_index).T.long()
        edge_attr = torch.tensor(edge_attr).unsqueeze(-1).to(torch.float32)

        candidate_idxs = x[:, 1] == 1
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            candidate_idxs=candidate_idxs,
        )
        data.validate(raise_on_error=True)
        return data


def generate_synthetic_election(
    num_voters: int,
    num_candidates: int,
    rng: np.random.Generator | None = None,
    utility_profile_alpha: float = 1.0,
):
    if rng is None:
        rng = np.random.default_rng()

    voter_utilities = rng.dirichlet(
        alpha=(utility_profile_alpha,) * num_candidates, size=num_voters
    )

    return ElectionData(
        num_voters=num_voters,
        num_candidates=num_candidates,
        voter_utilities=voter_utilities,
    )


def get_scoring_function_winners(scores: torch.Tensor):
    winners = torch.where(scores == torch.max(scores), 1, 0)
    winners = (winners / winners.sum()).to(torch.float32)
    return winners


def utility_matrix_to_graph(U):
    """
    Converts a utility matrix U (Voters x Candidates) into a PyTorch Geometric Data object.

    Parameters:
        U (torch.Tensor): A tensor of shape (..., num_voters, num_candidates).

    Returns:
        Data: A PyTorch Geometric Data object.
    """
    num_voters, num_candidates = U.size(-2), U.size(-1)

    # Node features: one-hot encoding for voters and candidates
    x_voters = torch.tensor([[1, 0]] * num_voters, dtype=torch.float)
    x_candidates = torch.tensor([[0, 1]] * num_candidates, dtype=torch.float)
    x = torch.cat([x_voters, x_candidates], dim=0)

    # Create edges
    voter_indices = torch.arange(num_voters).repeat_interleave(num_candidates)
    candidate_indices = torch.arange(num_candidates).repeat(num_voters)

    # Shift candidate indices to match node indexing
    candidate_indices += num_voters

    edge_index = torch.stack([voter_indices, candidate_indices], dim=0)

    # Edge attributes (utility values)
    edge_attr = U.flatten().unsqueeze(-1)

    # Candidate indices
    candidate_idxs = x[:, 1] == 1

    return Data(
        x=x, edge_index=edge_index, edge_attr=edge_attr, candidate_idxs=candidate_idxs
    )
