import numpy as np
import torch


class ElectionData:
    def __init__(
        self, num_voters: int, num_candidates: int, voter_utilities: np.ndarray
    ):
        super().__init__()
        self.num_voters = num_voters
        self.num_candidates = num_candidates

        assert voter_utilities.shape == (self.num_voters, self.num_candidates)
        self.voter_utilities = torch.from_numpy(voter_utilities)
        self.voter_preferences = torch.from_numpy(np.argsort(voter_utilities, axis=-1))


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
