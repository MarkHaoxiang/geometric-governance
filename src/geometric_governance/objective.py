import torch
from geometric_governance.data import ElectionData, get_scoring_function_winners


def compute_plurality(election_data: ElectionData) -> torch.Tensor:
    scores = election_data.rank_candidate_count[0]
    return get_scoring_function_winners(scores)


def compute_borda(election_data: ElectionData) -> torch.Tensor:
    scoring = torch.tensor(
        list(range(election_data.num_candidates, 0, -1)), dtype=torch.float32
    )
    scores = scoring @ election_data.rank_candidate_count
    return get_scoring_function_winners(scores)


def compute_copeland(election_data: ElectionData) -> torch.Tensor:
    scores = election_data.tournament_embedding.sum(dim=1)
    return get_scoring_function_winners(scores)


def compute_utilitarian(election_data: ElectionData) -> torch.Tensor:
    return election_data.voter_utilities.sum(dim=0)


def compute_nash(election_data: ElectionData) -> torch.Tensor:
    return election_data.voter_utilities.prod(dim=0)


def compute_rawlsian(election_data: ElectionData) -> torch.Tensor:
    return election_data.voter_utilities.min(dim=0).values


VotingRulesRegistry = {
    "plurality": compute_plurality,
    "borda": compute_borda,
    "copeland": compute_copeland,
}

WelfareObjectiveRegistry = {
    "utilitarian": compute_utilitarian,
    "nash": compute_nash,
    "rawlsian": compute_rawlsian,
}

VotingObjectiveRegistry = VotingRulesRegistry | WelfareObjectiveRegistry
