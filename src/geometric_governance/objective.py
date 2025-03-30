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


def compute_minimax(election_data: ElectionData) -> torch.Tensor:
    scores = election_data.pairwise_comparison.min(dim=1)[0]
    return get_scoring_function_winners(scores)


def compute_single_transferrable_vote(election_data: ElectionData) -> torch.Tensor:
    remaining_candidates = torch.ones(election_data.num_candidates, dtype=torch.bool)
    num_remaining_candidates = election_data.num_candidates
    votes = election_data.rank_candidate_count[0]
    majority = election_data.num_voters // 2 + 1
    voter_rank_remaining = [0 for _ in range(election_data.num_voters)]
    winners = torch.zeros(election_data.num_candidates, dtype=torch.float32)
    while num_remaining_candidates > 1:
        # Candidate has majority of votes
        for candidate in range(election_data.num_candidates):
            if votes[candidate] >= majority:
                winners[candidate] = 1.0
                return winners

        # Edge case: tie
        if votes[remaining_candidates].max() == votes[remaining_candidates].min():
            return get_scoring_function_winners(votes)

        # Candidate with fewest votes is eliminated
        min_votes = votes[remaining_candidates].min()
        losing_candidates: list[int] = []
        for candidate in range(election_data.num_candidates):
            if votes[candidate] == min_votes:
                losing_candidates.append(candidate)

        for candidate in losing_candidates:
            assert remaining_candidates[candidate]
            num_remaining_candidates -= 1
            remaining_candidates[candidate] = False

        # Redistribute votes
        votes = torch.zeros(election_data.num_candidates, dtype=torch.float32)
        for voter in range(election_data.num_voters):
            voter_rank_start = voter_rank_remaining[voter]
            for rank in range(voter_rank_start, election_data.num_candidates):
                candidate = election_data.voter_ranked_order[voter, rank]  # type: ignore
                if remaining_candidates[candidate]:
                    votes[candidate] += 1
                    break
                else:
                    voter_rank_remaining[voter] += 1

    assert num_remaining_candidates == 1
    return remaining_candidates.to(dtype=torch.float32)


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
    "minimax": compute_minimax,
    "stv": compute_single_transferrable_vote,
}

WelfareObjectiveRegistry = {
    "utilitarian": compute_utilitarian,
    "nash": compute_nash,
    "rawlsian": compute_rawlsian,
}

VotingObjectiveRegistry = VotingRulesRegistry | WelfareObjectiveRegistry
