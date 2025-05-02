import os
from typing import Literal
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader as TorchDataloader
from torch_geometric.loader import DataLoader as GraphDataloader
from tqdm import tqdm
from geometric_governance.util import (
    RangeOrValue,
    get_value,
    get_max,
    DATA_DIR as _DATA_DIR,
)
from geometric_governance.data import SetDataset, generate_dirichlet_election
from geometric_governance.objective import VotingRulesRegistry

DATA_DIR = os.path.join(_DATA_DIR, "learn_voting_rules")
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)


def generate_rule_dataset(
    dataset_size: int,
    num_voters_range: RangeOrValue,
    num_candidates_range: RangeOrValue,
    voting_rule: str,
    representation: Literal["set", "set_one_hot", "graph"],
    seed: int,
):
    rng = np.random.default_rng(seed=seed)
    dataset = []
    generated_count = 0
    with tqdm(range(dataset_size)) as pbar:
        while generated_count < dataset_size:
            num_voters = get_value(num_voters_range, rng)
            num_candidates = get_value(num_candidates_range, rng)

            election_data = generate_dirichlet_election(
                num_voters=num_voters, num_candidates=num_candidates, rng=rng
            )

            assert voting_rule in VotingRulesRegistry
            winners = VotingRulesRegistry[voting_rule](election_data)
            if winners.max() < 1.0:  # Tie
                continue

            match representation:
                case "graph":
                    graph = election_data.to_bipartite_graph(vote_data="ranking")
                    graph.winners = winners
                    dataset.append(graph)
                case "set":
                    pad_shape = get_max(num_candidates_range) - num_candidates
                    voter_preferences = election_data.voter_candidate_rank
                    voter_preferences = F.pad(voter_preferences, (0, pad_shape, 0, 0))
                    winners = F.pad(winners, (0, pad_shape))
                    dataset.append((voter_preferences, winners))
                case "set_one_hot":
                    M = get_max(num_candidates_range)
                    pad_shape = M - num_candidates
                    voter_preferences = election_data.voter_candidate_rank
                    voter_preferences_one_hot = F.one_hot(
                        voter_preferences, num_classes=M
                    ).reshape(num_voters, -1)
                    voter_preferences_one_hot = F.pad(
                        voter_preferences_one_hot, (0, pad_shape * M, 0, 0)
                    )
                    winners = F.pad(winners, (0, pad_shape))
                    dataset.append((voter_preferences_one_hot, winners))
                case _:
                    raise ValueError(f"Unknown representation {representation}.")

            generated_count += 1
            pbar.update(1)

    return dataset


type Dataloader = GraphDataloader | TorchDataloader


def load_dataloader(
    dataset_size: int,
    num_voters: RangeOrValue,
    num_candidates: RangeOrValue,
    dataloader_batch_size: int,
    voting_rule: str,
    representation: Literal["set", "set_one_hot", "graph"],
    seed: int,
    recompute: bool = True,
) -> tuple[Dataloader, Dataloader, Dataloader]:
    dataset_file = os.path.join(
        DATA_DIR,
        f"rule_dataset_{dataset_size}_{num_voters}_{num_candidates}_{representation}_{voting_rule}_{seed}.pt",
    )

    if os.path.exists(dataset_file) and not recompute:
        with open(dataset_file, "rb") as f:
            dataset = torch.load(f, weights_only=False)
    else:
        dataset = generate_rule_dataset(
            dataset_size,
            num_voters,
            num_candidates,
            voting_rule,
            representation,
            seed,
        )

        with open(dataset_file, "wb") as f:
            torch.save(dataset, f)

    if representation == "graph":
        dataloader = GraphDataloader(
            dataset, batch_size=dataloader_batch_size, shuffle=True
        )
    elif representation.startswith("set"):
        voter_preferences_list = [x[0] for x in dataset]
        winner_list = [x[1] for x in dataset]
        set_dataset = SetDataset(voter_preferences_list, winner_list)
        dataloader = TorchDataloader(
            set_dataset,
            batch_size=dataloader_batch_size,
            shuffle=True,
            collate_fn=SetDataset.collate_fn,
        )
    return dataloader
