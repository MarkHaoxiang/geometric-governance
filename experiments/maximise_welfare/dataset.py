import os
from typing import Literal
import numpy as np
import torch
from torch.utils.data import DataLoader as TorchDataloader
from torch_geometric.loader import DataLoader as GraphDataloader
from tqdm import tqdm
from geometric_governance.util import (
    RangeOrValue,
    get_value,
    DATA_DIR as _DATA_DIR,
)
from geometric_governance.data import generate_synthetic_election
from geometric_governance.objective import WelfareObjectiveRegistry

DATA_DIR = os.path.join(_DATA_DIR, "maximise_welfare")
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)


def generate_welfare_dataset(
    dataset_size: int,
    num_voters_range: RangeOrValue,
    num_candidates_range: RangeOrValue,
    vote_data: Literal["ranking", "utility"],
    welfare_rule: Literal["utilitarian", "nash", "rawlsian"],
    seed: int,
):
    rng = np.random.default_rng(seed=seed)
    dataset = []
    generated_count = 0
    with tqdm(range(dataset_size)) as pbar:
        while generated_count < dataset_size:
            num_voters = get_value(num_voters_range, rng)
            num_candidates = get_value(num_candidates_range, rng)

            election_data = generate_synthetic_election(
                num_voters=num_voters, num_candidates=num_candidates, rng=rng
            )

            assert welfare_rule in WelfareObjectiveRegistry
            welfare_fn = WelfareObjectiveRegistry[welfare_rule]
            winners = welfare_fn(election_data, get_winners=True)
            welfare_value = welfare_fn(election_data, get_winners=False)

            if winners.max() < 1.0:  # Tie
                continue

            graph = election_data.to_bipartite_graph(vote_data=vote_data)
            graph.winners = winners
            graph.welfare = welfare_value

            dataset.append(graph)

            generated_count += 1
            pbar.update(1)

    return dataset


type Dataloader = GraphDataloader | TorchDataloader


def load_dataloader(
    dataset_size: int,
    num_voters: RangeOrValue,
    num_candidates: RangeOrValue,
    dataloader_batch_size: int,
    vote_data: Literal["ranking", "utility"],
    welfare_rule: Literal["utilitarian", "nash", "rawlsian"],
    seed: int,
    recompute: bool = True,
) -> tuple[Dataloader, Dataloader, Dataloader]:
    dataset_file = os.path.join(
        DATA_DIR,
        f"welfare_dataset_{dataset_size}_{num_voters}_{num_candidates}_{vote_data}_{welfare_rule}_{seed}.pt",
    )

    if os.path.exists(dataset_file) and not recompute:
        with open(dataset_file, "rb") as f:
            dataset = torch.load(f, weights_only=False)
    else:
        dataset = generate_welfare_dataset(
            dataset_size,
            num_voters,
            num_candidates,
            vote_data,
            welfare_rule,
            seed,
        )

        with open(dataset_file, "wb") as f:
            torch.save(dataset, f)

    dataloader = GraphDataloader(
        dataset, batch_size=dataloader_batch_size, shuffle=True
    )

    return dataloader
