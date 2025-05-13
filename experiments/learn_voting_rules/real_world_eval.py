import os
from collections import defaultdict

import torch
from torch_geometric.data.batch import Batch

from geometric_governance.model.gevn import ElectionModel
from geometric_governance.util import OUTPUT_DIR, cuda

from dataset import generate_rule_dataset
from gen_table import fmt

# Note: This table generation only works on the local machine where the sweep was conducted

if __name__ == "__main__":
    rules = ["plurality", "borda", "copeland", "minimax", "stv"]
    model_sizes = ["small", "medium"]

    results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    dataset_size = 256

    for rule in rules:
        grenoble_dataset = generate_rule_dataset(
            dataset_size=dataset_size,
            voting_rule=rule,
            vote_source="grenoble",
            representation="graph",
            num_voters_range=50,
            num_candidates_range=5,
            seed=0,
        )
        grenoble_batch = Batch.from_data_list(grenoble_dataset).to(device=cuda)

        movielens_dataset = generate_rule_dataset(
            dataset_size=dataset_size,
            voting_rule=rule,
            vote_source="movielens",
            representation="graph",
            num_voters_range=100,
            num_candidates_range=10,
            seed=0,
        )
        movielens_batch = Batch.from_data_list(movielens_dataset).to(device=cuda)

        for size in model_sizes:
            experiment_folder = os.path.join(
                OUTPUT_DIR, f"graph-election-{rule}-{size}"
            )
            subfolders = [f.path for f in os.scandir(experiment_folder) if f.is_dir()]
            best_models: list[ElectionModel] = []
            for folder in subfolders:
                best_model_path = os.path.join(folder, "checkpoints", "model_best.pt")
                best_models.append(
                    torch.load(best_model_path, weights_only=False).to(device=cuda)
                )

            with torch.no_grad():
                for model in best_models:
                    for source, data in (
                        ("grenoble", grenoble_batch),
                        ("movielens", movielens_batch),
                    ):
                        election = model.election(data)
                        acc = (
                            (election.winners > 0) & (data.winners > 0)
                        ).sum().item() / dataset_size
                        results[rule][size][source].append(acc)

    # Construct columns

    table_str = f"""table(
        columns: 4,
        table.header(
            table.vline(stroke: 0.5pt), [],[], table.vline(stroke: 0.5pt),
            [GEVN Small], [GEVN Medium],
            table.vline(stroke: 0.5pt)
        ),
    {
        "".join(
            f'''
        [#rule("{rule}")],
        [Grenoble], {", ".join(f"[{fmt(results[rule][size]['grenoble'])}]" for size in model_sizes)},
        [Movielens], {", ".join(f"[{fmt(results[rule][size]['movielens'])}]" for size in model_sizes)},
        table.hline(stroke: 0.5pt),'''
            for rule in rules
        )
    }
    )
    """
    print(table_str)
