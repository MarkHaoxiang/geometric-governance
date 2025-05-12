import numpy as np
import wandb
import itertools

api = wandb.Api()

project_name = "geometric-governance"

datasets = ["dirichlet", "spatial", "movielens"]
vote_datas = ["ranking", "utility"]

for dataset, vote_data in itertools.product(datasets, vote_datas):
    welfare_rules = ["utilitarian", "nash", "rawlsian"]
    losses = ["welfare", "rule"]

    for i, welfare_rule in enumerate(welfare_rules):
        for j, loss in enumerate(losses):
            filters = {
                "displayName": f"{dataset}-{vote_data}-{welfare_rule}-{loss}-small-sum"
            }

            runs = api.runs(path="geometric-governance/maximise-welfare", filters=filters)

            test_accuracies = []
            num_epochs = 330

            for run in runs:
                test_accuracy = run.summary["test_accuracy"]
                test_accuracies.append(test_accuracy)
            test_accuracies = np.stack(test_accuracies)

            mean = np.mean(test_accuracies, axis=0)

            std = np.std(test_accuracies, axis=0)

            print(f"{dataset}-{vote_data}-{welfare_rule}-{loss}-small-sum")
            print(f"mean={mean} (std={std})")
            print()

