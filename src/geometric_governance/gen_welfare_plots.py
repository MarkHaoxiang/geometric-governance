import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb
from tqdm import tqdm
import itertools

api = wandb.Api()

project_name = "geometric-governance"

def moving_average(x, window_size):
    return np.convolve(x, np.ones(window_size), 'valid') / window_size 

window_size = 25 # Number of epochs to calculate moving average over

datasets = ["dirichlet", "spatial", "movielens"]
vote_datas = ["ranking", "utility"]

for dataset, vote_data in itertools.product(datasets, vote_datas):
    welfare_rules = ["utilitarian", "nash", "rawlsian"]
    losses = ["welfare", "rule"]

    fig, axs = plt.subplots(1, 3, figsize=(12, 4), sharex=True)

    for i, welfare_rule in enumerate(tqdm(welfare_rules)):
        ax = axs[i % 3]
        ax.set_title(welfare_rule)

        if i % 3 == 0:
            ax.set_ylabel("Validation Welfare")

        for j, loss in enumerate(losses):
            filters = {
                "displayName": f"{dataset}-{vote_data}-{welfare_rule}-{loss}-small-sum"
            }

            runs = api.runs(path="geometric-governance/maximise-welfare", filters=filters)

            validation_welfare = []
            num_epochs = 330

            for run in runs:
                history = run.history(keys=["val/welfare"], samples=num_epochs)
                validation_welfare.append(history["val/welfare"].to_numpy()[:num_epochs])
            validation_welfare = np.stack(validation_welfare)

            color = "blue" if loss=="welfare" else "red"
            mean = np.mean(validation_welfare, axis=0)
            mean = moving_average(mean, window_size)
            line, =  ax.plot(np.arange(num_epochs)[window_size - 1:], mean, color=color, linewidth=1.0)
            line_label = "Welfare Loss" if loss=="welfare" else "Rule Loss"
            line.set_label(line_label)

            std = np.std(validation_welfare, axis=0)
            std = moving_average(std, window_size)
            ax.fill_between(
                np.arange(num_epochs)[window_size - 1:],
                mean + std,
                mean - std,
                color=color,
                alpha=0.2,
                interpolate=True,
                linewidth=0
            )

        ax.set_xlabel("Epoch")
        ax.legend()

    fig.suptitle(f"{dataset.title()} dataset + {vote_data.title()} voting data", fontsize=16)
    plt.tight_layout()
    fig.savefig(f"{dataset}-{vote_data}-welfare_loss_plots", dpi=400)
