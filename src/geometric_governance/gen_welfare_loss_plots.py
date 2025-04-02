import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb
from tqdm import tqdm

api = wandb.Api()

utility_utilitarian_welfare_small_sum = ["vzfzzzke", "rt4xi1fl", "f30xzsru", "1jasppqj", "xp7cbjlj"], "Utilitarian (Welfare Loss)"
utility_nash_welfare_small_sum = ["q9nyzq1g", "diqsx0kw", "w3y3w5ii", "meljx9dt", "59rhnsr8"], "Nash (Welfare Loss)"
utility_rawlsian_welfare_small_sum = ["5zucxss3", "0a0lfxoy", "meqljdkd", "ri6z6ml5", "uxxszat6"], "Rawlsian (Welfare Loss)"

utility_utilitarian_rule_small_sum = ["olgo7t2w", "3absmnb8", "xw5o00st", "i4hr4vde", "257u6bi8"], "Utilitarian (Rule Loss)"
utility_nash_rule_small_sum = ["2tixdk3p", "nce36qp1", "rc0ra7ri", "ac93yplx", "vkv8eg57"], "Nash (Rule Loss)"
utility_rawlsian_rule_small_sum = ["wobn8x24", "jexom0or", "pmkx6me5", "pnakfb4g", "rie6b3pz"], "Rawlsian (Rule Loss)"

project_name = "geometric-governance"
to_plot = [
    utility_utilitarian_welfare_small_sum, utility_nash_welfare_small_sum, utility_rawlsian_welfare_small_sum,
    utility_utilitarian_rule_small_sum, utility_nash_rule_small_sum, utility_rawlsian_rule_small_sum
]

fig, axs = plt.subplots(2, 3, figsize=(12, 8), sharex=True)

for i, plot in enumerate(tqdm(to_plot)):
    run_ids, title = plot
    ax = axs[i // 3][i % 3]

    validation_welfare = []
    num_epochs = 640
    for run_id in run_ids:
        run = api.run(f"{wandb.Api().default_entity}/{project_name}/{run_id}")
        history = run.history(keys=["val/welfare"], samples=num_epochs)
        validation_welfare.append(history["val/welfare"].to_numpy()[:num_epochs])
    validation_welfare = np.stack(validation_welfare)

    ax.set_title(title)
    if i % 3 == 0:
        ax.set_ylabel("Validation Welfare")

    color = "blue"
    mean = np.mean(validation_welfare, axis=0)
    ax.plot(np.arange(num_epochs), mean, color=color, linewidth=1.0)

    std = np.std(validation_welfare, axis=0)
    ax.fill_between(
        np.arange(num_epochs),
        mean + std,
        mean - std,
        color=color,
        alpha=0.2,
        interpolate=True,
        linewidth=0
    )

for ax in axs[-1]:
    ax.set_xlabel("Epoch")

plt.tight_layout()
fig.savefig("welfare_loss_plots")
