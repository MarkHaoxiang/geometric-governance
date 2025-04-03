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

# Ranking
# Welfare Loss
ranking_utilitarian_welfare_small_sum = ["e633j43c", "dgdxw04n", "0jcuxn52", "059jax08", "58w1v0tn"], "Utilitarian"
ranking_nash_welfare_small_sum = ["d7e0nxa9", "f90638a1", "qlievumy", "9uzo3672", "y45yhnqw"], "Nash"
ranking_rawlsian_welfare_small_sum = ["dk8picj4", "zz2iwh4o", "py1qsmsj", "y9pt4sdk", "hcq84p0m"], "Rawlsian"

# Rule Loss
ranking_utilitarian_rule_small_sum = ["wbmj8opt", "9cxehvgw", "pxw811q1", "3b5ag1e2", "gv5337uc"], "Utilitarian"
ranking_nash_rule_small_sum = ["gyb8fxsr", "ab9pvxu4", "7loino67", "1y4pshou", "e8nr6rc2"], "Nash"
ranking_rawlsian_rule_small_sum = ["0gqzx2gl", "262ydrax", "nqyn8mfb", "ffo4hnnz", "pzxjxddz"], "Rawlsian"

project_name = "geometric-governance"
to_plot = [
    utility_utilitarian_welfare_small_sum, utility_nash_welfare_small_sum, utility_rawlsian_welfare_small_sum,
    utility_utilitarian_rule_small_sum, utility_nash_rule_small_sum, utility_rawlsian_rule_small_sum
]
to_plot = [
    ranking_utilitarian_welfare_small_sum, ranking_nash_welfare_small_sum, ranking_rawlsian_welfare_small_sum,
    ranking_utilitarian_rule_small_sum, ranking_nash_rule_small_sum, ranking_rawlsian_rule_small_sum
]

def moving_average(x, window_size):
    return np.convolve(x, np.ones(window_size), 'valid') / window_size 

window_size = 25 # Number of epochs to calculate moving average over

fig, axs = plt.subplots(1, 3, figsize=(12, 4), sharex=True)

for i, plot in enumerate(tqdm(to_plot)):
    run_ids, title = plot
    ax = axs[i % 3]

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
    if i >= 3:
        color = "red"
    mean = np.mean(validation_welfare, axis=0)
    mean = moving_average(mean, window_size)
    line, =  ax.plot(np.arange(num_epochs)[window_size - 1:], mean, color=color, linewidth=1.0)
    line_label = "Rule Loss" if i >=3 else "Welfare Loss"
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

for ax in axs:
    ax.set_xlabel("Epoch")
    ax.legend()

plt.tight_layout()
fig.savefig("welfare_loss_plots")
