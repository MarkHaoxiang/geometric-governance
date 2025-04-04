import matplotlib.pyplot as plt
import numpy as np
import wandb
from tqdm import tqdm

api = wandb.Api()

# Freeze, welfare
utilitarian_robust = ["1yoswtdj", "f4c7so09", "zblb7u6n", "y0he8pbl", "hkikoo76"]
utilitarian_train = ["qki3whke", "tasyw2we", "vqxhkklq", "c39rdwj3", "51f5rkp4"]
utilitarian_freeze = ["ex8rlxg1", "1mvyyh6o", "3iewlox9", "oss70i7j", "rv0lyho6"]

project_name = "geometric-governance"


def moving_average(x, window_size):
    return np.convolve(x, np.ones(window_size), "valid") / window_size


window_size = 25  # Number of epochs to calculate moving average over

fig, axs = plt.subplots(1, 2, figsize=(12, 4), sharex=True)
fig.suptitle("Strategic Adversarial Training (Utilitarian)")
axs[0].set_title("Social Welfare")
axs[1].set_title("GESN Rational Loss")

api = wandb.Api()
colors = ["blue", "red", "green"]

baseline_value = 6.01282

for i, (run_ids, label) in enumerate(
    tqdm(
        [
            (utilitarian_robust, "robust (freeze)"),
            (utilitarian_train, "robust (train)"),
            (utilitarian_freeze, "standard (freeze)"),
        ]
    )
):
    validation_welfare = []
    strategy_loss = []
    num_epochs = 400
    for run_id in run_ids:
        run = api.run(f"{api.default_entity}/{project_name}/{run_id}")
        history = run.history(
            keys=["val/welfare", "train/strategy_loss"], samples=num_epochs
        )
        validation_welfare.append(history["val/welfare"].to_numpy()[:num_epochs])
        strategy_loss.append(history["train/strategy_loss"].to_numpy()[:num_epochs])
    validation_welfare = np.stack(validation_welfare)
    strategy_loss = np.stack(strategy_loss)

    # Welfare
    color = colors[i]
    mean = np.mean(validation_welfare, axis=0)
    mean = moving_average(mean, window_size)

    ax = axs[0]
    (line,) = ax.plot(
        np.arange(num_epochs)[window_size - 1 :], mean, color=color, linewidth=1.0
    )
    line.set_label(label)

    std = np.std(validation_welfare, axis=0)
    std = moving_average(std, window_size)
    ax.fill_between(
        np.arange(num_epochs)[window_size - 1 :],
        mean + std,
        mean - std,
        color=color,
        alpha=0.2,
        interpolate=True,
        linewidth=0,
    )
    if i == 0:
        ax.axhline(
            y=baseline_value,
            color="black",
            linestyle="dotted",
            linewidth=1.0,
            label="honest voting",
        )

    # Strategy Loss
    mean = np.mean(strategy_loss, axis=0)
    mean = moving_average(mean, window_size)

    ax = axs[1]
    (line,) = ax.plot(
        np.arange(num_epochs)[window_size - 1 :], mean, color=color, linewidth=1.0
    )
    line.set_label(label)

    std = np.std(strategy_loss, axis=0)
    std = moving_average(std, window_size)
    ax.fill_between(
        np.arange(num_epochs)[window_size - 1 :],
        mean + std,
        mean - std,
        color=color,
        alpha=0.2,
        interpolate=True,
        linewidth=0,
    )

ax = axs[0]
ax.set_xlabel("Epoch")
ax.set_ylabel("Welfare")

ax = axs[1]
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")

handles, labels = axs[0].get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    loc="lower center",
    bbox_to_anchor=(0.5, -0.05),
    ncol=4,
    frameon=False,
)

plt.tight_layout()
fig.subplots_adjust(bottom=0.2)
fig.savefig("strategy_loss_plots.png", bbox_inches="tight", dpi=300)
