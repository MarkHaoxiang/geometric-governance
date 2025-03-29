import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb

api = wandb.Api()

rawlsian_run_ids = ["iabgtxya", "pcrnxh6g", "027297pq", "laibtjp3", "jnnjm4be"]
utilitarian_run_ids = ["23tiorc3", "7objypn4", "26lwt2en", "uhkfqg0q", "kd4opz1c"]
nash_run_ids = ["7r1byuv5", "mo4nofv5", "oozydlbw", "dzfdzdoq", "bhsaoz25"]
utilitarian_monotinicity_run_ids = ["j2fncrrc", "k2rj0cwx", "b2lnp9yc", "k4dl9gd7", "10qwlg4b"]

project_name = "geometric-governance"
run_ids = nash_run_ids
title = "Nash Welfare Validation Accuracy"

validation_accuracies = []
num_epochs = 200
for run_id in run_ids:
    run = api.run(f"{wandb.Api().default_entity}/{project_name}/{run_id}")
    history = run.history(keys=["val/accuracy"])
    validation_accuracies.append(history["val/accuracy"].to_numpy())
validation_accuracies = np.stack(validation_accuracies)

fig, (ax) = plt.subplots(1, 1, figsize=(12, 8))
ax.set_title(title)
ax.set_xlabel("Epoch")
ax.set_ylabel("Accuracy")

color = "blue"
mean = np.mean(validation_accuracies, axis=0)
ax.plot(np.arange(num_epochs), mean, color=color, linewidth=1.0)

std = np.std(validation_accuracies, axis=0)
breakpoint()
ax.fill_between(
    np.arange(num_epochs),
    mean + std,
    mean - std,
    color=color,
    alpha=0.2,
    interpolate=True,
    linewidth=0
)

fig.savefig(title)
breakpoint()
