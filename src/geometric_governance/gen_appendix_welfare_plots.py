import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb

api = wandb.Api()

utilitarian = ["23tiorc3", "7objypn4", "26lwt2en", "uhkfqg0q", "kd4opz1c"], "Utilitarian Welfare Validation Accuracy"
nash = ["7r1byuv5", "mo4nofv5", "oozydlbw", "dzfdzdoq", "bhsaoz25"], "Nash Welfare Validation Accuracy"
rawlsian_min = ["av2aj0z0", "ei94r82u", "91fnghy5", "wmujgxmw", "b052vinx"], "Rawlsian Welfare + Min. Aggr. Validation Accuracy"
rawlsian_long = ["iabgtxya", "pcrnxh6g", "027297pq", "laibtjp3", "jnnjm4be"], "Rawlsian Welfare + Extra Compute Validation Accuracy"

utilitarian_monotinicity = ["j2fncrrc", "k2rj0cwx", "b2lnp9yc", "k4dl9gd7", "10qwlg4b"], "Utilitarian Welfare + Monotonicity Loss Validation Accuracy"
nash_monotonicity = ["b4ds0xyi", "je9jsqyf", "lr87mw7l", "vkllkmkf", "wvqae6a1"], "Nash Welfare + Monotonicity Loss Validation Accuracy"
rawlsian_min_monotonicity = ["v0vk0w5m", "h2exf694", "oht0sdnd", "95w5qu7y", "mucen6as"], "Rawlsian Welfare + Min. Aggr. \n + Monotonicity Loss Validation Accuracy"
rawlsian_long_monotonicity = ["59e2u3zh", "dtbqj8av", "4hufe116", "wmrkkrh5", "yma4wo71"], "Rawlsian Welfare + Extra Compute \n + Monotonicity Loss Validation Accuracy"

project_name = "geometric-governance"
to_plot = [utilitarian, nash, rawlsian_long, rawlsian_min]
to_plot = [utilitarian_monotinicity, nash_monotonicity, rawlsian_long_monotonicity, rawlsian_min_monotonicity]

fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharex=True)

for i, plot in enumerate(to_plot):
    run_ids, title = plot
    ax = axs[i // 2][i % 2]

    validation_accuracies = []
    num_epochs = 200
    for run_id in run_ids:
        run = api.run(f"{wandb.Api().default_entity}/{project_name}/{run_id}")
        history = run.history(keys=["val/accuracy"])
        validation_accuracies.append(history["val/accuracy"].to_numpy()[:num_epochs])
    validation_accuracies = np.stack(validation_accuracies)

    ax.set_title(title)
    if i % 2 == 0:
        ax.set_ylabel("Accuracy")

    color = "blue"
    mean = np.mean(validation_accuracies, axis=0)
    ax.plot(np.arange(num_epochs), mean, color=color, linewidth=1.0)

    std = np.std(validation_accuracies, axis=0)
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

fig.savefig("welfare_monotonicity")
