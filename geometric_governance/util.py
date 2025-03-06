import os
from typing import Any, Literal
from datetime import datetime

import wandb
import numpy as np


RangeOrValue = tuple[int, int] | int

DATA_DIR = os.path.join(os.path.dirname(__file__), "../outputs")


class Logger:
    def __init__(
        self,
        experiment_name: str,
        config: dict | None = None,
        mode: Literal["online", "offline", "disabled"] = "online",
    ):
        super().__init__()
        now = datetime.now()
        dir = os.path.join(
            DATA_DIR,
            experiment_name,
            now.strftime("%Y-%m-%d %H-%M-%S"),
        )

        self.experiment_name = experiment_name
        self.dir = dir
        self.checkpoint_dir = os.path.join(dir, "checkpoints")

        os.makedirs(dir, exist_ok=False)
        os.makedirs(self.checkpoint_dir, exist_ok=False)

        wandb.init(
            project="geometric-governance",
            name=experiment_name,
            dir=self.dir,
            config=config,
            mode=mode,
        )

    def log(self, data: dict[str, Any]):
        wandb.log(data, commit=False)

    def commit(self):
        wandb.log({}, commit=True)


def get_value(v: RangeOrValue, rng: np.random.Generator) -> int:
    if isinstance(v, tuple):
        return rng.integers(low=v[0], high=v[1])
    return v
