import os
from typing import Any, Literal
from datetime import datetime

import wandb
import numpy as np


RangeOrValue = tuple[int, int] | int

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "../outputs")


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
            OUTPUT_DIR,
            experiment_name,
            now.strftime("%Y-%m-%d %H-%M-%S"),
        )

        self.experiment_name = experiment_name
        self.dir = dir
        self.checkpoint_dir = os.path.join(dir, "checkpoints")
        self.config = config
        self.mode = mode

        os.makedirs(dir, exist_ok=False)
        os.makedirs(self.checkpoint_dir, exist_ok=False)

    def log(self, data: dict[str, Any]):
        wandb.log(data, commit=False)

    def commit(self):
        wandb.log({}, commit=True)

    def begin(self):
        wandb.init(
            project="geometric-governance",
            name=self.experiment_name,
            dir=self.dir,
            config=self.config,
            mode=self.mode,
        )

    def close(self):
        wandb.finish()

    def __enter__(self):
        self.begin()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


def get_value(v: RangeOrValue, rng: np.random.Generator) -> int:
    if isinstance(v, tuple):
        return rng.integers(low=v[0], high=v[1])
    return v
