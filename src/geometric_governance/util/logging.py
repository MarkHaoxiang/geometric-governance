import os
from typing import Any, Literal
from datetime import datetime
import wandb

from geometric_governance.util.misc import OUTPUT_DIR, get_secrets


def get_default_entity() -> str | None:
    return get_secrets().get("wandb_entity", None)


class Logger:
    def __init__(
        self,
        project: str,
        experiment_name: str,
        entity: str | None = None,
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
        if config is not None:
            config["checkpoint_dir"] = self.checkpoint_dir
        self.mode = mode
        self.project_name = project
        self.entity = entity if entity else get_default_entity()

        os.makedirs(dir, exist_ok=False)
        os.makedirs(self.checkpoint_dir, exist_ok=False)

    def log(self, data: dict[str, Any]):
        wandb.log(data, commit=False)

    def commit(self):
        wandb.log({}, commit=True)

    def begin(self):
        wandb.init(
            entity=self.entity,
            project=self.project_name,
            name=self.experiment_name,
            dir=self.dir,
            config=self.config,
            mode=self.mode,
        )

    @property
    def summary(self):
        return wandb.run.summary

    def close(self):
        wandb.finish()

    def __enter__(self):
        self.begin()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
