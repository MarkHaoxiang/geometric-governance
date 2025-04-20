from typing import Literal
from pydantic import BaseModel
from geometric_governance.train import TrainingSchema
from geometric_governance.util import RangeOrValue


class Dataset(BaseModel):
    dataset_size: int
    num_voters: RangeOrValue
    num_candidates: RangeOrValue
    recompute: bool = False
    seed: int


class Config(BaseModel):
    # Dataset parameters
    voting_rule: str
    dataloader_batch_size: int
    train_dataset: Dataset
    val_dataset: Dataset
    test_dataset: Dataset

    # Training parameters
    train: TrainingSchema

    # Logging
    logging_checkpoint_interval: int
    logging_mode: Literal["online", "offline", "disabled"]

    # Special loss parameters
    monotonicity_loss_enable: bool
    monotonicity_loss_batch_size: int

    # Model parameters
    representation: Literal["graph", "set"]
    model_size: Literal["small", "medium"]

    # Experiment utility
    repeat_number: int | None = None
