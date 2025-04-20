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
    welfare_rule: Literal["utilitarian", "nash", "rawlsian"]
    vote_data: Literal["ranking", "utility"]
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
    welfare_loss_enable: bool

    # Model parameters
    model_size: Literal["small", "large"]
    model_aggr: str

    # Experiment utility
    repeat_number: int | None = None
