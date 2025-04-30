from typing import Literal
from pydantic import BaseModel
from geometric_governance.train import TrainingSchema
from geometric_governance.util import RangeOrValue

from geometric_governance.data import DatasetSource


class Dataset(BaseModel):
    dataset_size: int
    num_voters: RangeOrValue
    num_candidates: RangeOrValue
    seed: int
    shuffle: bool = True
    recompute: bool = False


class Model(BaseModel):
    use_manual_election: bool
    size: Literal["small", "large"]
    from_pretrained: None | str
    freeze_weights: bool


class Config(BaseModel):
    # Dataset parameters
    welfare_rule: Literal["utilitarian", "nash", "rawlsian"]
    vote_data: Literal["ranking", "utility"]
    vote_source: DatasetSource
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
    monotonicity_loss_calculate: bool
    monotonicity_loss_train: bool
    monotonicity_loss_batch_size: int

    # Model
    election_model: Model

    # Experiment utility
    repeat_number: int | None = None
    strategy_module_enable: bool
    strategy_p: float
