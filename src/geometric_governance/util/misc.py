import os
from typing import Type

from omegaconf import DictConfig, OmegaConf
import numpy as np
import torch


type RangeOrValue = tuple[int, int] | int

_FILE_DIR = os.path.dirname(__file__)
_SECRETS_FILE = os.path.join(_FILE_DIR, "../../../.secrets.yaml")
OUTPUT_DIR = os.path.join(_FILE_DIR, "../../../outputs")
DATA_DIR = os.path.join(_FILE_DIR, "../../../data")

cuda = torch.device(0) if torch.cuda.is_available() else torch.device("cpu")


def get_secrets():
    if not os.path.exists(_SECRETS_FILE):
        return {}
    return OmegaConf.to_container(OmegaConf.load(_SECRETS_FILE))


def get_value(v: RangeOrValue, rng: np.random.Generator) -> int:
    if isinstance(v, tuple):
        return int(rng.integers(low=v[0], high=v[1]))
    return v


def get_max(v: RangeOrValue) -> int:
    if isinstance(v, tuple):
        return v[1]
    elif isinstance(v, int):
        return v
    else:
        raise ValueError(f"Invalid value type {type(v)}")


def omega_to_pydantic[T](config: DictConfig, config_cls: Type[T]) -> T:
    """Converts Hydra config to Pydantic config."""
    config_dict = OmegaConf.to_object(config)  # type: ignore[assignment]
    return config_cls(**config_dict)  # type: ignore


def get_parameter_count(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
