import numpy as np

RangeOrValue = tuple[int, int] | int


def get_value(v: RangeOrValue, rng: np.random.Generator) -> int:
    if isinstance(v, tuple):
        return rng.integers(low=v[0], high=v[1])
    return v
