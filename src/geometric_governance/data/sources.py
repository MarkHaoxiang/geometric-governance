import os
from typing import Callable, Literal
from functools import cache
import numpy as np
from numpy import typing as npt
import pandas as pd

from geometric_governance.data.election_data import ElectionData
from geometric_governance.util import DATA_DIR


def generate_dirichlet_election(
    num_voters: int,
    num_candidates: int,
    rng: np.random.Generator | None = None,
    utility_profile_alpha: float = 1.0,
):
    if rng is None:
        rng = np.random.default_rng()

    voter_utilities = rng.dirichlet(
        alpha=(utility_profile_alpha,) * num_candidates, size=num_voters
    )

    return ElectionData(
        num_voters=num_voters,
        num_candidates=num_candidates,
        voter_utilities=voter_utilities,
    )


def generate_spatial_election(
    num_voters: int,
    num_candidates: int,
    rng: np.random.Generator | None = None,
    k: int = 3,
    f: Callable[
        [npt.NDArray[np.float32]], npt.NDArray[np.float32]
    ] = lambda x: np.maximum(0, 1 - x),
):
    if rng is None:
        rng = np.random.default_rng()
    voters = rng.uniform(low=0.0, high=1.0, size=(num_voters, k))
    candidates = rng.uniform(low=0.0, high=1.0, size=(num_candidates, k))

    diff = voters[:, np.newaxis, :] - candidates[np.newaxis, :, :]
    dist_matrix = np.linalg.norm(diff, axis=-1)

    voter_utilities = f(dist_matrix)

    return ElectionData(
        num_voters=num_voters,
        num_candidates=num_candidates,
        voter_utilities=voter_utilities,
    )


@cache
def _read_grenoble_data():
    df = pd.read_csv(os.path.join(DATA_DIR, "GrenobleData_2017_Presid+.csv"), sep=";")
    df.columns = [col.strip() for col in df.columns]
    EV_COLUMNS = [col for col in df.columns if col.startswith("EV")]
    EV_COLUMNS.remove("EV_OPINION")
    df = df[EV_COLUMNS]
    df = df.map(lambda x: x.strip() if isinstance(x, str) else x)
    df = df.replace(to_replace=["None"], value=np.nan)
    return df, EV_COLUMNS


def generate_grenoble_election(
    num_voters: int,
    num_candidates: int,
    rng: np.random.Generator | None = None,
):
    if rng is None:
        rng = np.random.default_rng()
    df, candidates = _read_grenoble_data()

    assert num_candidates <= len(candidates)
    candidates = rng.choice(candidates, size=num_candidates)
    df = df[candidates].dropna()
    df = df[~(df == 0).all(axis=1)]

    assert num_voters <= len(df)
    df = df.astype(float)
    voter_utilities = df.to_numpy()
    idxs = np.random.choice(voter_utilities.shape[0], num_voters, replace=False)
    voter_utilities = voter_utilities[idxs]

    return ElectionData(
        num_voters=num_voters,
        num_candidates=num_candidates,
        voter_utilities=voter_utilities,
    )


@cache
def _load_movielens_data(min_rating_num: int = 20_000):
    df = pd.read_csv(os.path.join(DATA_DIR, "ml-32m", "ratings.csv"))
    movie_counts = df.groupby("movieId").size()
    mask = movie_counts[movie_counts >= min_rating_num].index
    df_filtered = df[df["movieId"].isin(mask)]
    movie_ids = df_filtered["movieId"].unique()
    return df_filtered, movie_ids


def generate_movielens_dataset(
    num_voters: int,
    num_candidates: int,
    rng: np.random.Generator | None = None,
    min_rating_num: int = 20_000,
    max_retries: int = 10,
):
    if rng is None:
        rng = np.random.default_rng()
    df, movie_ids = _load_movielens_data(min_rating_num)

    while True:
        movie_idxs = rng.choice(movie_ids, size=num_candidates, replace=False)
        df_filtered = df[df["movieId"].isin(movie_idxs)]
        voter_counts = df_filtered.groupby("userId").size()
        voter_idxs = voter_counts[voter_counts >= num_candidates].index
        if len(voter_idxs) >= num_voters:
            break
        max_retries -= 1

        if max_retries <= 0:
            raise ValueError(
                f"""Unable to find a valid set of {num_candidates} candidates with at least {num_voters} voters. Consider increasing {min_rating_num}, decreasing {num_candidates}, or decreasing {num_voters}."""
            )

    voter_idxs = np.random.choice(voter_idxs, size=num_voters, replace=False)
    df_filtered = df_filtered[df_filtered["userId"].isin(voter_idxs)]

    voter_utilities = df_filtered.pivot_table(
        index="userId", columns="movieId", values="rating"
    ).to_numpy()

    row_perm = np.random.permutation(num_voters)
    col_perm = np.random.permutation(num_candidates)
    voter_utilities = voter_utilities[row_perm, :][:, col_perm]

    return ElectionData(
        num_voters=num_voters,
        num_candidates=num_candidates,
        voter_utilities=voter_utilities,
    )


type DatasetSource = Literal["dirichlet", "spatial", "grenoble", "movielens"]
DatasetRegistry: dict[DatasetSource, Callable] = {
    "dirichlet": generate_dirichlet_election,
    "spatial": generate_spatial_election,
    "grenoble": generate_grenoble_election,
    "movielens": generate_movielens_dataset,
}
