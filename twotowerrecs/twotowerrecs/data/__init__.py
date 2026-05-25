"""MovieLens dataset loader and preprocessor.

We use ML-25M because it's representative of "real" rec system data — long
tail, sparse, with timestamps. The 25M is overkill for prototyping but the
training script handles it; for fast iteration use ML-1M.

The data is downloaded once into `data/raw/` and processed into integer-encoded
parquet files in `data/processed/`. The mapping from raw IDs to integer
indices is the most important artifact — both the training and serving code
need consistent mappings.
"""
from __future__ import annotations

import io
import logging
import os
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

ML_25M_URL = "https://files.grouplens.org/datasets/movielens/ml-25m.zip"
ML_1M_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"


@dataclass
class MovieLensData:
    interactions: pd.DataFrame  # cols: user_idx, movie_idx, rating, timestamp
    movies: pd.DataFrame        # cols: movie_idx, movie_id, title, genres
    n_users: int
    n_movies: int

    @property
    def n_interactions(self) -> int:
        return len(self.interactions)


def download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        log.info("Using cached download at %s", dest)
        return dest
    log.info("Downloading %s -> %s ...", url, dest)
    with urllib.request.urlopen(url) as r, dest.open("wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    return dest


def load_movielens(
    variant: str = "1m",
    cache_dir: str | Path = "data/raw",
    rating_threshold: float = 3.5,
) -> MovieLensData:
    """Load MovieLens. variant is '1m' or '25m'.

    Only interactions with rating >= rating_threshold are kept (we treat this
    as implicit feedback / positive signal).
    """
    cache_dir = Path(cache_dir)
    if variant == "1m":
        zip_path = download(ML_1M_URL, cache_dir / "ml-1m.zip")
        df = _read_ml_1m(zip_path)
    elif variant == "25m":
        zip_path = download(ML_25M_URL, cache_dir / "ml-25m.zip")
        df = _read_ml_25m(zip_path)
    else:
        raise ValueError(f"Unknown variant {variant!r}")

    log.info("Loaded %d raw interactions", len(df))
    df = df[df["rating"] >= rating_threshold].copy()
    log.info("Filtered to %d interactions with rating >= %.1f",
             len(df), rating_threshold)

    # Integer-encode users and movies. Stable ordering matters for reproducibility.
    user_ids = sorted(df["user_id"].unique())
    movie_ids = sorted(df["movie_id"].unique())
    user_idx = {uid: i for i, uid in enumerate(user_ids)}
    movie_idx = {mid: i for i, mid in enumerate(movie_ids)}
    df["user_idx"] = df["user_id"].map(user_idx).astype(np.int32)
    df["movie_idx"] = df["movie_id"].map(movie_idx).astype(np.int32)

    # Build the movies table (we need titles + genres for the frontend)
    movies = _read_movies_table(zip_path)
    movies = movies[movies["movie_id"].isin(movie_idx.keys())].copy()
    movies["movie_idx"] = movies["movie_id"].map(movie_idx).astype(np.int32)

    return MovieLensData(
        interactions=df[["user_idx", "movie_idx", "rating", "timestamp"]],
        movies=movies[["movie_idx", "movie_id", "title", "genres"]],
        n_users=len(user_ids),
        n_movies=len(movie_ids),
    )


def _read_ml_1m(zip_path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as z:
        with z.open("ml-1m/ratings.dat") as f:
            df = pd.read_csv(
                f,
                sep="::",
                engine="python",
                names=["user_id", "movie_id", "rating", "timestamp"],
                encoding="latin-1",
            )
    return df


def _read_ml_25m(zip_path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as z:
        with z.open("ml-25m/ratings.csv") as f:
            df = pd.read_csv(f)
            df = df.rename(columns={"userId": "user_id", "movieId": "movie_id"})
    return df


def _read_movies_table(zip_path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as z:
        # Try both ML formats
        for name in z.namelist():
            if name.endswith("movies.dat"):
                with z.open(name) as f:
                    return pd.read_csv(
                        f, sep="::", engine="python",
                        names=["movie_id", "title", "genres"],
                        encoding="latin-1",
                    )
            if name.endswith("movies.csv"):
                with z.open(name) as f:
                    df = pd.read_csv(f)
                    return df.rename(columns={"movieId": "movie_id"})
    raise FileNotFoundError("Couldn't find movies table in zip")


def train_test_split_temporal(
    interactions: pd.DataFrame,
    test_frac: float = 0.1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out the most recent `test_frac` interactions per user as test.
    This is the right way to evaluate recsys — random splits leak future info."""
    interactions = interactions.sort_values(["user_idx", "timestamp"]).reset_index(drop=True)

    def _split(group):
        n = len(group)
        cut = max(1, int(n * (1 - test_frac)))
        return group.iloc[:cut], group.iloc[cut:]

    train_parts = []
    test_parts = []
    for _, g in interactions.groupby("user_idx"):
        tr, te = _split(g)
        train_parts.append(tr)
        test_parts.append(te)
    return pd.concat(train_parts), pd.concat(test_parts)
