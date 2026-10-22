"""Evaluation split - temporal per-user holdout (Milestone 3).

The production pipeline builds every user vector from *all* of that user's
ratings. Measuring a model with those vectors would score it on interactions
it has already seen, which inflates every metric (data leakage). This stage
produces an honest train/test split instead:

* for every user with at least ``min_ratings`` ratings, the most recent
  ``test_fraction`` of their ratings (by timestamp) is held out as the test
  set, and the rest is the training set;
* users with fewer ratings stay entirely in training (they cannot be
  evaluated without leaving them with no history at all).

The split is temporal, not random, because a recommender is always asked to
predict the *future* from the past: a random split would let the model learn
from ratings made after the ones it is tested on.

The training ratings are written in the same layout as the ``clean`` stage,
so the unchanged ``transform`` and ``materialize`` stages can run on them:

    <eval_root>/clean/ratings.parquet      training ratings
    <eval_root>/clean/items.parquet        catalog (copied)
    <eval_root>/clean/tags.parquet         tags (copied, when present)
    <eval_root>/holdout/ratings.parquet    held-out test ratings
    <eval_root>/holdout/split_report.json

It never touches the production artifacts.
"""

from __future__ import annotations

import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config.settings import Settings, load_settings
from data.pipeline import get_logger, write_report

LOGGER = get_logger("split")

#: Fraction of each eligible user's most recent ratings held out for testing.
DEFAULT_TEST_FRACTION = 0.2
#: Users with fewer ratings than this are kept entirely in training.
DEFAULT_MIN_RATINGS = 10


def temporal_holdout(
    ratings: pd.DataFrame,
    *,
    test_fraction: float = DEFAULT_TEST_FRACTION,
    min_ratings: int = DEFAULT_MIN_RATINGS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split ratings into (train, test) with a per-user temporal holdout.

    Ties on the timestamp are broken by item identifier, so the split is
    deterministic for a given input.
    """
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be between 0 and 1")
    if min_ratings < 2:
        raise ValueError("min_ratings must be at least 2")

    ordered = ratings.sort_values(["user_id", "timestamp", "item_id"], kind="mergesort")
    position = ordered.groupby("user_id").cumcount()
    size = ordered.groupby("user_id")["item_id"].transform("size")

    n_test = np.floor(size * test_fraction).astype(int).clip(lower=1)
    eligible = size >= min_ratings
    is_test = eligible & (position >= size - n_test)

    train = ordered[~is_test].reset_index(drop=True)
    test = ordered[is_test].reset_index(drop=True)
    return train, test


def run(
    settings: Settings | None = None,
    *,
    eval_root: Path,
    test_fraction: float = DEFAULT_TEST_FRACTION,
    min_ratings: int = DEFAULT_MIN_RATINGS,
) -> Mapping[str, Any]:
    """Read the production clean artifacts and write the evaluation split."""
    settings = settings or load_settings()
    source = settings.pipeline.interim_root / "clean"
    ratings_path = source / "ratings.parquet"
    if not ratings_path.exists():
        raise FileNotFoundError(f"missing {ratings_path}; run the offline pipeline first")

    ratings = pd.read_parquet(ratings_path)
    train, test = temporal_holdout(ratings, test_fraction=test_fraction, min_ratings=min_ratings)

    clean_dir = Path(eval_root) / "clean"
    holdout_dir = Path(eval_root) / "holdout"
    clean_dir.mkdir(parents=True, exist_ok=True)
    holdout_dir.mkdir(parents=True, exist_ok=True)

    train.to_parquet(clean_dir / "ratings.parquet", index=False)
    test.to_parquet(holdout_dir / "ratings.parquet", index=False)
    for name in ("items", "tags"):
        path = source / f"{name}.parquet"
        if path.exists():
            shutil.copyfile(path, clean_dir / f"{name}.parquet")

    summary: dict[str, Any] = {
        "stage": "split",
        "method": "temporal per-user holdout",
        "test_fraction": test_fraction,
        "min_ratings": min_ratings,
        "ratings_total": int(len(ratings)),
        "ratings_train": int(len(train)),
        "ratings_test": int(len(test)),
        "users_total": int(ratings["user_id"].nunique()),
        "users_evaluated": int(test["user_id"].nunique()),
        "latest_train_timestamp_median": int(train.groupby("user_id")["timestamp"].max().median()),
    }
    write_report(holdout_dir / "split_report.json", summary)
    LOGGER.info(
        "split train=%d test=%d users_evaluated=%d",
        summary["ratings_train"],
        summary["ratings_test"],
        summary["users_evaluated"],
    )
    return summary


__all__ = ["DEFAULT_MIN_RATINGS", "DEFAULT_TEST_FRACTION", "run", "temporal_holdout"]
