"""Temporal holdout used by the Milestone 3 evaluation."""

from __future__ import annotations

import pandas as pd
import pytest

from data.pipeline.split import temporal_holdout


def _ratings(user_sizes: dict[int, int]) -> pd.DataFrame:
    rows = []
    for user, size in user_sizes.items():
        for t in range(size):
            rows.append({"user_id": user, "item_id": 1000 + t, "rating": 4.0, "timestamp": t})
    return pd.DataFrame(rows)


def test_the_most_recent_ratings_go_to_test():
    train, test = temporal_holdout(_ratings({1: 10}), test_fraction=0.2, min_ratings=10)
    assert sorted(test["timestamp"]) == [8, 9]
    assert train["timestamp"].max() < test["timestamp"].min()


def test_train_and_test_are_disjoint_and_complete():
    ratings = _ratings({1: 20, 2: 15})
    train, test = temporal_holdout(ratings, test_fraction=0.2, min_ratings=10)
    assert len(train) + len(test) == len(ratings)
    merged = train.merge(test, on=["user_id", "item_id"])
    assert merged.empty


def test_users_below_the_minimum_stay_in_training():
    train, test = temporal_holdout(_ratings({1: 4, 2: 12}), min_ratings=10)
    assert 1 not in set(test["user_id"])
    assert (train["user_id"] == 1).sum() == 4


def test_every_eligible_user_keeps_at_least_one_test_rating():
    _, test = temporal_holdout(_ratings({1: 10}), test_fraction=0.01, min_ratings=10)
    assert len(test) == 1


@pytest.mark.parametrize("fraction", [0.0, 1.0, -0.1])
def test_invalid_fraction_is_rejected(fraction):
    with pytest.raises(ValueError):
        temporal_holdout(_ratings({1: 10}), test_fraction=fraction)
