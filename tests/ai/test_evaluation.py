"""Ranking metrics, checked against hand-computed values."""

from __future__ import annotations

import math

import pytest

from ai.evaluation import (
    MetricAccumulator,
    classification_accuracy,
    f1_score,
    mean_popularity_percentile,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)

RECOMMENDED = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
RELEVANT = {2, 9, 50, 60, 70}


def test_precision_and_recall_match_the_worked_example():
    assert precision_at_k(RECOMMENDED, RELEVANT, 10) == pytest.approx(0.2)
    assert recall_at_k(RECOMMENDED, RELEVANT, 10) == pytest.approx(0.4)


def test_f1_is_the_harmonic_mean():
    assert f1_score(0.2, 0.4) == pytest.approx(2 * 0.2 * 0.4 / 0.6)
    assert f1_score(0.0, 0.0) == 0.0


def test_ndcg_rewards_hits_near_the_top():
    # item 2 sits at position 2 (discount log2(3)), item 9 at position 9 (log2(10))
    expected_dcg = 1 / math.log2(3) + 1 / math.log2(10)
    ideal = sum(1 / math.log2(i + 2) for i in range(5))
    assert ndcg_at_k(RECOMMENDED, RELEVANT, 10) == pytest.approx(expected_dcg / ideal)
    assert ndcg_at_k([2, 9], {2, 9}, 2) == pytest.approx(1.0)
    assert ndcg_at_k([1, 2], {2}, 2) < ndcg_at_k([2, 1], {2}, 2)


def test_empty_relevant_set_scores_zero():
    assert recall_at_k(RECOMMENDED, set(), 10) == 0.0
    assert ndcg_at_k(RECOMMENDED, set(), 10) == 0.0


def test_accuracy_is_dominated_by_true_negatives():
    # 2 hits, 8 false positives, 3 false negatives, out of 10 000 items.
    accuracy = classification_accuracy(RECOMMENDED, RELEVANT, 10, 10_000)
    assert accuracy == pytest.approx((2 + 10_000 - 2 - 8 - 3) / 10_000)
    assert accuracy > 0.99


def test_accumulator_averages_and_tracks_coverage():
    acc = MetricAccumulator(k=2)
    acc.add([1, 2], {1})
    acc.add([3, 4], {9})
    summary = acc.summary(catalog_size=10)
    assert summary["precision@2"] == pytest.approx(0.25)
    assert summary["hit@2"] == pytest.approx(0.5)
    assert summary["catalog_coverage"] == pytest.approx(0.4)
    assert summary["users"] == 2


def test_popularity_percentile():
    ranks = {1: 1, 2: 2, 3: 3, 4: 4}
    assert mean_popularity_percentile([[1, 2]], ranks, 4) == pytest.approx(0.375)
