"""Offline ranking metrics for the recommendation strategies (Milestone 3).

Pure functions over item identifiers: no I/O, no model, no storage. They are
what ``benchmarks/evaluate_models.py`` applies to every strategy, and they
are unit tested against hand-computed values.

Why these metrics and not plain classification accuracy: a recommender picks
K items out of thousands. Predicting "not relevant" for every item already
scores ~99.9 % accuracy on MovieLens, which says nothing about the ranking.
``classification_accuracy`` is reported anyway, precisely to show that
effect; the metrics that carry information are the ranking ones below.

Definitions, for one user with recommended list ``R`` (length K) and the set
``T`` of held-out items the user rated positively:

* precision@K = |R ∩ T| / K
* recall@K    = |R ∩ T| / |T|
* F1@K        = harmonic mean of precision@K and recall@K
* hit@K       = 1 if |R ∩ T| > 0 else 0
* NDCG@K      = DCG / ideal DCG, with binary relevance and log2 discount, so a
  hit at position 1 is worth more than a hit at position 10.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field


@dataclass(frozen=True)
class UserMetrics:
    """Ranking metrics of one user's Top-K list."""

    precision: float
    recall: float
    f1: float
    ndcg: float
    hit: float
    hits: int


def precision_at_k(recommended: Sequence[int], relevant: set[int], k: int) -> float:
    if k <= 0:
        return 0.0
    hits = sum(1 for item in recommended[:k] if item in relevant)
    return hits / k


def recall_at_k(recommended: Sequence[int], relevant: set[int], k: int) -> float:
    if not relevant:
        return 0.0
    hits = sum(1 for item in recommended[:k] if item in relevant)
    return hits / len(relevant)


def f1_score(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def ndcg_at_k(recommended: Sequence[int], relevant: set[int], k: int) -> float:
    if not relevant or k <= 0:
        return 0.0
    dcg = sum(
        1.0 / math.log2(position + 2)
        for position, item in enumerate(recommended[:k])
        if item in relevant
    )
    ideal = sum(1.0 / math.log2(position + 2) for position in range(min(len(relevant), k)))
    return dcg / ideal if ideal > 0 else 0.0


def user_metrics(recommended: Sequence[int], relevant: set[int], k: int) -> UserMetrics:
    precision = precision_at_k(recommended, relevant, k)
    recall = recall_at_k(recommended, relevant, k)
    hits = sum(1 for item in recommended[:k] if item in relevant)
    return UserMetrics(
        precision=precision,
        recall=recall,
        f1=f1_score(precision, recall),
        ndcg=ndcg_at_k(recommended, relevant, k),
        hit=1.0 if hits > 0 else 0.0,
        hits=hits,
    )


def classification_accuracy(
    recommended: Sequence[int], relevant: set[int], k: int, catalog_size: int
) -> float:
    """Accuracy of the implied binary classifier "recommended = relevant".

    Every catalog item is a prediction: recommended items are predicted
    positive, the rest negative. Reported only to illustrate why accuracy is
    misleading for top-K recommendation (true negatives dominate).
    """
    if catalog_size <= 0:
        return 0.0
    top = set(recommended[:k])
    true_positive = len(top & relevant)
    false_positive = len(top) - true_positive
    false_negative = len(relevant) - true_positive
    true_negative = catalog_size - true_positive - false_positive - false_negative
    return (true_positive + true_negative) / catalog_size


@dataclass
class MetricAccumulator:
    """Averages per-user metrics; also tracks catalog coverage."""

    k: int
    n_users: int = 0
    totals: dict[str, float] = field(
        default_factory=lambda: dict.fromkeys(("precision", "recall", "f1", "ndcg", "hit"), 0.0)
    )
    recommended_items: set[int] = field(default_factory=set)

    def add(self, recommended: Sequence[int], relevant: set[int]) -> UserMetrics:
        metrics = user_metrics(recommended, relevant, self.k)
        self.n_users += 1
        for name in self.totals:
            self.totals[name] += getattr(metrics, name)
        self.recommended_items.update(recommended[: self.k])
        return metrics

    def summary(self, catalog_size: int) -> dict[str, float]:
        if self.n_users == 0:
            return {"users": 0}
        result: dict[str, float] = {
            f"{name}@{self.k}": round(total / self.n_users, 4)
            for name, total in self.totals.items()
        }
        result["users"] = self.n_users
        result["catalog_coverage"] = (
            round(len(self.recommended_items) / catalog_size, 4) if catalog_size else 0.0
        )
        return result


def mean_popularity_percentile(
    recommended_lists: Iterable[Sequence[int]], popularity_rank: Mapping[int, int], n_items: int
) -> float:
    """Average popularity percentile of the recommended items (0 = most popular).

    A value near 0 means the strategy mostly recommends blockbusters
    (popularity bias); higher values mean more long-tail recommendations.
    """
    values = [
        popularity_rank[item] / n_items
        for items in recommended_lists
        for item in items
        if item in popularity_rank
    ]
    return round(sum(values) / len(values), 4) if values else 0.0


__all__ = [
    "MetricAccumulator",
    "UserMetrics",
    "classification_accuracy",
    "f1_score",
    "mean_popularity_percentile",
    "ndcg_at_k",
    "precision_at_k",
    "recall_at_k",
    "user_metrics",
]
