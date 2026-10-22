"""Regression tests for the issues found when reviewing the Milestone 2 code."""

from __future__ import annotations

import numpy as np
from fastapi.testclient import TestClient

from ai.decorators import StrategyDecorator
from ai.decorators.caching_decorator import CachingDecorator, InMemoryCacheBackend
from ai.interfaces import UserFeatures, top_k
from ai.strategies.popularity_strategy import PopularityStrategy
from application.api import create_app
from config.settings import Settings
from messaging.feedback_history_service import FeedbackHistoryService
from messaging.interaction_event_queue import (
    InMemoryInteractionEventQueue,
    InteractionEvent,
    RetryBuffer,
)
from tests.conftest import StubStrategy


def test_top_k_never_pads_the_ranking_with_seen_items():
    scores = np.array([1.0, -np.inf, -np.inf], dtype=np.float32)
    ranking = top_k(scores, [1, 2, 3], k=3)
    assert [r.item_id for r in ranking] == [1]
    assert all(np.isfinite(r.score) for r in ranking)


def test_small_category_returns_fewer_items_instead_of_seen_ones():
    strategy = PopularityStrategy(
        np.array([1, 2, 3, 4]),
        np.array([4.0, 3.0, 2.0, 1.0]),
        category_rankings={"Drama": (np.array([1, 2, 3]), np.array([3.0, 2.0, 1.0]))},
    )
    user = UserFeatures(user_id=1, seen_items=(1, 2), preferred_category="Drama")
    assert [r.item_id for r in strategy.predict(user, 3)] == [3]


def test_cache_key_separates_strategies_sharing_a_backend(user_features):
    backend = InMemoryCacheBackend()

    class Other(StubStrategy):
        strategy_id = "other"

    first = CachingDecorator(StubStrategy([101]), backend=backend, ttl_seconds=60)
    second = CachingDecorator(Other([202]), backend=backend, ttl_seconds=60)
    assert first.cache_key(user_features, 5) != second.cache_key(user_features, 5)
    first.predict(user_features, 5)
    assert [r.item_id for r in second.predict(user_features, 5)] == [202]


def test_the_fallback_runs_through_the_decorator_chain(settings: Settings):
    with TestClient(create_app(settings)) as client:
        fallback = client.app.state.components.controller._fallback  # noqa: SLF001
    assert isinstance(fallback, StrategyDecorator)
    assert fallback.strategy_id == "popularity"


def test_the_worker_replays_the_retry_buffer(tmp_path):
    buffer = RetryBuffer(tmp_path / "retry.jsonl")
    buffer.append(InteractionEvent(user_id=1, event_type="click", item_id=5))

    class Repo:
        def __init__(self):
            self.records = []

        def append(self, record):
            self.records.append(record)
            return True

    repo = Repo()
    worker = FeedbackHistoryService(
        InMemoryInteractionEventQueue(), repo, retry_buffer=buffer, drain_interval_seconds=0
    )
    assert worker.maybe_drain_retry_buffer(now=10.0) == 1
    assert len(repo.records) == 1
    assert not buffer.path.exists()
