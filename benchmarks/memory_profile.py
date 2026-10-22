"""Memory profiling of the inference service components (Milestone 3).

    python benchmarks/memory_profile.py
    python benchmarks/memory_profile.py --requests 2000 --json

Uses ``tracemalloc`` (standard library) to answer three questions:

1. How much memory does each component allocate when it is built?
   (feature store, catalog/profile repositories, each strategy)
2. How much does the whole composed service hold?
3. Does serving requests leak memory? Python allocations are measured after
   a warm-up and again after N requests: a steady service should stay flat.

It also reports the process resident set size (RSS) through ``psutil``, which
includes memory that tracemalloc cannot see (NumPy/PyTorch native buffers).

Output: ``reports/memory_profile.json``.
"""

from __future__ import annotations

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse  # noqa: E402
import gc  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import tracemalloc  # noqa: E402
from collections.abc import Callable  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import numpy as np  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

MB = 1024 * 1024


def _rss_mb() -> float | None:
    try:
        import psutil

        return round(psutil.Process().memory_info().rss / MB, 1)
    except ImportError:
        return None


def _measure(build: Callable[[], Any]) -> tuple[Any, float]:
    """Return (object, MB allocated by Python while building it)."""
    gc.collect()
    before, _ = tracemalloc.get_traced_memory()
    obj = build()
    gc.collect()
    after, _ = tracemalloc.get_traced_memory()
    return obj, round((after - before) / MB, 2)


def run(requests: int, k: int) -> dict[str, Any]:
    from ai.decorators import build_chain
    from ai.model_factory import ModelFactory
    from application.recommendation_controller import RecommendationController
    from config.settings import load_settings
    from data.feature_store_manager import ParquetFeatureStoreManager
    from persistence.sql_catalog_repository import SqlCatalogRepository
    from persistence.sql_user_profile_repository import SqlUserProfileRepository

    settings = load_settings()
    rss_start = _rss_mb()
    tracemalloc.start()

    components: dict[str, float] = {}
    store, components["feature_store"] = _measure(
        lambda: ParquetFeatureStoreManager.from_settings(settings)
    )
    catalog, components["catalog_repository"] = _measure(
        lambda: SqlCatalogRepository.from_settings(settings)
    )
    profiles, components["profile_repository"] = _measure(
        lambda: SqlUserProfileRepository.from_settings(settings)
    )
    strategies: dict[str, Any] = {}
    for strategy_id in ModelFactory.available():
        strategy, components[f"strategy:{strategy_id}"] = _measure(
            lambda sid=strategy_id: ModelFactory.create(sid, settings, use_cache=False)
        )
        strategies[strategy_id] = strategy

    chain = list(settings.model.decorator_chain)
    active = settings.model.active_strategy
    controller = RecommendationController(
        strategy=build_chain(strategies[active], chain, settings=settings),
        feature_store=store,
        catalog_repository=catalog,
        profile_repository=profiles,
        fallback_strategy=build_chain(
            strategies[settings.model.fallback_strategy], chain, settings=settings
        ),
        settings=settings,
    )

    import pandas as pd

    known = pd.read_parquet(
        settings.feature_store.version_dir / "user_features.parquet", columns=["user_id"]
    )["user_id"].to_numpy()
    user_ids = [int(u) for u in np.random.default_rng(7).choice(known, size=requests)]

    for user_id in user_ids[:50]:
        controller.recommend(user_id, k)
    gc.collect()
    after_warmup, _ = tracemalloc.get_traced_memory()

    for user_id in user_ids:
        controller.recommend(user_id, k)
    gc.collect()
    after_requests, peak = tracemalloc.get_traced_memory()

    snapshot = tracemalloc.take_snapshot()
    top = [
        {"location": str(stat.traceback[0]), "mb": round(stat.size / MB, 2)}
        for stat in snapshot.statistics("filename")[:8]
    ]
    tracemalloc.stop()

    growth = (after_requests - after_warmup) / MB
    return {
        "requests": requests,
        "k": k,
        "active_strategy": active,
        "allocated_by_component_mb": components,
        "python_heap_after_warmup_mb": round(after_warmup / MB, 2),
        "python_heap_after_requests_mb": round(after_requests / MB, 2),
        "growth_during_requests_mb": round(growth, 3),
        "growth_per_1000_requests_kb": round(growth * 1024 * 1000 / max(requests, 1), 2),
        "python_heap_peak_mb": round(peak / MB, 2),
        "process_rss_start_mb": rss_start,
        "process_rss_end_mb": _rss_mb(),
        "largest_allocations_by_file": top,
        "users_in_feature_store": len(store),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python benchmarks/memory_profile.py")
    parser.add_argument("--requests", type=int, default=1000)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--reports-dir", default=str(REPO_ROOT / "reports"))
    args = parser.parse_args(argv)

    result = run(args.requests, args.k)
    out = Path(args.reports_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "memory_profile.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print("\nMemory profile (tracemalloc + RSS)")
        print("-" * 60)
        for name, mb in result["allocated_by_component_mb"].items():
            print(f"  {name:28} {mb:8.2f} MB")
        print("-" * 60)
        for key in (
            "python_heap_after_warmup_mb",
            "python_heap_after_requests_mb",
            "growth_during_requests_mb",
            "growth_per_1000_requests_kb",
            "python_heap_peak_mb",
            "process_rss_start_mb",
            "process_rss_end_mb",
        ):
            print(f"  {key:32} {result[key]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
