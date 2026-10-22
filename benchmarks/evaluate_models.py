"""Offline validation of the four strategies on a temporal holdout (Milestone 3).

    python benchmarks/evaluate_models.py              # full run (split, features, NCF, metrics)
    python benchmarks/evaluate_models.py --skip-ncf   # faster: no neural model training
    python benchmarks/evaluate_models.py --reuse      # reuse a previously built split

What it does, in order:

1. ``data.pipeline.split``: holds out the most recent 20 % of every eligible
   user's ratings (the "exam").
2. Re-runs the unchanged ``transform`` and ``materialize`` stages on the
   training ratings only, into ``artifacts/evaluation/`` (production
   artifacts are never touched).
3. Trains the NCF model on the training ratings only.
4. Asks every strategy for a Top-K list per test user and compares it with
   the held-out items the user rated positively (rating >= 3.5).
5. Reports precision, recall, F1, NDCG, hit rate, catalog coverage,
   popularity bias, accuracy (to show why it is misleading), a breakdown by
   user activity, and concrete failure cases.

Output: ``reports/evaluation_report.json`` and ``reports/evaluation_report.md``.
"""

from __future__ import annotations

import os

# Must be set before numpy is imported: ALS parallelises its own loops and a
# second BLAS thread pool underneath only adds contention.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from collections import Counter  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ai.evaluation import (  # noqa: E402
    MetricAccumulator,
    classification_accuracy,
    mean_popularity_percentile,
)
from ai.interfaces import IRecommendationStrategy, UserFeatures  # noqa: E402
from config.settings import Settings, load_settings  # noqa: E402
from errors import ColdStartError, ModelUnavailableError  # noqa: E402

STRATEGIES = ("popularity", "als", "content", "ncf")
SEGMENTS = ("low", "medium", "high")


def _point_settings_to(eval_root: Path) -> Settings:
    """Redirect every artifact path to the evaluation directory."""
    os.environ["RECO_INTERIM_ROOT"] = str(eval_root)
    os.environ["RECO_FEATURE_STORE_ROOT"] = str(eval_root / "feature_store")
    os.environ["RECO_FEATURE_STORE_VERSION"] = "eval"
    os.environ["RECO_WEIGHTS_ROOT"] = str(eval_root / "models")
    os.environ["RECO_MODEL_VERSION"] = "eval"
    return load_settings()


def build_evaluation_artifacts(eval_root: Path, *, skip_ncf: bool) -> dict[str, Any]:
    """Run split -> transform -> materialize -> NCF training on the train split."""
    from data.pipeline import materialize, split, transform

    timings: dict[str, Any] = {}
    production = load_settings()

    started = time.perf_counter()
    split_summary = split.run(production, eval_root=eval_root)
    timings["split_s"] = round(time.perf_counter() - started, 2)

    settings = _point_settings_to(eval_root)
    started = time.perf_counter()
    transform.run(settings)
    timings["transform_s"] = round(time.perf_counter() - started, 2)

    started = time.perf_counter()
    materialize.run(settings)
    timings["materialize_s"] = round(time.perf_counter() - started, 2)

    if not skip_ncf:
        from ai.strategies.deep_learning_strategy import train

        started = time.perf_counter()
        train(settings)
        timings["ncf_training_s"] = round(time.perf_counter() - started, 2)

    return {"split": split_summary, "timings": timings}


def _load_strategies(settings: Settings, skip_ncf: bool) -> dict[str, IRecommendationStrategy]:
    from ai.model_factory import ModelFactory

    loaded: dict[str, IRecommendationStrategy] = {}
    for strategy_id in STRATEGIES:
        if strategy_id == "ncf" and skip_ncf:
            continue
        try:
            loaded[strategy_id] = ModelFactory.create(strategy_id, settings, use_cache=False)
        except ModelUnavailableError as exc:
            print(f"  ! {strategy_id} unavailable: {exc.message}", file=sys.stderr)
    return loaded


def _segment_of(n_ratings: float, low_cut: float, high_cut: float) -> str:
    if n_ratings <= low_cut:
        return "low"
    if n_ratings <= high_cut:
        return "medium"
    return "high"


def evaluate(eval_root: Path, *, k: int, skip_ncf: bool) -> dict[str, Any]:
    from data.feature_store_manager import ParquetFeatureStoreManager

    settings = _point_settings_to(eval_root)
    threshold = settings.pipeline.positive_rating_threshold

    test = pd.read_parquet(eval_root / "holdout" / "ratings.parquet")
    train = pd.read_parquet(eval_root / "clean" / "ratings.parquet")
    items = pd.read_parquet(eval_root / "clean" / "items.parquet")
    popularity = pd.read_parquet(settings.feature_store.version_dir / "popularity.parquet")
    item_features = pd.read_parquet(
        settings.feature_store.version_dir / "item_features.parquet", columns=["item_id"]
    )

    catalog_size = int(len(items))
    trainable_items = set(item_features["item_id"].astype(int))
    titles = dict(zip(items["item_id"].astype(int), items["title"], strict=False))
    genres = dict(zip(items["item_id"].astype(int), items["genres"], strict=False))
    popularity_rank = {
        int(item): int(rank)
        for item, rank in zip(popularity["item_id"], popularity["rank"], strict=False)
    }

    relevant_by_user: dict[int, set[int]] = {
        int(user): set(group["item_id"].astype(int))
        for user, group in test[test["rating"] >= threshold].groupby("user_id")
    }
    train_counts = train.groupby("user_id").size()
    low_cut, high_cut = (float(q) for q in np.quantile(train_counts.to_numpy(), [1 / 3, 2 / 3]))

    store = ParquetFeatureStoreManager.from_settings(settings)
    strategies = _load_strategies(settings, skip_ncf)
    fallback = strategies["popularity"]

    results: dict[str, Any] = {}
    failures: dict[str, list[dict[str, Any]]] = {}
    for strategy_id, strategy in strategies.items():
        overall = MetricAccumulator(k)
        by_segment = {name: MetricAccumulator(k) for name in SEGMENTS}
        accuracy_total = 0.0
        fallbacks = 0
        lists: list[list[int]] = []
        zero_hit_users: list[tuple[int, list[int]]] = []

        started = time.perf_counter()
        for user_id, relevant in relevant_by_user.items():
            try:
                features = store.get_user_features(user_id)
            except ColdStartError:
                features = UserFeatures(user_id=user_id, is_cold_start=True)
            try:
                if features.is_cold_start:
                    raise ColdStartError("no history")
                ranking = strategy.predict(features, k)
            except (ColdStartError, ModelUnavailableError):
                # Same behaviour as the service: the fallback answers.
                fallbacks += 1
                ranking = fallback.predict(features, k)
            recommended = [entry.item_id for entry in ranking]
            lists.append(recommended)

            metrics = overall.add(recommended, relevant)
            segment = _segment_of(float(train_counts.get(user_id, 0)), low_cut, high_cut)
            by_segment[segment].add(recommended, relevant)
            accuracy_total += classification_accuracy(recommended, relevant, k, catalog_size)
            if metrics.hits == 0:
                zero_hit_users.append((user_id, recommended))
        elapsed = time.perf_counter() - started

        summary = overall.summary(catalog_size)
        summary["accuracy"] = round(accuracy_total / max(overall.n_users, 1), 4)
        summary["fallback_users"] = fallbacks
        summary["popularity_percentile"] = mean_popularity_percentile(
            lists, popularity_rank, len(popularity_rank)
        )
        summary["seconds"] = round(elapsed, 2)
        summary["by_segment"] = {
            name: acc.summary(catalog_size) for name, acc in by_segment.items()
        }
        results[strategy_id] = summary

        failures[strategy_id] = [
            _describe_failure(
                user_id, recommended, relevant_by_user[user_id], train, titles, genres
            )
            for user_id, recommended in zero_hit_users[:3]
        ]

    unreachable = sum(len(relevant - trainable_items) for relevant in relevant_by_user.values())
    total_relevant = sum(len(relevant) for relevant in relevant_by_user.values())
    return {
        "k": k,
        "positive_rating_threshold": threshold,
        "users_evaluated": len(relevant_by_user),
        "catalog_size": catalog_size,
        "segment_cuts_train_ratings": {"low_max": low_cut, "medium_max": high_cut},
        "relevant_items_total": total_relevant,
        "relevant_items_unseen_in_training": unreachable,
        "results": results,
        "failure_cases": failures,
    }


def _describe_failure(
    user_id: int,
    recommended: list[int],
    relevant: set[int],
    train: pd.DataFrame,
    titles: dict[int, str],
    genres: dict[int, str],
) -> dict[str, Any]:
    history = train[train["user_id"] == user_id]
    genre_counts: Counter[str] = Counter()
    for item in history["item_id"].astype(int):
        genre_counts.update(str(genres.get(item, "")).split("|"))
    return {
        "user_id": user_id,
        "train_ratings": int(len(history)),
        "top_train_genres": [g for g, _ in genre_counts.most_common(3) if g],
        "recommended": [titles.get(i, str(i)) for i in recommended[:5]],
        "held_out_liked": [titles.get(i, str(i)) for i in list(relevant)[:5]],
    }


def write_reports(report: dict[str, Any], out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "evaluation_report.json"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    k = report["k"]
    lines = [
        "# Model evaluation report (temporal holdout)",
        "",
        f"- Users evaluated: {report['users_evaluated']}",
        f"- K = {k}, relevant = held-out rating >= {report['positive_rating_threshold']}",
        f"- Held-out relevant items never seen in training (unreachable for ALS/content/NCF): "
        f"{report['relevant_items_unseen_in_training']} of {report['relevant_items_total']}",
        "",
        f"| Strategy | Precision@{k} | Recall@{k} | F1@{k} | NDCG@{k} | Hit@{k} | Coverage "
        "| Pop. percentile | Accuracy | Fallback users |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for strategy_id, r in report["results"].items():
        lines.append(
            f"| {strategy_id} | {r[f'precision@{k}']} | {r[f'recall@{k}']} | {r[f'f1@{k}']} "
            f"| {r[f'ndcg@{k}']} | {r[f'hit@{k}']} | {r['catalog_coverage']} "
            f"| {r['popularity_percentile']} | {r['accuracy']} | {r['fallback_users']} |"
        )
    lines += ["", f"## NDCG@{k} by user activity (training ratings)", ""]
    lines.append("| Strategy | low | medium | high |")
    lines.append("|---|---|---|---|")
    for strategy_id, r in report["results"].items():
        cells = [str(r["by_segment"][s].get(f"ndcg@{k}", "-")) for s in SEGMENTS]
        lines.append(f"| {strategy_id} | " + " | ".join(cells) + " |")
    md_path = out_dir / "evaluation_report.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python benchmarks/evaluate_models.py")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--eval-root", default=str(REPO_ROOT / "artifacts" / "evaluation"))
    parser.add_argument("--reports-dir", default=str(REPO_ROOT / "reports"))
    parser.add_argument("--skip-ncf", action="store_true", help="Do not train/evaluate NCF")
    parser.add_argument("--reuse", action="store_true", help="Reuse an existing evaluation split")
    args = parser.parse_args(argv)

    eval_root = Path(args.eval_root).resolve()
    build_info: dict[str, Any] = {}
    if not (args.reuse and (eval_root / "holdout" / "ratings.parquet").exists()):
        build_info = build_evaluation_artifacts(eval_root, skip_ncf=args.skip_ncf)

    report = evaluate(eval_root, k=args.k, skip_ncf=args.skip_ncf)
    report["build"] = build_info
    json_path, md_path = write_reports(report, Path(args.reports_dir))
    print(md_path.read_text(encoding="utf-8"))
    print(f"Reports written to {json_path} and {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
