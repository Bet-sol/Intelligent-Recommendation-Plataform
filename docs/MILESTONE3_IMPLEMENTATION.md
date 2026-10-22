# Milestone 3 — Implementation Guide

*Testing, Validation, and Deployment of AI Systems.* This document records
what was added on top of the Milestone 2 code base, **where** it lives, how to
run it, and what the measurements say. Every number below was produced by the
scripts listed next to it, on the development machine (Windows 11, CPython
3.13, CPU only, dataset `ml-latest-small`).

## 0. Summary of changes

| Area | New / changed files |
|---|---|
| Milestone 2 feedback: Spanish errors, error algorithm, error journal | `src/errors.py`, `src/application/error_journal.py`, `src/application/api.py`, `src/application/recommendation_controller.py`, `src/config/settings.py`, `docs/ERROR_HANDLING_ALGORITHM.md` |
| Milestone 2 code-review fixes | `src/ai/interfaces.py`, `src/ai/decorators/caching_decorator.py`, `src/application/api.py`, `src/messaging/feedback_history_service.py`, `src/messaging/interaction_event_queue.py`, `benchmarks/latency_p95.py` |
| Testing strategy | `tests/test_system_e2e.py`, `benchmarks/memory_profile.py`, plus new unit tests |
| AI model validation | `src/data/pipeline/split.py`, `src/ai/evaluation.py`, `benchmarks/evaluate_models.py`, `reports/evaluation_report.*` |
| CI/CD | `.github/workflows/ci.yml` |
| Containers | `Dockerfile`, `.dockerignore`, `docker-compose.yml`, `k8s/`, `src/application/server.py` (container entry point) |
| Scalability | `benchmarks/load_test.py`, `reports/load_test_w1.json`, `reports/load_test_w4.json`, `reports/load_test_docker_*.json` |
| Web interface | `src/presentation/static/index.html` (degradation reasons explained in Spanish) |

Test suite: **300 tests, all passing** (264 from Milestone 2 + 36 new).
`ruff`, `black --check` and `mypy` are clean.

## 1. Corrections and improvements from Milestone 2

### 1.1 Instructor feedback on error handling

| Feedback | Correction | Where |
|---|---|---|
| Error messages in English | Client-facing messages are translated to Spanish by stable error code (`MENSAJES_ES`). A code without a translation keeps its original message. Internal `message`/`detail` stay in English for developers and logs. | `src/errors.py`, handlers in `src/application/api.py` |
| No explicit error algorithm | Pseudocode, flowchart and decision table of the procedure the code already follows. | `docs/ERROR_HANDLING_ALGORITHM.md` |
| No error journal (bitácora) | `ErrorJournal`: append-only JSON Lines with timestamp, kind, code, status, route, user and message. Records the errors answered by the validation and domain handlers (400, 404 and the 5xx of platform errors) and every degradation (cold start, model or feature store down, event publication failure); a route that does not exist (404/405) or an unexpected exception is not journaled. | `src/application/error_journal.py`; path `RECO_ERROR_LOG_PATH` in `src/config/settings.py` |

Design decisions: Spanish only (the error codes are stable, so a second
language would be one dictionary entry per code); the journal is kept apart
from the FR-06 business history and from the operational `LoggingDecorator`
log; a failure to write the journal never fails a request.

### 1.2 Issues found when re-reviewing the Milestone 2 code

| Issue | Fix | Where | Test |
|---|---|---|---|
| `top_k` padded short rankings with already-seen items scored `-inf` (serialised as `"score": null`) | Masked items are never candidates; a short list is returned instead | `src/ai/interfaces.py` | `test_top_k_never_pads_the_ranking_with_seen_items` |
| The fallback strategy bypassed the decorator chain: cold-start requests were not cached, timed or logged | The fallback is wrapped by the same chain as the active strategy | `src/application/api.py`, `benchmarks/latency_p95.py` | `test_the_fallback_runs_through_the_decorator_chain` |
| The cache key did not include the strategy, so two strategies sharing Redis could read each other's rankings | Key is now `reco:topk:{strategy}:{user}:{k}:{model_version}` | `src/ai/decorators/caching_decorator.py` | `test_cache_key_separates_strategies_sharing_a_backend` |
| The retry buffer was written but never replayed (only a test called the replay) | The worker replays it every 30 s | `src/messaging/feedback_history_service.py`, `src/application/api.py` | `test_the_worker_replays_the_retry_buffer` |
| The RabbitMQ adapter shared one pika connection across threads (pika is not thread-safe) | Channel use is serialised with a lock | `src/messaging/interaction_event_queue.py` | Exercised against a real RabbitMQ in the `full` compose profile (§5) |

### 1.3 Implemented vs. target architecture

The Milestone 2 evaluation asked to separate clearly what runs today from
what belongs to the target design.

| Component | Running in this version | Target (production) | Status |
|---|---|---|---|
| Relational store | SQLite file | PostgreSQL | Adapter switches by `RECO_SQL_URL`; only SQLite exercised |
| History store | Local JSON Lines (default) | MongoDB | **Running** against a real MongoDB 7 in the `full` compose profile (`/health` reports `history_backend: mongodb`) |
| Cache | In-process LRU (default) | Redis | **Running** against a real Redis 7 in the `full` profile (`cache_backend: redis`) |
| Message broker | In-process queue (default) | RabbitMQ | **Running** against a real RabbitMQ 3 in the `full` profile (`queue_backend: rabbitmq`), consumed by the worker container |
| Content embeddings | TF-IDF + SVD | sentence-transformers | Selectable by `RECO_CONTENT_EMBEDDING_BACKEND` |
| Containers | Docker image + compose (both profiles) | Kubernetes | Image built; both compose profiles run and load-tested on Docker Desktop 4.93 (§5, §6); K8s manifests rendered with `kubectl kustomize`, not applied to a cluster |

## 2. Software testing strategy

| Level | What it checks | Where | How to run |
|---|---|---|---|
| Unit | Strategies, decorators, factory, metrics, split, journal, schemas | `tests/ai`, `tests/data`, `tests/presentation`, `tests/application` | `python -m pytest` |
| Integration | Composed FastAPI app in-process (TestClient) on a synthetic store | `tests/application/test_api.py` | `python -m pytest` |
| Failure injection | Feature store, cache, broker and model replaced by failing doubles: never a 500 | `tests/test_failure_injection.py` | `python -m pytest` |
| Architecture | No outer layer imports a concrete inner class | `tests/test_architecture_imports.py` | `python -m pytest` |
| **System (new)** | Real `uvicorn` process on the real MovieLens artifacts, over HTTP: health, Top-K, cache, 400 in Spanish + journal, 404, 202 | `tests/test_system_e2e.py` | `python -m pytest tests/test_system_e2e.py` |
| **Memory profiling (new)** | Allocation per component, heap growth over 1 000 requests, process RSS | `benchmarks/memory_profile.py` | `python benchmarks/memory_profile.py` |

### Memory profile (1 000 requests, `reports/memory_profile.json`)

| Component | Allocated |
|---|---|
| Feature store (609 users) | 1.69 MB |
| `als` strategy | 1.79 MB |
| `content` strategy | 1.79 MB |
| `popularity` strategy | 0.95 MB |
| `ncf` strategy (includes the PyTorch import) | 54.97 MB |
| Python heap after warm-up → after 1 000 requests | 66.06 → 66.41 MB |
| Process RSS (includes native NumPy/PyTorch memory) | 110.7 → 420.4 MB |

**Interpretation.** The neural model dominates memory: PyTorch alone is ~55 MB
of Python allocations and most of the native RSS. The heap grows by only
0.35 MB over 1 000 requests; that growth is the bounded LRU cache filling up
(at most 4 096 entries, ~600 distinct users here), not a leak. The feature
store is held in memory per process, which is cheap at 609 users but scales
linearly with the user count (see §6).

## 3. AI model validation and error analysis

### Method

The production feature store is built from **all** ratings, so evaluating on
it would reward the models for ratings they have already seen (data leakage).
`src/data/pipeline/split.py` instead holds out the **most recent 20 %** of
each user's ratings (users with ≥ 10 ratings) — a temporal split, because a
recommender always predicts the future from the past. The unchanged
`transform` and `materialize` stages, and the NCF training, are re-run on the
training part only, into `artifacts/evaluation/` (production artifacts are
never touched). Each strategy then produces a Top-10 list per user, compared
with the held-out items the user rated ≥ 3.5.

- Ratings: 80 860 train / 19 929 test; 594 users have at least one positive
  held-out item and are evaluated.
- 1 252 of the 12 117 relevant held-out items never appear in training, so no
  collaborative or neural model can recommend them (a hard ceiling on recall).

Run: `python benchmarks/evaluate_models.py` (≈ 15 s; `--skip-ncf` to skip the
neural model). Output: `reports/evaluation_report.json` and `.md`.

### Results (K = 10)

| Strategy | Precision | Recall | F1 | NDCG | Hit rate | Catalog coverage | Popularity percentile | Accuracy |
|---|---|---|---|---|---|---|---|---|
| popularity (baseline) | 0.020 | 0.013 | 0.012 | 0.023 | 0.147 | 1.3 % | 0.003 | 0.997 |
| **als** | **0.087** | **0.089** | **0.067** | **0.107** | **0.483** | 7.8 % | 0.156 | 0.997 |
| content | 0.012 | 0.014 | 0.010 | 0.013 | 0.113 | 13.5 % | 0.420 | 0.997 |
| ncf | 0.064 | 0.043 | 0.040 | 0.080 | 0.338 | 1.0 % | 0.025 | 0.997 |

NDCG@10 by user activity (training ratings; low ≤ 36, medium ≤ 102, high > 102):

| Strategy | low | medium | high |
|---|---|---|---|
| popularity | 0.015 | 0.013 | 0.039 |
| als | 0.100 | 0.095 | 0.126 |
| content | 0.014 | 0.012 | 0.015 |
| ncf | 0.053 | 0.057 | 0.129 |

### Interpretation

1. **Accuracy is useless here.** Every strategy, including the weakest, scores
   99.7 %: with 9 690 catalog items and 10 recommendations, true negatives
   dominate. That is why ranking metrics are the ones reported.
2. **ALS is the best model**: 4.3× the precision and 4.7× the NDCG of the
   popularity baseline, and at least one correct item for 48 % of users. It is
   the configured active strategy, and these results justify that choice.
3. **NCF is second in precision but recommends almost the same items to
   everyone** (1.0 % catalog coverage, popularity percentile 0.025): it has
   learned popularity more than personal taste, and it only competes with ALS
   for highly active users (NDCG 0.129 vs 0.126). With 3 training epochs and
   uniform negative sampling this is expected; more epochs and
   popularity-aware negative sampling are the next step.
4. **Content-based is below the baseline in accuracy terms but has the widest
   coverage and the most long-tail recommendations** (percentile 0.42). Its
   TF-IDF embeddings of titles and genres are too coarse to predict what a
   user will watch next; it is useful for diversity, not as the main model.
5. **Absolute values are low, which is normal for offline evaluation of
   recommenders**: the test set only counts the exact films a user happened to
   rate later, so a plausible recommendation the user never rated counts as a
   miss.

### Error analysis — failure cases

- **Plausible but not the exact titles (ALS).** User 2 (24 training ratings,
  Drama/Action/Crime) received *Fight Club*, *American History X*, *Gran
  Torino*; the held-out liked films were *Good Will Hunting*, *The Town*,
  *Mad Max: Fury Road*. The recommendations match the user's taste, but none
  is the specific film — the main limit of offline precision.
- **Popularity collapse (NCF).** Users 2 and 4 — very different histories —
  both received *Forrest Gump*, *The Lord of the Rings*, *Pulp Fiction /
  Shawshank*. This is the low-coverage problem seen in the metrics.
- **Low-activity users.** Every model scores lower for users with ≤ 36
  ratings; ALS degrades least (0.100 vs 0.126), the baseline and NCF most.
- **Unreachable items.** 10 % of the relevant held-out items are new to the
  catalog at test time; only a content model could ever score them.

These findings also prepare Unit 4 (bias): the popularity bias of NCF and of
the fallback is now measured, not assumed.

## 4. CI/CD

`.github/workflows/ci.yml` runs on every push to `main`, on every pull request
and on demand. `tests` runs only after `quality`; `benchmark` and `docker` run
in parallel after `tests`:

| Stage | Commands | Fails the build when |
|---|---|---|
| quality | `ruff check .`, `black --check .`, `mypy` | style, format or type error |
| tests | `pytest --cov=src` (CPU-only PyTorch) | any of the 300 tests fails |
| benchmark | `latency_p95.py --requests 500`, `memory_profile.py` | p95 > 200 ms (NFR-01) |
| docker | build the image, smoke-test that same image (`/health` and a recommendation); push it to GHCR only on `main` and only after the smoke test | image does not build or does not answer |

Reliability contribution: every commit gets a red or green verdict on style,
types, tests, latency and a startable image, and the published image is the
one that passed the smoke test. Test and benchmark reports are uploaded as
build artifacts, so every commit leaves evidence. The workflow does not by
itself block a push: `main` is not protected yet, so branch protection with
required status checks and a reviewed pull request is the proposed next
integration control.

**Run history.** The first GitHub run (commit `5b98138`, 22 Oct 2026) passed
`quality`, `tests` (300 passed, 84 % coverage) and `benchmark` (p95 1.196 ms),
but `docker` failed before building: the GHCR repository name contained the
uppercase owner `Michou001`. The job now lowercases the image name and pushes
only the image that passed the smoke test. Before the first push, the Python
jobs had been reproduced in a clean `python:3.12` container (300 tests, 84 %
coverage, p95 2.2 ms on the development machine); that local measurement and
the runner measurement are different environments and are reported apart.

## 5. Containerization and deployment

- **`Dockerfile`**: `python:3.12-slim`, CPU-only PyTorch, non-root user, health
  check on `/health`, writable state in `/app/data` (journal, history, retry
  buffer), `WORKERS` environment variable for uvicorn workers, started by
  `python -m application.server` (see bottleneck 5 in §6). The image also
  installs the Redis, RabbitMQ and MongoDB clients; without them the adapters
  would silently fall back to their in-process doubles. The source is
  run from `/app/src` rather than pip-installed, because `config/settings.py`
  derives every artifact path from its own location.
- **`docker-compose.yml`**: default profile = the API alone with in-process
  backends (the tested configuration). Profile `full` adds the target
  architecture: Redis, RabbitMQ, MongoDB and the FR-06 worker as its own
  container. Host ports are configurable (`API_PORT`, `API_FULL_PORT`).
- **`k8s/`**: ConfigMap, Deployment (2 replicas, readiness/liveness probes,
  CPU/memory requests and limits sized from the measurements of §6), Service,
  HorizontalPodAutoscaler (2–6 replicas at 70 % CPU) and Redis for a shared
  cache. Apply with `kubectl apply -k k8s/`.

**Is Kubernetes justified?** For the coursework dataset, one container
handles ~400 req/s; Kubernetes is not needed to serve it. It is justified for
the target scenario of NFR-02 (traffic peaks, several replicas, rolling
updates, self-healing). The manifests are provided for that scenario, with
one precondition: with more than one replica the history must move to
MongoDB and the cache to Redis, because local files and in-process caches are
not shared between pods.

**Verification status (Docker Desktop 4.93, WSL 2).**

| Check | Result |
|---|---|
| `docker compose build` | Image `recommendation-platform:latest` built (CPU-only PyTorch 2.14) |
| Default profile (`docker compose up`) | Container `healthy`; `/health` reports 7 343 catalog items and the four strategies; Top-K for user 1; `k=0` gives 400 in Spanish; unknown user gives 404 in Spanish; interaction gives 202; web UI at `/ui/` |
| Strategy switch | `RECO_ACTIVE_STRATEGY=ncf` makes the container serve `ncf`, no code change |
| Error journal | Written to the `/app/data` volume and preserved across a container restart |
| Profile `full` | `api-full` reports `cache_backend: redis`, `queue_backend: rabbitmq`, `history_backend: mongodb`; Top-K stored in Redis under `reco:topk:als:7:5:v1`; RabbitMQ queue drained to 0 by the worker; 15 933 interaction documents in MongoDB after the load tests |
| Kubernetes | `kubectl kustomize k8s/` renders 6 objects (ConfigMap, 2 Services, 2 Deployments, HPA); not applied, because no cluster is enabled on the machine |

## 6. Scalability and performance analysis

`benchmarks/load_test.py` spawns the server and runs N concurrent clients for
8 s per level (client and server on the same machine).

| Workers | Clients | Throughput (req/s) | p50 (ms) | p95 (ms) | p99 (ms) | Server CPU | Server RSS |
|---|---|---|---|---|---|---|---|
| 1 | 1 | 303 | 2.9 | 4.2 | 5.7 | 82 % | 184 MB |
| 1 | 4 | 399 | 8.9 | 13.7 | 20.7 | 133 % | 187 MB |
| 1 | 16 | 364 | 36.5 | 49.2 | 54.8 | 138 % | 193 MB |
| 1 | 32 | 303 | 72.4 | 91.1 | 119.5 | 108 % | 198 MB |
| 4 | 1 | 307 | 2.9 | 3.9 | 5.1 | 92 % | 779 MB |
| 4 | 4 | 794 | 4.4 | 6.6 | 10.8 | 269 % | 749 MB |
| 4 | 16 | 842 | 14.9 | 28.2 | 36.5 | 336 % | 757 MB |
| 4 | 32 | 719 | 28.9 | 58.8 | 74.8 | 295 % | 762 MB |

No errors at any level; p95 stays under 200 ms everywhere.

**Containers (Docker Desktop, 2 workers per container, 10 s per level).**
`load_test.py --base-url` against the published port (`host`) and from a
container on the compose network (`internal`); CPU/memory from `docker stats`.

| Target | Clients | req/s | p50 (ms) | p95 (ms) | p99 (ms) |
|---|---|---|---|---|---|
| default profile, host, before fix 5 | 1 | 49 | 7.8 | 52.6 | 56.1 |
| default profile, host | 1 | 211 | 4.2 | 5.6 | 7.3 |
| default profile, host | 4 | 424 | 8.7 | 12.5 | 15.7 |
| default profile, host | 32 | 269 | 92.6 | 139.9 | 176.8 |
| default profile, internal | 4 | 548 | 6.9 | 10.0 | 13.2 |
| default profile, internal | 32 | 338 | 92.7 | 126.3 | 157.2 |
| profile `full` (Redis/RabbitMQ/MongoDB), host | 1 | 199 | 4.4 | 7.3 | 9.6 |
| profile `full`, host | 4 | 340 | 10.8 | 15.4 | 19.8 |
| profile `full`, host | 32 | 233 | 110.3 | 149.8 | 169.6 |

Under load the `full` profile used ~1.7 CPU cores in the API container
(~510 MB), ~0.5 core in RabbitMQ, ~0.2 in the worker and ~0.1 in MongoDB and
Redis. The queue stayed drained: one worker keeps up with the API.

**Bottlenecks identified**

1. **One worker saturates at ~400 req/s using 1–1.4 CPU cores** (Python GIL):
   adding clients beyond 4 only adds queueing latency. Four workers raise the
   ceiling to ~840 req/s (2.1×; the load generator shares the machine, so the
   true ceiling is higher).
2. **Memory scales linearly with workers** (~190 MB each), because every
   worker loads PyTorch, the four strategies and the feature store. The neural
   model is most of it (§2).
3. **Per-process state does not scale horizontally**: in-process cache and
   queue, SQLite, and the local history file are per instance.
4. **Scoring is O(catalog)** per request; fine for 7 343 items, ~8× more work
   on `ml-25m`.
5. **Nagle's algorithm in multi-worker uvicorn (found in the container load
   test).** Every keep-alive request took a constant ~48 ms, even `/health`,
   while the in-process latency was ~1-2 ms. Isolated step by step: not the
   Windows port forwarding (same result inside the Docker network), not the
   HTTP parser (h11 and httptools equal), but `--workers > 1`: uvicorn binds
   the shared socket with `proto = 0`, so asyncio does not set `TCP_NODELAY`
   on accepted connections and Nagle collides with the client's delayed ACK.
   Fix: `src/application/server.py` sets `TCP_NODELAY` on the listening
   socket (inherited by accepted sockets on Linux). Result: p95 52.6 to 5.6 ms
   and 49 to 211 req/s with one client; 198 to 424 req/s with four.
6. **Beyond ~4 concurrent clients per 2-worker container** only queueing
   latency grows (throughput falls from 424 to ~270 req/s at 32 clients, with
   the load generator sharing the same CPUs). More replicas, not more clients
   per replica, is the scaling unit (HPA in `k8s/`).
7. **Production backends cost ~1 ms per request** (Redis round trip): median
   in-request latency 1.0 ms with the in-process cache vs 1.9 ms with Redis.
   Acceptable: it is what makes the cache shared between replicas.

**Optimizations, justified by the numbers**

- Scale with **processes/replicas, not threads**: the bottleneck is the GIL.
- **Load NCF only where it is the active strategy** (or serve it from a
  separate model service): it accounts for ~83 % of each worker's Python heap
  (55 of 66 MB) and it is not the active model.
- **Redis cache and MongoDB/PostgreSQL** before running several replicas
  (running in `docker-compose` profile `full`; Redis in `k8s/`).
- **Keep `TCP_NODELAY` on every connection** (`application.server`): a 4x
  throughput gain at low concurrency for a 50-line entry point.
- **Approximate nearest-neighbour index** only if the catalog grows by orders
  of magnitude; at the current size the measured p95 does not need it.

## 7. Technical evidence

| Evidence | File |
|---|---|
| Test results (300 passing) | `python -m pytest` output; JUnit/coverage uploaded by CI |
| Model evaluation | `reports/evaluation_report.json`, `reports/evaluation_report.md` |
| Memory profile | `reports/memory_profile.json` |
| Load tests | `reports/load_test_w1.json`, `reports/load_test_w4.json`, `reports/load_test_docker_local.json`, `reports/load_test_docker_full.json`, `reports/load_test_docker_internal.json` |
| Latency p95 | `python benchmarks/latency_p95.py` |
| Error journal sample | `artifacts/logs/errors.jsonl` (created at runtime) |
| Configurations | `Dockerfile`, `docker-compose.yml`, `k8s/`, `.github/workflows/ci.yml` |

## 8. How to reproduce everything

```bash
pip install -e ".[dev,als,deep]"
python -m pytest                              # 300 tests
python benchmarks/evaluate_models.py          # model validation
python benchmarks/memory_profile.py           # memory profiling
python benchmarks/load_test.py --workers 1    # scalability
python benchmarks/load_test.py --workers 4
python benchmarks/latency_p95.py              # NFR-01
docker compose up --build                     # container (API_PORT=8080 if 8000 is taken)
docker compose --profile full up --build      # + Redis, RabbitMQ, MongoDB, worker
python benchmarks/load_test.py --base-url http://127.0.0.1:8000 --label docker_local
kubectl apply -k k8s/                         # cluster (needs Kubernetes)
```
