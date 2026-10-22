"""Concurrent load test of the HTTP service (Milestone 3, scalability).

    python benchmarks/load_test.py                         # spawns a server, 1 worker
    python benchmarks/load_test.py --workers 4             # spawns a server, 4 workers
    python benchmarks/load_test.py --base-url http://127.0.0.1:8000 --label docker
    python benchmarks/load_test.py --concurrency 1 8 32 --duration 15

``latency_p95.py`` answers "how fast is one request". This script answers the
scalability questions it cannot: how many requests per second the service
sustains, how latency degrades as concurrent clients grow, and how much CPU
and memory the server uses while doing it.

For each concurrency level it runs N client threads for a fixed duration,
each sending ``GET /recommendations/{user}`` in a closed loop with users drawn
at random. It reports throughput, latency percentiles, error rate, and the
server's CPU and RSS (sampled with psutil over the whole process tree, so
multi-worker servers are measured correctly).

When the script spawns the server itself, every writable store is redirected
to a temporary directory, so the run never modifies the repository.

Output: ``reports/load_test_w<N>.json`` for a spawned server, or
``reports/load_test_<label>.json`` for an existing one (``--label``).
"""

from __future__ import annotations

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import socket  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import httpx  # noqa: E402
import numpy as np  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
TARGET_P95_MS = 200.0


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _known_users() -> list[int]:
    import pandas as pd

    path = REPO_ROOT / "artifacts" / "feature_store" / "v1" / "user_features.parquet"
    return [int(u) for u in pd.read_parquet(path, columns=["user_id"])["user_id"]]


def spawn_server(workers: int, cache: bool) -> tuple[subprocess.Popen[bytes], str, Path]:
    workdir = Path(tempfile.mkdtemp(prefix="reco-load-"))
    shutil.copyfile(REPO_ROOT / "artifacts" / "platform.db", workdir / "platform.db")
    port = _free_port()
    env = {
        **os.environ,
        "PYTHONPATH": str(REPO_ROOT / "src"),
        "RECO_SQL_URL": f"sqlite:///{(workdir / 'platform.db').as_posix()}",
        "RECO_DOCUMENT_STORE_ROOT": str(workdir / "documents"),
        "RECO_RETRY_BUFFER_PATH": str(workdir / "retry_buffer.jsonl"),
        "RECO_ERROR_LOG_PATH": str(workdir / "errors.jsonl"),
        "RECO_LOG_LEVEL": "WARNING",
    }
    if not cache:
        env["RECO_DECORATOR_CHAIN"] = "timing,logging"
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "application.api:app",
            "--port",
            str(port),
            "--workers",
            str(workers),
            "--log-level",
            "warning",
        ],
        cwd=REPO_ROOT,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    base_url = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise SystemExit("the spawned server exited during startup")
        try:
            if httpx.get(f"{base_url}/health", timeout=1.0).status_code == 200:
                return process, base_url, workdir
        except httpx.HTTPError:
            time.sleep(0.5)
    process.kill()
    raise SystemExit("the spawned server did not become healthy")


class ResourceSampler(threading.Thread):
    """Samples CPU % and RSS of a process and its children every 0.5 s."""

    def __init__(self, pid: int | None) -> None:
        super().__init__(daemon=True)
        self._pid = pid
        self._stop_event = threading.Event()
        self.cpu: list[float] = []
        self.rss_mb: list[float] = []

    def run(self) -> None:
        if self._pid is None:
            return
        import psutil

        root = psutil.Process(self._pid)
        # Process objects are kept across samples: cpu_percent() measures the
        # time since the previous call *on the same object*, so a fresh
        # object every sample would always report 0.
        tracked: dict[int, psutil.Process] = {root.pid: root}
        while not self._stop_event.is_set():
            try:
                for child in root.children(recursive=True):
                    tracked.setdefault(child.pid, child)
            except psutil.Error:
                pass
            cpu = 0.0
            rss = 0.0
            for proc in list(tracked.values()):
                try:
                    cpu += proc.cpu_percent(interval=None)
                    rss += proc.memory_info().rss
                except psutil.Error:
                    continue
            self.cpu.append(cpu)
            self.rss_mb.append(rss / (1024 * 1024))
            self._stop_event.wait(0.5)

    def stop(self) -> dict[str, float | None]:
        self._stop_event.set()
        self.join(timeout=2)
        samples = self.cpu[1:] or self.cpu  # the first cpu_percent call is always 0
        return {
            "server_cpu_percent_mean": round(float(np.mean(samples)), 1) if samples else None,
            "server_cpu_percent_max": round(float(np.max(samples)), 1) if samples else None,
            "server_rss_mb_max": round(float(np.max(self.rss_mb)), 1) if self.rss_mb else None,
        }


def run_level(
    base_url: str, concurrency: int, duration: float, users: list[int], k: int, pid: int | None
) -> dict[str, Any]:
    latencies: list[float] = []
    errors = 0
    degraded = 0
    lock = threading.Lock()
    stop_at = time.monotonic() + duration

    def client(seed: int) -> None:
        nonlocal errors, degraded
        rng = np.random.default_rng(seed)
        local: list[float] = []
        local_errors = 0
        local_degraded = 0
        with httpx.Client(base_url=base_url, timeout=10.0) as session:
            while time.monotonic() < stop_at:
                user = int(rng.choice(users))
                started = time.perf_counter()
                try:
                    response = session.get(f"/recommendations/{user}", params={"k": k})
                    elapsed = (time.perf_counter() - started) * 1000.0
                    if response.status_code == 200:
                        local.append(elapsed)
                        if response.json().get("degraded"):
                            local_degraded += 1
                    else:
                        local_errors += 1
                except httpx.HTTPError:
                    local_errors += 1
        with lock:
            latencies.extend(local)
            errors += local_errors
            degraded += local_degraded

    sampler = ResourceSampler(pid)
    sampler.start()
    threads = [threading.Thread(target=client, args=(seed,)) for seed in range(concurrency)]
    started = time.perf_counter()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    wall = time.perf_counter() - started
    resources = sampler.stop()

    total = len(latencies) + errors
    result: dict[str, Any] = {
        "concurrency": concurrency,
        "duration_s": round(wall, 2),
        "requests": total,
        "errors": errors,
        "error_rate": round(errors / total, 4) if total else 0.0,
        "degraded_responses": degraded,
        "throughput_rps": round(len(latencies) / wall, 1) if wall else 0.0,
    }
    if latencies:
        arr = np.asarray(latencies)
        p95 = float(np.percentile(arr, 95))
        result.update(
            {
                "p50_ms": round(float(np.percentile(arr, 50)), 2),
                "p95_ms": round(p95, 2),
                "p99_ms": round(float(np.percentile(arr, 99)), 2),
                "max_ms": round(float(arr.max()), 2),
                # Errors fail the verdict too: a fast error is not a pass.
                "verdict": "PASS" if p95 <= TARGET_P95_MS and errors == 0 else "FAIL",
            }
        )
    result.update(resources)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python benchmarks/load_test.py")
    parser.add_argument("--base-url", default=None, help="Use an already running server")
    parser.add_argument("--workers", type=int, default=1, help="uvicorn workers when spawning")
    parser.add_argument("--no-cache", action="store_true", help="Spawn the server without caching")
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 16, 32])
    parser.add_argument("--duration", type=float, default=10.0, help="Seconds per level")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--reports-dir", default=str(REPO_ROOT / "reports"))
    parser.add_argument("--label", default=None, help="Report name suffix (default: w<workers>)")
    args = parser.parse_args(argv)

    users = _known_users()
    process = None
    workdir = None
    if args.base_url:
        base_url, pid = args.base_url, None
    else:
        process, base_url, workdir = spawn_server(args.workers, cache=not args.no_cache)
        pid = process.pid

    try:
        levels = []
        for concurrency in args.concurrency:
            print(f"  concurrency={concurrency} ...", flush=True)
            levels.append(run_level(base_url, concurrency, args.duration, users, args.k, pid))
    finally:
        if process is not None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
        if workdir is not None:
            shutil.rmtree(workdir, ignore_errors=True)

    report = {
        "target_p95_ms": TARGET_P95_MS,
        "server": "spawned" if process is not None else base_url,
        "label": args.label,
        "workers": args.workers if process is not None else None,
        "caching": not args.no_cache,
        "levels": levels,
    }
    out = Path(args.reports_dir)
    out.mkdir(parents=True, exist_ok=True)
    label = args.label or ("external" if process is None else f"w{args.workers}")
    name = f"load_test_{label}{'_nocache' if args.no_cache else ''}.json"
    (out / name).write_text(json.dumps(report, indent=2), encoding="utf-8")

    columns = ("conc", "rps", "p50", "p95", "p99", "err", "cpu%", "rssMB")
    print("\n" + " ".join(f"{name:>8}" for name in columns))
    for level in levels:
        row = (
            level["concurrency"],
            level["throughput_rps"],
            level.get("p50_ms", "-"),
            level.get("p95_ms", "-"),
            level.get("p99_ms", "-"),
            level["errors"],
            level.get("server_cpu_percent_mean") or "-",
            level.get("server_rss_mb_max") or "-",
        )
        print(" ".join(f"{value!s:>8}" for value in row))
    print(f"\nReport written to {out / name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
