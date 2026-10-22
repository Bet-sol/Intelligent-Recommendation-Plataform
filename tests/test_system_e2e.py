"""System test: the real service, as a separate process, on the real artifacts.

Every other test builds the application in-process on a synthetic store. This
one starts ``uvicorn`` as a subprocess against the MovieLens artifacts shipped
in ``artifacts/`` and talks to it over HTTP, exactly like a client would. It
checks the whole path end to end: startup, health, a Top-K recommendation,
Fail Fast (400), an unknown user (404), interaction ingestion (202) and the
error journal written to disk.

Writable state (SQL database copy, history, retry buffer, error journal) is
redirected to a temporary directory so the run never modifies the repository.
The test is skipped when the materialised artifacts are absent.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = REPO_ROOT / "artifacts"

pytestmark = pytest.mark.skipif(
    not (ARTIFACTS / "feature_store" / "v1" / "user_features.parquet").exists()
    or not (ARTIFACTS / "platform.db").exists(),
    reason="materialised MovieLens artifacts are not present",
)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.fixture(scope="module")
def live_server(tmp_path_factory: pytest.TempPathFactory) -> Iterator[tuple[str, Path]]:
    workdir = tmp_path_factory.mktemp("system")
    shutil.copyfile(ARTIFACTS / "platform.db", workdir / "platform.db")
    journal = workdir / "errors.jsonl"
    port = _free_port()

    env = {
        **os.environ,
        "PYTHONPATH": str(REPO_ROOT / "src"),
        "RECO_SQL_URL": f"sqlite:///{(workdir / 'platform.db').as_posix()}",
        "RECO_DOCUMENT_STORE_ROOT": str(workdir / "documents"),
        "RECO_RETRY_BUFFER_PATH": str(workdir / "retry_buffer.jsonl"),
        "RECO_ERROR_LOG_PATH": str(journal),
        "RECO_LOG_LEVEL": "WARNING",
        "RECO_REDIS_URL": "",
        "RECO_RABBITMQ_URL": "",
        "RECO_MONGO_URL": "",
    }
    process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "application.api:app", "--port", str(port)],
        cwd=REPO_ROOT,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    base_url = f"http://127.0.0.1:{port}"
    try:
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            if process.poll() is not None:
                pytest.fail("the service exited during startup")
            try:
                if httpx.get(f"{base_url}/health", timeout=1.0).status_code == 200:
                    break
            except httpx.HTTPError:
                time.sleep(0.5)
        else:
            pytest.fail("the service did not become healthy within 120 s")
        yield base_url, journal
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()


def test_health_reports_the_real_catalog(live_server):
    base_url, _ = live_server
    body = httpx.get(f"{base_url}/health").json()
    assert body["status"] == "ok"
    assert body["feature_store_available"] is True
    assert body["catalog_items"] > 1000
    assert set(body["registered_strategies"]) >= {"als", "content", "ncf", "popularity"}


def test_a_known_user_receives_a_top_k_ranking(live_server):
    base_url, _ = live_server
    response = httpx.get(f"{base_url}/recommendations/1", params={"k": 10})
    assert response.status_code == 200
    body = response.json()
    assert len(body["items"]) == 10
    assert body["degraded"] is False
    assert [item["rank"] for item in body["items"]] == list(range(1, 11))
    scores = [item["score"] for item in body["items"]]
    assert scores == sorted(scores, reverse=True)
    assert all(item["title"] for item in body["items"])


def test_the_second_identical_request_is_served_from_cache(live_server):
    base_url, _ = live_server
    first = httpx.get(f"{base_url}/recommendations/2", params={"k": 5}).json()
    second = httpx.get(f"{base_url}/recommendations/2", params={"k": 5}).json()
    assert first["items"] == second["items"]


def test_invalid_input_is_rejected_in_spanish_and_journaled(live_server):
    base_url, journal = live_server
    response = httpx.get(f"{base_url}/recommendations/1", params={"k": 0})
    assert response.status_code == 400
    assert response.json()["message"].startswith("La solicitud")
    entries = [json.loads(line) for line in journal.read_text(encoding="utf-8").splitlines()]
    assert any(e["code"] == "validation_error" for e in entries)


def test_an_unknown_user_is_a_404(live_server):
    base_url, _ = live_server
    response = httpx.get(f"{base_url}/recommendations/999999")
    assert response.status_code == 404
    assert response.json()["code"] == "user_not_found"


def test_an_interaction_is_accepted(live_server):
    base_url, _ = live_server
    response = httpx.post(
        f"{base_url}/interactions", json={"user_id": 1, "event_type": "click", "item_id": 1}
    )
    assert response.status_code == 202
    assert response.json()["queued"] is True
