"""Bitácora de errores y mensajes en español."""

from __future__ import annotations

from fastapi.testclient import TestClient

from application.api import create_app
from application.error_journal import ErrorJournal
from config.settings import Settings
from errors import MENSAJES_ES, ColdStartError, ModelUnavailableError, UserNotFoundError
from persistence.interfaces import CatalogItem, UserProfile
from persistence.sql_catalog_repository import SqlCatalogRepository
from persistence.sql_user_profile_repository import SqlUserProfileRepository
from tests.conftest import N_ITEMS, N_USERS


def _seed(settings: Settings) -> None:
    profiles = SqlUserProfileRepository.from_settings(settings)
    for user_id in list(range(1, N_USERS + 1)) + [99]:
        profiles.save(UserProfile(user_id=user_id))
    SqlCatalogRepository.from_settings(settings).save_many(
        [CatalogItem(item_id=101 + i, title=f"Item {101 + i}") for i in range(N_ITEMS)]
    )


def test_journal_appends_one_json_line_per_entry(tmp_path):
    journal = ErrorJournal(tmp_path / "x" / "errors.jsonl")
    journal.record("error", "user_not_found", "msg", route="/r", user_id=7, status=404)
    journal.record("degradation", "cold_start", "msg2")
    entries = journal.read_all()
    assert [e["code"] for e in entries] == ["user_not_found", "cold_start"]
    assert entries[0]["user_id"] == 7 and entries[0]["timestamp"]


def test_journal_never_raises_when_the_path_is_unwritable(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    ErrorJournal(blocker / "errors.jsonl").record("error", "c", "m")


def test_every_error_code_has_a_spanish_message():
    for cls in (UserNotFoundError, ColdStartError, ModelUnavailableError):
        assert cls.code in MENSAJES_ES
        assert cls("english text").to_payload()["message"] == MENSAJES_ES[cls.code]


def test_unknown_code_falls_back_to_the_original_message():
    class Custom(ModelUnavailableError):
        code = "not_translated"

    assert Custom("original").to_payload()["message"] == "original"


def test_404_is_in_spanish_and_journaled(settings: Settings):
    _seed(settings)
    with TestClient(create_app(settings)) as client:
        response = client.get("/recommendations/4242?k=3")
        journal = client.app.state.components.journal
    assert response.status_code == 404
    assert response.json()["message"] == MENSAJES_ES["user_not_found"]
    entry = journal.read_all()[-1]
    assert (entry["kind"], entry["code"], entry["status"]) == ("error", "user_not_found", 404)
    assert entry["route"] == "/recommendations/4242"


def test_400_is_in_spanish_and_journaled(settings: Settings):
    with TestClient(create_app(settings)) as client:
        response = client.get("/recommendations/1?k=0")
        journal = client.app.state.components.journal
    assert response.status_code == 400
    assert response.json()["message"] == MENSAJES_ES["validation_error"]
    assert journal.read_all()[-1]["code"] == "validation_error"


def test_cold_start_degradation_is_journaled(settings: Settings):
    _seed(settings)
    with TestClient(create_app(settings)) as client:
        response = client.get("/recommendations/99?k=3")
        journal = client.app.state.components.journal
    assert response.status_code == 200 and response.json()["degraded"] is True
    entry = journal.read_all()[-1]
    assert (entry["kind"], entry["code"], entry["user_id"]) == ("degradation", "cold_start", 99)
