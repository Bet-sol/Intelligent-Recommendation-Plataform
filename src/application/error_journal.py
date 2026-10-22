"""ErrorJournal - bitácora de errores (append-only JSON Lines).

Records every error the service turns into a client response (4xx/5xx) and
every degradation it absorbs (cold start, model or feature store down, event
publication failure). It is *operational* evidence of what failed and when.
It is not the recommendation history of FR-06, which is business data written
by ``FeedbackHistoryService``; the two are kept apart on purpose.

Writing the journal must never be the reason a request fails, so ``record``
swallows its own I/O errors.
"""

from __future__ import annotations

import json
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class ErrorJournal:
    """Thread-safe append-only journal, one JSON object per line."""

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._lock = threading.Lock()

    @property
    def path(self) -> Path:
        return self._path

    def record(
        self,
        kind: str,
        code: str,
        message: str,
        *,
        route: str | None = None,
        user_id: int | None = None,
        status: int | None = None,
        detail: str | None = None,
        **extra: Any,
    ) -> None:
        """Append one entry. ``kind`` is ``error`` or ``degradation``."""
        entry = {
            "timestamp": datetime.now(UTC).isoformat(),
            "kind": kind,
            "code": code,
            "status": status,
            "route": route,
            "user_id": user_id,
            "message": message,
            "detail": detail,
            **extra,
        }
        try:
            with self._lock:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                with self._path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
        except OSError:
            pass

    def read_all(self) -> list[dict[str, Any]]:
        """Return every entry, skipping corrupt lines."""
        if not self._path.exists():
            return []
        entries: list[dict[str, Any]] = []
        with self._path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except ValueError:
                    continue
        return entries


__all__ = ["ErrorJournal"]
