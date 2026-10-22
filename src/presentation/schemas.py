"""Layer 1 - Presentation: request and response contracts (Table 7).

Declarative Pydantic v2 schemas for user identifiers, ranking sizes, and
interaction payloads. Because FastAPI validates against them before the
route body runs, Fail Fast (Section 3.6, NFR-04) is a framework feature here
rather than hand-written code: a malformed request is rejected before any
component of the AI layer is invoked.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated

from pydantic import BaseModel, Field, field_validator

#: Ranking sizes accepted by the endpoint. The upper bound exists so a single
#: request cannot ask the service to rank the whole catalog.
MIN_K = 1
MAX_K = 100

UserIdField = Annotated[int, Field(ge=1, description="Identificador del usuario que solicita")]
KField = Annotated[int, Field(ge=MIN_K, le=MAX_K, description="Cantidad de elementos a devolver")]


class RecommendedItem(BaseModel):
    """One entry of the Top-K ranking returned to the client."""

    item_id: int = Field(description="Identificador del elemento en el catálogo")
    rank: int = Field(ge=1, description="Posición en el ranking; 1 es la mejor")
    score: float = Field(description="Puntuación de relevancia de la estrategia activa")
    title: str | None = Field(default=None, description="Título en el catálogo, si se conoce")
    categories: list[str] = Field(
        default_factory=list, description="Géneros o categorías del contenido"
    )


class RecommendationResponse(BaseModel):
    """Response of ``GET /recommendations/{user_id}``.

    ``degraded`` and ``degradation_reason`` make the controlled degradation of
    NFR-05 visible to the client: the request still succeeds, and the client
    can tell that it was served by the fallback rather than by the configured
    model.
    """

    user_id: int
    k: int
    items: list[RecommendedItem]
    strategy_id: str = Field(description="Estrategia que generó el ranking")
    model_version: str
    feature_store_version: str
    degraded: bool = Field(
        default=False, description="Verdadero si se usó la estrategia de respaldo"
    )
    degradation_reason: str | None = Field(
        default=None, description="Código del error que activó el respaldo, si hubo degradación"
    )
    latency_ms: float = Field(
        default=0.0, description="Latencia de la solicitud medida en el servidor"
    )


class InteractionRequest(BaseModel):
    """Payload of ``POST /interactions`` (FR-02).

    Validated before the event reaches the queue, so malformed interaction
    data is rejected at the entry point instead of poisoning the history.
    """

    user_id: UserIdField
    event_type: str = Field(
        min_length=1, max_length=64, description="Tipo de evento: click, play, rating, ..."
    )
    item_id: int | None = Field(default=None, ge=1)
    recommended_items: list[int] = Field(default_factory=list, max_length=MAX_K)
    model_version: str | None = Field(default=None, max_length=64)
    metadata: dict[str, str] = Field(default_factory=dict)
    occurred_at: datetime | None = Field(default=None)

    @field_validator("event_type")
    @classmethod
    def _strip_event_type(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("no puede estar vacío")
        return cleaned

    @field_validator("recommended_items")
    @classmethod
    def _validate_items(cls, value: list[int]) -> list[int]:
        if any(item < 1 for item in value):
            raise ValueError("los identificadores deben ser positivos")
        return value


class InteractionAccepted(BaseModel):
    """Acknowledgement of an accepted interaction event."""

    event_id: str
    accepted: bool = True
    queued: bool = Field(
        description="Falso si el broker no aceptó el evento y quedó en el búfer de reintentos"
    )


class ErrorResponse(BaseModel):
    """Client-facing error payload, free of stack traces and internal ids."""

    code: str
    message: str
    detail: str | None = None


class HealthResponse(BaseModel):
    """Operational snapshot of the composed application."""

    status: str
    active_strategy: str
    registered_strategies: list[str]
    decorator_chain: list[str]
    feature_store_version: str
    feature_store_available: bool
    cache_backend: str
    queue_backend: str
    history_backend: str
    catalog_items: int
    user_profiles: int


__all__ = [
    "ErrorResponse",
    "HealthResponse",
    "InteractionAccepted",
    "InteractionRequest",
    "MAX_K",
    "MIN_K",
    "RecommendationResponse",
    "RecommendedItem",
]
