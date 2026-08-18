from fastapi import APIRouter, Depends

from src.api.dependencies import ModelStore, get_model_store
from src.api.schemas.response import HealthResponse, ReadyResponse

router = APIRouter(tags=["Health"])

_APP_VERSION = "0.1.0"


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness probe",
    description="Returns 200 if the server process is alive.",
)
async def health() -> HealthResponse:
    return HealthResponse(status="ok", version=_APP_VERSION)


@router.get(
    "/ready",
    response_model=ReadyResponse,
    summary="Readiness probe",
    description=(
        "Returns whether the model and preprocessor are loaded and the "
        "service is ready to accept prediction requests. "
        "Also exposes the active artifact version fingerprint, the ISO-8601 "
        "load timestamp, and the number of successful hot-reloads performed "
        "since server start."
    ),
)
async def ready(
    store: ModelStore = Depends(get_model_store),
) -> ReadyResponse:
    # Attempt to read enriched metadata from the active ArtifactContainer.
    # Falls back to None fields gracefully if no artifacts are loaded yet.
    artifact_version: str | None = None
    loaded_at: str | None = None

    if store.is_ready:
        try:
            container = store.get_artifacts()
            artifact_version = container.version
            loaded_at = container.loaded_at
        except RuntimeError:
            pass  # Between load cycles — report as not ready

    return ReadyResponse(
        ready=store.is_ready,
        model_loaded=store.is_ready,
        preprocessor_loaded=store.is_ready,
        artifact_version=artifact_version,
        reload_count=store.reload_count,
        loaded_at=loaded_at,
    )
