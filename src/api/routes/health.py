"""Health and readiness probes for RetainStack API."""

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
        "service is ready to accept prediction requests."
    ),
)
async def ready(
    store: ModelStore = Depends(get_model_store),
) -> ReadyResponse:
    return ReadyResponse(
        ready=store.is_ready,
        model_loaded=store.model is not None,
        preprocessor_loaded=store.preprocessor is not None,
    )
