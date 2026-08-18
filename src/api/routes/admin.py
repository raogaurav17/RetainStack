"""Admin management routes for the RetainStack prediction API.

Exposes a single privileged endpoint:

  POST /api/v1/model/reload
      Triggers a zero-downtime hot-reload of the model and preprocessor
      artifacts from disk.  The active model continues serving all in-flight
      requests without interruption.  On success the response includes the
      old and new artifact version fingerprints and the pre-flight latency,
      giving operators immediate confirmation that the swap succeeded and which
      artifact revision is now live.

Security note
-------------
In a production deployment this router should be protected behind an internal
network boundary or authenticated with a secret header / mTLS.  For this
project the endpoint is intentionally left unauthenticated to keep the demo
setup simple, but a ``Depends(verify_admin_token)`` guard can be wired in at
the router level without changing the handler logic.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from src.api.dependencies import ArtifactReloadError, ModelStore, get_model_store
from src.api.schemas.response import ReloadResponse
from src.logger.logger import get_logger

logger = get_logger("api.admin")

router = APIRouter(prefix="/model", tags=["Admin"])


@router.post(
    "/reload",
    response_model=ReloadResponse,
    responses={
        500: {"description": "Pre-flight validation failed — active model unchanged"},
        503: {"description": "Artifact files not found on disk"},
    },
    summary="Hot-reload model artifacts",
    description=(
        "Reload ``model.skops`` and ``preprocessor.skops`` from the artifact directory "
        "without restarting the server or dropping any in-flight prediction requests. "
        "\n\n"
        "**Three-phase protocol:**\n"
        "1. **Load** — Deserialise candidate `.skops` artifacts outside the lock.\n"
        "2. **Pre-flight** — Run a synthetic dry-run inference pass to validate "
        "schema compatibility and output shape.\n"
        "3. **Swap** — Atomically replace the active artifact pointer under a write lock. "
        "In-flight requests always finish against the snapshot they captured at their own "
        "start, so there is zero torn-read risk.\n\n"
        "If pre-flight validation fails the active model is **never** replaced."
    ),
)
async def reload_model(
    store: ModelStore = Depends(get_model_store),
) -> ReloadResponse:
    """Atomically hot-reload model artifacts with pre-flight validation."""

    # Capture the version fingerprint of the currently active model *before*
    # attempting the reload so we can report it in the response regardless of
    # whether the reload succeeds.
    previous_version: str | None = None
    if store.is_ready:
        try:
            previous_version = store.get_artifacts().version
        except RuntimeError:
            pass  # No active model — first load via this endpoint

    logger.info(
        "Hot-reload requested (previous_version=%s, reload_count=%d)",
        previous_version,
        store.reload_count,
    )

    try:
        old_container, new_container, preflight_latency_ms = store.reload()
    except FileNotFoundError as exc:
        logger.error("Hot-reload failed — artifact file missing: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Artifact file not found: {exc}",
        ) from exc
    except ArtifactReloadError as exc:
        logger.error("Hot-reload aborted — pre-flight validation failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Pre-flight validation failed — active model unchanged. Reason: {exc}",
        ) from exc
    except Exception as exc:
        logger.error("Hot-reload failed with unexpected error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected reload error: {exc}",
        ) from exc

    logger.info(
        "Hot-reload succeeded: %s → %s (preflight=%.1fms)",
        previous_version,
        new_container.version,
        preflight_latency_ms,
    )

    return ReloadResponse(
        status="ok",
        message=(
            f"Model hot-reload successful. "
            f"Active artifact: {new_container.version} (loaded at {new_container.loaded_at})."
        ),
        previous_version=previous_version,
        new_version=new_container.version,
        reload_count=store.reload_count,
        preflight_latency_ms=round(preflight_latency_ms, 2),
    )
