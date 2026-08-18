from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from src.api.batcher import get_batcher
from src.api.dependencies import get_model_store
from src.api.routes import health, predict
from src.api.routes import admin
from src.api.routes.predict import _RAW_FEATURE_ORDER
from src.logger.logger import get_logger

logger = get_logger("api.app")


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
    """Load model artifacts and start the dynamic batcher on startup;
    stop the batcher and clean up artifacts on shutdown."""
    store = get_model_store()
    batcher = get_batcher()
    try:
        store.load()
        batcher.start(store, _RAW_FEATURE_ORDER)
        logger.info("All artifacts loaded — API is ready to serve predictions.")
    except FileNotFoundError as exc:
        logger.warning("Startup: artifact missing (%s). Running in degraded mode.", exc)
    yield

    await batcher.stop()
    logger.info("Artifacts released — shutting down.")


def create_app() -> FastAPI:
    """Build and return the configured FastAPI application."""
    app = FastAPI(
        title="RetainStack API",
        description=(
            "Real-time purchase-intent prediction for e-commerce sessions. "
            "Powered by an XGBoost classifier with DVC-versioned artifacts."
        ),
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # Routes
    app.include_router(health.router, prefix="/api/v1")
    app.include_router(predict.router, prefix="/api/v1")
    app.include_router(admin.router, prefix="/api/v1")

    # Global exception handler
    @app.exception_handler(Exception)
    async def _unhandled_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        logger.error("Unhandled error on %s %s: %s", request.method, request.url.path, exc)
        return JSONResponse(
            status_code=500,
            content={"detail": "An unexpected error occurred."},
        )

    return app

app = create_app()
