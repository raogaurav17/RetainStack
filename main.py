import uvicorn

from src.logger.logger import get_logger

logger = get_logger("main")


def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Start the FastAPI prediction server."""
    logger.info("=== Starting RetainStack API Server on %s:%d ===", host, port)
    uvicorn.run("src.api.app:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    serve()
