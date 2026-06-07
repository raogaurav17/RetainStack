import os
from dataclasses import dataclass, field

import joblib

from src.Config import Config
from src.logger.logger import get_logger

logger = get_logger("api.dependencies")


@dataclass
class ModelStore:
    """Holds the loaded ML artifacts so they are shared across requests."""

    model: object | None = field(default=None, repr=False)
    preprocessor: object | None = field(default=None, repr=False)

    @property
    def is_ready(self) -> bool:
        return self.model is not None and self.preprocessor is not None

    def load(self) -> None:
        """Load model.pkl and preprocessor.pkl from the artifact directory."""
        artifact_dir = os.path.join(Config.DATA_DIR, Config.ARTIFACT_DIR)

        model_path = os.path.join(artifact_dir, "model.pkl")
        preprocessor_path = os.path.join(artifact_dir, "preprocessor.pkl")

        if not os.path.exists(model_path):
            logger.error("model.pkl not found at %s", model_path)
            raise FileNotFoundError(f"model.pkl not found at {model_path}")

        if not os.path.exists(preprocessor_path):
            logger.error("preprocessor.pkl not found at %s", preprocessor_path)
            raise FileNotFoundError(
                f"preprocessor.pkl not found at {preprocessor_path}"
            )

        self.model = joblib.load(model_path)
        logger.info("Model loaded from %s", model_path)

        self.preprocessor = joblib.load(preprocessor_path)
        logger.info("Preprocessor loaded from %s", preprocessor_path)


# Module-level singleton — populated by the lifespan handler in app.py.
_store = ModelStore()


def get_model_store() -> ModelStore:
    """FastAPI dependency that returns the shared ModelStore."""
    return _store
