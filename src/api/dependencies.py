import os
from dataclasses import dataclass, field

import skops.io as skio

from src.config import settings
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
        """Load model.skops and preprocessor.skops from the artifact directory."""
        artifact_dir = settings.artifact_path

        model_path = os.path.join(artifact_dir, "model.skops")
        preprocessor_path = os.path.join(artifact_dir, "preprocessor.skops")

        if not os.path.exists(model_path):
            logger.error("model.skops not found at %s", model_path)
            raise FileNotFoundError(f"model.skops not found at {model_path}")

        if not os.path.exists(preprocessor_path):
            logger.error("preprocessor.skops not found at %s", preprocessor_path)
            raise FileNotFoundError(
                f"preprocessor.skops not found at {preprocessor_path}"
            )

        trusted_model = skio.get_untrusted_types(file=model_path)
        self.model = skio.load(model_path, trusted=trusted_model)
        logger.info("Model loaded from %s", model_path)

        trusted_preprocessor = skio.get_untrusted_types(file=preprocessor_path)
        self.preprocessor = skio.load(preprocessor_path, trusted=trusted_preprocessor)
        logger.info("Preprocessor loaded from %s", preprocessor_path)


# Singleton instance populated during startup lifespan
_store = ModelStore()

def get_model_store() -> ModelStore:
    """FastAPI dependency that returns the shared ModelStore."""
    return _store
