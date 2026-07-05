import logging
from logging.handlers import RotatingFileHandler
from src.config import settings
import os

def get_logger(name: str) -> logging.Logger:
    """Initialize and configure a rotating file and stream logger."""
    os.makedirs(settings.LOG_DIR, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, settings.LOG_LEVEL))

    if not logger.handlers:
        file_handler = RotatingFileHandler(
            filename=os.path.join(settings.LOG_DIR, f"{name}.log"),
            maxBytes=settings.MAX_BYTES,
            backupCount=settings.BACKUP_COUNT,
            encoding="utf-8",
        )

        file_format = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(filename)s:%(lineno)d | %(message)s"
        )
        file_handler.setFormatter(file_format)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(file_format)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger

