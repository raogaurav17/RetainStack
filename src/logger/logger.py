import logging
from logging.handlers import RotatingFileHandler
from src.config import settings
import os

def get_logger(name: str) -> logging.Logger:
    """
    Function to define logger for constantly monitor
    various steps happening throughout the  pipeline
    :return: a Logger object which helps in logging
    """

    # Directory for storing log files
    os.makedirs(settings.LOG_DIR, exist_ok=True)

    # Initializing logger
    logger = logging.getLogger(name)

    #Sets the minimum logging level for the logger
    logger.setLevel(getattr(logging, settings.LOG_LEVEL))

    #Initialize log handler if not present already
    if not logger.handlers:
        file_handler = RotatingFileHandler(
            filename=os.path.join(settings.LOG_DIR, f"{name}.log"),
            maxBytes=settings.MAX_BYTES,
            backupCount=settings.BACKUP_COUNT,
            encoding="utf-8",
        )

        # Defining Formatter
        file_format = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(filename)s:%(lineno)d | %(message)s"
        )
        file_handler.setFormatter(file_format)

        # Defining Handler for console
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(file_format)

        # Adding Handlers to Logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger

