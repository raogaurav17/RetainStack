import os

class Config:

    # Logger Configs
    LOG_DIR: str = os.getenv("LOG_DIR", "logs")
    LOG_FILE: str = os.getenv("LOG_FILE", "app.log")
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "DEBUG")
    MAX_BYTES: int = int(os.getenv("LOG_MAX_BYTES", 5 * 1024 * 1024))  # 5 MB
    BACKUP_COUNT: int = int(os.getenv("LOG_BACKUP_COUNT", 3))


