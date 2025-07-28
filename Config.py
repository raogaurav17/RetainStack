import os

class Config:

    # Logger Configs
    LOG_DIR: str = os.getenv("LOG_DIR", "logs")
    LOG_FILE: str = os.getenv("LOG_FILE", "app.log")
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "DEBUG")
    MAX_BYTES: int = int(os.getenv("LOG_MAX_BYTES", 5 * 1024 * 1024))  # 5 MB
    BACKUP_COUNT: int = int(os.getenv("LOG_BACKUP_COUNT", 3))

    # Data Dir (Used in various Configs)
    DATA_DIR: str = os.getenv("DATA_DIR", "data")

    # Data Ingestion and Preprocessing Configs
    RAW_DATA_FILE: str = os.getenv("RAW_DATA_FILE", "raw_data.csv")
    TRAIN_DATA_FILE: str = os.getenv("TRAIN_DATA_DIR", "train_data.csv")
    TEST_DATA_FILE: str = os.getenv("TEST_DATA_DIR", "test_data.csv")
    TRAIN_TEST_SPLIT_RATIO: float = os.getenv("TRAIN_TEST_SPLIT_RATIO", 0.2)
    TRAIN_VAL_SPLIT_RATIO: float = os.getenv("TRAIN_TEST_SPLIT_RATIO", 0.2)




