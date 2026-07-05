import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Logger settings
    LOG_DIR: str = "logs"
    LOG_FILE: str = "app.log"
    LOG_LEVEL: str = "DEBUG"
    MAX_BYTES: int = 5 * 1024 * 1024  # 5 MB
    BACKUP_COUNT: int = 3

    # Data directory paths
    DATA_DIR: str = "data"

    # Dataset file configurations
    RAW_DATA_FILE: str = "raw_data.csv"
    TRAIN_DATA_FILE: str = "train_data.csv"
    TEST_DATA_FILE: str = "test_data.csv"
    TRAIN_TEST_SPLIT_RATIO: float = 0.2
    TRAIN_VAL_SPLIT_RATIO: float = 0.2

    # Processed data settings
    PROCESSED_DATA_DIR: str = "processed"

    # Artifact settings
    ARTIFACT_DIR: str = "artifact"

    # Dynamic batching configurations
    BATCH_MAX_SIZE: int = 32
    BATCH_TIMEOUT_MS: int = 50

    # MLflow configurations
    MLFLOW_TRACKING_URI: str = "sqlite:///mlflow.db"
    MLFLOW_EXPERIMENT_NAME: str = "RetainStack_Experiment"
    
    # Project paths
    PARAMS_FILE_PATH: str = "params.yaml"
    
    @property
    def artifact_path(self) -> str:
        return os.path.join(self.DATA_DIR, self.ARTIFACT_DIR)
        
    class Config:
        env_file = ".env"

settings = Settings()
