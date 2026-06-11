import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Logger Configs
    LOG_DIR: str = "logs"
    LOG_FILE: str = "app.log"
    LOG_LEVEL: str = "DEBUG"
    MAX_BYTES: int = 5 * 1024 * 1024  # 5 MB
    BACKUP_COUNT: int = 3

    # Data Dir (Used in various Configs)
    DATA_DIR: str = "data"

    # Data Ingestion and Preprocessing Configs
    RAW_DATA_FILE: str = "raw_data.csv"
    TRAIN_DATA_FILE: str = "train_data.csv"
    TEST_DATA_FILE: str = "test_data.csv"
    TRAIN_TEST_SPLIT_RATIO: float = 0.2
    TRAIN_VAL_SPLIT_RATIO: float = 0.2

    ## processed data config
    PROCESSED_DATA_DIR: str = "processed"

    # Artifacts config
    ARTIFACT_DIR: str = "artifact"

    # MLflow Configs
    MLFLOW_TRACKING_URI: str = "sqlite:///mlflow.db"
    MLFLOW_EXPERIMENT_NAME: str = "RetainStack_Experiment"
    
    # Custom project paths
    PARAMS_FILE_PATH: str = "params.yaml"
    
    @property
    def artifact_path(self) -> str:
        return os.path.join(self.DATA_DIR, self.ARTIFACT_DIR)
        
    class Config:
        env_file = ".env"

settings = Settings()
