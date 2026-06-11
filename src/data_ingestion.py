from src.logger.logger import get_logger
from src.config import settings
from src.utils import PipelineError
import os
import pandas as pd
from sklearn.model_selection import train_test_split

logger = get_logger("data_ingestion")


def ingest_data():
    # getting data files path
    raw_data_path = os.path.join(settings.DATA_DIR, settings.RAW_DATA_FILE)
    train_data_path = os.path.join(settings.DATA_DIR, settings.TRAIN_DATA_FILE)
    test_data_path = os.path.join(settings.DATA_DIR, settings.TEST_DATA_FILE)
    try:
        #creating training and testing data
        raw_df = pd.read_csv(raw_data_path)
        logger.info(f"fetched raw data from: {raw_data_path}")
        train_df, test_df = train_test_split(raw_df,
                                             test_size=settings.TRAIN_TEST_SPLIT_RATIO,
                                             random_state=42)
        train_df.to_csv(train_data_path, index=False)
        test_df.to_csv(test_data_path, index=False)
        logger.info(f"Saving training data to: {train_data_path}")
        logger.info(f"Saving testing data to: {test_data_path}")


    except FileNotFoundError as e:
        logger.error(f"raw data file not found at {raw_data_path}")
        raise PipelineError(f"raw data file not found at {raw_data_path}") from e

    except Exception as e:
        logger.error(e)
        raise PipelineError(f"Unexpected error in data ingestion: {e}") from e


if __name__ == "__main__":
    ingest_data()

