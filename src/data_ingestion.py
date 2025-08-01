from src.logger.logger import get_logger
from src.Config import Config
import os
import pandas as pd
from sklearn.model_selection import train_test_split

logger = get_logger("data_ingestion")


def ingest_data():
    # getting data files path
    raw_data_path = os.path.join(Config.DATA_DIR, Config.RAW_DATA_FILE)
    train_data_path = os.path.join(Config.DATA_DIR, Config.TRAIN_DATA_FILE)
    test_data_path = os.path.join(Config.DATA_DIR, Config.TEST_DATA_FILE)
    try:
        #creating training and testing data
        raw_df = pd.read_csv(raw_data_path)
        logger.info(f"fetched raw data from: {raw_data_path}")
        train_df, test_df = train_test_split(raw_df,
                                             test_size=Config.TRAIN_TEST_SPLIT_RATIO,
                                             random_state=42)
        train_df.to_csv(train_data_path, index=False)
        test_df.to_csv(test_data_path, index=False)
        logger.info(f"Saving training data to: {train_data_path}")
        logger.info(f"Saving testing data to: {test_data_path}")


    except FileNotFoundError:
        logger.error(f"raw data file not found at {raw_data_path}")

    except Exception as e:
        logger.error(e)


if __name__ == "__main__":
    ingest_data()

