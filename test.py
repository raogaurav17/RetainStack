##CODE FOR TESTING LOGGER
# from logger.logger import get_logger
#
# logger = get_logger(__name__)
#
# def divide(a, b):
#     logger.debug(f"Starting division: {a} / {b}")
#     try:
#         result = a / b
#         logger.info(f"Division successful: {result}")
#         return result
#     except ZeroDivisionError as e:
#         logger.error("Attempted division by zero")
#         logger.exception("Exception occurred in divide function")
#         return None
#
# def main():
#     logger.info("Logger test started")
#     divide(10, 2)   # Should log debug and info
#     divide(5, 0)    # Should log error and exception
#     logger.warning("This is a warning")
#     logger.critical("This is a critical message")
#     logger.info("Logger test complete")
#
# if __name__ == "__main__":
#     main()


import os
import pandas as pd
from Config import Config
from src.data_ingestion import ingest_data


# Cleanup before test (in case files exist)
train_path = os.path.join(Config.DATA_DIR, Config.TRAIN_DATA_FILE)
test_path = os.path.join(Config.DATA_DIR, Config.TEST_DATA_FILE)

if os.path.exists(train_path):
    os.remove(train_path)
if os.path.exists(test_path):
    os.remove(test_path)

# Act
ingest_data()

# Assert
assert os.path.exists(train_path), "Train file was not created"
assert os.path.exists(test_path), "Test file was not created"

# Validate content structure (optional)
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

assert not train_df.empty, "Train CSV is empty"
assert not test_df.empty, "Test CSV is empty"
assert train_df.shape[1] == test_df.shape[1], "Mismatch in column count between train and test"
