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

# #CODE FOR TESTING DATA INGESTION
# import os
# import pandas as pd
# from Config import Config
# from src.data_ingestion import ingest_data
#
#
# # Cleanup before test (in case files exist)
# train_path = os.path.join(Config.DATA_DIR, Config.TRAIN_DATA_FILE)
# test_path = os.path.join(Config.DATA_DIR, Config.TEST_DATA_FILE)
#
# if os.path.exists(train_path):
#     os.remove(train_path)
# if os.path.exists(test_path):
#     os.remove(test_path)
#
# # Act
# ingest_data()
#
# # Assert
# assert os.path.exists(train_path), "Train file was not created"
# assert os.path.exists(test_path), "Test file was not created"
#
# # Validate content structure (optional)
# train_df = pd.read_csv(train_path)
# test_df = pd.read_csv(test_path)
#
# assert not train_df.empty, "Train CSV is empty"
# assert not test_df.empty, "Test CSV is empty"
# assert train_df.shape[1] == test_df.shape[1], "Mismatch in column count between train and test"


# #CODE FOR TESTING DATA PREPROCESSING
# import os
# from Config import Config
# from src.data_preprocessing import preprocess_data
#
# def test_preprocessing():
#     print("📦 Running preprocessing pipeline...")
#     preprocess_data()
#
#     processed_dir = os.path.join(Config.DATA_DIR, Config.PROCESSED_DATA_DIR)
#     expected_files = [
#         "x_train.csv", "x_val.csv", "x_test.csv",
#         "y_train.csv", "y_val.csv", "y_test.csv"
#     ]
#
#     all_exist = True
#     for file in expected_files:
#         file_path = os.path.join(processed_dir, file)
#         if os.path.exists(file_path):
#             print(f"{file} generated successfully.")
#         else:
#             print(f"{file} is missing!")
#             all_exist = False
#
#     if all_exist:
#         print("\n Preprocessing completed and all files are present.")
#     else:
#         print("\n️ Preprocessing incomplete. Please check the logs for errors.")
#
# if __name__ == "__main__":
#     test_preprocessing()

# # Training module test
# import os
# from src.train import model_train
# from Config import Config
# import joblib
#
# def test_model_train():
#     # Run training
#     model_train()
#
#     # Check if model was saved
#     model_path = os.path.join(Config.DATA_DIR, Config.ARTIFACT_DIR, "model.pkl")
#     assert os.path.exists(model_path), "Model file was not saved."
#
#     # Check if it’s a valid model
#     model = joblib.load(model_path)
#     assert hasattr(model, "predict"), "Loaded model does not have 'predict' method."
#
# if __name__ == "__main__":
#     test_model_train()

# test for eval module
import os
import json
from src.Config import Config
from src.evaluate import evaluate_model
def test_evaluate_model():
    # Run evaluation
    evaluate_model()

    # Check that evaluation_metrics.json is created
    metrics_path = os.path.join(
        Config.DATA_DIR, Config.ARTIFACT_DIR, "evaluation_metrics.json"
    )
    assert os.path.exists(metrics_path), "Evaluation metrics JSON file not created."

    # Load and validate metrics
    with open(metrics_path, "r") as f:
        lines = f.readlines()
        assert len(lines) > 0
        metrics = json.loads(lines[0])
        assert "accuracy" in metrics
        assert 0.0 <= metrics["accuracy"] <= 1.0

if __name__ == "__main__":
    test_evaluate_model()
