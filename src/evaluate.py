import os
import pandas as pd
import joblib
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from Config import Config
from logger.logger import get_logger

logger = get_logger("evaluate")

def evaluate_model():
    """
    This function evaluates the trained model on test data.
    It logs accuracy, classification report, and confusion matrix.
    :return: None
    """
    try:
        logger.info("Model evaluation started...")

        # Load test data
        test_dir = os.path.join(Config.DATA_DIR, Config.PROCESSED_DATA_DIR)
        x_test_path = os.path.join(test_dir, "x_test.csv")
        y_test_path = os.path.join(test_dir, "y_test.csv")

        x_test = pd.read_csv(x_test_path)
        y_test = pd.read_csv(y_test_path)

        logger.debug(f"x_test shape: {x_test.shape}")
        logger.debug(f"y_test shape: {y_test.shape}")

        # Load model
        model_path = os.path.join(Config.DATA_DIR, Config.ARTIFACT_DIR, "model.pkl")
        if not os.path.exists(model_path):
            logger.error(f"Model file not found at {model_path}")
            return

        model = joblib.load(model_path)
        logger.debug("Model loaded successfully.")

        # Predict and evaluate
        y_pred = model.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)

        logger.info("Evaluation Metrics:")
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info("Classification Report:\n" + report)
        logger.info("Confusion Matrix:\n" + str(conf_matrix))

        # Optional: save metrics
        metrics_path = os.path.join(Config.DATA_DIR, Config.ARTIFACT_DIR, "evaluation_metrics.json")
        pd.DataFrame({
            "accuracy": [accuracy],
        }).to_json(metrics_path, orient="records", lines=True)
        logger.info(f"Evaluation metrics saved to {metrics_path}")

    except Exception as e:
        logger.error(e)


if __name__ == "__main__":
    evaluate_model()
