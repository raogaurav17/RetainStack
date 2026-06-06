import json
import os
import pandas as pd
import joblib
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from src.Config import Config
from src.logger.logger import get_logger

logger = get_logger("evaluate")

def evaluate_model():
    """
    This function evaluates the trained model on test data.
    It logs and persists accuracy, precision, recall, F1, ROC-AUC,
    the classification report, and the confusion matrix.
    :return: None
    """
    try:
        logger.info("Model evaluation started...")

        # Load test data
        test_dir = os.path.join(Config.DATA_DIR, Config.PROCESSED_DATA_DIR)
        x_test_path = os.path.join(test_dir, "x_test.csv")
        y_test_path = os.path.join(test_dir, "y_test.csv")

        x_test = pd.read_csv(x_test_path)
        y_test = pd.read_csv(y_test_path).squeeze()  # ensure 1-D Series

        logger.debug(f"x_test shape: {x_test.shape}")
        logger.debug(f"y_test shape: {y_test.shape}")

        # Load model
        model_path = os.path.join(Config.DATA_DIR, Config.ARTIFACT_DIR, "model.pkl")
        if not os.path.exists(model_path):
            logger.error(f"Model file not found at {model_path}")
            return

        model = joblib.load(model_path)
        logger.debug("Model loaded successfully.")

        # Predict
        y_pred = model.predict(x_test)
        y_prob = model.predict_proba(x_test)[:, 1]

        # Compute metrics
        accuracy  = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall    = recall_score(y_test, y_pred, zero_division=0)
        f1        = f1_score(y_test, y_pred, zero_division=0)
        roc_auc   = roc_auc_score(y_test, y_prob)
        report    = classification_report(y_test, y_pred)
        conf_mat  = confusion_matrix(y_test, y_pred).tolist()

        logger.info("Evaluation Metrics:")
        logger.info(f"Accuracy : {accuracy:.4f}")
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall   : {recall:.4f}")
        logger.info(f"F1 Score : {f1:.4f}")
        logger.info(f"ROC-AUC  : {roc_auc:.4f}")
        logger.info("Classification Report:\n" + report)
        logger.info("Confusion Matrix:\n" + str(conf_mat))

        # Persist metrics as a flat JSON for dvc metrics show
        metrics = {
            "accuracy":  round(accuracy,  4),
            "precision": round(precision, 4),
            "recall":    round(recall,    4),
            "f1":        round(f1,        4),
            "roc_auc":   round(roc_auc,   4),
            "confusion_matrix": conf_mat,
        }
        metrics_path = os.path.join(Config.DATA_DIR, Config.ARTIFACT_DIR, "evaluation_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Evaluation metrics saved to {metrics_path}")

    except Exception as e:
        logger.error(e)


if __name__ == "__main__":
    evaluate_model()

