import json
import os
import pandas as pd
import skops.io as skio
import mlflow
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from src.config import settings
from src.utils import PipelineError
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
        test_dir = os.path.join(settings.DATA_DIR, settings.PROCESSED_DATA_DIR)
        x_test_path = os.path.join(test_dir, "x_test.csv")
        y_test_path = os.path.join(test_dir, "y_test.csv")

        x_test = pd.read_csv(x_test_path)
        y_test = pd.read_csv(y_test_path).squeeze()  # ensure 1-D Series

        logger.debug(f"x_test shape: {x_test.shape}")
        logger.debug(f"y_test shape: {y_test.shape}")

        # Load model
        model_path = os.path.join(settings.artifact_path, "model.skops")
        if not os.path.exists(model_path):
            logger.error(f"Model file not found at {model_path}")
            raise PipelineError(f"Model file not found at {model_path}")

        trusted = skio.get_untrusted_types(file=model_path)
        model = skio.load(model_path, trusted=trusted)
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
        metrics_path = os.path.join(settings.artifact_path, "evaluation_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Evaluation metrics saved to {metrics_path}")

        # MLflow Tracking
        run_id_path = os.path.join(settings.artifact_path, "run_id.txt")
        if os.path.exists(run_id_path):
            with open(run_id_path, "r") as f:
                run_id = f.read().strip()
            
            mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
            mlflow.set_experiment(settings.MLFLOW_EXPERIMENT_NAME)
            
            with mlflow.start_run(run_id=run_id):
                mlflow.log_metrics({
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "roc_auc": roc_auc
                })
            logger.info(f"Evaluation metrics logged to MLflow run {run_id}")
        else:
            logger.warning(f"run_id.txt not found at {run_id_path}, skipping MLflow logging")

    except Exception as e:
        logger.error(e)
        raise PipelineError(f"Unexpected error in evaluation: {e}") from e


if __name__ == "__main__":
    evaluate_model()

