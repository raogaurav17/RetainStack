import os
import json
import pandas as pd
import mlflow
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

from src.config import settings
from src.utils import PipelineError, load_params
from src.logger.logger import get_logger

logger = get_logger("data_drift")

def detect_data_drift():
    """
    This function detects data drift between reference data (train) and current data (test).
    It generates an Evidently report and saves it as HTML and JSON, and logs metrics to MLflow.
    """
    try:
        logger.info("Data drift detection started...")

        # Load reference data (train) and current data (test)
        # We use test data here to represent 'new/incoming' data for monitoring purposes
        train_path = os.path.join(settings.DATA_DIR, settings.TRAIN_DATA_FILE)
        test_path = os.path.join(settings.DATA_DIR, settings.TEST_DATA_FILE)

        reference_data = pd.read_csv(train_path)
        current_data = pd.read_csv(test_path)
        
        # Load params to identify target column
        params = load_params(settings.PARAMS_FILE_PATH)
        target_col = params.get('data_preprocess', {}).get('target', 'Revenue')
        
        # Calculate drift on features by dropping the target column
        if target_col in reference_data.columns:
            reference_data = reference_data.drop(columns=[target_col])
        if target_col in current_data.columns:
            current_data = current_data.drop(columns=[target_col])

        # Generate Evidently Report
        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=reference_data, current_data=current_data)

        # Save Report
        artifact_dir = settings.artifact_path
        os.makedirs(artifact_dir, exist_ok=True)
        
        html_report_path = os.path.join(artifact_dir, "data_drift_report.html")
        json_report_path = os.path.join(artifact_dir, "data_drift_report.json")
        
        report.save_html(html_report_path)
        logger.info(f"Data drift HTML report saved at {html_report_path}")
        
        report.save_json(json_report_path)
        logger.info(f"Data drift JSON report saved at {json_report_path}")

        # Parse JSON to extract metrics for MLflow logging
        with open(json_report_path, "r") as f:
            drift_data = json.load(f)

        dataset_drift = drift_data["metrics"][0]["result"]["dataset_drift"]
        share_of_drifted_columns = drift_data["metrics"][0]["result"]["share_of_drifted_columns"]
        number_of_drifted_columns = drift_data["metrics"][0]["result"]["number_of_drifted_columns"]
        
        logger.info(f"Dataset Drift detected: {dataset_drift}")
        logger.info(f"Share of drifted columns: {share_of_drifted_columns}")
        logger.info(f"Number of drifted columns: {number_of_drifted_columns}")

        # Log to MLflow
        mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
        mlflow.set_experiment(settings.MLFLOW_EXPERIMENT_NAME)

        with mlflow.start_run(run_name="data_drift"):
            mlflow.log_metrics({
                "dataset_drift": int(dataset_drift),
                "share_of_drifted_columns": share_of_drifted_columns,
                "number_of_drifted_columns": number_of_drifted_columns
            })
            mlflow.log_artifact(html_report_path, artifact_path="drift_reports")
            mlflow.log_artifact(json_report_path, artifact_path="drift_reports")
            
        logger.info("Drift metrics and reports logged to MLflow")

    except Exception as e:
        logger.error(e)
        raise PipelineError(f"Unexpected error in data drift detection: {e}") from e


if __name__ == "__main__":
    detect_data_drift()
