import os
import json
import pandas as pd
import mlflow
from evidently import Report
from evidently.presets import DataDriftPreset

from src.config import settings
from src.utils import PipelineError, load_params
from src.logger.logger import get_logger

logger = get_logger("data_drift")

def detect_data_drift():
    """Detect data drift between reference (train) and current (test) datasets."""
    try:
        logger.info("Data drift detection started...")

        train_path = os.path.join(settings.DATA_DIR, settings.TRAIN_DATA_FILE)
        test_path = os.path.join(settings.DATA_DIR, settings.TEST_DATA_FILE)

        reference_data = pd.read_csv(train_path)
        current_data = pd.read_csv(test_path)
        
        params = load_params(settings.PARAMS_FILE_PATH)
        target_col = params.get('data_preprocess', {}).get('target', 'Revenue')
        
        # Calculate drift on features only by dropping the target column
        if target_col in reference_data.columns:
            reference_data = reference_data.drop(columns=[target_col])
        if target_col in current_data.columns:
            current_data = current_data.drop(columns=[target_col])

        # Generate Evidently data drift report
        report = Report(metrics=[DataDriftPreset()])
        snapshot = report.run(current_data=current_data, reference_data=reference_data)

        artifact_dir = settings.artifact_path
        os.makedirs(artifact_dir, exist_ok=True)
        
        html_report_path = os.path.join(artifact_dir, "data_drift_report.html")
        json_report_path = os.path.join(artifact_dir, "data_drift_report.json")
        
        snapshot.save_html(html_report_path)
        logger.info(f"Data drift HTML report saved at {html_report_path}")
        
        snapshot.save_json(json_report_path)
        logger.info(f"Data drift JSON report saved at {json_report_path}")

        # Parse JSON report to extract summary drift metrics
        with open(json_report_path, "r") as f:
            drift_data = json.load(f)

        # Locate the DriftedColumnsCount metric from the flat metrics list
        number_of_drifted_columns = 0
        share_of_drifted_columns = 0.0
        for metric in drift_data["metrics"]:
            if "DriftedColumnsCount" in metric["metric_name"]:
                number_of_drifted_columns = int(metric["value"]["count"])
                share_of_drifted_columns = float(metric["value"]["share"])
                break
        dataset_drift = number_of_drifted_columns > 0
        
        logger.info(f"Dataset Drift detected: {dataset_drift}")
        logger.info(f"Share of drifted columns: {share_of_drifted_columns}")
        logger.info(f"Number of drifted columns: {number_of_drifted_columns}")

        # Log metrics and HTML/JSON report artifacts to MLflow
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
