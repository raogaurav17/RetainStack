import pandas as pd
import xgboost as xgb
import os
import joblib
import yaml
import mlflow
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

from src.Config import Config
from src.logger.logger import get_logger

logger = get_logger("train")

def model_train():
    """
    This function is used to train the model and save the model
    :return: None
        """
    try:
        logger.info("Model training started...")

        #processed data paths
        processed_dir = os.path.join(Config.DATA_DIR, Config.PROCESSED_DATA_DIR)
        os.makedirs(processed_dir, exist_ok=True)  # Required before saving CSVs
        x_train = pd.read_csv(os.path.join(processed_dir, "x_train.csv"))
        x_val = pd.read_csv(os.path.join(processed_dir, "x_val.csv"))
        y_train = pd.read_csv(os.path.join(processed_dir, "y_train.csv"))
        y_val = pd.read_csv(os.path.join(processed_dir, "y_val.csv"))
        logger.debug("Training and validation data loaded...")

        # getting params
        with open("params.yaml", 'r') as stream:
            params = yaml.safe_load(stream)


        # defining model
        model = xgb.XGBClassifier(
            n_estimators=params['train']['n_estimators'],
            learning_rate=params['train']['learning_rate'],
            objective=params['train']['objective'],
            eval_metric=params['train']['eval_metric'],
            max_depth=params['train']['max_depth'],
            subsample=params['train']['subsample'],
            scale_pos_weight=params['train']['scale_pos_weight']
        )
        logger.debug("Model Initialized")

        # training model
        model.fit(x_train, y_train)
        logger.debug("Model Trained")

        # testing on validation data
        y_pred = model.predict(x_val)
        logger.info(classification_report(y_val, y_pred))
        accuracy = accuracy_score(y_val, y_pred)
        logger.info("Accuracy: {}".format(accuracy))
        logger.info(f"\nConfusion matrix: {confusion_matrix(y_val, y_pred)}")

        # saving model
        artifact_dir = os.path.join(Config.DATA_DIR, Config.ARTIFACT_DIR)
        os.makedirs(artifact_dir, exist_ok=True)
        model_path = os.path.join(artifact_dir, "model.pkl")
        joblib.dump(model, model_path)
        logger.info(f"Model saved successfully at {model_path}")

        # MLflow Tracking
        mlflow.set_tracking_uri(Config.MLFLOW_TRACKING_URI)
        mlflow.set_experiment(Config.MLFLOW_EXPERIMENT_NAME)
        
        with mlflow.start_run() as run:
            mlflow.log_params(params['train'])
            mlflow.sklearn.log_model(
                model,
                artifact_path="model"
            )
            
            # Save run_id for evaluate.py
            run_id_path = os.path.join(artifact_dir, "run_id.txt")
            with open(run_id_path, "w") as f:
                f.write(run.info.run_id)
            logger.info(f"MLflow run ID ({run.info.run_id}) saved to {run_id_path}")

    except Exception as e:
        logger.error(e)

if __name__ == "__main__":
    model_train()






