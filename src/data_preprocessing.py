from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from src.logger.logger import get_logger
import pandas as pd
from src.config import settings
from src.utils import PipelineError, load_params
import os
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import skops.io as skio

logger = get_logger("data_preprocessing")

def preprocess_data() -> None:
    """
    This function is used for preprocessing the raw data
    :return: None
    """
    try:
        # defining raw data file path
        train_data_path = os.path.join(settings.DATA_DIR, settings.TRAIN_DATA_FILE)
        test_data_path = os.path.join(settings.DATA_DIR, settings.TEST_DATA_FILE)
        logger.info(f"Preprocessing raw data from {train_data_path} and {test_data_path}...")

        # Converting data to DataFrame
        temp_train_data = pd.read_csv(train_data_path)
        logger.debug(f"Data Fetched Successfully from {train_data_path}")
        test_data = pd.read_csv(test_data_path)
        logger.debug(f"Data Fetched Successfully from {test_data_path}")

        # Splitting temporary train data to train and validation data
        train_data, val_data = train_test_split(temp_train_data, test_size=settings.TRAIN_VAL_SPLIT_RATIO, random_state=42)
        logger.debug(f"Train and test data split Successfully Successfully from {train_data_path}")

        # getting column names from params.yaml
        params = load_params(settings.PARAMS_FILE_PATH)
        target_col = params['data_preprocess']['target']
        
        # Splitting input and output column
        x_train = train_data.drop(columns=[target_col])
        x_val = val_data.drop(columns=[target_col])
        x_test = test_data.drop(columns=[target_col])
        y_train = train_data[target_col]
        y_val = val_data[target_col]
        y_test = test_data[target_col]
        logger.info(f"Train and test data split successfully into input and output data")

        cols = params['data_preprocess']['features']
        categorical_features = params['data_preprocess']['categorical_features']
        numerical_features = [i for i in cols if i not in categorical_features]
        logger.info("Features params accessed successfully")

        # Checking all columns are present or not
        missing_cols = [col for col in cols if col not in x_train.columns]
        if missing_cols:
            logger.error(f"Missing columns in training data: {missing_cols}")
            raise PipelineError(f"Missing columns in training data: {missing_cols}")


        #defining a ColumnTransformer object for pipeline
        preprocessor = ColumnTransformer(transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
            ('num', MinMaxScaler(), numerical_features)
        ])

        # Transforming columns
        preprocessor.fit(x_train)
        x_train = preprocessor.transform(x_train)
        x_val = preprocessor.transform(x_val)
        x_test = preprocessor.transform(x_test)
        logger.info(f"Features transformed Successfully into {len(cols)} features")
        logger.debug(f"x_train shape: {x_train.shape}, x_val shape: {x_val.shape}, x_test shape: {x_test.shape}")

        # Converting inputs back to DataFrame
        ct_ft = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
        all_ft = list(ct_ft) + numerical_features
        x_train = pd.DataFrame(x_train, columns=all_ft)
        x_val = pd.DataFrame(x_val, columns=all_ft)
        x_test = pd.DataFrame(x_test, columns=all_ft)

        # defining preprocessed dir
        processed_dir = os.path.join(settings.DATA_DIR, settings.PROCESSED_DATA_DIR)
        os.makedirs(processed_dir, exist_ok=True)  # Required before saving CSVs
        x_train.to_csv(os.path.join(processed_dir, "x_train.csv"), index=False)
        x_val.to_csv(os.path.join(processed_dir, "x_val.csv"), index=False)
        x_test.to_csv(os.path.join(processed_dir, "x_test.csv"), index=False)
        y_train.to_csv(os.path.join(processed_dir, "y_train.csv"), index=False)
        y_val.to_csv(os.path.join(processed_dir, "y_val.csv"), index=False)
        y_test.to_csv(os.path.join(processed_dir, "y_test.csv"), index=False)

        # Storing ColumnTransformer Artifact
        artifact_dir = settings.artifact_path
        os.makedirs(artifact_dir, exist_ok=True)
        preprocessor_path = os.path.join(artifact_dir, "preprocessor.skops")
        skio.dump(preprocessor, preprocessor_path)
        logger.info(f"Preprocessor saved successfully at {preprocessor_path}")


        logger.info(f"Data Preprocessing Successfully into {processed_dir}")
        return

    except FileNotFoundError as e:
        logger.error("Data file not found, Exiting....")
        raise PipelineError("Data file not found") from e
    except Exception as e:
        logger.error(f"Unexpected error: {e}, Exiting....")
        raise PipelineError(f"Unexpected error in preprocessing: {e}") from e

if __name__ == "__main__":
    preprocess_data()

