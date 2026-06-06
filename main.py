from src.data_ingestion import ingest_data
from src.data_preprocessing import preprocess_data
from src.train import model_train
from src.evaluate import evaluate_model
from src.logger.logger import get_logger

logger = get_logger("main")


def main():
    logger.info("=== RetainStack Pipeline Started ===")

    logger.info("--- Stage 1: Data Ingestion ---")
    ingest_data()

    logger.info("--- Stage 2: Data Preprocessing ---")
    preprocess_data()

    logger.info("--- Stage 3: Model Training ---")
    model_train()

    logger.info("--- Stage 4: Model Evaluation ---")
    evaluate_model()

    logger.info("=== RetainStack Pipeline Completed ===")


if __name__ == "__main__":
    main()
