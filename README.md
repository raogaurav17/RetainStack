# RetainStack

RetainStack is an end-to-end MLOps pipeline for predicting online customer purchase intent. It uses an XGBoost binary classifier trained on e-commerce session data, with DVC for pipeline reproducibility and data versioning, MLflow for experiment tracking, and GitHub Actions for CI/CD automation.

## Features

- **FastAPI Serving Layer** — Real-time prediction API with health/readiness probes, Pydantic validation, and auto-generated OpenAPI docs
- **DVC Pipeline** — Reproducible, parameterised stages for ingestion, preprocessing, training, and evaluation
- **MLflow Tracking** — Integrated experiment tracking for logging hyperparameters, model metrics, and artifacts to SQLite
- **Data Versioning** — Raw data and model artifacts tracked and stored on AWS S3 via DVC
- **XGBoost Classifier** — Tuned binary classifier with class-imbalance handling
- **Full Evaluation Metrics** — Accuracy, Precision, Recall, F1, ROC-AUC, and confusion matrix persisted as DVC metrics
- **Rotating File Logging** — Per-module logs written to `logs/` with configurable log level and rotation
- **CI/CD** — GitHub Actions workflow that pulls data, reproduces the pipeline, and pushes artifacts
- **Environment-configurable** — All paths, split ratios, and hyperparameters overridable via environment variables or `params.yaml`

---

## Dataset

RetainStack uses the **[Online Shoppers Purchasing Intention Dataset](https://archive.ics.uci.edu/ml/datasets/Online+Shoppers+Purchasing+Intention+Dataset)** from the UCI Machine Learning Repository.

| Property | Details |
|---|---|
| **Source** | UCI Machine Learning Repository |
| **Records** | ~12,330 web sessions |
| **Task** | Binary Classification |
| **Target Variable** | `Revenue` — whether the session ended in a purchase (`True`/`False`) |
| **Domain** | E-commerce / Web Analytics |

### Features Used

| Feature | Type | Description |
|---|---|---|
| `Administrative` | Integer | Number of administrative pages visited |
| `Administrative_Duration` | Float | Total time spent on administrative pages (seconds) |
| `Informational_Duration` | Float | Total time spent on informational pages (seconds) |
| `ProductRelated` | Integer | Number of product-related pages visited |
| `ProductRelated_Duration` | Float | Total time spent on product-related pages (seconds) |
| `BounceRates` | Float | Average bounce rate of pages visited |
| `ExitRates` | Float | Average exit rate of pages visited |
| `PageValues` | Float | Average page value of pages visited before a transaction |
| `Month` | Categorical | Month of the visit |
| `Revenue` | Boolean | **Target** — whether a purchase was completed |

---

## Project Structure

```text
RetainStack/
├── data/
│   ├── raw_data.csv           # Source dataset (DVC-tracked)
│   ├── train_data.csv         # Train split (DVC-tracked)
│   ├── test_data.csv          # Test split (DVC-tracked)
│   ├── processed/             # Preprocessed feature/label CSVs
│   └── artifact/
│       ├── preprocessor.skops # Fitted ColumnTransformer
│       ├── model.skops        # Trained XGBoost model
│       └── evaluation_metrics.json
├── Experiments/               # Exploratory notebooks
│   ├── data_exploration.ipynb
│   └── model_training.ipynb
├── src/
│   ├── api/
│   │   ├── app.py             # FastAPI application factory + lifespan
│   │   ├── dependencies.py    # Model/preprocessor loading & DI
│   │   ├── routes/
│   │   │   ├── health.py      # GET /health, /ready
│   │   │   └── predict.py     # POST /predict
│   │   └── schemas/
│   │       ├── request.py     # Pydantic input validation
│   │       └── response.py    # Pydantic response models
│   ├── logger/
│   │   └── logger.py          # Rotating file + console logger
│   ├── Config.py              # Centralised configuration (env-overridable)
│   ├── data_ingestion.py      # Loads raw data and produces train/test splits
│   ├── data_preprocessing.py  # Feature engineering and scaling
│   ├── train.py               # XGBoost model training
│   └── evaluate.py            # Full model evaluation + metric persistence
├── main.py                    # End-to-end pipeline orchestrator
├── dvc.yaml                   # DVC pipeline stage definitions
├── dvc.lock                   # DVC pipeline lock file
├── params.yaml                # Hyperparameters and feature config
├── pyproject.toml             # Project metadata and dependencies (uv)
└── README.md
```

---

## Setup Instructions

### Prerequisites

- Python ≥ 3.11
- [uv](https://github.com/astral-sh/uv) package manager
- AWS credentials with read/write access to the DVC S3 remote

### Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/raogaurav17/RetainStack.git
   cd RetainStack
   ```

2. **Create a virtual environment and install dependencies**
   ```bash
   uv sync
   ```

3. **Activate the virtual environment**
   ```bash
   source .venv/bin/activate      # Linux / macOS
   # .venv\Scripts\activate       # Windows
   ```

4. **Pull data and artifacts from the DVC remote**
   ```bash
   uv run dvc pull
   ```

---

## Running the Pipeline

### Option A — DVC (recommended, stage-level caching)

Reproduce only the stages that have changed:

```bash
uv run dvc repro
```

Force a full re-run:

```bash
uv run dvc repro --force
```

### Option B — Run individual stages manually

```bash
python -m src.data_ingestion
python -m src.data_preprocessing
python -m src.train
python -m src.evaluate
```

---

## API Server

RetainStack includes a FastAPI serving layer for real-time purchase-intent predictions.

### Start the server

```bash
uv run python main.py
```

Or directly via uvicorn:

```bash
uv run uvicorn src.api.app:app --host 127.0.0.1 --port 8000
```

Interactive API docs are available at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

### Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/v1/health` | Liveness probe — returns `{"status": "ok"}` |
| `GET` | `/api/v1/ready` | Readiness probe — confirms model & preprocessor are loaded |
| `POST` | `/api/v1/predict` | Predict purchase intent for a single session |

### Example prediction request

```bash
curl -s -X POST http://127.0.0.1:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Administrative": 0,
    "Administrative_Duration": 0.0,
    "Informational_Duration": 0.0,
    "ProductRelated": 53,
    "ProductRelated_Duration": 1482.5,
    "BounceRates": 0.02,
    "ExitRates": 0.05,
    "PageValues": 8.15,
    "Month": "Nov"
  }'
```

**Response:**

```json
{
  "prediction": 1,
  "purchase_probability": 0.5546,
  "confidence": "medium"
}
```

| Response Field | Description |
|---|---|
| `prediction` | `1` = purchase predicted, `0` = no purchase |
| `purchase_probability` | Model's probability estimate (0–1) |
| `confidence` | `high` (≥ 0.75), `medium` (≥ 0.40), or `low` (< 0.40) |

---

## Configuration

| File | Purpose |
|---|---|
| `params.yaml` | Feature list, categorical features, XGBoost hyperparameters |
| `src/Config.py` | Directory paths and split ratios — all values are overridable via environment variables |
| `dvc.yaml` | Pipeline stage definitions, dependencies, outputs, and metric declarations |

### Key Environment Variables

| Variable | Default | Description |
|---|---|---|
| `DATA_DIR` | `data` | Root directory for all data files |
| `TRAIN_TEST_SPLIT_RATIO` | `0.2` | Fraction of raw data held out as test set |
| `TRAIN_VAL_SPLIT_RATIO` | `0.2` | Fraction of training data held out as validation set |
| `LOG_LEVEL` | `DEBUG` | Logging verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `LOG_DIR` | `logs` | Directory where log files are written |

---

## Metrics & Results

Evaluation metrics are written to `data/artifact/evaluation_metrics.json` after each pipeline run. View them with:

```bash
uv run dvc metrics show
```

Compare across experiments or git commits:

```bash
uv run dvc metrics diff
```

Metrics tracked:

| Metric | Description |
|---|---|
| `accuracy` | Overall classification accuracy |
| `precision` | Precision on the positive class |
| `recall` | Recall on the positive class |
| `f1` | F1 score (harmonic mean of precision and recall) |
| `roc_auc` | Area under the ROC curve |
| `confusion_matrix` | 2×2 confusion matrix |

### MLflow UI

In addition to DVC metrics, you can visualize all experiments, parameters, and models using the MLflow UI. To launch the UI, run:

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Navigate to `http://127.0.0.1:5000` in your browser to view the `RetainStack_Experiment` and compare different runs.

---

## Future Improvements

- **Batch prediction endpoint** — `POST /api/v1/predict/batch` for bulk inference
- **Model hot-reload** — Reload artifacts without restarting the server after a pipeline re-run
- **Hyperparameter tuning** — Automated search with Optuna or scikit-learn `GridSearchCV`
- **Data validation** — Schema checks on ingested data with `pandera` or `great_expectations`
- **Drift monitoring** — Alerting when feature or prediction distributions shift in production
- **Extended feature set** — Evaluate dropped features (`VisitorType`, `Weekend`, `SpecialDay`, etc.)
- **Cloud deployment** — AWS SageMaker or GCP Vertex AI integration

---

## Author

Gaurav
