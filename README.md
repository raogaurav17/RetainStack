# RetainStack

RetainStack is an end-to-end MLOps project for predicting whether a visitor to an e-commerce website will complete a purchase. It combines a production-ready FastAPI serving layer, reproducible DVC pipeline, experiment tracking with MLflow, and robust deployment patterns such as dynamic batching and zero-downtime model reloads.

The project is designed to mirror a realistic ML system lifecycle: ingest data, preprocess features, train a model, evaluate it, track metrics, serve predictions, and continually validate the service under load.

## Why this project?

RetainStack is useful for learning and demonstration because it covers the full machine learning operations loop rather than only model training:

- Data pipeline orchestration with DVC
- Experiment tracking and model lineage with MLflow
- Secure artifact serialization with `skops`
- API serving with FastAPI and Pydantic validation
- Advanced inference patterns such as dynamic batching and hot-reload
- Operational checks with health probes, metrics, and stress testing

---

## Project highlights

- FastAPI serving layer with health/readiness endpoints and OpenAPI docs
- Server-side dynamic batching that groups concurrent single-session requests together
- Client-driven batch prediction endpoint that accepts up to 500 sessions per request
- Thread-safe, zero-downtime model hot-reload with artifact fingerprint verification
- Reproducible DVC stages for ingestion, preprocessing, training, evaluation, and drift detection
- MLflow tracking for parameters, metrics, and artifacts
- Secure model persistence using `skops` instead of unsafe pickle-based serialization
- DVC-managed data and artifact versioning with AWS S3 storage
- Automated evaluation metrics including accuracy, precision, recall, F1, ROC-AUC, and confusion matrix
- Drift reporting via Evidently and model monitoring summaries in MLflow
- Stress testing for concurrent traffic, validation failures, throughput spikes, and reload safety
- Rotating file logging and environment-driven configuration
- CI/CD workflow to reproduce the pipeline and publish artifacts

---

## Dataset

RetainStack uses the [Online Shoppers Purchasing Intention Dataset](https://archive.ics.uci.edu/ml/datasets/Online+Shoppers+Purchasing+Intention+Dataset) from the UCI Machine Learning Repository.

| Property | Details |
|---|---|
| Source | UCI Machine Learning Repository |
| Records | ~12,330 sessions |
| Task | Binary classification |
| Target | `Revenue` — whether the shopper completed a purchase |
| Domain | E-commerce / web analytics |

### Features used

| Feature | Type | Description |
|---|---|---|
| `Administrative` | Integer | Number of administrative pages visited |
| `Administrative_Duration` | Float | Total time spent on administrative pages |
| `Informational_Duration` | Float | Total time spent on informational pages |
| `ProductRelated` | Integer | Number of product-related pages visited |
| `ProductRelated_Duration` | Float | Time spent on product-related pages |
| `BounceRates` | Float | Average bounce rate |
| `ExitRates` | Float | Average exit rate |
| `PageValues` | Float | Average page value before conversion |
| `Month` | Categorical | Month of the session |
| `Revenue` | Boolean | Target label indicating a purchase |

---

## Repository structure

```text
RetainStack/
├── .github/
│   └── workflows/            # CI/CD automation
├── data/
│   ├── raw_data.csv          # Source dataset (DVC-tracked)
│   ├── train_data.csv        # Training split (DVC-tracked)
│   ├── test_data.csv         # Test split (DVC-tracked)
│   ├── processed/            # Preprocessed feature/label outputs
│   └── artifact/
│       ├── preprocessor.skops
│       ├── model.skops
│       ├── evaluation_metrics.json
│       ├── data_drift_report.html
│       └── data_drift_report.json
├── Experiments/
│   ├── data_exploration.ipynb
│   └── model_training.ipynb
├── src/
│   ├── api/
│   │   ├── app.py            # FastAPI app factory and lifespan
│   │   ├── batcher.py        # Dynamic batching logic
│   │   ├── dependencies.py   # Artifact store and hot-reload implementation
│   │   ├── routes/
│   │   │   ├── admin.py      # Model reload endpoint
│   │   │   ├── health.py     # Health and readiness probes
│   │   │   └── predict.py    # Prediction endpoints
│   │   └── schemas/
│   │       ├── request.py    # Request validation
│   │       └── response.py   # Response models
│   ├── logger/
│   │   └── logger.py         # Rotating logger
│   ├── config.py            # Environment-configurable settings
│   ├── data_ingestion.py    # Data loading and split generation
│   ├── data_preprocessing.py
│   ├── data_drift.py        # Drift detection and logging
│   ├── train.py             # Model training
│   └── evaluate.py          # Evaluation and metric export
├── main.py                  # API server entry point
├── stress_test.py           # Concurrent endpoint validation and load test
├── dvc.yaml                 # DVC pipeline configuration
├── dvc.lock                 # Locked dependency graph for the pipeline
├── params.yaml              # Hyperparameters and feature configuration
├── pyproject.toml           # Dependencies and package metadata
├── README.md
├── setup.py
├── mlflow.db
├── logs/
└── .venv/
```

---

## Quick start

### Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv)
- AWS credentials with access to the DVC S3 remote

### 1) Clone and install

```bash
git clone https://github.com/raogaurav17/RetainStack.git
cd RetainStack
uv sync
```

### 2) Activate the environment

```bash
source .venv/bin/activate
# Windows: .venv\Scripts\activate
```

### 3) Pull tracked data and artifacts

```bash
uv run dvc pull
```

### 4) Run the pipeline

#### Recommended: run the DVC pipeline

```bash
uv run dvc repro
```

To force a full rerun:

```bash
uv run dvc repro --force
```

#### Alternative: run individual stages

```bash
python -m src.data_ingestion
python -m src.data_preprocessing
python -m src.train
python -m src.evaluate
python -m src.data_drift
```

---

## Running the API

Start the service:

```bash
uv run python main.py
```

Or run via Uvicorn directly:

```bash
uv run uvicorn src.api.app:app --host 127.0.0.1 --port 8000
```

Open the generated docs at:

- [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

### Core API endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/v1/health` | Liveness probe |
| `GET` | `/api/v1/ready` | Readiness check confirming the model and preprocessor are loaded |
| `POST` | `/api/v1/predict` | Score a single session |
| `POST` | `/api/v1/predict/batch` | Score 1–500 sessions in one request |
| `POST` | `/api/v1/model/reload` | Swap in a new model/preprocessor pair without dropping in-flight requests |

### Example: single-session prediction

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

Sample response:

```json
{
  "prediction": 1,
  "purchase_probability": 0.5546,
  "confidence": "medium"
}
```

### Example: batch prediction

```bash
curl -s -X POST http://127.0.0.1:8000/api/v1/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "sessions": [
      {
        "Administrative": 0,
        "Administrative_Duration": 0.0,
        "Informational_Duration": 0.0,
        "ProductRelated": 53,
        "ProductRelated_Duration": 1482.5,
        "BounceRates": 0.02,
        "ExitRates": 0.05,
        "PageValues": 8.15,
        "Month": "Nov"
      },
      {
        "Administrative": 1,
        "Administrative_Duration": 30.0,
        "Informational_Duration": 5.0,
        "ProductRelated": 10,
        "ProductRelated_Duration": 300.0,
        "BounceRates": 0.10,
        "ExitRates": 0.15,
        "PageValues": 0.0,
        "Month": "Feb"
      }
    ]
  }'
```

Sample response:

```json
{
  "predictions": [
    {"prediction": 1, "purchase_probability": 0.5546, "confidence": "medium"},
    {"prediction": 0, "purchase_probability": 0.1823, "confidence": "low"}
  ],
  "total": 2
}
```

### Example: hot-reload endpoint

```bash
curl -s -X POST http://127.0.0.1:8000/api/v1/model/reload | python3 -m json.tool
```

Sample response:

```json
{
  "status": "ok",
  "message": "Model hot-reload successful. Active artifact: 20fc576ceb04 (loaded at 2026-08-18T15:44:40.999861+00:00).",
  "previous_version": "20fc576ceb04",
  "new_version": "20fc576ceb04",
  "reload_count": 1,
  "preflight_latency_ms": 23.54
}
```

Check readiness and artifact version:

```bash
curl -s http://127.0.0.1:8000/api/v1/ready | python3 -m json.tool
```

---

## Stress testing

The repository includes `stress_test.py`, a self-contained async script that exercises the API under load and checks correctness under concurrent traffic.

### Install optional support for rich output

```bash
uv add rich
```

### Run the default test suite

```bash
uv run python stress_test.py
```

### Example variations

```bash
# Higher concurrency and throughput burst
uv run python stress_test.py --total 500 --concurrency 100 --burst-duration 30

# Repeatable run with fixed seed
uv run python stress_test.py --seed 42
```

### Stress test phases

| Phase | Endpoint | What it validates |
|---|---|---|
| 1 | `GET /api/v1/health` | Liveness under concurrent requests |
| 2 | `GET /api/v1/ready` | Readiness and service availability |
| 3 | `POST /api/v1/predict` | Dynamic batching with single-session inference |
| 4 | `POST /api/v1/predict/batch` | Batch-size sweep |
| 5 | `POST /api/v1/predict/batch` | Maximum-capacity batch calls |
| 6 | `POST /api/v1/predict` | Throughput burst testing |
| 7 | `POST /api/v1/predict` | Input validation and 422 handling |
| 8 | `POST /api/v1/predict` + `POST /api/v1/model/reload` | Zero-downtime hot-reload safety under concurrent traffic |

### CLI flags

| Flag | Default | Description |
|---|---|---|
| `--host` | `127.0.0.1` | API host |
| `--port` | `8000` | API port |
| `--concurrency` | `50` | Max concurrent in-flight requests |
| `--total` | `200` | Requests per endpoint phase |
| `--batch-sizes` | `1 10 50 100` | Batch sizes used in the sweep |
| `--burst-duration` | `10.0` | Duration of the throughput burst |
| `--timeout` | `30.0` | Per-request timeout |
| `--seed` | — | Reproducible random seed |
| `--reload-rounds` | `5` | Number of concurrent reload attempts during the safety phase |

The script exits with code `0` when the non-validation phases achieve at least a 95% success rate and returns `1` otherwise.

---

## Configuration

| File | Purpose |
|---|---|
| `params.yaml` | Feature list, categorical settings, and model hyperparameters |
| `src/config.py` | Paths, split ratios, and batching settings |
| `dvc.yaml` | DVC stage graph and artifact outputs |

### Key environment variables

| Variable | Default | Description |
|---|---|---|
| `DATA_DIR` | `data` | Root directory housing raw and processed datasets |
| `TRAIN_TEST_SPLIT_RATIO` | `0.2` | Fraction reserved for the test split |
| `TRAIN_VAL_SPLIT_RATIO` | `0.2` | Fraction reserved for validation during training |
| `LOG_LEVEL` | `DEBUG` | Logging verbosity |
| `LOG_DIR` | `logs` | Location for rotating log files |
| `BATCH_MAX_SIZE` | `32` | Queue flush threshold for batched inference |
| `BATCH_TIMEOUT_MS` | `50` | Max time to wait before flushing a batch |

---

## Metrics and experiment tracking

Evaluation metrics are written to `data/artifact/evaluation_metrics.json` after each pipeline run.

View DVC metrics:

```bash
uv run dvc metrics show
```

Compare metrics across commits or runs:

```bash
uv run dvc metrics diff
```

Tracked metrics include:

| Metric | Description |
|---|---|
| `accuracy` | Overall classification accuracy |
| `precision` | Positive-class precision |
| `recall` | Positive-class recall |
| `f1` | F1 score |
| `roc_auc` | ROC AUC |
| `confusion_matrix` | 2×2 confusion matrix |

### MLflow UI

To launch the MLflow dashboard locally:

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Then open:

- `http://127.0.0.1:5000`

This lets you explore the `RetainStack_Experiment`, compare runs, and inspect model parameters and metrics.

---

## Author

Gaurav
