# RetainStack

RetainStack is an end-to-end MLOps pipeline for predicting and improving online customer retention. It leverages machine learning models, Data Version Control (DVC) for reproducibility, and Continuous Integration/Continuous Deployment (CI/CD) practices for automation and deployment.

**Note: This project is currently under development. Not all features are fully implemented.**

## Features

- **Data Version Control (DVC)**: Track and version datasets and machine learning models.
- **Data Preprocessing**: Clean, transform, and split raw data.
- **Machine Learning Modeling**: Train and evaluate predictive models.
- **Model Registry**: Store and version trained models.
- **Testing Suite**: Unit tests for pipeline components.
- **CI/CD Integration**: GitHub Actions for linting, testing, and model training.
- **Cloud-ready**: Configurable for deployment on AWS, Azure, or GCP.

## Dataset

RetainStack uses the **[Online Shoppers Purchasing Intention Dataset](https://archive.ics.uci.edu/ml/datasets/Online+Shoppers+Purchasing+Intention+Dataset)** from the UCI Machine Learning Repository.

| Property | Details |
|---|---|
| **Source** | UCI Machine Learning Repository |
| **Records** | ~12,330 web sessions |
| **Task** | Binary Classification |
| **Target Variable** | `Revenue` — whether the session ended in a purchase (`True`/`False`) |
| **Domain** | E-commerce / Web Analytics |

### Features

| Feature | Type | Description |
|---|---|---|
| `Administrative` | Integer | Number of administrative pages visited |
| `Administrative_Duration` | Float | Total time spent on administrative pages (seconds) |
| `Informational` | Integer | Number of informational pages visited |
| `Informational_Duration` | Float | Total time spent on informational pages (seconds) |
| `ProductRelated` | Integer | Number of product-related pages visited |
| `ProductRelated_Duration` | Float | Total time spent on product-related pages (seconds) |
| `BounceRates` | Float | Average bounce rate of pages visited |
| `ExitRates` | Float | Average exit rate of pages visited |
| `PageValues` | Float | Average page value of pages visited before a transaction |
| `SpecialDay` | Float | Closeness of the visit to a special day (e.g. Valentine's Day) |
| `Month` | Categorical | Month of the visit |
| `OperatingSystems` | Integer | Operating system used by the visitor |
| `Browser` | Integer | Browser used by the visitor |
| `Region` | Integer | Geographic region of the visitor |
| `TrafficType` | Integer | Traffic source type |
| `VisitorType` | Categorical | New visitor, returning visitor, or other |
| `Weekend` | Boolean | Whether the visit occurred on a weekend |
| `Revenue` | Boolean | **Target** — whether a purchase was completed |

## Project Structure

```text
RetainStack/
├── data/                      # Raw and processed datasets
├── Experments/                # Experiment tracking / notebooks
├── src/                       # Source code
│   ├── logger/
│   │   └── logger.py          # Logger module
│   ├── Config.py              # Configuration handling
│   ├── data_ingestion.py      # Ingestion scripts
│   ├── data_preprocessing.py  # Cleaning and splitting logic
│   ├── train.py               # Model training logic
│   └── evaluate.py            # Model evaluation
├── main.py                    # Main entry point
├── dvc.yaml                   # DVC pipeline definition
├── params.yaml                # Hyperparameters & config
├── pyproject.toml             # Python project metadata
├── setup.py                   # Setup script
├── requirements.txt           # Production dependencies
├── requirements-dev.txt       # Development dependencies
└── README.md                  # Project documentation
```

## Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/raogauarav17/RetainStack.git
   cd RetainStack
   ```

2. **Create a virtual environment**
   ```bash
   uv venv
   source .venv/bin/activate   # Linux/macOS
   # or for Windows:
   # .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   uv add -r requirements.txt
   ```

4. **Initialize and setup DVC**
   ```bash
   dvc init
   dvc pull
   ```

## Running the Pipeline

To execute the entire DVC pipeline:

```bash
dvc repro
```

To execute individual stages manually (e.g., data ingestion):

```bash
python src/data_ingestion.py
```

## Configuration

All configuration details (paths, split ratios, model parameters) are defined in the following files:

- `params.yaml`: Hyperparameters
- `config.py`: Directory structure definitions
- `dvc.yaml`: Pipeline stage configurations

## Testing

Run unit tests using `pytest`:

```bash
pytest test.py
```

## Results

Model performance metrics and evaluation visualizations are output to the `artifacts/` or `reports/` directory. DVC tracks metrics which can be displayed via:

```bash
dvc metrics show
```

## Future Improvements

- Implementation of Streamlit or FastAPI for model serving
- Integration with MLflow for comprehensive model tracking
- Addition of drift monitoring and alerting mechanisms
- Full cloud deployment integration (e.g., AWS SageMaker, GCP Vertex AI)

## Author

Gaurav
