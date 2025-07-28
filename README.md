# 📊 RetainStack

**retainStack** is an end-to-end MLOps pipeline for predicting and improving online customer retention. It leverages machine learning models, DVC for reproducibility, and CI/CD practices for automation and deployment.

---

## Under Development
 So all feautres are not implemented yet

## 🚀 Features

- 🔄 **Data Version Control (DVC)**: Track and version datasets and ML models.
- 🧹 **Data Preprocessing**: Clean, transform, and split raw data.
- 🧠 **ML Modeling**: Train and evaluate predictive models.
- 📦 **Model Registry**: Store and version trained models.
- 🧪 **Testing Suite**: Unit tests for pipeline components.
- ⚙️ **CI/CD Integration**: GitHub Actions for linting, testing, and model training.
- ☁️ **Cloud-ready**: Configurable for deployment on AWS, Azure, or GCP.

---

## 🗂️ Project Structure

```

retainStack/
│
├── data/                      # Raw and processed datasets
├── logger/
|   ├── logger.py              # Logger module for log monitoring
├── src/                       # Source code
│   ├── data\_ingestion.py     # Ingestion scripts
│   ├── data_preprocessing.py  # Cleaning and splitting logic
│   ├── training.py            # Model training logic
│   ├── evaluation.py          # Model evaluation
│   ├── utils/                 # Reusable helpers and logger
│  
├── dvc.yaml                   # DVC pipeline definition
├── params.yaml                # Hyperparameters & config
├── test.py                     # Unit tests
├── .github/workflows/         # CI/CD workflows
├── requirements.txt
├── README.md
└── .gitignore

````

---

## 📦 Setup Instructions

1. **Clone the repo**
   ```bash
   git clone https://github.com/raogauarav17/RetainStack.git
   cd RetainStack
    ```

2. **Create virtual environment**

   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Linux/Mac
   .venv\Scripts\activate      # Windows
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Set up DVC**

   ```bash
   dvc init
   dvc pull
   ```

---

## ⚙️ Running the Pipeline

To run the full DVC pipeline:

```bash
dvc repro
```

To run individual stages (e.g., data ingestion):

```bash
python src/data_ingestion.py
```

---

## 📁 Configuration

All configuration (paths, split ratios, model parameters) is defined in:

* `params.yaml` for hyperparameters
* `config.py` for directory structure
* `dvc.yaml` for pipeline stages

---


## 🧪 Testing

Run unit tests using:

```bash
pytest tests/
```

---

## 📊 Results

Model performance metrics and evaluation visualizations are saved in the `artifacts/` or `reports/` directory. DVC tracks metrics via `dvc metrics show`.

---

## 📌 Future Improvements

* Streamlit or FastAPI serving
* MLflow for model tracking
* Drift monitoring and alerting
* Full cloud deployment (SageMaker, Vertex AI)

---

## 👨‍💻 Author

**Gaurav**
---

