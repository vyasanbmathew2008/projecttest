# Medical Tabular ML + GenAI Project

> A three-week educational machine-learning workflow for dataset preparation, complete preprocessing, ML modelling, model evaluation, deployment, and GenAI implementation using tabular CSV datasets.

This project uses one retained medical CSV dataset: the heart-disease dataset. The Week 1–3 workflow is configured specifically for this dataset. Week 3 deployment is implemented in `app.py`, not as a separate notebook.

> **Important:** This is an educational prototype, not a medical device or diagnostic system. Predictions must not be used as a diagnosis or substitute for qualified professional review.

## Project features

- Dataset profiling and initial data preprocessing
- Missing-value, duplicate, schema, and data-type analysis
- Leakage-safe numeric and categorical preprocessing
- Logistic regression and random forest baseline models
- Balanced-accuracy, macro-F1, ROC-AUC, classification-report, and confusion-matrix evaluation
- Saved complete scikit-learn model pipelines with Joblib
- Streamlit deployment interface for tabular CSV models
- Prediction confidence and class-probability output
- Optional safety-focused GenAI explanation layer
- Deterministic fallback explanation when no GenAI API key is configured

## Three-week workflow

### Week 1 — Dataset & Initial Data Preprocessing

Notebook: [`notebooks/01_week1_dataset_initial_preprocessing.ipynb`](notebooks/01_week1_dataset_initial_preprocessing.ipynb)

This phase establishes the data contract for the CSV datasets:

- Dataset dimensions and column inventory
- Column-name normalization
- Data types and categorical-value inspection
- Missing-cell analysis
- Duplicate-row analysis
- Numeric distributions and initial plots
- Selection of one supervised target for modelling

Local output (not committed): `artifacts/week1_dataset_profile.csv`

### Week 2 — Complete Preprocessing & ML Modelling

Notebook: [`notebooks/02_week2_complete_preprocessing_ml_modelling.ipynb`](notebooks/02_week2_complete_preprocessing_ml_modelling.ipynb)

This phase builds the complete modelling pipeline:

- Stratified train/test split
- Numeric imputation and standardization
- Categorical/text imputation and one-hot encoding
- Logistic regression baseline
- Random forest baseline
- Balanced-accuracy comparison
- Serialization of the best complete pipeline

Local outputs (not committed):

- `artifacts/week2_model_results.csv`
- `artifacts/models/<dataset>_<model>.joblib`

### Week 3 — Model Evaluation and Deployment with GenAI Implementation

Week 3 deployment is implemented in [`app.py`](app.py), using the Joblib model exported by Week 2 and the Gemini API for optional explanations.

This phase follows the deployment-oriented structure of the supplied README:

- Load the Week 2 Joblib model pipeline
- Use the trained tabular model for prediction
- Return predicted class, confidence, and class probabilities
- Add an optional safety-focused Gemini explanation layer
- Fall back to a deterministic explanation when no API key is available
- Deploy the tabular model through Streamlit

The app reads the local Week 2 model artifact from `artifacts/models/` after Week 2 has been run.

## Model pipeline

```text
Tabular CSV Dataset
        │
        ▼
Initial Data Profiling
        │
        ▼
Train/Test Split
        │
        ▼
Numeric + Categorical Preprocessing
        │
        ▼
Logistic Regression / Random Forest
        │
        ▼
Model Evaluation
        │
        ▼
Joblib Pipeline Artifact
        │
        ▼
Streamlit Deployment
        │
        ▼
Prediction + Optional GenAI Explanation
```

## Repository structure

```text
.
├── app.py                       # Streamlit deployment application
├── data/
│   └── raw/                    # Tabular CSV datasets
├── notebooks/
│   ├── 01_week1_dataset_initial_preprocessing.ipynb
│   └── 02_week2_complete_preprocessing_ml_modelling.ipynb
├── docs/
│   ├── README_week1.md
│   └── README_week2.md
├── requirements.txt
└── README.md
```

## Datasets

The committed tabular CSV files are stored in `data/raw/`:

- `heart_disease.csv`

The current default Week 2 and Week 3 task is:

```text
Dataset: heart_disease.csv
Target:  Heart Disease Status
```

The notebooks and Streamlit application are configured for `heart_disease.csv` with target `Heart Disease Status`.

## Installation

```bash
git clone https://github.com/vyasanbmathew2008/projecttest.git
cd projecttest
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

## Run the notebooks

Start JupyterLab from the repository root:

```bash
jupyter lab
```

Run the notebooks in order:

1. Week 1 — profile the CSV datasets.
2. Week 2 — train and save the best model pipeline.
3. Week 3 — run `app.py` to deploy the saved model with optional Gemini explanations.

## Run the Streamlit deployment

```bash
streamlit run app.py
```

The application:

- Lets you select one of the supported CSV datasets
- Trains a tabular model from the selected dataset
- Displays validation balanced accuracy
- Accepts numeric and categorical inputs
- Shows a prediction and class probabilities
- Provides an optional explanation text field
- Uses the GenAI explanation layer only when `GEMINI_API_KEY` is configured
- Provides a deterministic safety-focused explanation otherwise
- Allows the prediction result to be downloaded as JSON

> Run the Streamlit file with `streamlit run app.py`, not `python app.py`.

## GenAI implementation

The Streamlit app uses the Gemini API only when `GEMINI_API_KEY` is available. The implementation imports the SDK with `from google import genai` and calls `genai.Client(...).models.generate_content(...)`. The explanation layer sends prediction metadata and limited user context to Gemini, and is instructed not to diagnose, invent facts, or recommend treatment.

Create a local `.env` file from [`.env.example`](.env.example). The application loads this file automatically. The `.env` file is ignored by Git and must never be committed.

Optional Gemini settings are:

```dotenv
GEMINI_API_KEY=your-key
GEMINI_MODEL=gemini-2.0-flash
```

Do not include personal or sensitive medical information in prompts. If the service is unavailable, the application uses a deterministic fallback explanation.

## Technology stack

- Python
- Pandas and NumPy
- Scikit-learn
- Joblib
- Matplotlib and Seaborn
- JupyterLab
- Streamlit
- Optional Gemini API for GenAI explanations

## Educational disclaimer

The datasets, models, metrics, and explanations in this repository are for learning and experimentation. They have not been validated for clinical use, safety, fairness, or deployment in a healthcare setting.
