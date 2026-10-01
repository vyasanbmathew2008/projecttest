# Medical Tabular ML + GenAI Project

> A three-week educational machine-learning workflow for dataset preparation, complete preprocessing, ML modelling, model evaluation, deployment, and GenAI implementation using tabular CSV datasets.

This project uses one retained medical CSV dataset: the heart-disease dataset. The Week 1–3 workflow is configured specifically for this dataset.

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

Output: `artifacts/week1_dataset_profile.csv`

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

Outputs:

- `artifacts/week2_model_results.csv`
- `artifacts/models/<dataset>_<model>.joblib`

### Week 3 — Model Evaluation and Deployment with GenAI Implementation

Notebook: [`notebooks/03_week3_evaluation_deployment_genai.ipynb`](notebooks/03_week3_evaluation_deployment_genai.ipynb)

This phase follows the deployment-oriented structure of the supplied README:

- Load the frozen Week 2 model
- Evaluate it on an untouched stratified test split
- Generate a classification report and confusion matrix
- Review false positives and false negatives
- Define a reusable `predict_record()` prediction contract
- Return predicted class, confidence, and class probabilities
- Save evaluation metadata
- Add an optional safety-focused GenAI explanation layer
- Fall back to a deterministic explanation when no API key is available
- Deploy the tabular model through Streamlit

Output: `artifacts/week3_evaluation.json`

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
├── app/
│   └── streamlit_app.py        # Streamlit deployment application
├── data/
│   └── raw/                    # Tabular CSV datasets
├── notebooks/
│   ├── 01_week1_dataset_initial_preprocessing.ipynb
│   ├── 02_week2_complete_preprocessing_ml_modelling.ipynb
│   └── 03_week3_evaluation_deployment_genai.ipynb
├── docs/
│   ├── README_week1.md
│   ├── README_week2.md
│   └── README_week3.md
├── artifacts/                  # Profiles, metrics, and model files
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
```

## Run the notebooks

Start JupyterLab from the repository root:

```bash
jupyter lab
```

Run the notebooks in order:

1. Week 1 — profile the CSV datasets.
2. Week 2 — train and save the best model pipeline.
3. Week 3 — evaluate the saved model and test the deployment contract.

## Run the Streamlit deployment

```bash
streamlit run app/streamlit_app.py
```

The application:

- Lets you select one of the supported CSV datasets
- Trains a tabular model from the selected dataset
- Displays validation balanced accuracy
- Accepts numeric and categorical inputs
- Shows a prediction and class probabilities
- Provides an optional explanation text field
- Uses the GenAI explanation layer only when `OPENAI_API_KEY` is configured
- Provides a deterministic safety-focused explanation otherwise
- Allows the prediction result to be downloaded as JSON

> Run the Streamlit file with `streamlit run app/streamlit_app.py`, not `python app/streamlit_app.py`.

## GenAI implementation

The Week 3 notebook and Streamlit app use an OpenAI-compatible client only when `OPENAI_API_KEY` is available. The explanation layer receives prediction metadata and limited user context, and is instructed not to diagnose, invent facts, or recommend treatment.

Optional environment variables are documented in [`.env.example`](.env.example):

```bash
export OPENAI_API_KEY="your-key"
export OPENAI_API_BASE="your-compatible-endpoint"
export OPENAI_MODEL="gpt-4o-mini"
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
- Optional OpenAI-compatible client for GenAI explanations

## Educational disclaimer

The datasets, models, metrics, and explanations in this repository are for learning and experimentation. They have not been validated for clinical use, safety, fairness, or deployment in a healthcare setting.
