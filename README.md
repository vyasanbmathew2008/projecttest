# Medical Tabular ML + GenAI Project

> A three-week educational machine-learning workflow for dataset preparation, complete preprocessing, ML modelling, model evaluation, deployment, and GenAI implementation using tabular CSV datasets.

This project uses four independent medical CSV datasets. Each dataset has its own target, preprocessing pipeline, and pickle model. The Week 3 deployment is implemented in `app.py`, not as a separate notebook.

> **Important:** This is an educational prototype, not a medical device or diagnostic system. Predictions must not be used as a diagnosis or substitute for qualified professional review.

## Project features

- Dataset profiling and initial data preprocessing for all four CSV datasets
- Missing-value, duplicate, schema, and data-type analysis
- Leakage-safe numeric and categorical preprocessing
- Logistic regression and random forest baseline models
- Balanced-accuracy, macro-F1, ROC-AUC, classification-report, and confusion-matrix evaluation
- Saved complete scikit-learn model pipelines with pickle
- Streamlit deployment interface for tabular CSV models
- Prediction confidence and class-probability output
- Optional safety-focused GenAI explanation layer
- Deterministic fallback explanation when no GenAI API key is configured

## Three-week workflow

### Week 1 — Dataset & Initial Data Preprocessing

Notebook: [`notebooks/01_02_week1_week2_preprocessing_ml_modelling.ipynb`](notebooks/01_02_week1_week2_preprocessing_ml_modelling.ipynb)

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

Notebook: [`notebooks/01_02_week1_week2_preprocessing_ml_modelling.ipynb`](notebooks/01_02_week1_week2_preprocessing_ml_modelling.ipynb)

This phase builds the complete modelling pipeline:

- Stratified train/test split
- Numeric imputation and standardization
- Categorical/text imputation and one-hot encoding
- Logistic regression baseline
- Random forest baseline
- Balanced-accuracy comparison
- Serialization of the best complete pipeline, feature schema, and model metadata into one pickle bundle

Local outputs (not committed):

- `artifacts/week2_model_results.csv`
- `artifacts/models/<dataset>.pkl` (one pickle bundle per dataset)

### Week 3 — Model Evaluation and Deployment with GenAI Implementation

Week 3 deployment is implemented in [`app.py`](app.py), using the pickle model exported by Week 2 and the Gemini API for optional explanations.

This phase follows the deployment-oriented structure of the supplied README:

- Load the Week 2 pickle model pipeline
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
pickle Pipeline Artifact
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
│   └── 01_02_week1_week2_preprocessing_ml_modelling.ipynb
├── docs/
│   ├── README_week1.md
│   ├── README_week2.md
│   └── README_week3.md
├── requirements.txt
└── README.md
```

## Datasets

The committed tabular CSV files are stored in `data/raw/`:

- `heart_disease.csv`

The four dataset targets are:

| Dataset | Target |
|---|---|
| `heart_disease.csv` | `Heart Disease Status` |
| `diabetes_dataset.csv` | `Target` |
| `lung_disease_data.csv` | `Recovered` |
| `health_dataset.csv` | `Disease` |

The Week 2 notebook generates one `.pkl` bundle per dataset. `app.py` lets you select a disease and loads the matching bundle. The generated bundles are ignored by Git so you can manually add them when desired.

## Installation

```bash
git clone https://github.com/vyasanbmathew2008/projecttest.git
cd projecttest
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

## Open notebooks in Google Colab

The notebooks automatically download the four CSV datasets from the public GitHub raw-data links when the files are not available locally.

- [Open Week 1 in Google Colab](https://colab.research.google.com/github/vyasanbmathew2008/projecttest/blob/main/notebooks/01_02_week1_week2_preprocessing_ml_modelling.ipynb)
- [Open Week 2 in Google Colab](https://colab.research.google.com/github/vyasanbmathew2008/projecttest/blob/main/notebooks/01_02_week1_week2_preprocessing_ml_modelling.ipynb)
- [Open the Colab guide](README_COLAB.md)

Run all cells in the merged notebook to create the four `.pkl` files under `artifacts/models/`.

## Run the notebooks

Start JupyterLab from the repository root:

```bash
jupyter lab
```

Run the notebooks in order:

1. Week 1 — profile the CSV datasets.
2. Week 2 — continue in the same merged notebook to train all four datasets and save one `.pkl` bundle per dataset.
3. Week 3 — run `app.py`, select a disease, and deploy its saved `.pkl` model with optional Gemini explanations.

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
- pickle
- Matplotlib and Seaborn
- JupyterLab
- Streamlit
- Optional Gemini API for GenAI explanations

## Educational disclaimer

The datasets, models, metrics, and explanations in this repository are for learning and experimentation. They have not been validated for clinical use, safety, fairness, or deployment in a healthcare setting.
