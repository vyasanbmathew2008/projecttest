# Medical AI Predictor — ML + GenAI

> An educational healthcare AI project that combines tabular machine-learning disease prediction with a Gemini-powered, patient-friendly explanation layer.

This project uses four medical CSV datasets. Each dataset is cleaned and preprocessed with a leakage-safe scikit-learn pipeline, trained using baseline ML models, evaluated, and exported as a pickle model. The Streamlit application loads these saved models and provides an easy-to-use prediction interface with optional Gemini explanations.

> **Important:** This is an educational prototype, not a medical device or diagnostic system. Predictions can be incorrect and must not be used as a diagnosis or substitute for qualified professional medical advice.

## Project features

- Four medical tabular datasets with a standardized target name: `Disease`
- Dataset profiling and preprocessing
- Missing-value and duplicate handling
- Numeric and categorical preprocessing
- Logistic regression and random forest model comparison
- Balanced-accuracy and other classification metrics during training
- Complete scikit-learn pipeline serialization with pickle
- Saved models stored in a single `models/` directory
- Streamlit prediction interface
- Prediction confidence and class-probability overview
- Gemini-powered patient-friendly explanation
- Practical guidance about what to do next
- Emergency warning signs for relevant predicted conditions
- Deterministic fallback explanation when Gemini is unavailable
- Gemini model selector with `gemini-3.5-flash-lite` as the default configured model

## Supported datasets

| Dataset | Model file | Standard target |
|---|---|---|
| Heart disease | `models/heart_disease.pkl` | `Disease` |
| Diabetes | `models/diabetes_dataset.pkl` | `Disease` |
| Lung disease | `models/lung_disease_data.pkl` | `Disease` |
| Infectious disease / symptoms | `models/health_dataset.pkl` | `Disease` |

The training code accepts the original target-column names used by the datasets and standardizes them internally to `Disease`.

For the heart-disease dataset, the original yes/no target values are also normalized to human-readable labels such as `Heart Disease` and `No Heart Disease`.

## Workflow

### Week 1 — Dataset & Initial Data Preprocessing

Notebook: [`notebooks/01_02_week1_week2_preprocessing_ml_modelling.ipynb`](notebooks/01_02_week1_week2_preprocessing_ml_modelling.ipynb)

This phase covers:

- Dataset dimensions and column inspection
- Column-name normalization
- Data-type and categorical-value inspection
- Missing-value analysis
- Duplicate-row analysis
- Numeric distributions and initial visualization
- Selection and standardization of the supervised target

### Week 2 — Complete Preprocessing & ML Modelling

The notebook and `train.py` build the complete modelling pipeline:

- Train/test splitting
- Numeric imputation and scaling
- Categorical/text imputation and encoding
- Logistic regression baseline
- Random forest baseline
- Model evaluation
- Selection of the better-performing baseline
- Serialization of the complete pipeline and metadata into a pickle bundle

Generated model files are stored directly in:

```text
models/
├── heart_disease.pkl
├── diabetes_dataset.pkl
├── lung_disease_data.pkl
└── health_dataset.pkl
```

### Week 3 — Streamlit Deployment + GenAI

Week 3 is implemented in [`app.py`](app.py).

The application:

1. Loads a saved pickle model from `models/`
2. Collects the required user information
3. Generates an AI-assisted disease prediction
4. Displays confidence and class probabilities when available
5. Sends the prediction and optional non-identifying context to Gemini
6. Displays a patient-friendly explanation and practical next steps
7. Shows relevant emergency warning signs

The application does **not** display internal model implementation details such as the algorithm name, pickle filename, dataset metadata, or backend configuration.

## Model pipeline

```text
Medical CSV Dataset
        │
        ▼
Data Profiling & Cleaning
        │
        ▼
Target Standardization → Disease
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
Saved Pickle Model
        │
        ▼
Streamlit Prediction
        │
        ▼
Gemini Patient-Friendly Explanation
```

## Repository structure

```text
.
├── app.py                       # Streamlit deployment application
├── train.py                     # Local preprocessing, training, and model export
├── models/                      # Generated pickle models
├── data/
│   └── raw/                     # Raw tabular CSV datasets
├── notebooks/
│   └── 01_02_week1_week2_preprocessing_ml_modelling.ipynb
├── docs/
│   ├── README_week1.md
│   ├── README_week2.md
│   └── README_week3.md
├── requirements.txt
├── .env.example
└── README.md
```

## Installation

```bash
git clone https://github.com/vyasanbmathew2008/projecttest.git
cd projecttest

python -m venv .venv
```

### Activate the virtual environment

Linux/macOS:

```bash
source .venv/bin/activate
```

Windows:

```powershell
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Train the models locally

Run:

```bash
python train.py
```

This processes all four datasets and creates:

```text
models/heart_disease.pkl
models/diabetes_dataset.pkl
models/lung_disease_data.pkl
models/health_dataset.pkl
```

To train only one dataset:

```bash
python train.py --dataset health_dataset
```

The available dataset names are:

```text
heart_disease
diabetes_dataset
lung_disease_data
health_dataset
```

You can also specify custom data and model directories:

```bash
python train.py --data-dir data/raw --model-dir models
```

## Run the Streamlit application

After generating the pickle models:

```bash
streamlit run app.py
```

The app allows you to:

- Select a health area
- Enter the required health information
- Enter symptoms for the infectious-disease/symptom model
- Add optional non-identifying context
- Receive an AI-assisted prediction
- View prediction confidence and class probabilities
- Select a Gemini model
- Get a plain-language explanation and next-step guidance

> Run the application with `streamlit run app.py`, not `python app.py`.

## Gemini GenAI integration

The app uses the Gemini API to generate the plain-language explanation after a prediction.

The default configured model is:

```text
gemini-3.5-flash-lite
```

The sidebar also provides a model selector and a custom model ID option.

Create a local `.env` file using [`.env.example`](.env.example):

```dotenv
GEMINI_API_KEY=your-key
GEMINI_MODEL=gemini-3.5-flash-lite
```

The `.env` file should remain local and must never be committed to Git.

The Gemini prompt is designed to:

- Explain the predicted condition in simple language
- Describe reasonable next steps
- Identify relevant emergency warning signs
- Avoid claiming that the user definitely has the condition
- Avoid prescribing medication or dosage
- Avoid inventing patient-specific information
- Avoid exposing internal ML implementation details

If a Gemini API key is not configured or the Gemini service is unavailable, the application uses a deterministic fallback explanation.

> Do not enter personal or sensitive medical information into the optional context field.

## Google Colab

The Week 1–2 notebook can be opened in Google Colab:

- [Open the notebook in Google Colab](https://colab.research.google.com/github/vyasanbmathew2008/projecttest/blob/main/notebooks/01_02_week1_week2_preprocessing_ml_modelling.ipynb)
- [Open the Colab guide](README_COLAB.md)

Run the notebook cells to perform the preprocessing, modelling, evaluation, and model export workflow.

## Technology stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Pickle
- Matplotlib
- Seaborn
- Jupyter / Google Colab
- Streamlit
- Google Gemini API

## Educational disclaimer

The datasets, models, metrics, predictions, and Gemini-generated explanations in this repository are for learning and experimentation.

They have not been clinically validated for diagnosis, treatment, safety, fairness, or healthcare deployment. An AI prediction may be wrong. Anyone with symptoms or health concerns should seek advice from a qualified healthcare professional.
