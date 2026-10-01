# Medical ML + GenAI Dataset Project

A reproducible three-week educational workflow for tabular medical datasets, baseline machine-learning models, evaluation, and a Streamlit demo. The project keeps raw data, notebooks, application code, documentation, and generated artifacts in separate locations.

> **Disclaimer:** This is an educational prototype, not a medical device or diagnostic system. Do not use its outputs as a diagnosis or substitute for qualified professional review.

## Repository layout

```text
.
├── app/                  # Streamlit application
│   └── streamlit_app.py
├── data/
│   ├── raw/              # Source CSV datasets
│   └── images/           # Optional, ignored image inputs
├── notebooks/            # Ordered analysis and modelling workflow
├── docs/                 # Per-week notebook guides
├── artifacts/            # Generated profiles, metrics, and model files
├── requirements.txt
└── README.md
```

## Quick start

```bash
git clone https://github.com/vyasanbmathew2008/projecttest.git
cd projecttest
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

The app trains a model from the selected CSV in `data/raw/`, accepts structured inputs, keeps an optional uploaded image as a separate input, and provides a downloadable JSON result. If `OPENAI_API_KEY` is configured, the optional explanation section can use the configured OpenAI-compatible endpoint; otherwise it uses a deterministic fallback.

## Analysis workflow

Run JupyterLab from the repository root and execute the notebooks in order:

1. [Week 1 — dataset profiling](notebooks/01_week1_dataset_initial_preprocessing.ipynb) → profile artifacts
2. [Week 2 — preprocessing and modelling](notebooks/02_week2_complete_preprocessing_ml_modelling.ipynb) → model and comparison results
3. [Week 3 — evaluation and deployment contract](notebooks/03_week3_evaluation_deployment_genai.ipynb) → evaluation metadata

See the [notebook guides](docs/) for inputs, outputs, and completion checklists.

## Data and reproducibility

The committed CSVs are stored in `data/raw/`. Optional image datasets belong in `data/images/` and are intentionally ignored because of size, licensing, and access requirements. Do not commit credentials, personally identifiable information, or large raw image archives.

The disease CSVs represent different tasks and should not be blindly concatenated. Select one dataset/target pair, document the choice, and treat categorical/text features and image pixels as separate pipeline branches.

## Data sources

- [Lung Disease X-Ray Dataset](https://www.kaggle.com/datasets/fatemehmehrparvar/lung-disease)
- [20 Skin Diseases Dataset](https://www.kaggle.com/datasets/haroonalam16/20-skin-diseases-dataset)
- [Skin Disease Dataset 2](https://www.kaggle.com/datasets/pacificrm/skindiseasedataset)

Download only data you are permitted to use.
