# Multimodal ML + GenAI Dataset Project

This project is organized as a reproducible **three-week workflow** for the four tabular medical CSV files in the repository root plus optional image inputs. Images are not committed because of size, licensing, and Kaggle access requirements.

## Three-week plan

| Week | Notebook | Deliverable |
|---|---|---|
| 1 | [`notebooks/01_week1_dataset_initial_preprocessing.ipynb`](notebooks/01_week1_dataset_initial_preprocessing.ipynb) | Dataset inventory, schema/missingness/duplicate checks, initial plots, optional image inventory |
| 2 | [`notebooks/02_week2_complete_preprocessing_ml_modelling.ipynb`](notebooks/02_week2_complete_preprocessing_ml_modelling.ipynb) | Leakage-safe preprocessing, baseline models, model comparison, serialized pipeline |
| 3 | [`notebooks/03_week3_evaluation_deployment_genai.ipynb`](notebooks/03_week3_evaluation_deployment_genai.ipynb) | Test evaluation, error analysis, prediction contract, separate image branch, optional explanation layer |

Per-notebook guides are in [`docs/`](docs/).

## Streamlit app: clone and run

The app is [`app.py`](app.py). It trains a model automatically from the CSV files, supports the four datasets, accepts structured numeric and categorical/text values, optionally accepts an image upload, and provides a downloadable JSON result.

```bash
git clone https://github.com/vyasanbmathew2008/projecttest.git
cd projecttest
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The browser interface opens locally, normally at `http://localhost:8501`. No model download is required. The app caches the dataset and selected model during the session. If `OPENAI_API_KEY` is configured, the optional explanation section can use the configured OpenAI-compatible endpoint; without it, the app provides a deterministic safety-focused explanation.

The current serialized repository model is not required for the app to start: the app retrains from the selected CSV, which makes a fresh clone reproducible. The uploaded image is displayed and tracked as a separate input; it is not silently mixed into the tabular model. Add and validate a dedicated image model before connecting image predictions.

## Dataset paths and image inputs

Run JupyterLab or Streamlit from this repository root. The notebooks expect CSVs at `./*.csv`. Optional images can be extracted to `data/images/<label>/image.jpg`; the notebooks also detect `images/`, `Lung X-Ray Image/`, and `skin-disease-images/`. Folder names become candidate image labels.

Kaggle sources from the original README:

- [Lung Disease X-Ray Dataset](https://www.kaggle.com/datasets/fatemehmehrparvar/lung-disease)
- [20 Skin Diseases Dataset](https://www.kaggle.com/datasets/haroonalam16/20-skin-diseases-dataset)
- [Skin Disease Dataset 2](https://www.kaggle.com/datasets/pacificrm/skindiseasedataset)

Download only data you are permitted to use. Do not commit credentials, personally identifiable information, or large raw image archives.

## Notebook quick start

```bash
jupyter lab
```

Run the notebooks in order. Week 1 writes a profile to `artifacts/`; Week 2 writes `artifacts/models/` and model results; Week 3 consumes that model and writes evaluation metadata.

## Modelling boundary

The CSVs describe different medical tasks and should not be blindly concatenated. Choose one dataset/target pair, document the choice, and treat categorical/text features and image pixels as separate pipeline branches. This is an educational prototype, not a medical device or diagnostic system.

## Dependencies

See [`requirements.txt`](requirements.txt). The project uses pandas, scikit-learn, matplotlib, seaborn, joblib, Pillow, Streamlit, JupyterLab, and an optional OpenAI-compatible client.
