# Multimodal ML + GenAI Dataset Project

This project is organized as a reproducible **three-week workflow** for the four tabular medical CSV files in the repository root plus optional image inputs. Images are not committed because of size, licensing, and Kaggle access requirements.

## Three-week plan

| Week | Notebook | Deliverable |
|---|---|---|
| 1 | [`notebooks/01_week1_dataset_initial_preprocessing.ipynb`](notebooks/01_week1_dataset_initial_preprocessing.ipynb) | Dataset inventory, schema/missingness/duplicate checks, initial plots, optional image inventory |
| 2 | [`notebooks/02_week2_complete_preprocessing_ml_modelling.ipynb`](notebooks/02_week2_complete_preprocessing_ml_modelling.ipynb) | Leakage-safe preprocessing, baseline models, model comparison, serialized pipeline |
| 3 | [`notebooks/03_week3_evaluation_deployment_genai.ipynb`](notebooks/03_week3_evaluation_deployment_genai.ipynb) | Test evaluation, error analysis, prediction contract, separate image branch, optional GenAI explanation |

Per-notebook guides are in [`docs/`](docs/).

## Dataset paths and image inputs

Run JupyterLab from this repository root. The notebooks expect CSVs at `./*.csv`. Optional images can be extracted to `data/images/<label>/image.jpg`; the notebooks also detect `images/`, `Lung X-Ray Image/`, and `skin-disease-images/`. Folder names become candidate image labels.

Kaggle sources from the original README:

- [Lung Disease X-Ray Dataset](https://www.kaggle.com/datasets/fatemehmehrparvar/lung-disease)
- [20 Skin Diseases Dataset](https://www.kaggle.com/datasets/haroonalam16/20-skin-diseases-dataset)
- [Skin Disease Dataset 2](https://www.kaggle.com/datasets/pacificrm/skindiseasedataset)

Download only data you are permitted to use. Do not commit credentials, personally identifiable information, or large raw image archives.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

Run the notebooks in order. Week 1 writes a profile to `artifacts/`; Week 2 writes `artifacts/models/` and model results; Week 3 consumes that model and writes evaluation metadata. Set `OPENAI_API_KEY` only if you want the optional GenAI explanation call.

## Modelling boundary

The CSVs describe different medical tasks and should not be blindly concatenated. Choose one dataset/target pair, document the choice, and treat categorical/text features and image pixels as separate pipeline branches. This is an educational prototype, not a medical device or diagnostic system.

## Dependencies

See [`requirements.txt`](requirements.txt). The notebooks were written for Python 3.10+ and use pandas, scikit-learn, matplotlib, seaborn, joblib, Pillow, and the optional OpenAI-compatible client.
