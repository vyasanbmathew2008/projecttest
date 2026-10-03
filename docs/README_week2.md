# Week 2 — Complete Preprocessing and ML Modelling

Notebook: [`../notebooks/01_02_week1_week2_preprocessing_ml_modelling.ipynb`](../notebooks/01_02_week1_week2_preprocessing_ml_modelling.ipynb)

The notebook trains the four datasets independently, using these targets:

| Dataset | Target |
|---|---|
| `heart_disease.csv` | `Heart Disease Status` |
| `diabetes_dataset.csv` | `Target` |
| `lung_disease_data.csv` | `Recovered` |
| `health_dataset.csv` | `Disease` |

For each dataset, the notebook performs a stratified split, fits numeric and categorical preprocessing, compares logistic regression with a random forest, and exports a metadata-rich pickle bundle. Rows with missing target values are excluded before splitting. Malformed rows in the uploaded health CSV are skipped safely.

## Outputs

Local outputs are ignored by Git:

- `../artifacts/models/heart_disease.pkl`
- `../artifacts/models/diabetes_dataset.pkl`
- `../artifacts/models/lung_disease_data.pkl`
- `../artifacts/models/health_dataset.pkl`
- `../artifacts/week2_model_results.csv`

Each pickle bundle contains the trained pipeline, target, classes, feature names, input schema, model name, and validation scores. `app.py` loads these bundles and does not read the training CSV files.

## Completion checklist

- Confirm all four targets are appropriate.
- Review class balance and validation metrics.
- Confirm all four `.pkl` files exist in `artifacts/models/`.
- Run `streamlit run app.py` and test each disease option.
