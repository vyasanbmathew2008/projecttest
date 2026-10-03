# Google Colab guide

This project can be run in Google Colab without manually uploading the CSV files. The notebooks download the datasets from the public GitHub raw-data directory when they are not already present in the Colab runtime.

## Open the notebooks

- [Week 1 — Dataset and Initial Data Preprocessing](https://colab.research.google.com/github/vyasanbmathew2008/projecttest/blob/main/notebooks/01_week1_dataset_initial_preprocessing.ipynb)
- [Week 2 — Complete Preprocessing and ML Modelling](https://colab.research.google.com/github/vyasanbmathew2008/projecttest/blob/main/notebooks/02_week2_complete_preprocessing_ml_modelling.ipynb)

## Run order

1. Open Week 1 and use **Runtime → Run all** to download and profile the datasets.
2. Open Week 2 and use **Runtime → Run all** to train the four disease models.
3. The notebook writes these local files in the Colab runtime:

```text
artifacts/models/heart_disease.pkl
artifacts/models/diabetes_dataset.pkl
artifacts/models/lung_disease_data.pkl
artifacts/models/health_dataset.pkl
```

4. Download the `.pkl` files from the Colab file browser if you want to add them manually to GitHub.
5. To run the Streamlit app locally, place the four `.pkl` files in `artifacts/models/` and run:

```bash
streamlit run app.py
```

## Dataset source

The notebooks download from:

```text
https://raw.githubusercontent.com/vyasanbmathew2008/projecttest/main/data/raw/
```

The datasets are kept independent and use these targets:

| Dataset | Target |
|---|---|
| `heart_disease.csv` | `Heart Disease Status` |
| `diabetes_dataset.csv` | `Target` |
| `lung_disease_data.csv` | `Recovered` |
| `health_dataset.csv` | `Disease` |

The generated pickle files contain the preprocessing pipeline, feature schema, classes, target metadata, and validation scores.
