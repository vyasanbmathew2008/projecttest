# Week 2 notebook guide — Complete Preprocessing and ML Modelling

Notebook: [`../notebooks/02_week2_complete_preprocessing_ml_modelling.ipynb`](../notebooks/02_week2_complete_preprocessing_ml_modelling.ipynb)

## Purpose

The default task is `heart_disease.csv` with target `Heart Disease Status`. The notebook performs a stratified split before fitting transformations, imputes numeric and categorical values inside a scikit-learn pipeline, one-hot encodes text/categorical values, compares logistic regression with random forest, and serializes the best complete pipeline.

## Inputs and outputs

Change `DATASET_NAME` and `TARGET_COLUMN` near the top for another CSV. The notebook reads from `../` and writes `../artifacts/week2_model_results.csv` and `../artifacts/models/<dataset>_<model>.joblib`.

## Completion checklist

Confirm the target is appropriate, review class imbalance, retain preprocessing and estimator together, and do not merge disease datasets without a justified target definition. The optional image block only verifies image availability; an image model must be validated separately.
