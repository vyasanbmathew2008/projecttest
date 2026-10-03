# Week 2 notebook guide — Complete Preprocessing and ML Modelling

Notebook: [`../notebooks/02_week2_complete_preprocessing_ml_modelling.ipynb`](../notebooks/02_week2_complete_preprocessing_ml_modelling.ipynb)

## Purpose

The default task is `heart_disease.csv` with target `Heart Disease Status`. The notebook performs a stratified split before fitting transformations, imputes numeric and categorical values inside a scikit-learn pipeline, one-hot encodes text/categorical values, compares logistic regression with random forest, and serializes the best complete pipeline.

## Inputs and outputs

The notebook is configured for `heart_disease.csv` and `Heart Disease Status`. The notebook reads the training CSV from `../data/raw/` and writes the local, ignored outputs `../artifacts/week2_model_results.csv` and `../artifacts/models/<dataset>_<model>.pkl`.

## Completion checklist

Confirm the target is appropriate, review class imbalance, retain preprocessing and estimator together, verify the pickle export, and do not merge unrelated disease targets.
