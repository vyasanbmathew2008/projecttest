# Separate image pipeline guide

Notebook: [`../notebooks/04_separate_image_pipeline_skin_xray.ipynb`](../notebooks/04_separate_image_pipeline_skin_xray.ipynb)

## Purpose

This notebook creates two independent baseline image classifiers: one for skin-disease images and one for chest X-rays. Images are resized to grayscale vectors and passed through a balanced logistic-regression pipeline. The image branch is deliberately separate from the tabular CSV models.

## Inputs

Place permitted, extracted datasets under `../data/images/` using one class folder per label:

```text
../data/images/skin_disease/<class_name>/image.jpg
../data/images/xray/<class_name>/image.jpg
```

The notebook also recognizes `skin-disease-images/` and `Lung X-Ray Image/` under `data/images/`. See the Kaggle links in the main README.

## Outputs

- `../artifacts/models/image_skin_disease_logistic_regression.joblib`
- `../artifacts/models/image_xray_logistic_regression.joblib`
- `../artifacts/image_pipeline_results.csv`

If a dataset is missing, the notebook reports it as skipped rather than mixing it with another dataset.

## Safety checklist

Review class definitions, duplicate or patient-level leakage, class imbalance, image quality, and performance by class. This baseline is for education and experimentation only; it is not a diagnostic model.
