# Week 1 notebook guide — Dataset and Initial Data Preprocessing

Notebook: [`../notebooks/01_week1_dataset_initial_preprocessing.ipynb`](../notebooks/01_week1_dataset_initial_preprocessing.ipynb)

## Purpose

This notebook establishes the data contract before modelling. It discovers the CSV files in the project root, normalizes column-name whitespace, reports dimensions, duplicates, missingness, data types, cardinality, and numeric distributions, and inventories optional image files by folder label.

## Inputs and outputs

Inputs are `../diabetes_dataset.csv`, `../heart_disease.csv`, `../infectious_disease.csv`, `../lung_disease_data.csv`, plus optional image folders under `../data/images/`. Outputs are `../artifacts/week1_dataset_profile.csv` and, when images exist, `../artifacts/image_inventory.csv`.

## Completion checklist

Select a single supervised target, document its meaning and class balance, decide how missing and duplicate rows will be handled, and verify image folder labels for the planned task.
