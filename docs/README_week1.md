# Week 1 notebook guide — Dataset and Initial Data Preprocessing

Notebook: [`../notebooks/01_week1_dataset_initial_preprocessing.ipynb`](../notebooks/01_week1_dataset_initial_preprocessing.ipynb)

## Purpose

This notebook establishes the data contract before modelling. It discovers the CSV files under `data/raw/`, normalizes column-name whitespace, reports dimensions, duplicates, missingness, data types, cardinality, and numeric distributions.

## Inputs and outputs

Inputs are `../data/raw/diabetes_dataset.csv`, `../data/raw/heart_disease.csv`, `../data/raw/infectious_disease.csv`, `../data/raw/lung_disease_data.csv`. Outputs are `../artifacts/week1_dataset_profile.csv`.

## Completion checklist

Select a single supervised target, document its meaning and class balance, decide how missing and duplicate rows will be handled.
