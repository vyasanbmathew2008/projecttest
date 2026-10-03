# Week 1 notebook guide — Dataset and Initial Data Preprocessing

Notebook: [`../notebooks/01_02_week1_week2_preprocessing_ml_modelling.ipynb`](../notebooks/01_02_week1_week2_preprocessing_ml_modelling.ipynb)

## Purpose

This notebook establishes the data contract before modelling. It discovers the CSV files under `data/raw/`, normalizes column-name whitespace, reports dimensions, duplicates, missingness, data types, cardinality, and numeric distributions.

## Inputs and outputs

Input is `../data/raw/heart_disease.csv`. Output is `../artifacts/week1_dataset_profile.csv`.

## Completion checklist

Select a single supervised target, document its meaning and class balance, decide how missing and duplicate rows will be handled.
