# Week 3 notebook guide — Evaluation, Deployment, and GenAI

Notebook: [`../notebooks/03_week3_evaluation_deployment_genai.ipynb`](../notebooks/03_week3_evaluation_deployment_genai.ipynb)

## Purpose

This notebook loads the Week 2 artifact, evaluates it on an untouched stratified test split, reports balanced accuracy/macro-F1/ROC-AUC where applicable, plots a confusion matrix, inspects errors, and defines `predict_record()` as a deployment contract. `multimodal_predict()` accepts an optional image path without mixing image bytes into the tabular pipeline.

## GenAI layer

`generate_explanation()` uses an OpenAI-compatible endpoint only when `OPENAI_API_KEY` is available. It sends prediction metadata and field names rather than raw narrative by default, includes safety instructions, and falls back deterministically if the call fails. Keep sensitive data out of prompts unless your approved environment permits it.

## Inputs and outputs

The notebook expects a model under `../artifacts/models/` and writes `../artifacts/week3_evaluation.json`.

## Completion checklist

Review false positives and false negatives, test the prediction contract, document model/version and threshold choices, and complete privacy, security, bias, and human-review checks before deployment.
