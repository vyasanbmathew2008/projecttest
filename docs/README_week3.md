# Week 3 — Model Evaluation and Deployment with GenAI Implementation

Notebook: [`../notebooks/03_week3_evaluation_deployment_genai.ipynb`](../notebooks/03_week3_evaluation_deployment_genai.ipynb)

## Purpose

Week 3 completes the machine-learning workflow by evaluating the frozen Week 2 model, defining a deployment contract, connecting the prediction flow to the Streamlit application, and implementing an optional safety-focused GenAI explanation layer.

## Week 3 features

- Load the serialized Week 2 model pipeline
- Evaluate the model on an untouched stratified test split
- Generate a classification report
- Calculate balanced accuracy and macro-F1
- Calculate ROC-AUC for eligible binary tasks
- Display a confusion matrix
- Inspect incorrect predictions
- Define `predict_record()` for deployment
- Return prediction confidence and class probabilities
- Save evaluation metadata as JSON
- Generate a cautious optional explanation with an OpenAI-compatible client
- Use a deterministic fallback when no API key is configured

## Evaluation pipeline

```text
Saved Week 2 Joblib Pipeline
            │
            ▼
Untouched Stratified Test Split
            │
            ▼
Predictions and Class Probabilities
            │
            ▼
Classification Report + Metrics
            │
            ▼
Error Analysis and Confusion Matrix
            │
            ▼
Deployment Prediction Contract
```

## Deployment contract

`predict_record(record)` accepts one dictionary-like tabular record and returns metadata containing:

- Dataset name
- Predicted class
- Model filename
- Prediction confidence when available
- Class probabilities when available

The contract is demonstrated in the notebook and is implemented in the Streamlit application at [`../app/streamlit_app.py`](../app/streamlit_app.py).

## GenAI implementation

`generate_explanation()` is optional and runs through an OpenAI-compatible endpoint only when `OPENAI_API_KEY` is set. The prompt is restricted to prediction metadata and field names. Its instructions are to:

- Explain cautiously
- State uncertainty
- Avoid diagnosis
- Avoid treatment recommendations
- Avoid inventing facts
- Require human review

When the key is missing or the service fails, the notebook returns a deterministic safety-focused explanation instead.

Keep sensitive medical information out of prompts unless your approved environment permits it.

## Inputs and outputs

Input model:

```text
../artifacts/models/<dataset>_<model>.joblib
```

Evaluation output:

```text
../artifacts/week3_evaluation.json
```

## Run the deployment

From the repository root:

```bash
streamlit run app/streamlit_app.py
```

The Streamlit application allows the user to select a supported CSV dataset, enter structured numeric and categorical values, view a prediction and probabilities, read the optional explanation, and download the result JSON.

## Completion checklist

- Review false positives and false negatives.
- Review balanced accuracy, macro-F1, ROC-AUC, and the confusion matrix where applicable.
- Test `predict_record()` with a representative row.
- Confirm the model and target metadata are correct.
- Review privacy, security, bias, and human-review requirements.
- Treat the application as an educational prototype, not a diagnostic tool.
