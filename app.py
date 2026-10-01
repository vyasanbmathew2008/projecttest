from pathlib import Path
import json
import os
import re

import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from google import genai

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / '.env')
DATA_DIR = ROOT / 'data' / 'raw'
DATASETS = {
    "Heart disease": ("heart_disease.csv", "Heart Disease Status"),
}

st.set_page_config(page_title="Medical ML Prediction", page_icon="", layout="wide")


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result.columns = [re.sub(r"\s+", " ", str(col)).strip() for col in result.columns]
    return result


@st.cache_data(show_spinner=False)
def load_dataset(dataset_label: str):
    filename, target = DATASETS[dataset_label]
    path = DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset: {path}")
    try:
        df = pd.read_csv(path, on_bad_lines="warn")
    except TypeError:
        df = pd.read_csv(path, error_bad_lines=False)
    df = clean_columns(df).drop_duplicates()
    if target not in df.columns:
        raise ValueError(f"Target '{target}' is not present in {filename}")
    return df, target


@st.cache_resource(show_spinner="Training model from the repository dataset...")
def train_model(dataset_label: str):
    df, target = load_dataset(dataset_label)
    X = df.drop(columns=[target])
    y = df[target].astype(str).str.strip()
    if y.nunique() < 2:
        raise ValueError("The selected dataset needs at least two target classes.")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    numeric = X_train.select_dtypes(include=np.number).columns.tolist()
    categorical = [col for col in X_train.columns if col not in numeric]
    preprocessor = ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "categorical_text",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            ),
        ]
    )
    candidates = {
        "Logistic regression": LogisticRegression(
            max_iter=1500, class_weight="balanced", random_state=42
        ),
        "Random forest": RandomForestClassifier(
            n_estimators=250,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        ),
    }
    scores = {}
    fitted = {}
    for name, estimator in candidates.items():
        pipeline = Pipeline([("preprocessor", preprocessor), ("model", estimator)])
        pipeline.fit(X_train, y_train)
        scores[name] = balanced_accuracy_score(y_test, pipeline.predict(X_test))
        fitted[name] = pipeline
    best_name = max(scores, key=scores.get)
    return {
        "pipeline": fitted[best_name],
        "model_name": best_name,
        "scores": scores,
        "columns": X.columns.tolist(),
        "target": target,
        "classes": sorted(y.unique().tolist()),
    }


def make_input_form(df: pd.DataFrame, target: str):
    values = {}
    st.subheader("Structured and text inputs")
    st.caption("Categorical fields are handled as text features by the model. Missing values are imputed inside the pipeline.")
    columns = [col for col in df.columns if col != target]
    for index, column in enumerate(columns):
        series = df[column]
        label = str(column)
        if pd.api.types.is_numeric_dtype(series):
            default = float(series.median()) if series.notna().any() else 0.0
            minimum = float(series.min()) if series.notna().any() else -1e6
            maximum = float(series.max()) if series.notna().any() else 1e6
            if not np.isfinite(minimum): minimum = -1e6
            if not np.isfinite(maximum): maximum = 1e6
            if minimum == maximum: maximum = minimum + 1.0
            values[column] = st.number_input(
                label, min_value=minimum, max_value=maximum, value=min(max(default, minimum), maximum), key=f"field_{index}"
            )
        else:
            options = [str(value) for value in series.dropna().astype(str).unique()[:100]]
            options = options or [""]
            values[column] = st.selectbox(label, options, key=f"field_{index}")
    return values


def explain_prediction(prediction: dict, text_context: str) -> str:
    fallback = (
        f"Predicted class: {prediction['prediction']}\n\n"
        f"Confidence: {prediction.get('confidence', 'not available')}\n\n"
        "This is a machine-learning output, not a diagnosis. Review the input and consult a qualified professional."
    )
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return fallback
    try:
        client = genai.Client(api_key=api_key)
        request = {
            "prediction": prediction,
            "user_context": text_context[:2000],
        }
        prompt = (
            "Explain this tabular ML prediction cautiously using only the supplied metadata. "
            "Do not diagnose, invent facts, or recommend treatment. State uncertainty and require human review.\n\n"
            + json.dumps(request, default=str)
        )
        response = client.models.generate_content(
            model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
            contents=prompt,
        )
        return response.text
    except Exception as exc:
        return fallback + f"\n\nExplanation service unavailable: {type(exc).__name__}."


st.title("Medical Dataset ML Prediction")
st.write("A local Streamlit interface for the repository's trained tabular models. Text/categorical values are sent through the structured-data pipeline.")

with st.sidebar:
    st.header("Model settings")
    selected_dataset = st.selectbox("Dataset", list(DATASETS))
    st.markdown("**Run locally**")
    st.code("streamlit run app.py", language="bash")
    st.caption("The app trains from the CSV files automatically. No external model download is required.")
    st.caption(f"Gemini API: {'configured' if os.getenv('GEMINI_API_KEY') else 'not configured (using fallback)'}")

try:
    df, target = load_dataset(selected_dataset)
    model_info = train_model(selected_dataset)
except Exception as exc:
    st.error(str(exc))
    st.stop()

col_a, col_b = st.columns([2, 1])
with col_b:
    st.metric("Rows", f"{len(df):,}")
    st.metric("Features", f"{len(df.columns) - 1:,}")
    st.metric("Model", model_info["model_name"])
    st.write("Validation balanced accuracy")
    st.json({name: round(score, 4) for name, score in model_info["scores"].items()})

with col_a:
    with st.form("prediction_form"):
        record = make_input_form(df, target)
        text_context = st.text_area(
            "Text context for the explanation layer (optional)",
            placeholder="Add non-identifying notes only. This text is not used as a model feature.",
        )
        submitted = st.form_submit_button("Predict")

if submitted:
    row = pd.DataFrame([record], columns=model_info["columns"])
    pipeline = model_info["pipeline"]
    label = pipeline.predict(row)[0]
    result = {
        "dataset": selected_dataset,
        "prediction": str(label),
        "model": model_info["model_name"],
        "target": target,
    }
    if hasattr(pipeline, "predict_proba"):
        probabilities = pipeline.predict_proba(row)[0]
        result["confidence"] = round(float(np.max(probabilities)), 4)
        result["class_probabilities"] = {
            str(cls): round(float(prob), 4)
            for cls, prob in zip(pipeline.classes_, probabilities)
        }

    st.subheader("Prediction")
    st.success(f"Predicted class: {result['prediction']}")
    st.json(result)
    st.subheader("Explanation")
    st.write(explain_prediction(result, text_context))
    st.download_button(
        "Download result JSON",
        data=json.dumps(result, indent=2),
        file_name="prediction_result.json",
        mime="application/json",
    )

st.divider()
st.caption("Educational prototype only. Do not use this output as a diagnosis or as a substitute for qualified professional review.")
