from pathlib import Path
import json
import os
import pickle

import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from google import genai


ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

GEMINI_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.8-flash",
    "gemini-2.5-pro",
]

MODELS = {
    "Heart disease": "heart_disease",
    "Diabetes": "diabetes_dataset",
    "Lung disease": "lung_disease_data",
    "Infectious disease / symptoms": "health_dataset",
}

st.set_page_config(page_title="Medical AI Predictor", page_icon="🩺", layout="wide", initial_sidebar_state="expanded")


@st.cache_resource(show_spinner="Loading the Week 2 pickle model...")
def load_model(dataset_label: str):
    dataset_stem = MODELS[dataset_label]
    model_dir = ROOT / "artifacts" / "models"
    model_path = model_dir / f"{dataset_stem}.pkl"
    if not model_path.exists():
        raise FileNotFoundError(
            "No pickle model was found. Run the Week 2 notebook first so it creates "
            f"{model_path}"
        )
    with model_path.open("rb") as handle:
        bundle = pickle.load(handle)

    if not isinstance(bundle, dict) or "pipeline" not in bundle or "feature_schema" not in bundle:
        raise ValueError(
            "The pickle file does not contain the expected model bundle. "
            "Re-run the Week 2 notebook to generate the current .pkl format."
        )

    return {
        **bundle,
        "model_path": model_path,
        "scores": bundle.get("scores", {}),
    }


def make_input_form(model_info: dict):
    values = {}
    feature_schema = model_info["feature_schema"]
    columns = model_info["columns"]

    if model_info.get("dataset_name") == "health_dataset":
        st.subheader("Enter symptoms")
        st.caption("Select the symptoms that best match the user input. The health model maps them to a predicted disease.")
        symptom_options = sorted({
            option
            for column in columns
            for option in feature_schema.get(column, {}).get("options", [])
            if option
        })
        selected = st.multiselect(
            "Symptoms from the dataset",
            symptom_options,
            max_selections=len(columns),
            placeholder="Choose one or more symptoms",
        )
        typed = st.text_area(
            "Enter symptoms manually",
            placeholder="Example: fever, cough, fatigue",
            help="Separate multiple symptoms with commas. Known symptoms are used as model features; unknown text is still passed safely through the pipeline.",
        )
        typed_symptoms = [item.strip() for item in typed.split(",") if item.strip()]
        selected = list(dict.fromkeys(selected + typed_symptoms))[:len(columns)]
        for index, column in enumerate(columns):
            values[column] = selected[index] if index < len(selected) else np.nan
        return values

    st.subheader("Structured and text inputs")
    st.caption("Inputs and field types are loaded from the Week 2 pickle model bundle.")
    for index, column in enumerate(columns):
        info = feature_schema[column]
        label = str(column)
        if info["kind"] == "numeric":
            minimum = float(info["min"])
            maximum = float(info["max"])
            default = float(info["default"])
            if not np.isfinite(minimum):
                minimum = -1e6
            if not np.isfinite(maximum):
                maximum = 1e6
            if minimum == maximum:
                maximum = minimum + 1.0
            values[column] = st.number_input(
                label,
                min_value=minimum,
                max_value=maximum,
                value=min(max(default, minimum), maximum),
                key=f"field_{index}",
            )
        else:
            options = info.get("options", [""]) or [""]
            values[column] = st.selectbox(label, options, key=f"field_{index}")
    return values


def explain_prediction(prediction: dict, text_context: str, gemini_model: str) -> str:
    fallback = (
        f"Predicted disease: {prediction['prediction']}\n\n"
        f"Confidence: {prediction.get('confidence', 'not available')}\n\n"
        "This is a machine-learning output, not a diagnosis. Review the input and consult a qualified professional."
    )
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return fallback
    try:
        client = genai.Client(api_key=api_key)
        request = {"prediction": prediction, "user_context": text_context[:2000]}
        prompt = (
            "Give a short, plain-language description in no more than two sentences for this disease prediction. "
            "Use only the supplied metadata. Do not diagnose, invent facts, or recommend treatment. "
            "State that professional medical review is required.\n\n"
            + json.dumps(request, default=str)
        )
        response = client.models.generate_content(
            model=gemini_model,
            contents=prompt,
        )
        return response.text
    except Exception:
        return "Gemini explanation is currently unavailable. Check your Gemini API key and selected model, then try again."


st.title("🩺 Medical AI Predictor")
st.markdown("### AI-powered disease prediction")
st.write("Choose a dataset, enter the required information, and get a model prediction with a simple Gemini explanation.")

with st.sidebar:
    st.header("Model settings")
    selected_dataset = st.selectbox("Dataset", list(MODELS))
    st.markdown("**Run locally**")
    st.code("streamlit run app.py", language="bash")
    st.caption("Model, target, classes, feature names, and input settings are loaded from the selected pickle model.")
    st.divider()
    configured_model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    if configured_model not in GEMINI_MODELS:
        configured_model = "gemini-2.5-flash"
    gemini_choice = st.selectbox(
        "Gemini model",
        ["Default / configured", *GEMINI_MODELS, "Custom model ID"],
        index=0,
        help="Choose the Gemini model used for the short prediction description.",
    )
    if gemini_choice == "Custom model ID":
        gemini_model = st.text_input(
            "Custom Gemini model ID",
            value=configured_model,
            placeholder="Example: gemini-2.5-flash",
            help="Enter a model ID supported by your Gemini API account.",
        ).strip() or configured_model
    elif gemini_choice == "Default / configured":
        gemini_model = configured_model
    else:
        gemini_model = gemini_choice
    st.caption(f"Selected Gemini: {gemini_model}")
    st.caption(f"Gemini API: {'configured' if os.getenv('GEMINI_API_KEY') else 'not configured (using fallback)'}")

try:
    model_info = load_model(selected_dataset)
except Exception as exc:
    st.error(str(exc))
    st.stop()

col_a, col_b = st.columns([2, 1])
with col_b:
    st.metric("Features", f"{len(model_info['columns']):,}")
    st.metric("Model", str(model_info["model_name"]).replace("_", " ").title())
    st.caption(f"Loaded: {model_info['model_path'].name}")
    st.write("Validation balanced accuracy")
    for name, score in model_info["scores"].items():
        st.metric(name.replace("_", " ").title(), f"{score:.1%}")

with col_a:
    with st.form("prediction_form"):
        record = make_input_form(model_info)
        text_context = st.text_area(
            "Text context for the explanation layer (optional)",
            placeholder="Add non-identifying notes only. This text is not used as a model feature.",
        )
        submitted = st.form_submit_button("🔍 Predict disease", use_container_width=True)

if submitted:
    row = pd.DataFrame([record], columns=model_info["columns"])
    pipeline = model_info["pipeline"]
    label = pipeline.predict(row)[0]
    result = {
        "dataset": selected_dataset,
        "prediction": str(label),
        "model": model_info["model_name"],
        "target": model_info["target"],
        "model_file": model_info["model_path"].name,
    }
    if model_info.get("dataset_name") == "health_dataset":
        result["input_symptoms"] = [str(value) for value in record.values() if pd.notna(value) and str(value).strip()]
    if hasattr(pipeline, "predict_proba"):
        probabilities = pipeline.predict_proba(row)[0]
        result["confidence"] = round(float(np.max(probabilities)), 4)
        result["class_probabilities"] = {
            str(cls): round(float(prob), 4)
            for cls, prob in zip(pipeline.classes_, probabilities)
        }

    st.divider()
    st.markdown("## 🧪 Prediction result")
    result_col1, result_col2 = st.columns([2, 1])
    with result_col1:
        st.success(f"Predicted disease: {result['prediction']}")
    with result_col2:
        confidence = result.get("confidence")
        if confidence is not None:
            st.metric("Confidence", f"{confidence:.1%}")
        st.caption(f"Model: {str(result['model']).replace('_', ' ').title()}")

    if result.get("class_probabilities"):
        st.markdown("**Prediction probabilities**")
        probability_cols = st.columns(len(result["class_probabilities"]))
        for index, (class_name, probability) in enumerate(result["class_probabilities"].items()):
            with probability_cols[index]:
                st.metric(class_name, f"{probability:.1%}")

    with st.expander("Prediction details", expanded=False):
        st.write(f"**Dataset:** {result['dataset']}")
        st.write(f"**Target:** {result['target']}")
        st.write(f"**Model file:** {result['model_file']}")
        if result.get("input_symptoms"):
            st.write("**Input symptoms:** " + ", ".join(result["input_symptoms"]))

    st.markdown("### 🤖 Gemini explanation")
    st.info(explain_prediction(result, text_context, gemini_model))

st.divider()
st.caption("Educational prototype only. Do not use this output as a diagnosis or as a substitute for qualified professional review.")
