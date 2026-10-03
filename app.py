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

    st.subheader("Your information")
    st.caption("Enter the information below to receive an AI-assisted health prediction.")
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
        f"{prediction['prediction']} is the condition the AI system identified from the information provided. "
        "This result is only an AI-based prediction, not a medical diagnosis, and should be reviewed by a qualified healthcare professional."
    )
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return fallback
    try:
        client = genai.Client(api_key=api_key)
        request = {"prediction": prediction, "user_context": text_context[:2000]}
        prompt = (
            "Write a natural, patient-friendly explanation for the predicted condition. "
            "Return only 2 short sentences. Start by briefly explaining what the predicted condition generally refers to. "
            "Then explain that the AI identified patterns associated with that condition from the information provided. "
            "Do NOT mention the algorithm, model type, dataset, pickle file, feature names, confidence score, probability, backend, or technical implementation. "
            "Do not diagnose, claim certainty, invent patient-specific facts, or recommend treatment. "
            "End by stating that the result is not a diagnosis and should be reviewed by a qualified healthcare professional.\n\n"
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
st.markdown("### Understand your health information with AI")
st.write(
    "Provide the requested information and receive an AI-assisted prediction with a clear, easy-to-understand explanation."
)
st.info(
    "For educational use only. This tool does not replace a medical examination, professional advice, or diagnosis."
)

with st.sidebar:
    st.header("Prediction settings")
    selected_dataset = st.selectbox("Health area", list(MODELS))
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
    st.caption("Used only to generate the plain-language explanation.")

try:
    model_info = load_model(selected_dataset)
except Exception as exc:
    st.error(str(exc))
    st.stop()

with st.form("prediction_form"):
    record = make_input_form(model_info)
    text_context = st.text_area(
        "Additional information (optional)",
        placeholder="Add any non-identifying information you would like the AI to consider for the explanation.",
    )
    submitted = st.form_submit_button("🔍 Check prediction", use_container_width=True)

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
    st.markdown("## 🧪 Your result")
    result_col1, result_col2 = st.columns([2.5, 1])
    with result_col1:
        st.success(f"### {result['prediction']}")
        st.caption("This is the condition identified by the AI from the information provided.")
    with result_col2:
        confidence = result.get("confidence")
        if confidence is not None:
            st.metric("Prediction confidence", f"{confidence:.1%}")

    if result.get("class_probabilities"):
        st.markdown("#### Prediction overview")
        probability_cols = st.columns(len(result["class_probabilities"]))
        for index, (class_name, probability) in enumerate(result["class_probabilities"].items()):
            with probability_cols[index]:
                st.metric(class_name, f"{probability:.1%}")

    st.markdown("### 🤖 What this result means")
    st.info(explain_prediction(result, text_context, gemini_model))
    st.caption(
        "Important: An AI prediction can be incorrect. If you have symptoms or health concerns, consult a qualified healthcare professional."
    )

st.divider()
st.caption("Educational prototype only. Do not use this output as a diagnosis or as a substitute for qualified professional review.")
