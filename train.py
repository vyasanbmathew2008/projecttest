"""Train all disease models locally and export pickle bundles for app.py.

Run from the repository root:
    python train.py

Train one dataset only:
    python train.py --dataset health_dataset
"""

from __future__ import annotations

import argparse
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = ROOT / "data" / "raw"
DEFAULT_MODEL_DIR = ROOT / "models"
DEFAULT_PROCESSED_DIR = ROOT / "data" / "processed"
RANDOM_STATE = 42
EXCLUDED_FEATURE_COLUMNS = {"recovered"}
LUNG_EXCLUDED_FEATURE_COLUMNS = {"treatment type"}

DATASET_CONFIG = {
    "heart_disease": {
        "file": "heart_disease.csv",
        "target_aliases": ["Disease", "Heart Disease Status"],
    },
    "diabetes_dataset": {
        "file": "diabetes_dataset.csv",
        "target_aliases": ["Disease", "Target"],
    },
    "lung_disease_data": {
        "file": "lung_disease_data.csv",
        "target_aliases": ["Disease", "Disease Type"],
        "merge_files": ["lung_cancer.csv"],
    },
    "health_dataset": {
        "file": "health_dataset.csv",
        "target_aliases": ["Disease"],
    },
}

TARGET_COLUMN = "Disease"


def read_table(path: Path) -> pd.DataFrame:
    """Read an uploaded CSV while skipping malformed rows in health_dataset.csv."""
    return pd.read_csv(path, engine="python", on_bad_lines="skip")


def clean_table(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize headers and remove exact duplicate rows."""
    cleaned = df.copy()
    cleaned.columns = [re.sub(r"\s+", " ", str(column)).strip() for column in cleaned.columns]
    return cleaned.drop_duplicates().reset_index(drop=True)


def make_preprocessor(X_train: pd.DataFrame) -> tuple[ColumnTransformer, list[str]]:
    numeric = X_train.select_dtypes(include=np.number).columns.tolist()
    categorical = [column for column in X_train.columns if column not in numeric]
    preprocessor = ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            ),
        ]
    )
    return preprocessor, numeric


def build_feature_schema(X: pd.DataFrame, numeric_columns: list[str]) -> dict:
    schema = {}
    for column in X.columns:
        series = X[column]
        if column in numeric_columns:
            numeric_values = pd.to_numeric(series, errors="coerce").dropna()
            schema[column] = {
                "kind": "numeric",
                "default": float(numeric_values.median()) if len(numeric_values) else 0.0,
                "min": float(numeric_values.min()) if len(numeric_values) else -1e6,
                "max": float(numeric_values.max()) if len(numeric_values) else 1e6,
            }
        else:
            options = [str(value) for value in series.dropna().astype(str).unique()[:100]]
            schema[column] = {"kind": "categorical", "options": options or [""]}
    return schema



def prepare_lung_dataset(data_dir: Path, processed_dir: Path) -> pd.DataFrame:
    """Merge the original lung-disease data with the lung-cancer dataset before training."""
    base = clean_table(read_table(data_dir / "lung_disease_data.csv"))
    cancer_path = data_dir / "lung_cancer.csv"
    cancer = clean_table(read_table(cancer_path))

    if "Disease Type" in base.columns and "Disease" not in base.columns:
        base = base.rename(columns={"Disease Type": "Disease"})
    if "LUNG_CANCER" not in cancer.columns:
        raise ValueError(
            f"{cancer_path.name} must contain a LUNG_CANCER target column. "
            f"Columns found: {list(cancer.columns)}"
        )
    cancer = cancer.rename(columns={"LUNG_CANCER": "Disease"})
    cancer["Disease"] = (
        cancer["Disease"].astype("string").str.strip().str.lower()
        .map({"yes": "Lung Cancer", "no": "No Lung Cancer", "1": "Lung Cancer", "0": "No Lung Cancer"})
        .fillna(cancer["Disease"].astype("string").str.strip())
    )
    cancer = cancer.rename(columns={
        "GENDER": "Gender", "AGE": "Age", "SMOKING": "Smoking",
        "YELLOW_FINGERS": "Yellow Fingers", "CHRONIC DISEASE": "Chronic Disease",
        "FATIGUE ": "Fatigue", "ALLERGY ": "Allergy", "WHEEZING": "Wheezing",
        "ALCOHOL CONSUMING": "Alcohol Consuming", "COUGHING": "Coughing",
        "SHORTNESS OF BREATH": "Shortness of Breath",
        "SWALLOWING DIFFICULTY": "Swallowing Difficulty", "CHEST PAIN": "Chest Pain",
    })
    base = base.drop(columns=["Treatment Type", "Recovered"], errors="ignore")
    cancer = cancer.drop(columns=["Recovered"], errors="ignore")
    base["Dataset Source"] = "lung_disease_data"
    cancer["Dataset Source"] = "lung_cancer"
    merged = clean_table(pd.concat([base, cancer], ignore_index=True, sort=False))
    processed_dir.mkdir(parents=True, exist_ok=True)
    merged_path = processed_dir / "lung_merged_dataset.csv"
    merged.to_csv(merged_path, index=False)
    print(f"Merged lung dataset: {len(base)} + {len(cancer)} = {len(merged)} rows -> {merged_path}")
    return merged


def train_dataset(dataset_name: str, data_dir: Path, model_dir: Path) -> Path:
    config = DATASET_CONFIG[dataset_name]
    source_path = data_dir / config["file"]
    if not source_path.exists():
        raise FileNotFoundError(f"Dataset not found: {source_path}")

    print(f"\n===== {dataset_name} =====")
    df = clean_table(read_table(source_path))
    target = next((column for column in config["target_aliases"] if column in df.columns), None)
    if target is None:
        raise ValueError(
            f"No disease target found in {source_path.name}. "
            f"Expected one of: {config['target_aliases']}; columns: {list(df.columns)}"
        )

    if target != TARGET_COLUMN:
        df = df.rename(columns={target: TARGET_COLUMN})
    target = TARGET_COLUMN

    # Convert dataset-specific binary heart-disease labels into meaningful
    # disease classes while keeping the common target column name "Disease".
    if dataset_name == "heart_disease":
        df[TARGET_COLUMN] = (
            df[TARGET_COLUMN]
            .astype("string")
            .str.strip()
            .str.lower()
            .map({
                "yes": "Heart Disease",
                "no": "No Heart Disease",
                "1": "Heart Disease",
                "0": "No Heart Disease",
            })
            .fillna(df[TARGET_COLUMN].astype("string").str.strip())
        )

    # Exclude post-outcome columns from model inputs to prevent data leakage.
    # Column matching is case-insensitive, so Recovered / recovered / RECOVERED are all excluded.
    excluded_names = set(EXCLUDED_FEATURE_COLUMNS)
    if dataset_name == "lung_disease_data":
        excluded_names.update(LUNG_EXCLUDED_FEATURE_COLUMNS)
        excluded_names.add("dataset source")
    excluded_columns = [
        column for column in df.columns
        if str(column).strip().lower() in excluded_names
    ]
    if excluded_columns:
        print(f"Excluding leakage-prone feature columns: {excluded_columns}")
    X = df.drop(columns=[target, *excluded_columns], errors="ignore")
    y = df[target].astype("string").str.strip()
    valid_target = df[target].notna()
    X = X.loc[valid_target].reset_index(drop=True)
    y = y.loc[valid_target].reset_index(drop=True).astype(str)
    if X.shape[1] == 0:
        raise ValueError(f"{dataset_name} has no usable feature columns after exclusions")
    if y.nunique() < 2:
        raise ValueError(f"{dataset_name} must contain at least two target classes")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    preprocessor, numeric_columns = make_preprocessor(X_train)
    candidates = {
        "logistic_regression": LogisticRegression(
            max_iter=1500,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=50,
            max_depth=12,
            class_weight="balanced",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
    }

    scores = {}
    fitted = {}
    for model_name, estimator in candidates.items():
        pipeline = Pipeline(
            [("preprocessor", preprocessor), ("model", estimator)]
        )
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        score = balanced_accuracy_score(y_test, predictions)
        scores[model_name] = float(score)
        fitted[model_name] = pipeline
        print(f"{model_name}: balanced accuracy={score:.4f}")
        print(classification_report(y_test, predictions, zero_division=0))

    best_name = max(scores, key=scores.get)
    model_bundle = {
        "pipeline": fitted[best_name],
        "model_name": best_name,
        "dataset_name": dataset_name,
        "target": target,
        "columns": X.columns.tolist(),
        "classes": sorted(y.unique().tolist()),
        "feature_schema": build_feature_schema(X, numeric_columns),
        "scores": scores,
    }

    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"{dataset_name}.pkl"
    with model_path.open("wb") as handle:
        pickle.dump(model_bundle, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved pickle: {model_path}")
    return model_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train disease models and export pickle bundles")
    parser.add_argument(
        "--dataset",
        choices=["all", *DATASET_CONFIG.keys()],
        default="all",
        help="Train all datasets or one selected dataset (default: all)",
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected = DATASET_CONFIG.keys() if args.dataset == "all" else [args.dataset]
    generated = [train_dataset(name, args.data_dir, args.model_dir) for name in selected]
    print("\nGenerated pickle files:")
    for path in generated:
        print(f"- {path}")


if __name__ == "__main__":
    main()
