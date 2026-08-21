"""Cached loaders the dashboard uses to read pipeline outputs and models.

Every loader takes the source file's mtime as a plain (hashed) argument, so
Streamlit's cache invalidates automatically after a pipeline re-run, without
needing a manual cache-clear.
"""
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config  # noqa: E402


def _file_mtime(path: Path) -> float:
    return path.stat().st_mtime if path.exists() else -1.0


def available_datasets() -> list[str]:
    return sorted(p.stem.replace("_summary", "") for p in config.RESULTS_DIR.glob("*_summary.json"))


@st.cache_data(show_spinner=False)
def _load_json(path: str, mtime: float) -> dict:
    with open(path) as f:
        return json.load(f)


def summary(dataset: str) -> dict:
    path = config.RESULTS_DIR / f"{dataset}_summary.json"
    return _load_json(str(path), _file_mtime(path))


def metrics(dataset: str) -> dict:
    path = config.RESULTS_DIR / f"{dataset}_metrics.json"
    return _load_json(str(path), _file_mtime(path))


def shap_importance(dataset: str) -> dict:
    path = config.RESULTS_DIR / f"{dataset}_shap_importance.json"
    return _load_json(str(path), _file_mtime(path))


@st.cache_data(show_spinner="Loading results...")
def _load_results(path: str, mtime: float) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["timestamp"])


def load_results(dataset: str) -> pd.DataFrame:
    path = config.RESULTS_DIR / f"{dataset}_anomaly_results.csv"
    return _load_results(str(path), _file_mtime(path))


MODEL_KEYS = ["scaler", "random_forest", "xgboost", "isolation_forest"]


@st.cache_resource(show_spinner="Loading trained models...")
def _load_models(dataset: str, newest_mtime: float) -> dict:
    loaded = {}
    for key in MODEL_KEYS:
        path = config.MODELS_DIR / f"{dataset}_{key}.joblib"
        if path.exists():
            loaded[key] = joblib.load(path)

    ae_path = config.MODELS_DIR / f"{dataset}_autoencoder.keras"
    if ae_path.exists():
        from tensorflow.keras.models import load_model
        loaded["autoencoder"] = load_model(ae_path)

    return loaded


def models_for(dataset: str) -> dict:
    newest = max(
        (_file_mtime(config.MODELS_DIR / f"{dataset}_{k}.joblib") for k in MODEL_KEYS),
        default=-1.0,
    )
    return _load_models(dataset, newest)
