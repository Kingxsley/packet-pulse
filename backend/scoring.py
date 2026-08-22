"""Threshold application and TP/FP/FN/TN outcome labeling.

Framework-agnostic (pure pandas/numpy), shared by the Model Comparison and
Live Scoring routes.
"""
import numpy as np
import pandas as pd

# Autoencoder dropped from the app-facing model set: it was the weakest or
# tied-weakest model on both datasets, and TensorFlow was by far the
# heaviest dependency in the project (see README). src/models.py still has
# the training code, and `run.py --with-autoencoder` still trains it for
# local/CLI comparison, but the deployed app never loads or offers it.
MODEL_ORDER = ["Isolation Forest", "Random Forest", "XGBoost"]
MODEL_COLORS = {"Isolation Forest": "#ffb020", "Random Forest": "#3fd0c9", "XGBoost": "#5b8def"}

DEFAULT_THRESHOLDS = {"Random Forest": 0.5, "XGBoost": 0.5, "Isolation Forest": 20}

OUTCOME_LABELS = {
    (1, 1): "True Positive",
    (0, 1): "False Positive",
    (1, 0): "False Negative",
    (0, 0): "True Negative",
}


def pred_col(model: str) -> str:
    return f"pred_{model.lower().replace(' ', '_')}"


def score_col(model: str) -> str:
    return f"score_{model.lower().replace(' ', '_')}"


def apply_thresholds(df: pd.DataFrame, thresholds: dict) -> pd.DataFrame:
    """thresholds: {"Random Forest": 0..1, "XGBoost": 0..1,
    "Isolation Forest": 1..50 (% flagged), "Autoencoder": 1..50 (% flagged)}"""
    df = df.copy()
    if "score_random_forest" in df:
        df["pred_random_forest"] = (df["score_random_forest"] >= thresholds["Random Forest"]).astype(int)
    if "score_xgboost" in df:
        df["pred_xgboost"] = (df["score_xgboost"] >= thresholds["XGBoost"]).astype(int)
    if "score_isolation_forest" in df:
        cutoff = np.percentile(df["score_isolation_forest"], 100 - thresholds["Isolation Forest"])
        df["pred_isolation_forest"] = (df["score_isolation_forest"] >= cutoff).astype(int)
    if "score_autoencoder" in df:
        cutoff = np.percentile(df["score_autoencoder"], 100 - thresholds["Autoencoder"])
        df["pred_autoencoder"] = (df["score_autoencoder"] >= cutoff).astype(int)
    return df


def outcomes(y_true: pd.Series, y_pred: pd.Series) -> pd.Series:
    return pd.Series(
        [OUTCOME_LABELS[(int(t), int(p))] for t, p in zip(y_true, y_pred)],
        index=y_true.index,
    )


def outcome_counts(y_true: pd.Series, y_pred: pd.Series) -> dict:
    counts = outcomes(y_true, y_pred).value_counts()
    return {k: int(counts.get(k, 0)) for k in
            ["True Positive", "False Positive", "False Negative", "True Negative"]}


def severity_tier(score) -> str:
    """Calibrated for the 0-1 probability models (Random Forest, XGBoost).
    Isolation Forest/Autoencoder scores are unbounded anomaly scores, not
    probabilities, so tiers there are an approximation: anything above 1
    reads as Critical rather than being clamped or scaled against a
    per-batch distribution."""
    try:
        s = float(score)
    except (TypeError, ValueError):
        return "Medium"
    if s >= 0.9:
        return "Critical"
    if s >= 0.7:
        return "High"
    if s >= 0.4:
        return "Medium"
    return "Low"
