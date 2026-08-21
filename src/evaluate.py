"""Evaluation metrics and SHAP explainability."""
import numpy as np
import pandas as pd
from sklearn.metrics import (confusion_matrix, f1_score, precision_score,
                              recall_score, roc_auc_score, roc_curve,
                              precision_recall_curve)


def compute_metrics(y_true, y_pred, y_proba=None) -> dict:
    metrics = {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    if y_proba is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
    return metrics


def confusion(y_true, y_pred):
    return confusion_matrix(y_true, y_pred)


def roc_points(y_true, y_proba):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    return fpr, tpr


def pr_points(y_true, y_proba):
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    return precision, recall


def shap_feature_importance(xgb_model, X: pd.DataFrame) -> pd.Series:
    """Compute mean |SHAP value| per feature.

    The original script called `explainer.shap_values(X)` and assumed a
    single 2D array back. Depending on the installed shap/xgboost versions
    this can instead return a list (one array per class) or a shap
    `Explanation` object -- both would break `pd.DataFrame(shap_values, ...)`
    downstream. This normalizes all three cases.
    """
    import shap

    explainer = shap.TreeExplainer(xgb_model)
    raw = explainer.shap_values(X)

    if isinstance(raw, list):
        values = raw[-1]  # positive-class contributions for binary classification
    elif hasattr(raw, "values"):
        values = raw.values
    else:
        values = raw

    values = np.asarray(values)
    if values.ndim == 3:
        values = values[..., -1]

    return pd.DataFrame(values, columns=X.columns).abs().mean().sort_values(ascending=False)
