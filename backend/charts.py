"""Plotly figure builders. Every function returns a JSON-safe {"data": [...],
"layout": {...}} dict, rendered client-side with Plotly.newPlot/react so the
Model Comparison page can redraw charts on threshold changes without a full
page reload.
"""
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve

from .scoring import MODEL_COLORS

DARK = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Public Sans, sans-serif", color="#8a94a3", size=12),
    xaxis=dict(gridcolor="#232a35", zerolinecolor="#3a4452", linecolor="#3a4452"),
    yaxis=dict(gridcolor="#232a35", zerolinecolor="#3a4452", linecolor="#3a4452"),
    margin=dict(l=48, r=20, t=36, b=40),
    legend=dict(bgcolor="rgba(0,0,0,0)"),
)


def _payload(fig: go.Figure) -> dict:
    fig.update_layout(**DARK)
    return json.loads(fig.to_json())


def feature_distribution(df: pd.DataFrame, feature: str) -> dict:
    fig = go.Figure()
    for label, name, color in [(0, "Normal", "#3fd0c9"), (1, "Attack", "#e5484d")]:
        subset = df.loc[df["label"] == label, feature]
        fig.add_trace(go.Histogram(x=subset, name=name, opacity=0.6, histnorm="probability density",
                                    marker_color=color, nbinsx=50))
    fig.update_layout(barmode="overlay", title=feature, showlegend=False, height=260)
    return _payload(fig)


def time_series(df: pd.DataFrame, anomaly_cols: dict) -> dict:
    """anomaly_cols: {column_name: (display_name, color)}"""
    fig = go.Figure()
    fig.add_trace(go.Scattergl(x=df["timestamp"], y=df["inter_arrival_time"], mode="markers",
                                marker=dict(color="#3a4452", size=4), name="traffic", opacity=0.5))
    for col, (label, color) in anomaly_cols.items():
        subset = df[df[col] == 1]
        fig.add_trace(go.Scattergl(x=subset["timestamp"], y=subset["inter_arrival_time"], mode="markers",
                                    marker=dict(color=color, size=6), name=label))
    attacks = df[df["label"] == 1]
    fig.add_trace(go.Scattergl(x=attacks["timestamp"], y=attacks["inter_arrival_time"], mode="markers",
                                marker=dict(color="#edeff2", size=5, symbol="x"), name="labeled attack"))
    fig.update_layout(height=420, xaxis_title="time", yaxis_title="inter_arrival_time")
    return _payload(fig)


def scatter(df: pd.DataFrame) -> dict:
    fig = go.Figure()
    for label, name, color in [(0, "Normal", "#3fd0c9"), (1, "Attack", "#e5484d")]:
        subset = df[df["label"] == label]
        fig.add_trace(go.Scattergl(x=subset["inter_arrival_time"], y=subset["request_rate"], mode="markers",
                                    marker=dict(color=color, size=5, opacity=0.5), name=name))
    fig.update_layout(height=420, xaxis_title="inter_arrival_time", yaxis_title="request_rate")
    return _payload(fig)


def comparison_bar(metrics: dict) -> dict:
    models = list(metrics.keys())
    fig = go.Figure()
    fig.add_bar(name="F1", x=models, y=[metrics[m]["f1"] for m in models], marker_color="#3fd0c9")
    fig.add_bar(name="ROC-AUC", x=models, y=[metrics[m].get("roc_auc", 0) for m in models], marker_color="#ffb020")
    fig.update_layout(barmode="group", yaxis_range=[0, 1], height=340)
    return _payload(fig)


def shap_bar(shap_values: dict) -> dict:
    items = sorted(shap_values.items(), key=lambda kv: kv[1])
    fig = go.Figure(go.Bar(x=[v for _, v in items], y=[k for k, _ in items], orientation="h",
                            marker_color="#ffb020"))
    fig.update_layout(height=300, xaxis_title="mean |SHAP value|")
    return _payload(fig)


def confusion_fig(y_true, y_pred, title: str) -> dict:
    cm = confusion_matrix(y_true, y_pred)
    fig = go.Figure(go.Heatmap(
        z=cm, x=["Normal", "Attack"], y=["Normal", "Attack"],
        colorscale=[[0, "#12161d"], [1, "#3fd0c9"]], showscale=False,
        text=cm, texttemplate="%{text:,}", textfont=dict(color="#edeff2"),
    ))
    fig.update_layout(title=title, height=280, xaxis_title="Predicted", yaxis_title="True",
                       yaxis=dict(autorange="reversed"))
    return _payload(fig)


def roc_pr_figs(test_df: pd.DataFrame, models_with_scores: list[str]) -> tuple[dict, dict]:
    roc_fig, pr_fig = go.Figure(), go.Figure()
    for model in models_with_scores:
        sc = f"score_{model.lower().replace(' ', '_')}"
        if sc not in test_df:
            continue
        color = MODEL_COLORS[model]
        fpr, tpr, _ = roc_curve(test_df["label"], test_df[sc])
        roc_fig.add_scatter(x=fpr, y=tpr, mode="lines", name=model, line=dict(color=color))
        precision, recall, _ = precision_recall_curve(test_df["label"], test_df[sc])
        pr_fig.add_scatter(x=recall, y=precision, mode="lines", name=model, line=dict(color=color))
    roc_fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="#3a4452"), name="Random")
    roc_fig.update_layout(height=340, xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
    pr_fig.update_layout(height=340, xaxis_title="Recall", yaxis_title="Precision")
    return _payload(roc_fig), _payload(pr_fig)
