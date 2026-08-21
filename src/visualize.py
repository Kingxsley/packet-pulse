"""Figure generation. All figures are saved to disk instead of the original
script's blocking `plt.show()` / `fig.show()` calls, which hang indefinitely
in a non-interactive/headless run (see README "Fixes" item 8)."""
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns

from . import config


def feature_distributions(df, features, dataset_name: str):
    fig, axes = plt.subplots(1, len(features), figsize=(5 * len(features), 4))
    for ax, col in zip(axes, features):
        sns.histplot(data=df, x=col, hue="label", stat="density", common_norm=False, bins=50, ax=ax)
        ax.set_title(f"{col} distribution")
    fig.tight_layout()
    path = config.FIGURES_DIR / f"{dataset_name}_feature_distributions.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def confusion_matrix_plot(cm, model_name: str, dataset_name: str):
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title(f"{model_name} Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.tight_layout()
    safe = model_name.lower().replace(" ", "_")
    path = config.FIGURES_DIR / f"{dataset_name}_confusion_{safe}.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def pr_curve_plot(pr_results: dict, dataset_name: str):
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, (precision, recall) in pr_results.items():
        ax.plot(recall, precision, label=name)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves (Test Set)")
    ax.legend()
    fig.tight_layout()
    path = config.FIGURES_DIR / f"{dataset_name}_precision_recall.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def roc_curve_plot(roc_results: dict, dataset_name: str):
    fig = go.Figure()
    for name, (fpr, tpr, auc) in roc_results.items():
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{name} (AUC={auc:.3f})"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash"), name="Random"))
    fig.update_layout(title="ROC Curves (Test Set)", xaxis_title="False Positive Rate",
                       yaxis_title="True Positive Rate")
    path = config.FIGURES_DIR / f"{dataset_name}_roc_curves.html"
    fig.write_html(path)
    return path


def time_series_plot(df, anomaly_cols: dict, dataset_name: str):
    fig = px.line(df, x=df.index, y="inter_arrival_time", title="Inter-Arrival Time with Anomalies")
    for col, color in anomaly_cols.items():
        subset = df[df[col] == 1]
        fig.add_scatter(x=subset.index, y=subset["inter_arrival_time"], mode="markers",
                         name=col.replace("anomaly_", "").title(),
                         marker=dict(color=color, size=6))
    subset = df[df["label"] == 1]
    fig.add_scatter(x=subset.index, y=subset["inter_arrival_time"], mode="markers",
                     name="Labeled Attacks", marker=dict(color="green", size=7, symbol="x"))
    fig.update_layout(xaxis_title="Time", yaxis_title="Inter-Arrival Time")
    path = config.FIGURES_DIR / f"{dataset_name}_timeseries.html"
    fig.write_html(path)
    return path


def scatter_plot(df, dataset_name: str):
    fig = px.scatter(df, x="inter_arrival_time", y="request_rate", color="anomaly_xgboost",
                      title="Request Rate vs. Inter-Arrival Time",
                      labels={"inter_arrival_time": "Inter-Arrival Time", "request_rate": "Request Rate"},
                      color_continuous_scale=["blue", "cyan"])
    subset = df[df["label"] == 1]
    fig.add_scatter(x=subset["inter_arrival_time"], y=subset["request_rate"], mode="markers",
                     name="Labeled Attacks", marker=dict(color="green", size=7, symbol="x"))
    path = config.FIGURES_DIR / f"{dataset_name}_scatter.html"
    fig.write_html(path)
    return path
