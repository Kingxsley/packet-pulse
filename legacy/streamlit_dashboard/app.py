"""Packet Pulse dashboard: the network traffic anomaly detection engine.

Reads the artifacts produced by `run.py` (outputs/results, outputs/models)
and presents them interactively: metrics, live-adjustable detection
thresholds, ROC/PR curves, SHAP feature importance, and a live-scoring tab
that runs uploaded packet data through the persisted models.

Built and originally run locally with Streamlit (`streamlit run dashboard/app.py`);
it now deploys straight from GitHub to Railway on every push, with Postgres
persisting anything imported through Live Scoring.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import (confusion_matrix, f1_score, precision_score,
                              precision_recall_curve, recall_score, roc_curve)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import data_access as da  # noqa: E402
import db  # noqa: E402
import theme  # noqa: E402
from src import config  # noqa: E402
from src.features import engineer_features  # noqa: E402

DATASET_LABELS = {
    "dns": "DNS capture (306k rows)",
    "dos": "DoS capture, raw (112k rows)",
    "dos_clean": "DoS capture, cleaned (112k rows)",
}
MODEL_ORDER = ["Isolation Forest", "Autoencoder", "Random Forest", "XGBoost"]
MODEL_COLORS = {"Isolation Forest": "#ffb020", "Autoencoder": "#e5484d",
                 "Random Forest": "#3fd0c9", "XGBoost": "#5b8def"}

st.set_page_config(page_title="Packet Pulse", layout="wide", page_icon="📡")
theme.inject()


# --------------------------------------------------------------------------
# Sidebar: dataset selection, thresholds, retraining
# --------------------------------------------------------------------------
def sidebar():
    st.sidebar.title("🛰️ Anomaly Detection")

    datasets = da.available_datasets()
    if not datasets:
        st.sidebar.error("No results found. Run `python run.py --dataset dos_clean` first.")
        st.stop()

    dataset = st.sidebar.selectbox(
        "Dataset", datasets, format_func=lambda d: DATASET_LABELS.get(d, d)
    )

    st.sidebar.subheader("Detection thresholds")
    st.sidebar.caption("Adjust each model's operating point without retraining.")
    thresholds = {
        "Random Forest": st.sidebar.slider("Random Forest probability", 0.0, 1.0, 0.5, 0.01),
        "XGBoost": st.sidebar.slider("XGBoost probability", 0.0, 1.0, 0.5, 0.01),
        "Isolation Forest": st.sidebar.slider("Isolation Forest sensitivity (% flagged)", 1, 50, 20, 1),
        "Autoencoder": st.sidebar.slider("Autoencoder sensitivity (% flagged)", 1, 50, 5, 1),
    }

    with st.sidebar.expander("Retrain pipeline"):
        st.caption("Re-runs feature engineering + all 4 models on the raw data. "
                   "Takes ~1-3 minutes depending on dataset and options.")
        retrain_dataset = st.selectbox(
            "Dataset to retrain", list(config.DATASET_FILES),
            format_func=lambda d: DATASET_LABELS.get(d, d), key="retrain_ds",
        )
        tune = st.checkbox("Hyperparameter tuning (--tune, slower)", value=False)
        skip_ae = st.checkbox("Skip autoencoder (faster)", value=False)
        if st.button("Run pipeline now", type="primary"):
            from src.pipeline import run as run_pipeline
            with st.spinner(f"Training on '{retrain_dataset}'... this will take a while."):
                run_pipeline(retrain_dataset, tune=tune, train_autoencoder=not skip_ae)
            st.cache_data.clear()
            st.cache_resource.clear()
            st.success("Done. Refreshing...")
            st.rerun()

    return dataset, thresholds


# --------------------------------------------------------------------------
# Threshold application
# --------------------------------------------------------------------------
def apply_thresholds(df: pd.DataFrame, thresholds: dict) -> pd.DataFrame:
    """Recompute each model's binary flag at the user-chosen operating point."""
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


def pred_col(model: str) -> str:
    return f"pred_{model.lower().replace(' ', '_')}"


def score_col(model: str) -> str:
    return f"score_{model.lower().replace(' ', '_')}"


OUTCOME_LABELS = {
    (1, 1): "True Positive",
    (0, 1): "False Positive",
    (1, 0): "False Negative",
    (0, 0): "True Negative",
}
OUTCOME_STYLE = {
    "True Positive": "background-color: rgba(46, 160, 67, 0.35)",
    "False Positive": "background-color: rgba(219, 109, 40, 0.35)",
    "False Negative": "background-color: rgba(219, 40, 40, 0.35)",
    "True Negative": "",
}


def outcomes(y_true, y_pred) -> pd.Series:
    """Per-row TP/FP/FN/TN label. Attack vs. normal is `y_pred`; correctness
    against the ground truth is what distinguishes true/false positive/negative."""
    return pd.Series(
        [OUTCOME_LABELS[(int(t), int(p))] for t, p in zip(y_true, y_pred)],
        index=y_true.index,
    )


def outcome_counts(y_true, y_pred) -> dict:
    counts = outcomes(y_true, y_pred).value_counts()
    return {k: int(counts.get(k, 0)) for k in
            ["True Positive", "False Positive", "False Negative", "True Negative"]}


# --------------------------------------------------------------------------
# Tabs
# --------------------------------------------------------------------------
def tab_overview(dataset, df, thresholds):
    summ = da.summary(dataset)
    stored_metrics = da.metrics(dataset)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{summ['rows']:,}")
    c2.metric("Attack rate", f"{summ['attack_rate_full']:.2%}")
    c3.metric("Train / test rows", f"{summ['train_rows']:,} / {summ['test_rows']:,}")
    c4.metric("Training time", f"{summ['elapsed_sec']:.0f}s")

    st.caption(f"Features used: {', '.join(summ['features'])}")

    st.subheader("Stored test-set metrics (from the last training run)")
    metrics_df = pd.DataFrame(stored_metrics).T
    st.dataframe(metrics_df.style.format("{:.4f}"), use_container_width=True)

    fig = go.Figure()
    for metric_name in ["f1", "roc_auc"]:
        fig.add_bar(name=metric_name.upper(), x=metrics_df.index,
                     y=metrics_df[metric_name], text=metrics_df[metric_name].round(3))
    fig.update_layout(barmode="group", title="Model comparison", yaxis_range=[0, 1])
    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "Metrics above are from the last training run at the *default* threshold. "
        "Use the **Model Comparison** tab to see metrics recomputed live at your "
        "chosen thresholds from the sidebar."
    )


def _sample(df, n=15000):
    return df.sample(n, random_state=config.RANDOM_STATE) if len(df) > n else df


def tab_explore(dataset, df):
    st.caption(f"{len(df):,} rows loaded. Plots below sample up to 15,000 points for responsiveness.")
    features = config.FEATURES
    plot_df = _sample(df)

    st.subheader("Feature distributions: attack vs. normal")
    cols = st.columns(len(features))
    for c, feat in zip(cols, features):
        fig = px.histogram(plot_df, x=feat, color="label", barmode="overlay",
                            histnorm="probability density", nbins=50,
                            color_discrete_map={0: "#1f77b4", 1: "#d62728"})
        fig.update_layout(title=feat, showlegend=False, height=300, margin=dict(t=40, b=10))
        c.plotly_chart(fig, use_container_width=True)

    st.subheader("Traffic over time")
    model_choice = st.selectbox("Highlight predictions from", MODEL_ORDER, key="explore_model")
    anomaly_col = f"anomaly_{model_choice.lower().replace(' ', '_')}"
    fig = px.scatter(plot_df.sort_values("timestamp"), x="timestamp", y="inter_arrival_time",
                      color=plot_df[anomaly_col].map({0: "normal (predicted)", 1: "flagged (predicted)"}),
                      opacity=0.5, color_discrete_map={"normal (predicted)": "#c7c7c7",
                                                        "flagged (predicted)": MODEL_COLORS[model_choice]})
    true_attacks = plot_df[plot_df["label"] == 1]
    fig.add_scatter(x=true_attacks["timestamp"], y=true_attacks["inter_arrival_time"], mode="markers",
                     name="labeled attack", marker=dict(color="black", size=5, symbol="x"))
    fig.update_layout(height=450, legend_title="")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Request rate vs. inter-arrival time")
    fig = px.scatter(plot_df, x="inter_arrival_time", y="request_rate", color="label",
                      opacity=0.5, color_continuous_scale=["#1f77b4", "#d62728"])
    st.plotly_chart(fig, use_container_width=True)


def tab_model_comparison(dataset, df, thresholds):
    test_df = apply_thresholds(df[df["split"] == "test"], thresholds)

    st.subheader("Live metrics at current thresholds (test set)")
    rows = []
    for model in MODEL_ORDER:
        pc = pred_col(model)
        if pc not in test_df:
            continue
        y_true, y_pred = test_df["label"], test_df[pc]
        counts = outcome_counts(y_true, y_pred)
        rows.append({
            "model": model,
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "true positives": counts["True Positive"],
            "false positives": counts["False Positive"],
            "false negatives": counts["False Negative"],
        })
    metrics_table = pd.DataFrame(rows).set_index("model")
    st.dataframe(
        metrics_table.style.format({
            "precision": "{:.4f}", "recall": "{:.4f}", "f1": "{:.4f}",
            "true positives": "{:,.0f}", "false positives": "{:,.0f}", "false negatives": "{:,.0f}",
        }),
        use_container_width=True,
    )
    st.caption("A **false positive** is normal traffic wrongly flagged as an attack (an analyst "
               "chasing a ghost); a **false negative** is a real attack that slipped through unflagged.")

    st.subheader("Confusion matrices")
    cols = st.columns(4)
    for c, model in zip(cols, MODEL_ORDER):
        pc = pred_col(model)
        if pc not in test_df:
            c.info(f"{model}: not available")
            continue
        y_true, y_pred = test_df["label"], test_df[pc]
        cm = confusion_matrix(y_true, y_pred)
        fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                         labels=dict(x="Predicted", y="True"),
                         x=["Normal", "Attack"], y=["Normal", "Attack"])
        fig.update_layout(title=model, height=320, margin=dict(t=40, b=10), coloraxis_showscale=False)
        c.plotly_chart(fig, use_container_width=True)
        counts = outcome_counts(y_true, y_pred)
        c.caption(f"🔺 {counts['False Positive']:,} false positives · "
                  f"🔻 {counts['False Negative']:,} false negatives")

    st.subheader("ROC & Precision-Recall curves (test set)")
    roc_col, pr_col = st.columns(2)
    roc_fig, pr_fig = go.Figure(), go.Figure()
    for model in MODEL_ORDER:
        sc = score_col(model)
        if sc not in test_df:
            continue
        fpr, tpr, _ = roc_curve(test_df["label"], test_df[sc])
        roc_fig.add_scatter(x=fpr, y=tpr, mode="lines", name=model,
                             line=dict(color=MODEL_COLORS[model]))
        precision, recall, _ = precision_recall_curve(test_df["label"], test_df[sc])
        pr_fig.add_scatter(x=recall, y=precision, mode="lines", name=model,
                            line=dict(color=MODEL_COLORS[model]))
    roc_fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="gray"), name="Random")
    roc_fig.update_layout(title="ROC", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
    pr_fig.update_layout(title="Precision-Recall", xaxis_title="Recall", yaxis_title="Precision")
    roc_col.plotly_chart(roc_fig, use_container_width=True)
    pr_col.plotly_chart(pr_fig, use_container_width=True)


def tab_feature_importance(dataset):
    shap_vals = da.shap_importance(dataset)
    st.caption("Mean |SHAP value| per feature for the XGBoost model — how much each "
               "feature drives the model's attack/normal decision, on average.")
    series = pd.Series(shap_vals).sort_values(ascending=True)
    fig = px.bar(series, orientation="h", labels={"value": "mean |SHAP value|", "index": "feature"})
    fig.update_layout(showlegend=False, height=350)
    st.plotly_chart(fig, use_container_width=True)


LIVE_REQUIRED_COLS = ["source_ip", "dest_ip", "source_port", "dest_port",
                       "protocol", "packet_length", "timestamp", "inter_arrival_time"]


def tab_live_scoring(dataset, raw_df, thresholds):
    st.write("Upload raw packet data (CSV or JSON) with columns "
             f"`{', '.join(LIVE_REQUIRED_COLS)}` (+ optional `label`), "
             "and it will be scored using the models currently trained for "
             f"**{DATASET_LABELS.get(dataset, dataset)}**.")
    st.caption("Isolation Forest/Autoencoder sensitivity is a percentile of the "
               "uploaded batch's own scores, so it's most meaningful on batches "
               "of at least a few dozen packets rather than single rows.")

    if db.enabled():
        with st.expander("Import history (Postgres)", expanded=False):
            history = db.list_imports()
            if history.empty:
                st.caption("No imports yet. Score something below and it'll show up here.")
            else:
                st.dataframe(history, use_container_width=True, hide_index=True)
    else:
        st.caption("Import history is off (no DATABASE_URL set): imports won't persist across restarts.")

    demo = st.button("Try it with a random sample from this dataset")
    uploaded = st.file_uploader("Upload packets", type=["csv", "json"])

    input_df, source_label = None, None
    if demo:
        cols = LIVE_REQUIRED_COLS + (["label"] if "label" in raw_df else [])
        input_df = raw_df[cols].sample(min(300, len(raw_df)), random_state=None).reset_index(drop=True)
        source_label = "Random sample"
    elif uploaded is not None:
        try:
            input_df = (pd.read_json(uploaded) if uploaded.name.endswith(".json")
                        else pd.read_csv(uploaded))
            source_label = uploaded.name
        except Exception as e:
            st.error(f"Could not parse file: {e}")
            return

    if input_df is None:
        return

    missing = [c for c in LIVE_REQUIRED_COLS if c not in input_df.columns]
    if missing:
        st.error(f"Missing required columns: {missing}")
        return

    with st.spinner("Scoring..."):
        input_df["timestamp"] = pd.to_datetime(input_df["timestamp"])
        featured = engineer_features(input_df.sort_values("timestamp").reset_index(drop=True))
        artifacts = da.models_for(dataset)
        if "scaler" not in artifacts:
            st.error("No trained models found for this dataset. Train it from the sidebar first.")
            return

        X_scaled = artifacts["scaler"].transform(featured[config.FEATURES])
        scored = featured.copy()

        if "random_forest" in artifacts:
            proba = artifacts["random_forest"].predict_proba(X_scaled)[:, 1]
            scored["score_random_forest"] = proba
        if "xgboost" in artifacts:
            proba = artifacts["xgboost"].predict_proba(X_scaled)[:, 1]
            scored["score_xgboost"] = proba
        if "isolation_forest" in artifacts:
            scored["score_isolation_forest"] = -artifacts["isolation_forest"].score_samples(X_scaled)
        if "autoencoder" in artifacts:
            from src.models import autoencoder_scores
            scored["score_autoencoder"] = autoencoder_scores(artifacts["autoencoder"], X_scaled)

        scored = apply_thresholds(scored, thresholds)

    st.subheader(f"Scored {len(scored)} packets")
    flag_model = st.selectbox("Flag attacks using", MODEL_ORDER, key="live_flag_model")
    flag_pc = pred_col(flag_model)

    if flag_pc in scored:
        scored["Flagged"] = scored[flag_pc].map({1: "Attack", 0: "Normal"})
        n_flagged = int(scored[flag_pc].sum())
        if "label" in scored:
            scored["Outcome"] = outcomes(scored["label"], scored[flag_pc])
            counts = outcome_counts(scored["label"], scored[flag_pc])
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("True positives", counts["True Positive"], help="Real attacks correctly flagged")
            m2.metric("False positives", counts["False Positive"], help="Normal traffic wrongly flagged as an attack")
            m3.metric("False negatives", counts["False Negative"], help="Real attacks that slipped through unflagged")
            m4.metric("True negatives", counts["True Negative"], help="Normal traffic correctly left alone")
        else:
            st.metric(f"Flagged as attack by {flag_model}", f"{n_flagged} / {len(scored)}")

    flag_cols = [pred_col(m) for m in MODEL_ORDER if pred_col(m) in scored]
    display_cols = ["timestamp", "source_ip", "dest_ip", "protocol", "packet_length"] + \
        (["label"] if "label" in scored else []) + \
        (["Flagged", "Outcome"] if "Outcome" in scored else ["Flagged"] if "Flagged" in scored else []) + \
        flag_cols
    view = scored[display_cols]
    if "Outcome" in view:
        st.dataframe(
            view.style.apply(lambda col: col.map(OUTCOME_STYLE) if col.name == "Outcome" else [""] * len(col)),
            use_container_width=True, height=350,
        )
    else:
        st.dataframe(view, use_container_width=True, height=350)

    metrics_for_db = None
    if "label" in scored:
        rows = []
        for model in MODEL_ORDER:
            pc = pred_col(model)
            if pc in scored:
                rows.append({
                    "model": model,
                    "precision": precision_score(scored["label"], scored[pc], zero_division=0),
                    "recall": recall_score(scored["label"], scored[pc], zero_division=0),
                    "f1": f1_score(scored["label"], scored[pc], zero_division=0),
                })
        st.caption("Sample included ground-truth labels: quick accuracy check across all four models.")
        st.dataframe(pd.DataFrame(rows).set_index("model").style.format("{:.4f}"), use_container_width=True)
        metrics_for_db = {r["model"]: {k: v for k, v in r.items() if k != "model"} for r in rows}

    if db.enabled() and flag_pc in scored:
        import_id = db.save_import(dataset, source_label, flag_model, scored, flag_pc, metrics_for_db)
        st.caption(f"Saved to Postgres as import #{import_id}. Check the history panel above next time.")

    st.download_button("Download scored data as CSV", scored.to_csv(index=False),
                        file_name=f"{dataset}_scored.csv", mime="text/csv")


# --------------------------------------------------------------------------
def main():
    dataset, thresholds = sidebar()
    df = da.load_results(dataset)

    theme.header()
    st.markdown('<div class="pp-eyebrow">MBIS5015 Capstone · Network Anomaly Detection</div>',
                unsafe_allow_html=True)
    st.caption(f"Isolation Forest, Autoencoder, Random Forest & XGBoost, all scoring "
               f"**{DATASET_LABELS.get(dataset, dataset)}** side by side.")

    overview, explore, compare, importance, live = st.tabs(
        ["Overview", "Explore Data", "Model Comparison", "Feature Importance", "Live Scoring"]
    )
    with overview:
        tab_overview(dataset, df, thresholds)
    with explore:
        tab_explore(dataset, df)
    with compare:
        tab_model_comparison(dataset, df, thresholds)
    with importance:
        tab_feature_importance(dataset)
    with live:
        tab_live_scoring(dataset, df, thresholds)


if __name__ == "__main__":
    main()
