"""End-to-end anomaly detection pipeline for one dataset."""
import json
import time

import joblib
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from . import config, evaluate, models, visualize
from .data_loader import load_dataset
from .features import engineer_features

SHAP_SAMPLE_SIZE = 5000  # cap SHAP computation cost on large datasets


def run(dataset_name: str, tune: bool = False, train_autoencoder: bool = True) -> dict:
    t0 = time.time()
    print(f"\n{'=' * 60}\nDataset: {dataset_name}\n{'=' * 60}")

    print("Loading data...")
    df = load_dataset(dataset_name)
    print(f"  {len(df):,} rows")

    print("Engineering features...")
    df = engineer_features(df)
    features = config.FEATURES

    print("\nLabel statistics:")
    stats = df.groupby("label")[features].agg(["mean", "std", "count"])
    print(stats)

    print("\nMann-Whitney U test (attack vs. normal):")
    for col in features:
        stat, p = mannwhitneyu(df.loc[df["label"] == 1, col], df.loc[df["label"] == 0, col])
        print(f"  {col}: U={stat:.2f}, p={p:.2e}")

    visualize.feature_distributions(df, features, dataset_name)

    # --- Split ---
    df_train, df_test = train_test_split(
        df, test_size=0.3, random_state=config.RANDOM_STATE, stratify=df["label"]
    )
    df_full = pd.concat([df_train, df_test]).sort_index()

    X_train, y_train = df_train[features], df_train["label"]
    X_test, y_test = df_test[features], df_test["label"]
    X_full = df_full[features]

    # A single scaler fit only on the training split is used everywhere.
    # The original script fit a *separate* scaler on the full (train+test)
    # data for the unsupervised models -- meaning Isolation Forest and the
    # Autoencoder were trained and evaluated on the identical rows with no
    # held-out set, while Random Forest/XGBoost got a proper split. That
    # inconsistency made cross-model comparison unfair; all four models now
    # share one train-fit scaler and are evaluated the same way.
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_full_scaled = scaler.transform(X_full)

    attack_rate = float(y_train.mean())
    print(f"\nTraining attack rate: {attack_rate:.4f}")

    predictions = {}   # name -> full-data predictions (pd.Series indexed like df_full)
    probabilities = {}  # name -> full-data probability of attack (or None)

    # --- Isolation Forest (unsupervised, fit on training split only) ---
    print("\nTraining Isolation Forest...")
    contamination = min(max(attack_rate, 0.01), 0.5)
    iso_forest = models.train_isolation_forest(X_train_scaled, contamination)
    iso_scores_full = -iso_forest.score_samples(X_full_scaled)  # higher = more anomalous
    predictions["Isolation Forest"] = pd.Series(
        (iso_forest.predict(X_full_scaled) == -1).astype(int), index=df_full.index
    )
    probabilities["Isolation Forest"] = pd.Series(iso_scores_full, index=df_full.index)

    # --- Autoencoder (unsupervised, trained only on normal training rows) ---
    autoencoder = None
    if train_autoencoder:
        print("Training Autoencoder...")
        normal_train = X_train_scaled[y_train.values == 0]
        autoencoder = models.train_autoencoder(normal_train)
        recon_error_full = models.autoencoder_scores(autoencoder, X_full_scaled)
        recon_error_train_normal = models.autoencoder_scores(autoencoder, normal_train)
        threshold = np.percentile(recon_error_train_normal, 95)
        predictions["Autoencoder"] = pd.Series(
            (recon_error_full > threshold).astype(int), index=df_full.index
        )
        probabilities["Autoencoder"] = pd.Series(recon_error_full, index=df_full.index)
    else:
        print("Skipping Autoencoder (TensorFlow unavailable or disabled).")

    # --- Random Forest ---
    print("Training Random Forest...")
    rf_model = models.train_random_forest(X_train_scaled, y_train, tune=tune)
    predictions["Random Forest"] = pd.Series(rf_model.predict(X_full_scaled), index=df_full.index)
    probabilities["Random Forest"] = pd.Series(
        rf_model.predict_proba(X_full_scaled)[:, 1], index=df_full.index
    )

    # --- XGBoost ---
    print("Training XGBoost...")
    xgb_model = models.train_xgboost(X_train_scaled, y_train, tune=tune)
    predictions["XGBoost"] = pd.Series(xgb_model.predict(X_full_scaled), index=df_full.index)
    probabilities["XGBoost"] = pd.Series(
        xgb_model.predict_proba(X_full_scaled)[:, 1], index=df_full.index
    )

    df_full["split"] = "train"
    df_full.loc[df_test.index, "split"] = "test"

    for name, preds in predictions.items():
        key = name.lower().replace(" ", "_")
        df_full[f"anomaly_{key}"] = preds.reindex(df_full.index)
        if name in probabilities:
            df_full[f"score_{key}"] = probabilities[name].reindex(df_full.index)

    # --- SHAP ---
    print("\nComputing SHAP feature importance (XGBoost)...")
    shap_sample = pd.DataFrame(X_full_scaled, columns=features, index=df_full.index)
    if len(shap_sample) > SHAP_SAMPLE_SIZE:
        shap_sample = shap_sample.sample(SHAP_SAMPLE_SIZE, random_state=config.RANDOM_STATE)
    shap_importance = evaluate.shap_feature_importance(xgb_model, shap_sample)
    print(shap_importance)

    # --- Evaluate every model on the held-out test split ---
    print("\nTest-set metrics:")
    metrics = {}
    roc_results = {}
    pr_results = {}
    for name, preds in predictions.items():
        test_preds = preds.loc[df_test.index]
        m = evaluate.compute_metrics(y_test, test_preds)

        if name in probabilities:
            test_proba = probabilities[name].loc[df_test.index]
            m["roc_auc"] = float(roc_auc_score(y_test, test_proba))
            fpr, tpr = evaluate.roc_points(y_test, test_proba)
            roc_results[name] = (fpr, tpr, m["roc_auc"])
            precision, recall = evaluate.pr_points(y_test, test_proba)
            pr_results[name] = (precision, recall)

        metrics[name] = m
        print(f"  {name}: " + ", ".join(f"{k}={v:.4f}" for k, v in m.items() if v is not None))

        cm = evaluate.confusion(y_test, test_preds)
        visualize.confusion_matrix_plot(cm, name, dataset_name)

    visualize.pr_curve_plot(pr_results, dataset_name)
    visualize.roc_curve_plot(roc_results, dataset_name)
    visualize.time_series_plot(
        df_full, {f"anomaly_{n.lower().replace(' ', '_')}": c for n, c in
                  zip(predictions, ["orange", "purple", "blue", "cyan"])},
        dataset_name,
    )
    visualize.scatter_plot(df_full, dataset_name)

    # --- Persist artifacts ---
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, config.MODELS_DIR / f"{dataset_name}_scaler.joblib")
    joblib.dump(rf_model, config.MODELS_DIR / f"{dataset_name}_random_forest.joblib")
    joblib.dump(xgb_model, config.MODELS_DIR / f"{dataset_name}_xgboost.joblib")
    joblib.dump(iso_forest, config.MODELS_DIR / f"{dataset_name}_isolation_forest.joblib")
    if autoencoder is not None:
        autoencoder.save(config.MODELS_DIR / f"{dataset_name}_autoencoder.keras")

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df_full.to_csv(config.RESULTS_DIR / f"{dataset_name}_anomaly_results.csv", index=False)
    with open(config.RESULTS_DIR / f"{dataset_name}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(config.RESULTS_DIR / f"{dataset_name}_shap_importance.json", "w") as f:
        json.dump({k: float(v) for k, v in shap_importance.to_dict().items()}, f, indent=2)

    elapsed = time.time() - t0
    summary = {
        "dataset": dataset_name,
        "rows": int(len(df_full)),
        "features": features,
        "attack_rate_train": attack_rate,
        "attack_rate_full": float(df_full["label"].mean()),
        "train_rows": int(len(df_train)),
        "test_rows": int(len(df_test)),
        "isolation_forest_contamination": contamination,
        "autoencoder_trained": autoencoder is not None,
        "elapsed_sec": elapsed,
    }
    with open(config.RESULTS_DIR / f"{dataset_name}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nDone in {elapsed:.1f}s. Results saved to {config.RESULTS_DIR}")
    return {"metrics": metrics, "shap_importance": shap_importance.to_dict(),
            "summary": summary, "elapsed_sec": elapsed}
