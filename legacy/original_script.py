#final Version.

# Install required packages
!pip install influxdb-client pandas scikit-learn xgboost plotly shap seaborn tensorflow numpy scipy

# Import libraries
from influxdb_client import InfluxDBClient
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve
from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from scipy.stats import mannwhitneyu
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import shap
import warnings
from influxdb_client.client.warnings import MissingPivotFunction

# Suppress warnings
warnings.simplefilter("ignore", MissingPivotFunction)

# --- Fetch Data ---
print("🔍 Fetching data...")
url = "https://us-east-1-1.aws.cloud2.influxdata.com"
token = "REDACTED-SEE-README-this-was-a-live-InfluxDB-token-committed-in-source-rotate-it"
org = "Anormally Detection"
bucket = "realtime_dns"

client = InfluxDBClient(url=url, token=token, org=org, timeout=30_000)

# Query data
print("📦 Pulling data (last 7 days)...")
query = f'''
from(bucket: "{bucket}")
  |> range(start: -7d)
  |> filter(fn: (r) => r._measurement == "dns")
  |> filter(fn: (r) => r._field == "dns_rate" or r._field == "inter_arrival_time" or r._field == "label")
  |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
  |> limit(n: 10000)
'''
try:
    tables = client.query_api().query_data_frame(query)
    df = pd.concat(tables) if isinstance(tables, list) else tables
    print(f"✅ Retrieved {len(df)} rows:")
    print(df.head())
except Exception as e:
    print(f"⚠️ Query failed: {e}. Using sample data...")
    data = {
        "_time": [
            "2025-06-04 00:52:54.452882+00:00",
            "2025-06-04 00:53:38.385039+00:00",
            "2025-06-04 00:54:20.425892+00:00",
            "2025-06-04 00:54:56.326488+00:00",
            "2025-06-04 00:54:56.331338+00:00"
        ],
        "dns_rate": [2, 1, 1, 1, 2],
        "inter_arrival_time": [0.019380, 43.932157, 42.040853, 35.900596, 0.004850],
        "label": [0, 0, 0, 0, 1]
    }
    df = pd.DataFrame(data)
    df["_time"] = pd.to_datetime(df["_time"])

client.close()

# --- Preprocess Data ---
print("\n🛠️ Preprocessing data...")
features = ["inter_arrival_time", "request_rate"]

# Check columns
required_cols = ["inter_arrival_time", "dns_rate", "label"]
missing_cols = [col for col in required_cols if col not in df.columns]
if missing_cols:
    raise ValueError(f"Missing columns: {missing_cols}. Available: {list(df.columns)}")

# Validate input data
print("\n🔍 Input Data Summary:")
print(df[required_cols].describe())
if df[required_cols].isna().any().any():
    print("⚠️ NaN values in input data. Filling with median...")
    for col in required_cols:
        df[col] = df[col].fillna(df[col].median())

# Clip inter_arrival_time
if df["inter_arrival_time"].le(0).any():
    print("⚠️ Replacing non-positive inter_arrival_time with 0.001...")
    df["inter_arrival_time"] = df["inter_arrival_time"].clip(lower=0.001)

# Feature engineering
df["request_rate"] = 1 / df["inter_arrival_time"]
df["request_rate"] = df["request_rate"].replace([np.inf, -np.inf], np.nan).fillna(df["request_rate"].median())

# Check derived features
print("\n🔍 Derived Features Summary:")
print(df[features].describe())
if df[features].isna().any().any():
    print("⚠️ NaN values in derived features. Filling with median...")
    for col in features:
        df[col] = df[col].fillna(df[col].median())

# Label statistics
print("\n🔍 Label Statistics:")
stats = df.groupby("label")[["inter_arrival_time", "request_rate"]].agg(["mean", "std", "count"])
print(stats)
print("\n🔍 Sample Attacks (20):")
print(df[df["label"] == 1][["inter_arrival_time", "request_rate", "label"]].sample(20, random_state=42))
print("\n🔍 Sample Normal (20):")
print(df[df["label"] == 0][["inter_arrival_time", "request_rate", "label"]].sample(20, random_state=42))

# Statistical test
print("\n🔍 Mann-Whitney U Test (Attack vs. Normal):")
for col in features:
    stat, p = mannwhitneyu(df[df["label"] == 1][col], df[df["label"] == 0][col])
    print(f"{col}: U={stat:.2f}, p={p:.2e}")

# Feature distribution plots
plt.figure(figsize=(12, 4))
for i, col in enumerate(features, 1):
    plt.subplot(1, len(features), i)
    sns.histplot(data=df, x=col, hue="label", stat="density", common_norm=False, bins=50)
    plt.title(f"{col} Distribution")
plt.tight_layout()
plt.show()

# --- Data for Unsupervised Models ---
X_unsupervised = df[features]
scaler_unsupervised = StandardScaler()
X_unsupervised_scaled = scaler_unsupervised.fit_transform(X_unsupervised)

# Check NaN/inf
if np.any(np.isnan(X_unsupervised_scaled)) or np.any(np.isinf(X_unsupervised_scaled)):
    print("⚠️ NaN/inf values in unsupervised scaled features:")
    print(pd.DataFrame(X_unsupervised_scaled, columns=features).describe())
    raise ValueError("NaN or inf values detected in unsupervised preprocessing.")

# --- Data for Supervised Models ---
df_train, df_test = train_test_split(df, test_size=0.3, random_state=42, stratify=df["label"])

# Normalize training data
X_train = df_train[features]
y_train = df_train["label"]
scaler_supervised = StandardScaler()
X_train_scaled = scaler_supervised.fit_transform(X_train)

# Normalize test data
X_test = df_test[features]
y_test = df_test["label"]
X_test_scaled = scaler_supervised.transform(X_test)

# Recombine for full data
df_full = pd.concat([df_train, df_test]).sort_index()
X_full = df_full[features]
X_full_scaled = scaler_supervised.transform(X_full)

# --- Check Attack Prevalence ---
attack_rate = df_full["label"].mean()
print(f"\n📈 Attack rate: {attack_rate:.4f} ({int(attack_rate * len(df_full))} attacks)")

# Class distribution
print("\n🔍 Training Class Distribution:")
print(pd.Series(y_train).value_counts(normalize=True))

# Feature correlation
print("\n🔍 Feature Correlation:")
print(df_full[features].corr())

# --- Unsupervised Models ---
print("\n🤖 Training unsupervised models...")
# Isolation Forest
iso_forest = IsolationForest(contamination=0.3, max_features=2, random_state=42)
iso_forest.fit(X_unsupervised_scaled)
df_full["anomaly_iso"] = (iso_forest.predict(X_unsupervised_scaled) == -1).astype(int)

# Autoencoder
normal_data = X_unsupervised_scaled[df_full["label"] == 0]
autoencoder = Sequential([
    Dense(16, activation="relu", input_shape=(len(features),)),
    Dropout(0.2),
    Dense(8, activation="relu"),
    Dropout(0.2),
    Dense(16, activation="relu"),
    Dense(len(features), activation="linear")
])
autoencoder.compile(optimizer="adam", loss="mse")
early_stopping = EarlyStopping(monitor="loss", patience=5, restore_best_weights=True)
autoencoder.fit(normal_data, normal_data, epochs=50, batch_size=32, callbacks=[early_stopping], verbose=0)
reconstruction = autoencoder.predict(X_unsupervised_scaled, verbose=0)
reconstruction_error = np.mean((X_unsupervised_scaled - reconstruction) ** 2, axis=1)
threshold = np.percentile(reconstruction_error[df_full["label"] == 0], 95)
df_full["anomaly_autoencoder"] = (reconstruction_error > threshold).astype(int)

# --- Supervised Models ---
print("\n🤖 Training supervised models...")
# Random Forest
rf_params = {
    "n_estimators": [100, 200],
    "max_depth": [10, 15],
    "min_samples_split": [5, 10],
    "min_samples_leaf": [8, 10],
    "class_weight": ["balanced"]
}
rf_model = GridSearchCV(RandomForestClassifier(random_state=42), rf_params, cv=5, scoring="f1")
rf_model.fit(X_train_scaled, y_train)
print(f"Best Random Forest params: {rf_model.best_params_}")
df_full["anomaly_rf"] = rf_model.predict(X_full_scaled)

# XGBoost
xgb_params = {
    "n_estimators": [100, 200],
    "max_depth": [3, 5],
    "min_child_weight": [15, 20],
    "learning_rate": [0.01, 0.05],
    "scale_pos_weight": [(len(y_train) - sum(y_train)) / sum(y_train)]
}
xgb_model = GridSearchCV(XGBClassifier(eval_metric="logloss", random_state=42), xgb_params, cv=5, scoring="f1")
xgb_model.fit(X_train_scaled, y_train)
print(f"Best XGBoost params: {xgb_model.best_params_}")
df_full["anomaly_xgb"] = xgb_model.predict(X_full_scaled)

# --- SHAP Analysis ---
print("\n🔎 SHAP Feature Importance...")
explainer = shap.TreeExplainer(xgb_model.best_estimator_)
shap_values = explainer.shap_values(X_full_scaled)
shap_summary = pd.DataFrame(shap_values, columns=features).abs().mean()
print(shap_summary.sort_values(ascending=False))

# --- Evaluate Models ---
print("\n📊 Comparing models...")
models = {
    "Isolation Forest": df_full["anomaly_iso"],
    "Autoencoder": df_full["anomaly_autoencoder"],
    "Random Forest": df_full["anomaly_rf"],
    "XGBoost": df_full["anomaly_xgb"]
}
metrics = {}
roc_results = {}
pr_results = {}
for name, preds in models.items():
    test_preds = preds[df_test.index] if name in ["Random Forest", "XGBoost"] else preds
    metrics[name] = {
        "Full Precision": precision_score(df_full["label"], preds, zero_division=0),
        "Full Recall": recall_score(df_full["label"], preds),
        "Full F1": f1_score(df_full["label"], preds),
        "Test F1": f1_score(y_test, test_preds) if name in ["Random Forest", "XGBoost"] else None
    }
    if name in ["Random Forest", "XGBoost"]:
        probs = (rf_model.predict_proba(X_full_scaled)[:, 1] if name == "Random Forest" else xgb_model.predict_proba(X_full_scaled)[:, 1])
        test_probs = probs[df_test.index]
        metrics[name]["Test ROC-AUC"] = roc_auc_score(y_test, test_probs)
        roc_results[name] = roc_auc_score(df_full["label"], probs)
        precision, recall, _ = precision_recall_curve(y_test, test_probs)
        pr_results[name] = {"precision": precision, "recall": recall}

for name, metric in metrics.items():
    print(f"{name} Metrics:", {k: round(v, 4) for k, v in metric.items() if v is not None})
for name, auc in roc_results.items():
    print(f"{name} Full ROC-AUC:", round(auc, 4))

# Precision-Recall Curve
plt.figure(figsize=(8, 6))
for name in ["Random Forest", "XGBoost"]:
    plt.plot(pr_results[name]["recall"], pr_results[name]["precision"], label=name)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curves (Test Set)")
plt.legend()
plt.show()

# --- Misclassification Analysis ---
print("\n🔍 Misclassifications (XGBoost):")
misclassified = df_full[df_full["anomaly_xgb"] != df_full["label"]][features + ["label", "anomaly_xgb"]]
print(misclassified.head(20))
print("\n🔍 Misclassification Statistics:")
print(misclassified[features].describe())

# --- Visualizations ---
def create_visualizations(df, features):
    # Time series plot
    fig1 = px.line(df, x=df.index, y="inter_arrival_time", title="Inter-Arrival Time with Anomalies")
    for model, color in [("anomaly_iso", "orange"), ("anomaly_autoencoder", "purple"), ("anomaly_rf", "blue"), ("anomaly_xgb", "cyan")]:
        fig1.add_scatter(x=df[df[model] == 1].index, y=df[df[model] == 1]["inter_arrival_time"],
                         mode="markers", name=f"{model.replace('anomaly_', '').title()}",
                         marker=dict(color=color, size=6))
    fig1.add_scatter(x=df[df["label"] == 1].index, y=df[df["label"] == 1]["inter_arrival_time"],
                     mode="markers", name="Labeled Attacks", marker=dict(color="green", size=7, symbol="x"))
    fig1.update_layout(xaxis_title="Time", yaxis_title="Inter-Arrival Time")

    # Scatter plot
    fig2 = px.scatter(df, x="inter_arrival_time", y="request_rate", color="anomaly_xgb",
                      title="Request Rate vs. Inter-Arrival Time",
                      labels={"inter_arrival_time": "Inter-Arrival Time", "request_rate": "Request Rate"},
                      color_continuous_scale=["blue", "cyan"])
    fig2.add_scatter(x=df[df["label"] == 1]["inter_arrival_time"], y=df[df["label"] == 1]["request_rate"],
                     mode="markers", name="Labeled Attacks", marker=dict(color="green", size=7, symbol="x"))

    # ROC curve
    fig3 = go.Figure()
    for name, model in [("Random Forest", rf_model), ("XGBoost", xgb_model)]:
        probs = model.predict_proba(X_test_scaled)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, probs)
        auc = roc_auc_score(y_test, probs)
        fig3.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{name} (AUC={auc:.2f})"))
    fig3.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash"), name="Random"))
    fig3.update_layout(title="ROC Curves (Test Set)", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")

    # Anomaly table
    anomaly_cols = ["anomaly_iso", "anomaly_autoencoder", "anomaly_rf", "anomaly_xgb"]
    anomaly_df = df[df[anomaly_cols].any(axis=1)][features + anomaly_cols + ["label"]].head(20)

    return fig1, fig2, fig3, anomaly_df

# Run visualizations
fig1, fig2, fig3, anomaly_df = create_visualizations(df_full, features)
fig1.show()
fig2.show()
fig3.show()
print("\n📋 Anomaly Table (Top 20):")
print(anomaly_df)

# Confusion matrices
print("\n📉 Confusion Matrices (Full Data):")
for name, preds in models.items():
    cm = confusion_matrix(df_full["label"], preds)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

# Save results
df_full.to_csv("dns_anomaly_results.csv")
print("\n💾 Results saved to dns_anomaly_results.csv")