# Network Traffic Anomaly Detection

Rebuilt, working version of the MBIS5015 capstone prototype **"Anomaly
Detection in Network Traffic using Machine Learning for Proactive Cyber
Threat Mitigation."** The original deliverable was a single 339-line script
(`legacy/original_script.py`) that could not run against the data shipped
alongside it and depended on a live cloud database with a hardcoded API
token. This project restructures it into a runnable pipeline over the local
packet captures, and fixes the bugs described below.

> **⚠️ Security note:** the original script contained a live InfluxDB Cloud
> API token committed in plaintext. It has been redacted from
> `legacy/original_script.py` in this project, but **the original credential
> was exposed in the zip you provided — you should rotate/revoke that token
> in your InfluxDB Cloud account now**, since anyone who had that file could
> read/write your bucket.

## What it does

Detects malicious traffic (DNS floods / DoS) in packet-capture data using
four models trained side-by-side:

- **Isolation Forest** (unsupervised)
- **Autoencoder** (unsupervised, Keras/TensorFlow)
- **Random Forest** (supervised)
- **XGBoost** (supervised, with SHAP feature-importance explanation)

## Project structure

```
anomaly_detection_project/
├── run.py                  # CLI entry point
├── requirements.txt
├── src/
│   ├── config.py           # paths, feature list, hyperparameters
│   ├── data_loader.py      # loads the local JSON/CSV packet captures
│   ├── features.py         # feature engineering
│   ├── models.py           # model training (Isolation Forest, Autoencoder, RF, XGBoost)
│   ├── evaluate.py         # metrics, confusion matrix, SHAP
│   ├── visualize.py        # saves figures to outputs/figures
│   └── pipeline.py         # orchestrates one end-to-end run
├── dashboard/
│   ├── app.py               # Streamlit dashboard
│   └── data_access.py       # cached loaders for results/models
├── data/raw/                # DNSpackets_output.json, DOSpackets_output.json, Clean_DOS_Capstone.csv
├── outputs/
│   ├── figures/             # PNG/HTML charts per run
│   ├── results/             # per-row predictions (CSV) + metrics/summary/SHAP (JSON)
│   └── models/              # trained model + scaler artifacts (joblib / .keras)
└── legacy/original_script.py  # the original script, kept for reference (token redacted)
```

## Setup

```bash
cd anomaly_detection_project
python -m venv .venv
.venv/Scripts/activate       # Windows
pip install -r requirements.txt
```

This was built and tested with **Python 3.11** (TensorFlow does not yet
publish wheels for very new Python releases such as 3.14, so avoid the
newest interpreter on your machine if you have multiple installed).

## Usage

```bash
python run.py --dataset dos_clean      # fast, recommended default (112k rows)
python run.py --dataset dns            # 306k-row DNS capture
python run.py --dataset both           # runs dns + dos_clean back to back
python run.py --dataset dos_clean --tune          # enable hyperparameter search
python run.py --dataset dns --no-autoencoder      # skip TensorFlow if not installed
```

Each run prints per-model test-set metrics and writes:
- `outputs/results/<dataset>_anomaly_results.csv` — every row with each model's prediction, anomaly score, train/test split assignment, and the ground-truth label
- `outputs/results/<dataset>_metrics.json` — precision/recall/F1/ROC-AUC per model
- `outputs/results/<dataset>_summary.json` — row counts, attack rate, split sizes, training time
- `outputs/results/<dataset>_shap_importance.json` — mean |SHAP value| per feature (XGBoost)
- `outputs/figures/` — feature distributions, confusion matrices, ROC/PR curves, an interactive time-series plot and scatter plot
- `outputs/models/` — the fitted scaler and all four trained models, so a serving layer can load them directly

## Dashboard

A Streamlit dashboard (the charter's "Interactive Dashboard" deliverable)
sits on top of whatever `run.py` has produced so far — run the pipeline
for at least one dataset first, then:

```bash
streamlit run dashboard/app.py
```

Open http://localhost:8501. It has five tabs:

- **Overview** — row counts, attack rate, and the stored test-set metrics/comparison chart for all four models.
- **Explore Data** — feature distributions, an interactive time-series view of traffic with predicted/true anomalies highlighted, and a request-rate vs. inter-arrival-time scatter plot.
- **Model Comparison** — confusion matrices, ROC and Precision-Recall curves, all **recomputed live** as you move the per-model threshold sliders in the sidebar (probability threshold for Random Forest/XGBoost, "% flagged" sensitivity for Isolation Forest/Autoencoder) — this is the tool for exploring the precision/recall tradeoff flagged as a limitation in the CLI results.
- **Feature Importance** — the SHAP bar chart for XGBoost.
- **Live Scoring** — upload a CSV/JSON of raw packets (same schema as the training data, `label` optional) and it's run through the persisted scaler + all four models to produce anomaly flags, with a one-click "try it with a random sample from this dataset" option if you don't have a file handy, and a CSV download of the scored output.

The sidebar also has a **Retrain pipeline** control that calls the same
`src/pipeline.run()` used by `run.py` directly from the UI (any dataset,
with or without `--tune`/the autoencoder) — useful after adding new data to
`data/raw/` without leaving the browser. It blocks the UI for the duration
of training (~1–3 minutes), which is acceptable for this batch-retraining
use case but is not the same thing as the charter's real-time streaming
pipeline (see "Not (yet) implemented" below).

## Data

Three packet captures were provided, all sharing the same schema
(`source_ip, dest_ip, source_port, dest_port, protocol, packet_length,
timestamp, inter_arrival_time, label`):

| dataset | file | rows | attack rate |
|---|---|---|---|
| `dns` | `DNSpackets_output.json` | 306,838 | ~37% |
| `dos` | `DOSpackets_output.json` | 112,865 | ~0.1% |
| `dos_clean` | `Clean_DOS_Capstone.csv` | 112,864 | ~0.1% (deduplicated/cleaned export of `dos`) |

## Fixes made vs. the original script

1. **Leaked credential.** The original hardcoded an InfluxDB Cloud API token
   in source. Removed entirely — the pipeline now runs fully offline against
   the local packet captures instead of requiring cloud access.
2. **Crashing column check.** The original required a `dns_rate` column that
   doesn't exist anywhere in the data you provided (it was an InfluxDB-only
   pre-aggregated field) — running it against these files raised
   `ValueError: Missing columns: ['dns_rate']` immediately. `dns_rate` was
   also never actually used in the model's `features` list, so the check was
   dead code checking the wrong thing. Replaced with a `packet_rate` feature
   computed directly from packet timestamps (rolling 1-second packet count).
3. **Inconsistent clipping.** `inter_arrival_time` was only floored when
   `.le(0).any()` happened to be true for the whole column; a single bad row
   shouldn't gate whether the fix applies at all. Now applied unconditionally.
4. **Unfair unsupervised evaluation.** Isolation Forest and the Autoencoder
   were fit on the *entire* dataset (train + test combined) with no held-out
   split, while Random Forest/XGBoost got a proper train/test split — an
   apples-to-oranges comparison that also let unsupervised metrics leak test
   rows into fitting. All four models now share one train-fit scaler and are
   evaluated identically on the same held-out test split.
5. **Impractical grid search.** `GridSearchCV` with a 16-combination grid ×
   5-fold CV ran unconditionally for both Random Forest and XGBoost, which is
   very slow on a 300k-row dataset. Tuning is now opt-in via `--tune`
   (`RandomizedSearchCV`, 3-fold, 8 iterations); a single reasonable default
   hyperparameter set is used otherwise so a full run finishes in ~2 minutes.
6. **Fragile SHAP handling.** `explainer.shap_values(X)` was assumed to
   always return one 2D array; depending on the installed `shap`/`xgboost`
   versions it can instead return a list or an `Explanation` object, breaking
   `pd.DataFrame(shap_values, ...)`. `evaluate.shap_feature_importance` now
   normalizes all three shapes. SHAP is also capped at a 5,000-row sample on
   large datasets to keep runtime bounded.
7. **Blocking plots.** Every chart called `plt.show()` / `fig.show()`, which
   hangs indefinitely outside an interactive notebook. All figures are now
   saved to `outputs/figures/` instead.
8. **Fixed contamination.** Isolation Forest's `contamination=0.3` was a
   constant unrelated to the actual attack rate in the data (0.1%–37%
   depending on dataset). It's now derived from the training split's label
   prevalence (clamped to `[0.01, 0.5]`), while still not using labels to fit
   the model itself.
9. **No persisted artifacts.** The original discarded every trained model
   after the script exited. Trained models and the fitted scaler are now
   saved under `outputs/models/` (`joblib` for Isolation Forest/RF/XGBoost,
   Keras format for the autoencoder), so they can be loaded by a serving
   layer without retraining — a step toward the charter's "Model Serving"
   deliverable.

## Results

Test-set metrics from a full run of both datasets (defaults, no `--tune`):

**`dos_clean`** (112,864 rows, ~0.1% attacks — extreme class imbalance):

| model | precision | recall | F1 | ROC-AUC |
|---|---|---|---|---|
| Isolation Forest | 0.030 | 0.227 | 0.052 | 0.779 |
| Autoencoder | 0.019 | 0.750 | 0.036 | 0.939 |
| Random Forest | 0.456 | 0.932 | 0.612 | 0.991 |
| XGBoost | 0.092 | 0.932 | 0.168 | 0.990 |

**`dns`** (306,838 rows, ~37% attacks):

| model | precision | recall | F1 | ROC-AUC |
|---|---|---|---|---|
| Isolation Forest | 0.585 | 0.581 | 0.583 | 0.810 |
| Autoencoder | 0.865 | 0.538 | 0.663 | 0.844 |
| Random Forest | 0.9999 | 1.000 | 0.9999 | 1.000 |
| XGBoost | 0.9999 | 1.000 | 0.9999 | 1.000 |

**Observations / caveats:**
- On `dos_clean`, XGBoost/Isolation Forest/Autoencoder all show high recall
  but low precision — with only ~145 attack rows out of 112k, even a very
  good model produces many false positives at the default 0.5 probability
  threshold. If you need higher precision for this dataset, raise the
  classification threshold above 0.5 using `predict_proba` on the saved
  model rather than the default `.predict()`.
- The near-perfect scores on `dns` are not a bug, but worth treating with
  suspicion for a real deployment: SHAP shows `packet_length` alone accounts
  for the overwhelming majority of the XGBoost model's decisions on this
  dataset (`request_rate`/`inter_arrival_time` contribute almost nothing).
  This means the simulated DNS attack traffic in this capture is trivially
  separable by packet size — consistent with the Project Charter's own
  caveat that "testing is restricted to simulated data (not real enterprise
  traffic)." Treat these numbers as an upper bound, not evidence the model
  will generalize to real-world DNS attacks that don't have such a
  distinctive packet size.

## Not (yet) implemented

The Project Charter also scopes a **FastAPI backend** ("to fetch anomaly
data") and **containerized deployment**. The Streamlit dashboard above reads
the pipeline's output files and loaded models directly rather than going
through an API layer, and the "Retrain pipeline" control runs training
synchronously in-process rather than as a background job — both reasonable
simplifications for a single-user local dashboard, but a FastAPI service in
front of `outputs/` and `src/pipeline.py` would be the natural next step if
this needs to serve multiple dashboard clients or a real ingestion pipeline.
Docker packaging for the app + dashboard is also not set up yet.
