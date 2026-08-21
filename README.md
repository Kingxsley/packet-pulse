# Packet Pulse

Network traffic anomaly detection for our MBIS5015 capstone, "Anomaly
Detection in Network Traffic using Machine Learning for Proactive Cyber
Threat Mitigation." Four models score DNS and DoS packet captures side by
side, a Streamlit dashboard lets you dig into every prediction and tune
detection thresholds live, and a landing page ties it together.

- **Live dashboard:** https://dashboard-production-8fac.up.railway.app
- **Landing page:** https://kingxsley.github.io/packet-pulse/
- **Repo:** https://github.com/Kingxsley/packet-pulse

Deploys straight from this repo to Railway on every push. Postgres sits
next to it so anything imported through Live Scoring sticks around instead
of disappearing when the tab closes.

## What's in here

- **Isolation Forest** and **Autoencoder** (unsupervised) learn what normal
  traffic looks like without labels
- **Random Forest** and **XGBoost** (supervised) learn the attacks directly,
  with SHAP explaining every XGBoost call
- A **dashboard** with live threshold tuning, per-row true/false
  positive/negative labeling, and a Live Scoring tab anyone can drop a
  dataset into
- A **landing page** with the real numbers from the last training run, not
  placeholder copy

## Project structure

```
anomaly_detection_project/
├── run.py                  # CLI entry point for the training pipeline
├── requirements.txt
├── Procfile                 # Railway start command
├── src/
│   ├── config.py            # paths, feature list, hyperparameters
│   ├── data_loader.py       # loads the local JSON/CSV packet captures
│   ├── features.py          # feature engineering
│   ├── models.py            # model training (Isolation Forest, Autoencoder, RF, XGBoost)
│   ├── evaluate.py          # metrics, confusion matrix, SHAP
│   ├── visualize.py         # saves figures to outputs/figures
│   └── pipeline.py          # orchestrates one end-to-end run
├── dashboard/
│   ├── app.py                # Streamlit dashboard
│   ├── data_access.py        # cached loaders for results/models
│   ├── db.py                 # Postgres persistence for Live Scoring imports
│   └── theme.py               # shared brand styling
├── data/raw/                 # DNSpackets_output.json, DOSpackets_output.json, Clean_DOS_Capstone.csv
├── outputs/
│   ├── figures/               # PNG/HTML charts per run
│   ├── results/                # per-row predictions (CSV) + metrics/summary/SHAP (JSON)
│   └── models/                 # trained model + scaler artifacts (joblib / .keras)
├── website/index.html          # landing page source
├── docs/index.html             # copy of the landing page GitHub Pages serves
└── legacy/original_script.py   # our first prototype script, kept for reference
```

## Setup

```bash
git clone https://github.com/Kingxsley/packet-pulse
cd packet-pulse
python -m venv .venv
.venv/Scripts/activate       # Windows
pip install -r requirements.txt
```

We built and tested this on **Python 3.11** (TensorFlow doesn't publish
wheels for very new Python releases yet, so if you've got 3.13/3.14 on your
machine too, use 3.11 for this project).

## Running the pipeline

```bash
python run.py --dataset dos_clean      # fast, recommended default (112k rows)
python run.py --dataset dns            # 306k-row DNS capture
python run.py --dataset both           # runs dns + dos_clean back to back
python run.py --dataset dos_clean --tune          # enable hyperparameter search
python run.py --dataset dns --no-autoencoder      # skip TensorFlow if not installed
```

Each run prints per-model test-set metrics and writes:
- `outputs/results/<dataset>_anomaly_results.csv`: every row with each model's prediction, anomaly score, train/test split assignment, and the ground-truth label
- `outputs/results/<dataset>_metrics.json`: precision/recall/F1/ROC-AUC per model
- `outputs/results/<dataset>_summary.json`: row counts, attack rate, split sizes, training time
- `outputs/results/<dataset>_shap_importance.json`: mean |SHAP value| per feature (XGBoost)
- `outputs/figures/`: feature distributions, confusion matrices, ROC/PR curves, an interactive time-series plot and scatter plot
- `outputs/models/`: the fitted scaler and all four trained models

## Dashboard

```bash
streamlit run dashboard/app.py
```

That's the command we originally ran locally while building this. The
version at the live link above is the same app, just deployed to Railway so
we're not the only ones who can open it. Five tabs:

- **Overview**: row counts, attack rate, and the stored test-set metrics/comparison chart for all four models
- **Explore Data**: feature distributions, an interactive time-series view of traffic with predicted/true anomalies highlighted, and a request-rate vs. inter-arrival-time scatter plot
- **Model Comparison**: confusion matrices, ROC and Precision-Recall curves, all recomputed live as you move the per-model threshold sliders in the sidebar. Also shows exact false-positive/false-negative counts, not just precision/recall, since those are the numbers that actually matter when you're deciding how sensitive to make this
- **Feature Importance**: the SHAP bar chart for XGBoost
- **Live Scoring**: upload a CSV/JSON of raw packets (same schema as the training data, `label` optional) and it runs through the persisted scaler + all four models. Every row gets flagged attack/normal, and when you've got ground truth, each one is also labeled true/false positive/negative so you can see exactly what the model got wrong. Imports are saved to Postgres, so you can come back later and see what's been tested

The sidebar also has a **Retrain pipeline** control that runs the same
pipeline as `run.py`, straight from the browser, for any dataset (with or
without `--tune`/the autoencoder). It blocks the UI while training runs
(roughly 1 to 4 minutes), which is fine for a demo but isn't a background
job queue.

## Data

Three packet captures, all sharing the same schema (`source_ip, dest_ip,
source_port, dest_port, protocol, packet_length, timestamp,
inter_arrival_time, label`):

| dataset | file | rows | attack rate |
|---|---|---|---|
| `dns` | `DNSpackets_output.json` | 306,838 | ~37% |
| `dos` | `DOSpackets_output.json` | 112,865 | ~0.1% |
| `dos_clean` | `Clean_DOS_Capstone.csv` | 112,864 | ~0.1% (deduplicated/cleaned export of `dos`) |

## Results

Test-set metrics from a full run of both datasets (defaults, no `--tune`):

**`dos_clean`** (112,864 rows, ~0.1% attacks, extreme class imbalance):

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

A couple of things worth knowing before you trust these numbers:

- On `dos_clean`, recall stays high but precision drops hard at the default
  0.5 threshold, because there are only ~145 attack rows out of 112k. Use
  the Model Comparison tab's sliders to see the actual tradeoff instead of
  reading one row off this table.
- The near-perfect `dns` scores aren't a bug, but we wouldn't ship them
  without a second look: SHAP shows `packet_length` alone drives almost all
  of the XGBoost decisions on this capture, meaning the simulated attack
  traffic here is trivially separable by packet size. That's a property of
  how the capture was generated, not proof the model would catch a real
  DNS flood that doesn't have such an obvious size signature. Treat these
  as an upper bound.

## What this doesn't do yet, and how a company would actually use it

There's no InfluxDB, no live traffic feed, no streaming anything. This
scores whatever data you hand it, either the packet captures baked into the
repo or a file you upload through Live Scoring. It isn't watching real
network traffic right now.

Our original prototype tried to pull data live from an InfluxDB Cloud
bucket. We dropped that (see `legacy/original_script.py` for what it used
to look like) because it meant every run depended on a specific cloud
account being online, and honestly, we didn't have real traffic flowing
into it either, just recorded captures. Scoring the captures directly, in a
pipeline we actually understand end to end, got us a working system instead
of a half-connected one.

If a company wanted to run this for real, the missing piece is an ingestion
layer between "packets hitting the network" and our `engineer_features`
step: something like a packet capture agent or a network tap feeding a
queue (Kafka, or yes, back to something like InfluxDB) that this pipeline
polls or subscribes to, scoring each record as it arrives instead of in a
batch. The four models and the feature engineering wouldn't need to
change, they're already fast enough for that. What would need to change:

- A real ingestion source instead of static files
- Flagged attacks pushing to something a human sees fast (Slack, PagerDuty,
  email) instead of sitting in a dashboard tab waiting to be opened
- Multi-user auth, since right now anyone with the URL can see everyone
  else's imports
- A background job queue for retraining, instead of blocking the UI thread

The Postgres database we just added is a step in that direction. It's the
first piece of this system that's a real database instead of files on
disk, and it's what an alerting pipeline would build on top of.

## What we fixed from the original script

The first version of this (`legacy/original_script.py`) was a single
script that pulled from InfluxDB and had a live API token committed
straight into it. It also didn't actually run against the packet captures
we ended up using: it required a `dns_rate` column that only existed in
the InfluxDB schema, not in any file we had. A few other things we caught
along the way:

- Isolation Forest and the Autoencoder were being fit and evaluated on the
  same rows, no held-out split, while Random Forest and XGBoost got a
  proper train/test split. Not a fair comparison. All four now share one
  split.
- `GridSearchCV` ran an unconditional 16-combination search over 5 folds
  for both Random Forest and XGBoost, which crawls on 300k rows. Tuning is
  opt-in now via `--tune`.
- `shap_values()` output shape depends on the installed shap/xgboost
  version and the original code assumed one specific shape. We normalize
  all three shapes it can actually return.
- Every chart called `plt.show()`, which just hangs outside a notebook.
  Figures save to `outputs/figures/` instead.
- Isolation Forest's contamination was a hardcoded 0.3 regardless of the
  actual attack rate (0.1% to 37% depending on dataset). It's derived from
  the training split now.
- Nothing got saved. Trained models are persisted under `outputs/models/`
  so they can be loaded without retraining, which is also what makes Live
  Scoring possible.
