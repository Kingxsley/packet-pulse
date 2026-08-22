# Packet Pulse

Network traffic anomaly detection for our MBIS5015 capstone, "Anomaly
Detection in Network Traffic using Machine Learning for Proactive Cyber
Threat Mitigation." The front door is a live threat monitor: real captured
packets replayed through four trained models in near real time, scored one
at a time, with a running feed and a threat level that escalates when
traffic turns hostile. The metrics, charts, and threshold tuning that used
to be the whole app are still there, just repositioned as supporting
analytics behind the thing that's actually supposed to be watching traffic.

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
- An **app** with live threshold tuning, per-row true/false
  positive/negative labeling, and a Live Scoring page anyone can drop a
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
├── backend/
│   ├── main.py                # FastAPI routes
│   ├── data_access.py         # cached loaders for results/models
│   ├── db.py                  # Postgres persistence for Live Scoring imports
│   ├── scoring.py             # threshold application, TP/FP/FN/TN labeling
│   ├── charts.py               # Plotly figure builders
│   ├── siem.py                  # CEF export + Splunk HEC forwarder
│   ├── templates/               # Jinja2 pages, same design system as the landing page
│   └── static/app.css           # shared CSS tokens
├── data/raw/                 # DNSpackets_output.json, DOSpackets_output.json, Clean_DOS_Capstone.csv
├── outputs/
│   ├── figures/               # PNG/HTML charts per CLI run
│   ├── results/                # per-row predictions (CSV) + metrics/summary/SHAP (JSON)
│   └── models/                 # trained model + scaler artifacts (joblib / .keras)
├── website/index.html          # landing page source
├── docs/index.html             # copy of the landing page GitHub Pages serves
└── legacy/
    ├── original_script.py       # our first prototype script, kept for reference
    └── streamlit_dashboard/     # the Streamlit app this backend replaced
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

## The app

```bash
uvicorn backend.main:app --reload
```

Open http://localhost:8000. This used to be a Streamlit app
(`streamlit run dashboard/app.py`), and we built the first version that way
because it got something working fast. We moved off it once the UI stopped
being good enough: Streamlit only lets you restyle its own widgets through
injected CSS, which is exactly as fragile as it sounds, and it never looked
like more than a data-science notebook no matter how much CSS we threw at
it. The FastAPI app in `backend/` renders real HTML with Jinja2, using the
same design tokens as the landing page, so the app and the marketing site
are actually one visual product instead of two. The old Streamlit version
is kept in `legacy/streamlit_dashboard/` and still runs if you want it
(`pip install streamlit`, then `streamlit run legacy/streamlit_dashboard/app.py`).

Four items in the nav, because a SOC analyst using this daily shouldn't
have to think about which of eight tabs has what they need. Model
internals (metrics tables, SHAP, threshold tuning, feature scatter plots)
are real and still fully working, they're just not primary navigation
anymore: they live one click away, behind Settings, under a collapsed
"Model performance" panel, for whoever actually needs to validate the
models rather than the people responding to what they flag.

Every page's dataset picker also offers **Combined: DNS + DoS**, which
merges both trained datasets (tagged with `source_dataset` so you can
always tell which capture a row came from) instead of making you pick one
attack surface to watch at a time. Dashboard round-robins between the two
captures so both are visibly live at once, and Incidents/exports can
filter or include both. Combined view only makes sense on those two pages
(comparing models across two separate training runs' metrics wouldn't
mean anything), so picking it elsewhere just falls back to a real dataset.

- **Dashboard** (`/monitor`): the live feed. Replays a dataset's packets in timestamp order through a Server-Sent Events stream, scoring each one against the persisted models as it goes, plus a total-incidents count and a top-attacking-sources panel above the feed. A stat strip tracks packets scanned/attacks flagged/flag rate for the session, and a threat-level pill (NOMINAL/ELEVATED/CRITICAL) escalates based on the rolling flag rate. This is a replay of real recorded captures at a controllable speed, not a live tap on an actual network. See "What this doesn't do yet" below for what real packet capture would take
- **Incidents**: every packet the selected model flagged, newest first, paginated, each with a severity tier (Critical/High/Medium/Low, from the model's score) and a **status you can change** (New / Investigating / Resolved / False Positive), persisted in Postgres and keyed to the physical packet, not the model you're currently viewing it through, so triage work doesn't reset when you switch models. Filter by status, export the current filter as CSV or CEF (the SIEM-standard format), or forward as a batch to a Splunk HTTP Event Collector URL + token you provide, matching Splunk's documented HEC request shape exactly. We haven't tested that against a live Splunk instance (don't have one), but tested the actual HTTP mechanics against a mock endpoint: request goes out, response comes back, errors are caught and reported instead of crashing
- **Scan** (`/live-scoring`): upload a CSV/JSON of raw packets (same schema as the training data, `label` optional) and it runs through the persisted scaler + all four models. Every row gets flagged attack/normal, and when you've got ground truth, each one is also labeled true/false positive/negative so you can see exactly what the model got wrong. Imports are saved to Postgres, so you can come back later and see what's been tested
- **Settings**: dataset stats, an **Automation** panel with the retrain trigger (see below), and a collapsed **Model performance** panel linking to Overview, Explore Data, Model Comparison, and Feature Importance, unchanged and fully working, just not cluttering the main nav

Retraining runs the same pipeline as `run.py`, in a background thread with
status polling, for any dataset (with or without `--tune`/the autoencoder).
Training takes roughly 1 to 4 minutes. Rather than a button someone has to
remember to click, `POST /api/retrain` (`dataset`/`tune`/`autoencoder` form
fields) is meant to be hit by a scheduler, a Railway cron service, GitHub
Actions, whatever a real deployment already uses for scheduled jobs, so
retraining stays a background job instead of a page someone has to babysit.

## Data

Two packet captures, both sharing the same schema (`source_ip, dest_ip,
source_port, dest_port, protocol, packet_length, timestamp,
inter_arrival_time, label`):

| dataset | file | rows | attack rate |
|---|---|---|---|
| `dns` | `DNSpackets_output.json` | 306,838 | ~37% |
| `dos_clean` | `Clean_DOS_Capstone.csv` | 112,864 | ~0.1% |

There used to be a third, `dos`: the raw, uncleaned DoS capture that
`dos_clean` is a deduplicated export of. We dropped it (see "What we
removed" below) since nothing in the app ever used it differently from
`dos_clean`, it was just 34MB of dead weight in the repo.

## Results

Test-set metrics from a full run of both datasets (defaults, no `--tune`):

**`dos_clean`** (112,864 rows, ~0.1% attacks, extreme class imbalance):

| model | precision | recall | F1 | ROC-AUC |
|---|---|---|---|---|
| Isolation Forest | 0.030 | 0.227 | 0.052 | 0.779 |
| Random Forest | 0.456 | 0.932 | 0.612 | 0.991 |
| XGBoost | 0.092 | 0.932 | 0.168 | 0.990 |

**`dns`** (306,838 rows, ~37% attacks):

| model | precision | recall | F1 | ROC-AUC |
|---|---|---|---|---|
| Isolation Forest | 0.585 | 0.581 | 0.583 | 0.810 |
| Random Forest | 0.9999 | 1.000 | 0.9999 | 1.000 |
| XGBoost | 0.9999 | 1.000 | 0.9999 | 1.000 |

**Random Forest is the default "flag with" model** across the Threat
Monitor, Incidents, and Live Scoring pages. It's the highest-F1 model on
both datasets, and it isn't close on `dos_clean`: 0.612 vs. XGBoost's 0.168,
because XGBoost's recall is just as high but its precision collapses under
that dataset's 1-in-800 attack rate. Every page still lets you switch to
any of the four models; this is just the default, not the only option.

A couple of things worth knowing before you trust these numbers:

- On `dos_clean`, recall stays high but precision drops hard at the default
  0.5 threshold, because there are only ~145 attack rows out of 112k. Use
  the Model Comparison page's sliders to see the actual tradeoff instead of
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

- A real ingestion source instead of static files (see below)
- Multi-user auth, since right now anyone with the URL can see everyone
  else's imports
- A background job queue for retraining, instead of blocking the UI thread

Two things we already shipped that used to be on this list: Postgres
(`backend/db.py`) gives the system a real database instead of files on
disk, and the Incidents page's CEF export + Splunk HEC forwarder
(`backend/siem.py`) is a real path for flagged attacks to reach a human
fast, through whatever SIEM/alerting stack a company already runs, instead
of sitting in a browser tab waiting to be opened.

### Adding a real ingestion source

This is the actual gap, and it's worth being specific about what closing
it would take rather than leaving it vague:

1. **Something has to read the wire.** A normal server only sees traffic
   addressed to it, so you need either a SPAN/mirror port on the switch (or
   a physical network tap), or, for a cloud deployment, the provider's
   traffic-mirroring feature (AWS Traffic Mirroring, Azure vTap, GCP Packet
   Mirroring), since you can't tap a virtual network the way you'd tap a cable.
2. **[Zeek](https://zeek.org/)** pointed at that interface writes a
   `conn.log` with source/dest IP and port, protocol, byte counts, and
   timestamps for every connection: almost exactly `src/config.py`'s
   `REQUIRED_RAW_COLS`, and a lot less work than parsing raw packets
   ourselves.
3. **A small adapter script** tails `conn.log`, renames Zeek's fields to
   ours, batches a few seconds of rows, and `POST`s them as JSON to
   `/api/live-score` (`backend/main.py`), which already does feature
   engineering and scoring, no new ML code needed.
4. **Swap the Threat Monitor's source.** `/api/monitor-stream` currently
   reads a static CSV; pointed at a real feed, it'd read from a queue
   (Redis Streams or Kafka) that the adapter pushes into instead, and drop
   the "replay of recorded captures" framing since it'd actually be live.

None of this is implemented. It's the honest next step, not a feature.

## Configuration

Everything here degrades gracefully when unset, so local dev
(`uvicorn backend.main:app --reload`) never needs any of it configured.

| Variable | What it does |
|---|---|
| `APP_PASSWORD` | Turns on the login gate. Unset means the app is fully public, which is fine for local dev and is how this ran for most of the build, but not how it should stay deployed |
| `SESSION_SECRET` | Signs the login session cookie. Falls back to `APP_PASSWORD` if unset, which is fine, but set it separately if you'd rather not reuse the login password as a signing key |
| `DATABASE_URL` | Postgres connection string. Without it, Live Scoring imports and Incidents triage status just don't persist; everything else still works |
| `ALERT_WEBHOOK_URL` | A Slack incoming webhook URL (or anything that accepts `{"text": "..."}`). Without it, Critical-severity flags don't push anywhere; you'd only see them by visiting Incidents |

`APP_PASSWORD` is the one that actually matters for a real deployment: a
public URL with a database anyone can mutate (wipe the live model via
Retrain, edit anyone else's triage status) was the single biggest gap this
project had once it stopped being a private demo.

## What we removed, and why

Not everything we built earned a permanent place. Two things came out
once we looked at this as a real system instead of a place to show off
every idea we'd had:

- **The Autoencoder.** It was the weakest or tied-weakest model on both
  datasets (F1 0.036 on `dos_clean`, worse than Isolation Forest; 0.660 on
  `dns`, worst of the four), and TensorFlow was by a wide margin the
  heaviest dependency in the project, a ~1GB install that made every
  Railway build take 5+ minutes on its own. Cutting it dropped build time
  and image size substantially for a model that wasn't winning anywhere.
  The training code didn't go anywhere (`src/models.py`, still callable via
  `run.py --with-autoencoder` for local comparison), it's just not part of
  the deployed app's model set (`backend/scoring.py`) or dependencies
  (`requirements.txt`) anymore.
- **The raw `dos` capture.** `Clean_DOS_Capstone.csv` is a deduplicated
  export of it, and nothing in the app ever used the raw version
  differently, it was just a second, worse copy of the same data taking up
  34MB in the repo.

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
