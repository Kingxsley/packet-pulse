"""Central configuration for the anomaly detection pipeline."""
from pathlib import Path

# --- Paths ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
RESULTS_DIR = OUTPUT_DIR / "results"
MODELS_DIR = OUTPUT_DIR / "models"

DATASET_FILES = {
    "dns": DATA_DIR / "DNSpackets_output.json",
    "dos": DATA_DIR / "DOSpackets_output.json",
    "dos_clean": DATA_DIR / "Clean_DOS_Capstone.csv",
}

# --- Reproducibility ---
RANDOM_STATE = 42

# --- Feature engineering ---
# The original script assumed a pre-aggregated "dns_rate" field that does not
# exist in the raw packet captures shipped with this project (see README,
# "Fixes" section, item 2). We derive equivalent rate-based features directly
# from packet timestamps instead.
RATE_WINDOW = "1s"          # rolling window used to compute packet_rate
MIN_INTER_ARRIVAL = 0.001   # floor for inter_arrival_time to avoid div-by-zero

FEATURES = ["inter_arrival_time", "request_rate", "packet_rate", "packet_length"]
REQUIRED_RAW_COLS = ["source_ip", "dest_ip", "source_port", "dest_port",
                      "protocol", "packet_length", "timestamp",
                      "inter_arrival_time", "label"]

# --- Model hyperparameters (defaults; --tune enables search instead) ---
ISO_FOREST_PARAMS = dict(n_estimators=200, max_features=1.0, random_state=RANDOM_STATE)

AUTOENCODER_PARAMS = dict(epochs=30, batch_size=64, patience=5)

RF_PARAMS = dict(
    n_estimators=200, max_depth=15, min_samples_split=10,
    min_samples_leaf=8, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1,
)

XGB_PARAMS = dict(
    n_estimators=200, max_depth=5, min_child_weight=15,
    learning_rate=0.05, eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=-1,
)

# Reduced search spaces used only when --tune is passed (the original
# unconditional GridSearchCV over 16 combinations x 5-fold CV per model was
# impractical on a 300k-row dataset; see README "Fixes" item 5).
RF_TUNE_GRID = {
    "n_estimators": [100, 200],
    "max_depth": [10, 15],
    "min_samples_leaf": [8, 10],
    "class_weight": ["balanced"],
}
XGB_TUNE_GRID = {
    "n_estimators": [100, 200],
    "max_depth": [3, 5],
    "learning_rate": [0.01, 0.05],
}
TUNE_CV = 3
TUNE_N_ITER = 8
