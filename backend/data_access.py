"""Cached loaders for pipeline outputs and trained models.

A plain in-memory cache keyed on the source file's mtime, so results
refresh automatically after `run.py` (or a retrain) writes new files,
without needing a framework-specific cache like Streamlit's.
"""
import json
import sys
import threading
from pathlib import Path

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config  # noqa: E402

_lock = threading.Lock()
_json_cache: dict[str, tuple[float, dict]] = {}
_df_cache: dict[str, tuple[float, pd.DataFrame]] = {}
_model_cache: dict[str, tuple[float, dict]] = {}


def _file_mtime(path: Path) -> float:
    return path.stat().st_mtime if path.exists() else -1.0


def available_datasets() -> list[str]:
    return sorted(p.stem.replace("_summary", "") for p in config.RESULTS_DIR.glob("*_summary.json"))


def _load_json(path: Path) -> dict:
    mtime = _file_mtime(path)
    key = str(path)
    with _lock:
        cached = _json_cache.get(key)
        if cached and cached[0] == mtime:
            return cached[1]
    with open(path) as f:
        data = json.load(f)
    with _lock:
        _json_cache[key] = (mtime, data)
    return data


def summary(dataset: str) -> dict:
    return _load_json(config.RESULTS_DIR / f"{dataset}_summary.json")


def metrics(dataset: str) -> dict:
    return _load_json(config.RESULTS_DIR / f"{dataset}_metrics.json")


def shap_importance(dataset: str) -> dict:
    return _load_json(config.RESULTS_DIR / f"{dataset}_shap_importance.json")


def load_results(dataset: str) -> pd.DataFrame:
    path = config.RESULTS_DIR / f"{dataset}_anomaly_results.csv"
    mtime = _file_mtime(path)
    key = str(path)
    with _lock:
        cached = _df_cache.get(key)
        if cached and cached[0] == mtime:
            return cached[1]
    df = pd.read_csv(path, parse_dates=["timestamp"])
    with _lock:
        _df_cache[key] = (mtime, df)
    return df


MODEL_KEYS = ["scaler", "random_forest", "xgboost", "isolation_forest"]


def models_for(dataset: str) -> dict:
    newest = max(
        (_file_mtime(config.MODELS_DIR / f"{dataset}_{k}.joblib") for k in MODEL_KEYS),
        default=-1.0,
    )
    with _lock:
        cached = _model_cache.get(dataset)
        if cached and cached[0] == newest:
            return cached[1]

    loaded = {}
    for key in MODEL_KEYS:
        path = config.MODELS_DIR / f"{dataset}_{key}.joblib"
        if path.exists():
            loaded[key] = joblib.load(path)

    with _lock:
        _model_cache[dataset] = (newest, loaded)
    return loaded


def load_combined_results() -> pd.DataFrame:
    """Merges every trained dataset's results into one frame tagged with
    `source_dataset`, for the cross-surface Threat Monitor / Incidents views.
    Not cached separately -- it's cheap, built from already-cached frames."""
    frames = []
    for name in available_datasets():
        df = load_results(name).copy()
        df["source_dataset"] = name
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def invalidate(dataset: str) -> None:
    """Drops cached entries for a dataset after a retrain."""
    with _lock:
        _model_cache.pop(dataset, None)
        for d in (_json_cache, _df_cache):
            for key in [k for k in d if dataset in k]:
                d.pop(key, None)
