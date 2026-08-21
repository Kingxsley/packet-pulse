"""Load raw packet-capture data (JSON or CSV) into a normalized DataFrame.

The original script pulled data from an InfluxDB Cloud bucket (with a hardcoded
API token committed in source -- see README "Fixes" item 1). This project
instead reads the local packet captures that were shipped alongside it, which
also removes the hard dependency on live cloud access to run the pipeline.
"""
import json
from pathlib import Path

import pandas as pd

from . import config


def _finalize(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in config.REQUIRED_RAW_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Available: {list(df.columns)}")

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["label"] = df["label"].astype(int)
    return df


def load_json(path: Path) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8") as f:
        records = json.load(f)
    return _finalize(pd.DataFrame.from_records(records))


def load_csv(path: Path) -> pd.DataFrame:
    return _finalize(pd.read_csv(path))


def load_dataset(name: str) -> pd.DataFrame:
    """name: one of 'dns', 'dos', 'dos_clean'."""
    if name not in config.DATASET_FILES:
        raise ValueError(f"Unknown dataset '{name}'. Options: {list(config.DATASET_FILES)}")
    path = config.DATASET_FILES[name]
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    if path.suffix == ".json":
        return load_json(path)
    return load_csv(path)
