"""Feature engineering for network traffic anomaly detection."""
import numpy as np
import pandas as pd

from . import config


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive model features from raw packet fields.

    Fixes vs. the original script (see README "Fixes"):
      - inter_arrival_time is unconditionally floored (the original only
        clipped when *any* value was <= 0, which is fragile -- a single
        malformed record shouldn't gate whether clipping happens at all).
      - request_rate infinities/NaNs are resolved the same way for every
        dataset instead of relying on it happening to work out.
      - packet_rate replaces the InfluxDB-only "dns_rate" field the original
        script required but never actually used in its `features` list --
        it was dead validation code against a column that doesn't exist in
        these raw packet captures and would crash immediately.
    """
    df = df.copy()

    df["inter_arrival_time"] = df["inter_arrival_time"].clip(lower=config.MIN_INTER_ARRIVAL)

    df["request_rate"] = 1.0 / df["inter_arrival_time"]
    df["request_rate"] = df["request_rate"].replace([np.inf, -np.inf], np.nan)
    df["request_rate"] = df["request_rate"].fillna(df["request_rate"].median())

    packet_rate = (
        df.set_index("timestamp")["packet_length"]
        .rolling(config.RATE_WINDOW)
        .count()
    )
    df["packet_rate"] = packet_rate.to_numpy()

    for col in config.FEATURES:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    return df
