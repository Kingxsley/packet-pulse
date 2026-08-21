"""Persists uploaded/imported datasets to Postgres so anyone testing the
Live Scoring tab can come back and see prior imports, and so demo data
survives a redeploy instead of living only in the browser session.

Degrades gracefully to "persistence disabled" when DATABASE_URL isn't set
(e.g. running locally without a database), rather than crashing the app.
"""
import json
import os
from contextlib import contextmanager
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

DATABASE_URL = os.environ.get("DATABASE_URL")


def enabled() -> bool:
    return bool(DATABASE_URL)


@contextmanager
def _connection():
    import psycopg2
    conn = psycopg2.connect(DATABASE_URL)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


@st.cache_resource(show_spinner=False)
def init_db():
    """Creates the imports table once per app process."""
    if not enabled():
        return False
    with _connection() as conn, conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS imports (
                id SERIAL PRIMARY KEY,
                created_at TIMESTAMPTZ NOT NULL,
                dataset TEXT NOT NULL,
                source_label TEXT NOT NULL,
                flag_model TEXT NOT NULL,
                row_count INTEGER NOT NULL,
                flagged_count INTEGER NOT NULL,
                has_labels BOOLEAN NOT NULL,
                metrics JSONB,
                preview JSONB
            )
        """)
    return True


def save_import(dataset: str, source_label: str, flag_model: str,
                 scored: pd.DataFrame, flag_col: str, metrics: dict | None) -> int | None:
    """Stores a summary + a small preview of a scored upload. Returns the new row id."""
    if not enabled():
        return None
    init_db()

    preview_cols = [c for c in ["timestamp", "source_ip", "dest_ip", "protocol",
                                 "packet_length", "label", "Flagged", "Outcome"] if c in scored]
    preview = scored[preview_cols].head(50).astype(str).to_dict(orient="records")

    with _connection() as conn, conn.cursor() as cur:
        cur.execute(
            """INSERT INTO imports
               (created_at, dataset, source_label, flag_model, row_count,
                flagged_count, has_labels, metrics, preview)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING id""",
            (
                datetime.now(timezone.utc), dataset, source_label, flag_model,
                int(len(scored)), int(scored[flag_col].sum()), bool("label" in scored),
                json.dumps(metrics) if metrics else None, json.dumps(preview),
            ),
        )
        return cur.fetchone()[0]


def list_imports(limit: int = 15) -> pd.DataFrame:
    if not enabled():
        return pd.DataFrame()
    init_db()
    with _connection() as conn:
        return pd.read_sql(
            """SELECT id, created_at, dataset, source_label, flag_model,
                      row_count, flagged_count, has_labels
               FROM imports ORDER BY created_at DESC LIMIT %s""",
            conn, params=(limit,),
        )


def load_import(import_id: int) -> dict | None:
    if not enabled():
        return None
    with _connection() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT dataset, source_label, flag_model, metrics, preview FROM imports WHERE id = %s",
            (import_id,),
        )
        row = cur.fetchone()
    if row is None:
        return None
    dataset, source_label, flag_model, metrics, preview = row
    return {
        "dataset": dataset, "source_label": source_label, "flag_model": flag_model,
        "metrics": metrics, "preview": pd.DataFrame(preview),
    }
