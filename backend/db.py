"""Persists uploaded/imported datasets to Postgres so anyone testing the
Live Scoring page can come back and see prior imports, and so demo data
survives a redeploy instead of living only in a browser session.

Degrades gracefully to "persistence disabled" when DATABASE_URL isn't set
(e.g. running locally without a database), rather than crashing the app.
"""
import json
import os
import threading
from contextlib import contextmanager
from datetime import datetime, timezone

import pandas as pd

DATABASE_URL = os.environ.get("DATABASE_URL")
_init_lock = threading.Lock()
_initialized = False


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


def init_db() -> bool:
    global _initialized
    if not enabled():
        return False
    with _init_lock:
        if _initialized:
            return True
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
            cur.execute("""
                CREATE TABLE IF NOT EXISTS incident_status (
                    incident_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL DEFAULT 'New',
                    note TEXT,
                    updated_at TIMESTAMPTZ NOT NULL
                )
            """)
        _initialized = True
        return True


def save_import(dataset: str, source_label: str, flag_model: str,
                 scored: pd.DataFrame, flag_col: str, metrics: dict | None) -> int | None:
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


STATUS_CHOICES = ["New", "Investigating", "Resolved", "False Positive"]


def get_statuses(incident_ids: list[str]) -> dict[str, dict]:
    """Returns {incident_id: {"status": ..., "note": ..., "updated_at": ...}}
    for whichever of the given ids have a row; callers default the rest to
    "New" themselves, since most incidents never get touched."""
    if not enabled() or not incident_ids:
        return {}
    init_db()
    with _connection() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT incident_id, status, note, updated_at FROM incident_status WHERE incident_id = ANY(%s)",
            (list(incident_ids),),
        )
        return {row[0]: {"status": row[1], "note": row[2], "updated_at": row[3]} for row in cur.fetchall()}


def set_status(incident_id: str, status: str, note: str | None = None) -> None:
    if not enabled():
        return
    init_db()
    if status not in STATUS_CHOICES:
        status = "New"
    with _connection() as conn, conn.cursor() as cur:
        cur.execute(
            """INSERT INTO incident_status (incident_id, status, note, updated_at)
               VALUES (%s, %s, %s, %s)
               ON CONFLICT (incident_id) DO UPDATE
               SET status = EXCLUDED.status, note = EXCLUDED.note, updated_at = EXCLUDED.updated_at""",
            (incident_id, status, note, datetime.now(timezone.utc)),
        )


def all_statuses(dataset: str) -> dict[str, str]:
    """All touched statuses for a dataset's incidents, keyed by incident_id.
    Cheap because incident_status only ever holds rows someone actually
    triaged, which is a small fraction of total incidents -- untouched
    incidents are implicitly "New" and never get a row here."""
    if not enabled():
        return {}
    init_db()
    with _connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT incident_id, status FROM incident_status WHERE incident_id LIKE %s", (f"{dataset}:%",))
        return dict(cur.fetchall())


def list_imports(limit: int = 15) -> list[dict]:
    if not enabled():
        return []
    init_db()
    with _connection() as conn, conn.cursor() as cur:
        cur.execute(
            """SELECT id, created_at, dataset, source_label, flag_model,
                      row_count, flagged_count, has_labels
               FROM imports ORDER BY created_at DESC LIMIT %s""",
            (limit,),
        )
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]
