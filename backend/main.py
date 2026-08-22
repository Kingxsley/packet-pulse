"""Packet Pulse: FastAPI + server-rendered HTML replacement for the retired
Streamlit dashboard (legacy/streamlit_dashboard/). Reuses src/ untouched.
"""
import hashlib
import sys
import threading
import uuid
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sklearn.metrics import f1_score, precision_score, recall_score
from starlette.middleware.sessions import SessionMiddleware

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend import alerts, auth, charts, data_access as da, db, siem  # noqa: E402
from backend.scoring import (DEFAULT_THRESHOLDS, MODEL_COLORS, MODEL_ORDER,  # noqa: E402
                              apply_thresholds, outcome_counts, pred_col,
                              score_col, severity_tier)
from src import config  # noqa: E402
from src.features import engineer_features  # noqa: E402

DATASET_LABELS = {
    "dns": "DNS capture (306k rows)",
    "dos_clean": "DoS capture, cleaned (112k rows)",
    "combined": "Combined: DNS + DoS",
}
LIVE_REQUIRED_COLS = ["source_ip", "dest_ip", "source_port", "dest_port",
                       "protocol", "packet_length", "timestamp", "inter_arrival_time"]

app = FastAPI(title="Packet Pulse")
app.mount("/static", StaticFiles(directory=str(Path(__file__).parent / "static")), name="static")
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

_jobs: dict[str, dict] = {}


@app.middleware("http")
async def require_login(request: Request, call_next):
    path = request.url.path
    if not auth.enabled() or not auth.is_protected(path):
        return await call_next(request)
    if request.session.get("authed"):
        return await call_next(request)
    if path.startswith("/api/"):
        return JSONResponse({"error": "Login required."}, status_code=401)
    return RedirectResponse(f"/login?next={path}")


# Added after the auth-check middleware above so it ends up as the
# outermost layer (Starlette runs the most-recently-added middleware
# first), meaning request.session is already populated by the time
# require_login reads it.
app.add_middleware(SessionMiddleware, secret_key=auth.SESSION_SECRET, same_site="lax")


@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request, next: str = "/"):
    return templates.TemplateResponse(request, "login.html", {"next": next, "error": False})


@app.post("/login")
def login_submit(request: Request, password: str = Form(...), next: str = Form("/")):
    if auth.check_password(password):
        request.session["authed"] = True
        return RedirectResponse(next or "/", status_code=303)
    return templates.TemplateResponse(request, "login.html", {"next": next, "error": True}, status_code=401)


@app.post("/logout")
def logout(request: Request):
    request.session.clear()
    return RedirectResponse("/login", status_code=303)


def _selectable_datasets() -> list[str]:
    available = da.available_datasets()
    return available + ["combined"] if len(available) > 1 else available


def _base_ctx(request: Request, dataset: str, active: str) -> dict:
    return {
        "request": request, "dataset": dataset, "datasets": _selectable_datasets(),
        "dataset_labels": DATASET_LABELS, "active": active, "base_path": request.url.path,
        "auth_enabled": auth.enabled(), "authed": bool(request.session.get("authed")),
    }


def _validate_dataset(dataset: str, allow_combined: bool = False) -> str:
    available = da.available_datasets()
    if not available:
        raise HTTPException(503, "No trained datasets found. Run `python run.py --dataset dos_clean` first.")
    if allow_combined and dataset == "combined" and len(available) > 1:
        return "combined"
    if dataset not in available:
        dataset = available[0]
    return dataset


def _results_for(dataset: str) -> pd.DataFrame:
    return da.load_combined_results() if dataset == "combined" else da.load_results(dataset)


@app.get("/", response_class=RedirectResponse)
def root():
    available = da.available_datasets()
    default = "dos_clean" if "dos_clean" in available else (available[0] if available else "dos_clean")
    return RedirectResponse(f"/monitor?dataset={default}")


@app.get("/monitor", response_class=HTMLResponse)
def monitor(request: Request, dataset: str = "dos_clean"):
    dataset = _validate_dataset(dataset, allow_combined=True)
    flagged = _incidents_df(dataset, "Random Forest")
    top_attackers = (
        flagged["source_ip"].value_counts().head(5).reset_index()
        if not flagged.empty else pd.DataFrame(columns=["source_ip", "count"])
    )
    top_attackers.columns = ["source_ip", "count"]
    ctx = _base_ctx(request, dataset, "monitor")
    ctx.update(model_order=MODEL_ORDER, top_attackers=top_attackers.to_dict(orient="records"),
                total_incidents=len(flagged))
    return templates.TemplateResponse(request, "monitor.html", ctx)


def _row_payload(row, pc: str, sc: str) -> dict:
    return {
        "timestamp": row["timestamp"].isoformat(),
        "source_ip": row["source_ip"], "dest_ip": row["dest_ip"],
        "source_port": int(row["source_port"]), "dest_port": int(row["dest_port"]),
        "protocol": row["protocol"], "packet_length": int(row["packet_length"]),
        "source_dataset": row.get("source_dataset"),
        "label": int(row["label"]) if "label" in row and pd.notna(row["label"]) else None,
        "flagged": int(row[pc]) if pc in row and pd.notna(row[pc]) else None,
        "score": float(row[sc]) if sc in row and pd.notna(row[sc]) else None,
    }


@app.get("/api/monitor-stream")
async def monitor_stream(request: Request, dataset: str, flag_model: str = "Random Forest", speed: int = 8):
    import asyncio
    import json

    dataset = _validate_dataset(dataset, allow_combined=True)
    if flag_model not in MODEL_ORDER:
        flag_model = "Random Forest"
    pc, sc = pred_col(flag_model), score_col(flag_model)
    interval = 1.0 / max(min(speed, 50), 1)

    if dataset == "combined":
        # Round-robin between each source dataset so both attack surfaces
        # are visibly "live" together, instead of playing one capture
        # start-to-finish and only then starting the other.
        sources = [
            apply_thresholds(da.load_results(name), DEFAULT_THRESHOLDS)
            .sort_values("timestamp").reset_index(drop=True).assign(source_dataset=name)
            for name in da.available_datasets()
        ]
    else:
        sources = [apply_thresholds(da.load_results(dataset), DEFAULT_THRESHOLDS)
                   .sort_values("timestamp").reset_index(drop=True)]

    async def gen():
        counters = [0] * len(sources)
        src_i = 0
        while True:
            if await request.is_disconnected():
                break
            df = sources[src_i]
            row = df.iloc[counters[src_i] % len(df)]
            counters[src_i] += 1
            src_i = (src_i + 1) % len(sources)
            payload = _row_payload(row, pc, sc)
            if payload["flagged"] and severity_tier(payload["score"]) == "Critical":
                alerts.alert_critical_packet(
                    _incident_id(dataset, payload), DATASET_LABELS.get(dataset, dataset), flag_model,
                    payload["source_ip"], payload["dest_ip"], payload["score"] or 0.0,
                )
            yield f"data: {json.dumps(payload)}\n\n"
            await asyncio.sleep(interval)

    return StreamingResponse(gen(), media_type="text/event-stream", headers={
        "Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive",
    })


@app.get("/overview", response_class=HTMLResponse)
def overview(request: Request, dataset: str = "dos_clean"):
    dataset = _validate_dataset(dataset)
    metrics = da.metrics(dataset)
    ctx = _base_ctx(request, dataset, "overview")
    ctx.update(summary=da.summary(dataset), metrics=metrics,
                comparison_chart=charts.comparison_bar(metrics))
    return templates.TemplateResponse(request, "overview.html", ctx)


@app.get("/settings", response_class=HTMLResponse)
def settings(request: Request, dataset: str = "dos_clean"):
    dataset = _validate_dataset(dataset)
    ctx = _base_ctx(request, dataset, "settings")
    ctx.update(summary=da.summary(dataset), alerts_enabled=alerts.enabled())
    return templates.TemplateResponse(request, "settings.html", ctx)


@app.get("/explore", response_class=HTMLResponse)
def explore(request: Request, dataset: str = "dos_clean", highlight: str = "XGBoost"):
    dataset = _validate_dataset(dataset)
    if highlight not in MODEL_ORDER:
        highlight = "XGBoost"
    df = da.load_results(dataset)
    plot_df = df.sample(15000, random_state=config.RANDOM_STATE) if len(df) > 15000 else df

    dist_charts = [charts.feature_distribution(plot_df, f) for f in config.FEATURES]
    anomaly_col = f"anomaly_{highlight.lower().replace(' ', '_')}"
    ts_chart = charts.time_series(plot_df.sort_values("timestamp"),
                                   {anomaly_col: (highlight, MODEL_COLORS[highlight])})
    scatter_chart = charts.scatter(plot_df)

    ctx = _base_ctx(request, dataset, "explore")
    ctx.update(row_count=len(df), features=config.FEATURES, model_order=MODEL_ORDER, highlight=highlight,
                dist_charts=dist_charts, timeseries_chart=ts_chart, scatter_chart=scatter_chart)
    return templates.TemplateResponse(request, "explore.html", ctx)


@app.get("/compare", response_class=HTMLResponse)
def compare(request: Request, dataset: str = "dos_clean"):
    dataset = _validate_dataset(dataset)
    return templates.TemplateResponse(request, "compare.html", _base_ctx(request, dataset, "compare"))


@app.get("/api/compare")
def api_compare(dataset: str, rf: float = 0.5, xgb: float = 0.5, iso: float = 20):
    dataset = _validate_dataset(dataset)
    df = da.load_results(dataset)
    test_df = apply_thresholds(df[df["split"] == "test"], {
        "Random Forest": rf, "XGBoost": xgb, "Isolation Forest": iso,
    })

    metrics_rows, confusion, models_with_scores = [], {}, []
    for model in MODEL_ORDER:
        pc = pred_col(model)
        if pc not in test_df:
            continue
        y_true, y_pred = test_df["label"], test_df[pc]
        counts = outcome_counts(y_true, y_pred)
        metrics_rows.append({
            "model": model,
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "tp": counts["True Positive"], "fp": counts["False Positive"], "fn": counts["False Negative"],
        })
        confusion[model.lower().replace(" ", "_")] = charts.confusion_fig(y_true, y_pred, model)
        if score_col(model) in test_df:
            models_with_scores.append(model)

    roc_fig, pr_fig = charts.roc_pr_figs(test_df, models_with_scores)
    return JSONResponse({"metrics": metrics_rows, "confusion": confusion, "roc": roc_fig, "pr": pr_fig})


@app.get("/importance", response_class=HTMLResponse)
def importance(request: Request, dataset: str = "dos_clean"):
    dataset = _validate_dataset(dataset)
    shap_values = da.shap_importance(dataset)
    ctx = _base_ctx(request, dataset, "importance")
    ctx.update(chart=charts.shap_bar(shap_values))
    return templates.TemplateResponse(request, "importance.html", ctx)


@app.get("/live-scoring", response_class=HTMLResponse)
def live_scoring(request: Request, dataset: str = "dos_clean"):
    dataset = _validate_dataset(dataset)
    ctx = _base_ctx(request, dataset, "live")
    ctx.update(db_enabled=db.enabled(), history=db.list_imports() if db.enabled() else [])
    return templates.TemplateResponse(request, "live_scoring.html", ctx)


@app.get("/api/history", response_class=HTMLResponse)
def api_history(request: Request, dataset: str = "dos_clean"):
    return templates.TemplateResponse(request, "_history_table.html", {"history": db.list_imports()})


@app.post("/api/live-score")
async def api_live_score(dataset: str = Form(...), flag_model: str = Form(...),
                          file: UploadFile | None = File(None)):
    dataset = _validate_dataset(dataset)

    if file is not None:
        try:
            content = await file.read()
            import io
            input_df = (pd.read_json(io.BytesIO(content)) if file.filename.endswith(".json")
                        else pd.read_csv(io.BytesIO(content)))
            source_label = file.filename
        except Exception as e:
            return JSONResponse({"error": f"Could not parse file: {e}"}, status_code=400)
    else:
        raw = da.load_results(dataset)
        cols = LIVE_REQUIRED_COLS + (["label"] if "label" in raw else [])
        input_df = raw[cols].sample(min(300, len(raw))).reset_index(drop=True)
        source_label = "Random sample"

    missing = [c for c in LIVE_REQUIRED_COLS if c not in input_df.columns]
    if missing:
        return JSONResponse({"error": f"Missing required columns: {missing}"}, status_code=400)

    input_df["timestamp"] = pd.to_datetime(input_df["timestamp"])
    featured = engineer_features(input_df.sort_values("timestamp").reset_index(drop=True))
    artifacts = da.models_for(dataset)
    if "scaler" not in artifacts:
        return JSONResponse({"error": "No trained models found for this dataset."}, status_code=400)

    X_scaled = artifacts["scaler"].transform(featured[config.FEATURES])
    scored = featured.copy()
    if "random_forest" in artifacts:
        scored["score_random_forest"] = artifacts["random_forest"].predict_proba(X_scaled)[:, 1]
    if "xgboost" in artifacts:
        scored["score_xgboost"] = artifacts["xgboost"].predict_proba(X_scaled)[:, 1]
    if "isolation_forest" in artifacts:
        scored["score_isolation_forest"] = -artifacts["isolation_forest"].score_samples(X_scaled)

    scored = apply_thresholds(scored, DEFAULT_THRESHOLDS)

    flag_pc = pred_col(flag_model)
    if flag_pc not in scored:
        return JSONResponse({"error": f"{flag_model} isn't available for this dataset."}, status_code=400)
    scored["Flagged"] = scored[flag_pc].map({1: "Attack", 0: "Normal"})
    has_labels = "label" in scored

    from backend.scoring import outcomes
    counts, metrics_out, import_id = None, None, None
    if has_labels:
        scored["Outcome"] = outcomes(scored["label"], scored[flag_pc])
        counts = outcome_counts(scored["label"], scored[flag_pc])
        metrics_out = {}
        for model in MODEL_ORDER:
            pc = pred_col(model)
            if pc in scored:
                metrics_out[model] = {
                    "precision": float(precision_score(scored["label"], scored[pc], zero_division=0)),
                    "recall": float(recall_score(scored["label"], scored[pc], zero_division=0)),
                    "f1": float(f1_score(scored["label"], scored[pc], zero_division=0)),
                }

    if db.enabled():
        import_id = db.save_import(dataset, source_label, flag_model, scored, flag_pc, metrics_out)

    flag_sc = score_col(flag_model)
    if alerts.enabled() and flag_sc in scored:
        critical_count = int((scored.loc[scored[flag_pc] == 1, flag_sc].map(severity_tier) == "Critical").sum())
        alerts.alert_scan_summary(source_label, DATASET_LABELS.get(dataset, dataset), flag_model,
                                   critical_count, len(scored))

    preview_cols = ["timestamp", "source_ip", "dest_ip", "protocol", "packet_length", "Flagged"] + \
        (["label", "Outcome"] if has_labels else [])
    preview = scored[preview_cols].head(200).astype(str).to_dict(orient="records")

    return JSONResponse({
        "rows": len(scored), "flagged": int(scored[flag_pc].sum()), "has_labels": has_labels,
        "counts": counts, "metrics": metrics_out, "preview": preview, "import_id": import_id,
    })


_retrain_locks: dict[str, threading.Lock] = {}
_retrain_locks_guard = threading.Lock()


def _retrain_lock(dataset: str) -> threading.Lock:
    with _retrain_locks_guard:
        return _retrain_locks.setdefault(dataset, threading.Lock())


def _dataset_artifact_paths(dataset: str) -> list[Path]:
    """Every file a dataset currently owns under outputs/, for backup/restore."""
    return list(config.RESULTS_DIR.glob(f"{dataset}_*")) + list(config.MODELS_DIR.glob(f"{dataset}_*"))


def _backup_dataset(dataset: str) -> Path:
    import shutil
    backup_dir = config.OUTPUT_DIR / "_retrain_backup" / dataset
    if backup_dir.exists():
        shutil.rmtree(backup_dir)
    backup_dir.mkdir(parents=True, exist_ok=True)
    for p in _dataset_artifact_paths(dataset):
        shutil.copy2(p, backup_dir / p.name)
    return backup_dir


def _restore_dataset(dataset: str, backup_dir: Path) -> None:
    import shutil
    if not backup_dir.exists():
        return
    for p in _dataset_artifact_paths(dataset):
        p.unlink(missing_ok=True)
    for p in backup_dir.iterdir():
        dest_dir = config.MODELS_DIR if p.suffix in (".joblib", ".keras") else config.RESULTS_DIR
        shutil.copy2(p, dest_dir / p.name)


def _run_retrain(job_id: str, dataset: str, tune: bool, autoencoder: bool):
    from src.pipeline import run as run_pipeline

    lock = _retrain_lock(dataset)
    if not lock.acquire(blocking=False):
        _jobs[job_id] = {"status": "error", "message": f"A retrain for {dataset} is already running."}
        return
    try:
        backup_dir = _backup_dataset(dataset)
        try:
            run_pipeline(dataset, tune=tune, train_autoencoder=autoencoder)
            da.invalidate(dataset)
            _jobs[job_id] = {"status": "done"}
        except Exception as e:
            _restore_dataset(dataset, backup_dir)
            da.invalidate(dataset)
            _jobs[job_id] = {"status": "error", "message": f"Training failed, restored the previous model: {e}"}
    finally:
        lock.release()


@app.post("/api/retrain")
def api_retrain(dataset: str = Form(...), tune: bool = Form(False), autoencoder: bool = Form(False)):
    dataset = dataset if dataset in config.DATASET_FILES else "dos_clean"
    job_id = str(uuid.uuid4())
    _jobs[job_id] = {"status": "running"}
    threading.Thread(target=_run_retrain, args=(job_id, dataset, tune, autoencoder), daemon=True).start()
    return JSONResponse({"job_id": job_id})


@app.get("/api/jobs/{job_id}")
def api_job_status(job_id: str):
    return JSONResponse(_jobs.get(job_id, {"status": "unknown"}))


INCIDENT_COLS = ["timestamp", "source_ip", "source_port", "dest_ip", "dest_port",
                  "protocol", "packet_length", "label", "score", "source_dataset"]


def _incident_id(dataset: str, row: dict) -> str:
    """Identity for a flagged packet, independent of which model you're
    currently viewing it through -- triage status on an incident shouldn't
    reset just because you switched the flag model."""
    ds = row.get("source_dataset") or dataset
    key = f"{ds}|{row['timestamp']}|{row['source_ip']}|{row['source_port']}|{row['dest_ip']}|{row['dest_port']}"
    return f"{dataset}:{hashlib.md5(key.encode()).hexdigest()[:12]}"


def _incidents_df(dataset: str, flag_model: str) -> pd.DataFrame:
    df = apply_thresholds(_results_for(dataset), DEFAULT_THRESHOLDS)
    pc, sc = pred_col(flag_model), score_col(flag_model)
    if pc not in df:
        return df.iloc[0:0]
    flagged = df[df[pc] == 1].copy()
    flagged["score"] = flagged[sc] if sc in flagged else None
    flagged["severity"] = flagged["score"].map(severity_tier)
    flagged["incident_id"] = [
        _incident_id(dataset, r) for r in flagged[["timestamp", "source_ip", "source_port",
                                                     "dest_ip", "dest_port"] +
                                                    (["source_dataset"] if "source_dataset" in flagged else [])
                                                    ].to_dict(orient="records")
    ]
    return flagged.sort_values("timestamp", ascending=False)


@app.get("/incidents", response_class=HTMLResponse)
def incidents(request: Request, dataset: str = "dos_clean", flag_model: str = "Random Forest",
              status: str = "", page: int = 1):
    dataset = _validate_dataset(dataset, allow_combined=True)
    if flag_model not in MODEL_ORDER:
        flag_model = "Random Forest"
    df = _incidents_df(dataset, flag_model)

    touched = db.all_statuses(dataset) if db.enabled() else {}
    if status:
        if status == "New":
            df = df[~df["incident_id"].isin(touched)]
        else:
            keep_ids = {i for i, s in touched.items() if s == status}
            df = df[df["incident_id"].isin(keep_ids)]

    page_size = 50
    total = len(df)
    total_pages = max(1, -(-total // page_size))
    page = max(1, min(page, total_pages))
    page_df = df.iloc[(page - 1) * page_size: page * page_size].copy()
    cols = [c for c in INCIDENT_COLS if c in page_df.columns] + ["severity", "incident_id"]
    if "timestamp" in page_df:
        page_df["timestamp"] = page_df["timestamp"].astype(str)
    rows = page_df[cols].where(pd.notna(page_df[cols]), None).to_dict(orient="records")
    for r in rows:
        r["status"] = touched.get(r["incident_id"], "New")

    ctx = _base_ctx(request, dataset, "incidents")
    ctx.update(flag_model=flag_model, model_order=MODEL_ORDER, rows=rows, total=total,
                page=page, total_pages=total_pages, has_source_col="source_dataset" in df.columns,
                status_filter=status, status_choices=db.STATUS_CHOICES, db_enabled=db.enabled())
    return templates.TemplateResponse(request, "incidents.html", ctx)


@app.post("/api/incidents/status")
def update_incident_status(incident_id: str = Form(...), status: str = Form(...), note: str = Form("")):
    if not db.enabled():
        return JSONResponse({"ok": False, "message": "Postgres not configured."}, status_code=503)
    db.set_status(incident_id, status, note or None)
    return JSONResponse({"ok": True, "incident_id": incident_id, "status": status})


@app.get("/api/incidents/export.csv")
def export_incidents_csv(dataset: str = "dos_clean", flag_model: str = "Random Forest", limit: int = 5000):
    dataset = _validate_dataset(dataset, allow_combined=True)
    df = _incidents_df(dataset, flag_model).head(limit)
    cols = [c for c in INCIDENT_COLS if c in df.columns] + ["severity", "incident_id"]
    return PlainTextResponse(df[cols].to_csv(index=False), media_type="text/csv", headers={
        "Content-Disposition": f'attachment; filename="{dataset}_incidents.csv"',
    })


@app.get("/api/incidents/export.cef")
def export_incidents_cef(dataset: str = "dos_clean", flag_model: str = "Random Forest", limit: int = 5000):
    dataset = _validate_dataset(dataset, allow_combined=True)
    df = _incidents_df(dataset, flag_model).head(limit)
    return PlainTextResponse(siem.to_cef(df, flag_model), media_type="text/plain", headers={
        "Content-Disposition": f'attachment; filename="{dataset}_incidents.cef"',
    })


@app.post("/api/incidents/forward-siem")
def forward_siem(dataset: str = Form(...), flag_model: str = Form(...),
                  hec_url: str = Form(...), hec_token: str = Form(...), limit: int = Form(200)):
    dataset = _validate_dataset(dataset, allow_combined=True)
    df = _incidents_df(dataset, flag_model).head(min(limit, 500))
    if df.empty:
        return JSONResponse({"ok": False, "count": 0, "body": "No incidents matched this model/dataset."})
    events = siem.to_hec_events(df, flag_model)
    result = siem.send_to_hec(hec_url, hec_token, events)
    result["count"] = len(events)
    return JSONResponse(result)


@app.get("/health")
def health():
    return {"status": "ok"}
