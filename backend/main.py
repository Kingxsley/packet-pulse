"""Packet Pulse: FastAPI + server-rendered HTML replacement for the retired
Streamlit dashboard (legacy/streamlit_dashboard/). Reuses src/ untouched.
"""
import sys
import threading
import uuid
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sklearn.metrics import f1_score, precision_score, recall_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend import charts, data_access as da, db  # noqa: E402
from backend.scoring import (MODEL_COLORS, MODEL_ORDER, apply_thresholds,  # noqa: E402
                              outcome_counts, pred_col, score_col)
from src import config  # noqa: E402
from src.features import engineer_features  # noqa: E402

DATASET_LABELS = {
    "dns": "DNS capture (306k rows)",
    "dos": "DoS capture, raw (112k rows)",
    "dos_clean": "DoS capture, cleaned (112k rows)",
}
LIVE_REQUIRED_COLS = ["source_ip", "dest_ip", "source_port", "dest_port",
                       "protocol", "packet_length", "timestamp", "inter_arrival_time"]

app = FastAPI(title="Packet Pulse")
app.mount("/static", StaticFiles(directory=str(Path(__file__).parent / "static")), name="static")
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

_jobs: dict[str, dict] = {}


def _base_ctx(request: Request, dataset: str, active: str) -> dict:
    return {
        "request": request, "dataset": dataset, "datasets": da.available_datasets(),
        "dataset_labels": DATASET_LABELS, "active": active, "base_path": request.url.path,
    }


def _validate_dataset(dataset: str) -> str:
    available = da.available_datasets()
    if not available:
        raise HTTPException(503, "No trained datasets found. Run `python run.py --dataset dos_clean` first.")
    if dataset not in available:
        dataset = available[0]
    return dataset


@app.get("/", response_class=RedirectResponse)
def root():
    available = da.available_datasets()
    default = "dos_clean" if "dos_clean" in available else (available[0] if available else "dos_clean")
    return RedirectResponse(f"/overview?dataset={default}")


@app.get("/overview", response_class=HTMLResponse)
def overview(request: Request, dataset: str = "dos_clean"):
    dataset = _validate_dataset(dataset)
    metrics = da.metrics(dataset)
    ctx = _base_ctx(request, dataset, "overview")
    ctx.update(summary=da.summary(dataset), metrics=metrics,
                comparison_chart=charts.comparison_bar(metrics))
    return templates.TemplateResponse(request, "overview.html", ctx)


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
def api_compare(dataset: str, rf: float = 0.5, xgb: float = 0.5, iso: float = 20, ae: float = 5):
    dataset = _validate_dataset(dataset)
    df = da.load_results(dataset)
    test_df = apply_thresholds(df[df["split"] == "test"], {
        "Random Forest": rf, "XGBoost": xgb, "Isolation Forest": iso, "Autoencoder": ae,
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
    if "autoencoder" in artifacts:
        from src.models import autoencoder_scores
        scored["score_autoencoder"] = autoencoder_scores(artifacts["autoencoder"], X_scaled)

    scored = apply_thresholds(scored, {"Random Forest": 0.5, "XGBoost": 0.5, "Isolation Forest": 20, "Autoencoder": 5})

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

    preview_cols = ["timestamp", "source_ip", "dest_ip", "protocol", "packet_length", "Flagged"] + \
        (["label", "Outcome"] if has_labels else [])
    preview = scored[preview_cols].head(200).astype(str).to_dict(orient="records")

    return JSONResponse({
        "rows": len(scored), "flagged": int(scored[flag_pc].sum()), "has_labels": has_labels,
        "counts": counts, "metrics": metrics_out, "preview": preview, "import_id": import_id,
    })


def _run_retrain(job_id: str, dataset: str, tune: bool, autoencoder: bool):
    from src.pipeline import run as run_pipeline
    try:
        run_pipeline(dataset, tune=tune, train_autoencoder=autoencoder)
        da.invalidate(dataset)
        _jobs[job_id] = {"status": "done"}
    except Exception as e:
        _jobs[job_id] = {"status": "error", "message": str(e)}


@app.post("/api/retrain")
def api_retrain(dataset: str = Form(...), tune: bool = Form(False), autoencoder: bool = Form(True)):
    dataset = dataset if dataset in config.DATASET_FILES else "dos_clean"
    job_id = str(uuid.uuid4())
    _jobs[job_id] = {"status": "running"}
    threading.Thread(target=_run_retrain, args=(job_id, dataset, tune, autoencoder), daemon=True).start()
    return JSONResponse({"job_id": job_id})


@app.get("/api/jobs/{job_id}")
def api_job_status(job_id: str):
    return JSONResponse(_jobs.get(job_id, {"status": "unknown"}))


@app.get("/health")
def health():
    return {"status": "ok"}
