"""SIEM interoperability: Common Event Format (CEF) export and a Splunk
HTTP Event Collector (HEC) compatible forwarder.

CEF is the de facto standard most SIEMs (Splunk, QRadar, ArcSight,
LogRhythm) can parse natively or via a generic CEF input. HEC is Splunk's
own documented ingestion API (POST JSON to <url>/services/collector/event
with an `Authorization: Splunk <token>` header) -- any tool that speaks
HEC (there are compatible collectors beyond Splunk itself) works the same
way. We haven't tested this against a live Splunk instance (don't have
one), but the request shapes below match Splunk's published HEC spec
exactly, so a real HEC token + URL should just work.
"""
import json

import pandas as pd


def _severity(score) -> int:
    """CEF severity is 0-10. Our scores aren't all probabilities (Isolation
    Forest/Autoencoder emit unbounded anomaly scores), so this clamps
    whatever we have into range rather than assuming a 0-1 probability."""
    try:
        s = float(score)
    except (TypeError, ValueError):
        return 5
    return max(0, min(10, round(s * 10) if 0 <= s <= 1 else 7))


def to_cef_line(row: dict, flag_model: str) -> str:
    severity = _severity(row.get("score"))
    dataset = row.get("source_dataset", row.get("dataset", ""))
    fields = {
        "src": row.get("source_ip", ""),
        "spt": row.get("source_port", ""),
        "dst": row.get("dest_ip", ""),
        "dpt": row.get("dest_port", ""),
        "proto": row.get("protocol", ""),
        "act": "flagged",
        "rt": row.get("timestamp", ""),
    }
    labeled = {"cs1": ("ModelScore", row.get("score", "")),
               "cs2": ("Dataset", dataset),
               "cs3": ("TrueLabel", row.get("label", ""))}
    ext = " ".join(f"{k}={v}" for k, v in fields.items() if v != "" and v is not None)
    for key, (label, value) in labeled.items():
        if value != "" and value is not None:
            ext += f" {key}={value} {key}Label={label}"
    model_id = flag_model.lower().replace(" ", "_")
    return f"CEF:0|PacketPulse|AnomalyDetection|1.0|{model_id}|Anomalous traffic flagged|{severity}|{ext}"


def _records(df: pd.DataFrame) -> list[dict]:
    """Timestamps come through as pandas Timestamp objects, which json.dumps
    chokes on -- stringify before converting to records."""
    df = df.copy()
    if "timestamp" in df:
        df["timestamp"] = df["timestamp"].astype(str)
    return df.to_dict(orient="records")


def to_cef(df: pd.DataFrame, flag_model: str) -> str:
    return "\n".join(to_cef_line(row, flag_model) for row in _records(df))


def to_hec_events(df: pd.DataFrame, flag_model: str, sourcetype: str = "packetpulse:incident") -> list[dict]:
    """Splunk HEC event batch: https://docs.splunk.com/Documentation/Splunk/latest/Data/FormateventsforHTTPEventCollector"""
    events = []
    for row in _records(df):
        events.append({
            "sourcetype": sourcetype,
            "event": {
                "model": flag_model,
                "source_ip": row.get("source_ip"), "source_port": row.get("source_port"),
                "dest_ip": row.get("dest_ip"), "dest_port": row.get("dest_port"),
                "protocol": row.get("protocol"), "packet_length": row.get("packet_length"),
                "score": row.get("score"), "dataset": row.get("source_dataset", row.get("dataset")),
                "true_label": row.get("label"), "timestamp": row.get("timestamp"),
            },
        })
    return events


def send_to_hec(hec_url: str, hec_token: str, events: list[dict], timeout: int = 10) -> dict:
    import requests

    url = hec_url.rstrip("/")
    if not url.endswith("/services/collector/event"):
        url = f"{url}/services/collector/event"
    body = "\n".join(json.dumps(e) for e in events)
    try:
        resp = requests.post(
            url, data=body, timeout=timeout,
            headers={"Authorization": f"Splunk {hec_token}", "Content-Type": "application/json"},
        )
        return {"ok": resp.ok, "status_code": resp.status_code, "body": resp.text[:500]}
    except requests.RequestException as e:
        return {"ok": False, "status_code": None, "body": str(e)}
