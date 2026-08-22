"""Best-effort webhook alerting for Critical-severity flags.

Posts Slack-compatible {"text": ...} payloads (works directly with Slack
incoming webhooks; most other chat/webhook tools accept the same shape or
ignore the extra field). Configured via ALERT_WEBHOOK_URL only -- not
exposed as a UI setting, since where security alerts go is an admin/infra
decision, not something any logged-in visitor should be able to redirect.

Disabled entirely when the env var isn't set. Failures are swallowed: a
broken webhook should never take down packet scoring.
"""
import os
import threading

WEBHOOK_URL = os.environ.get("ALERT_WEBHOOK_URL")
_sent_ids: set[str] = set()
_sent_lock = threading.Lock()


def enabled() -> bool:
    return bool(WEBHOOK_URL)


def _post(text: str) -> None:
    import requests
    try:
        requests.post(WEBHOOK_URL, json={"text": text}, timeout=5)
    except requests.RequestException:
        pass


def _fire(text: str) -> None:
    if not enabled():
        return
    threading.Thread(target=_post, args=(text,), daemon=True).start()


def alert_critical_packet(incident_id: str, dataset_label: str, model: str, source_ip: str,
                           dest_ip: str, score: float) -> None:
    """Fires at most once per incident_id per process lifetime, so the live
    Threat Monitor replaying the same capture on a loop doesn't re-alert on
    every pass."""
    with _sent_lock:
        if incident_id in _sent_ids:
            return
        _sent_ids.add(incident_id)
    _fire(f":rotating_light: *Critical* flag on {dataset_label} ({model}): "
          f"{source_ip} -> {dest_ip}, score {score:.3f}. Incident `{incident_id}`.")


def alert_scan_summary(source_label: str, dataset_label: str, model: str, critical_count: int, total: int) -> None:
    if critical_count <= 0:
        return
    _fire(f":rotating_light: Scan of *{source_label}* against {dataset_label} ({model}) found "
          f"*{critical_count}* Critical-severity packet(s) out of {total} scored.")
