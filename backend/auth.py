"""Shared-password session gate.

Not real multi-user auth (no accounts, no roles) -- a single password that
guards the one thing on this app that can destroy state: retraining a
model (Settings) overwrites the artifacts everyone's traffic is being
scored against. Everything else -- watching the live Dashboard, running a
Scan, triaging Incidents -- stays open so anyone can try the product
without an account.

Disabled entirely when APP_PASSWORD isn't set, so local dev
(`uvicorn backend.main:app --reload`) never needs it configured.
"""
import os

APP_PASSWORD = os.environ.get("APP_PASSWORD")
SESSION_SECRET = os.environ.get("SESSION_SECRET") or APP_PASSWORD or "dev-only-insecure-secret"

# Only these need a login. Everything else on the app is intentionally public.
PROTECTED_PATHS = {"/settings"}
PROTECTED_PREFIXES = ("/api/retrain",)


def enabled() -> bool:
    return bool(APP_PASSWORD)


def is_protected(path: str) -> bool:
    return path in PROTECTED_PATHS or path.startswith(PROTECTED_PREFIXES)


def check_password(candidate: str) -> bool:
    return enabled() and candidate == APP_PASSWORD
