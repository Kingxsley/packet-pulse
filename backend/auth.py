"""Shared-password session gate.

Not real multi-user auth (no accounts, no roles) -- a single password that
closes the actual hole this app had while public: anyone with the URL
could wipe the live model via Retrain, or edit anyone else's incident
triage status. A password gate is the right size fix for "one team behind
one login," not "many customers with different permissions."

Disabled entirely when APP_PASSWORD isn't set, so local dev
(`uvicorn backend.main:app --reload`) never needs it configured.
"""
import os

APP_PASSWORD = os.environ.get("APP_PASSWORD")
SESSION_SECRET = os.environ.get("SESSION_SECRET") or APP_PASSWORD or "dev-only-insecure-secret"

PUBLIC_PATHS = {"/login", "/health"}


def enabled() -> bool:
    return bool(APP_PASSWORD)


def is_public(path: str) -> bool:
    return path in PUBLIC_PATHS or path.startswith("/static")


def check_password(candidate: str) -> bool:
    return enabled() and candidate == APP_PASSWORD
