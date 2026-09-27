"""
config.py  -  Central model + runtime configuration. Change via .env.
"""

from __future__ import annotations
import os
from dotenv import find_dotenv, load_dotenv


def _load_env_files() -> None:
    """Load .env settings. Earlier sources win; real environment variables beat all.

    1. .env in the directory you run the command from (or a parent of it).
       A bare load_dotenv() searches upward from THIS file instead, which stops
       working once the package is installed: it looks in site-packages and
       silently falls back to mock mode.
    2. $SNAP_USER_DATA/.env when running as a snap - a per-user location that
       strict confinement always allows.
    3. .env next to the source tree, for running from a checkout.
    """
    candidates = [find_dotenv(usecwd=True)]
    if os.getenv("SNAP_USER_DATA"):
        candidates.append(os.path.join(os.environ["SNAP_USER_DATA"], ".env"))
    candidates.append(find_dotenv())
    for path in candidates:
        if path and os.path.isfile(path):
            load_dotenv(path)   # never overrides values that are already set


_load_env_files()

USE_MOCK_LLM: bool = os.getenv("USE_MOCK_LLM", "true").strip().lower() == "true"
MODEL_PROVIDER: str = os.getenv("MODEL_PROVIDER", "anthropic")
DEFAULT_MODEL: str = os.getenv("DEFAULT_MODEL", "claude-3-5-haiku-latest")

# Anchor for "today" so "before my 1:1" is answerable. Defaults to the mock
# calendar's day. Set to a real date (or wire to datetime.now) when you add a
# real calendar.
REFERENCE_DATE: str = os.getenv("REFERENCE_DATE", "2026-07-08")


def model_name_for(role: str) -> str:
    return os.getenv(f"{role.upper()}_MODEL", DEFAULT_MODEL)


def get_model(role: str):
    from langchain.chat_models import init_chat_model
    return init_chat_model(model_name_for(role), model_provider=MODEL_PROVIDER)
