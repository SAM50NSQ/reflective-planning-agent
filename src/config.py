"""
config.py  -  Central model + runtime configuration. Change via .env.
"""

from __future__ import annotations
import os
from dotenv import load_dotenv

load_dotenv()

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
