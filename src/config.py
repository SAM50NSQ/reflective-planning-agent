"""
config.py  -  Central model + runtime configuration. Change models via .env,
never by editing node code.

Design goals (your requirements):
  - Provider-agnostic: swap Claude / Gemini / OpenAI / self-hosted by changing
    strings in .env. Done via LangChain's `init_chat_model`.
  - Per-ROLE model names, not one global. PLANNER_MODEL and JUDGE_MODEL can
    differ, so a cheap planner + a stronger judge is a .env change, not a
    refactor. (The judge, added in Stage 2, is the component most likely to
    need a stronger model on a tight budget.)
  - A mock switch so you can develop the whole graph for $0 and only spend on
    deliberate end-to-end runs.
"""

from __future__ import annotations
import os
from dotenv import load_dotenv

load_dotenv()  # read .env into the environment

# --- runtime switches ---
# USE_MOCK_LLM=true  -> nodes return canned responses, ZERO API cost. Default on
# so a fresh clone runs without a key. Set to false in .env for live runs.
USE_MOCK_LLM: bool = os.getenv("USE_MOCK_LLM", "true").strip().lower() == "true"

# Provider LangChain routes to. "anthropic" needs `langchain-anthropic` + key.
MODEL_PROVIDER: str = os.getenv("MODEL_PROVIDER", "anthropic")

# Default model string if a role-specific one is not set.
# NOTE: verify the exact model id for your provider/account. This default is a
# placeholder alias and may be wrong for your setup. [Guessing on exact id]
DEFAULT_MODEL: str = os.getenv("DEFAULT_MODEL", "claude-3-5-haiku-latest")


def model_name_for(role: str) -> str:
    """Resolve the model string for a role, e.g. role='planner' reads
    PLANNER_MODEL from .env, falling back to DEFAULT_MODEL."""
    return os.getenv(f"{role.upper()}_MODEL", DEFAULT_MODEL)


def get_model(role: str):
    """Return a provider-agnostic chat model for the given role.

    Only call this on the LIVE path; in mock mode nodes never touch it, so a
    missing key or provider package never breaks development runs.
    """
    from langchain.chat_models import init_chat_model  # imported lazily
    return init_chat_model(model_name_for(role), model_provider=MODEL_PROVIDER)
