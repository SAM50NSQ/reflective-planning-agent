"""
logging_setup.py  -  One place to configure verbose, per-step logging.

Set LOG_LEVEL in .env (DEBUG shows tool args/results + token counts per call;
INFO shows each node/step; WARNING shows only problems). Every module logs
under the 'agent.*' namespace so you can trace exactly which step a failure
came from.
"""

from __future__ import annotations
import logging
import os


def setup() -> logging.Logger:
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-5s | %(name)-16s | %(message)s",
        datefmt="%H:%M:%S",
    )
    # keep noisy libraries quiet unless we really want DEBUG
    for noisy in ("httpx", "httpcore", "anthropic", "langchain", "langgraph",
                  "urllib3", "asyncio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return logging.getLogger("agent")
