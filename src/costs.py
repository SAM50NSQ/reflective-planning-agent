"""
costs.py  -  A tiny cost meter so you SEE spend accumulate on a $5 budget.

It sums input/output tokens across the run and estimates USD. Prices are
centralised here; correct them against your provider's pricing page. In mock
mode nothing is added, so a run reports $0.

IMPORTANT: the price numbers below are estimates and may be out of date.
Verify at https://www.anthropic.com/pricing (or your provider). [Guessing]
"""

from __future__ import annotations

# USD per 1,000,000 tokens. VERIFY THESE. [Guessing]
PRICES = {
    "haiku":   {"input": 0.80, "output": 4.00},
    "sonnet":  {"input": 3.00, "output": 15.00},
    "default": {"input": 1.00, "output": 5.00},
}


def _rate(model_name: str) -> dict:
    m = (model_name or "").lower()
    if "haiku" in m:
        return PRICES["haiku"]
    if "sonnet" in m:
        return PRICES["sonnet"]
    return PRICES["default"]


class CostTracker:
    def __init__(self) -> None:
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.model = None

    def add(self, model_name: str, input_tokens: int, output_tokens: int) -> None:
        self.model = model_name
        self.calls += 1
        self.input_tokens += int(input_tokens or 0)
        self.output_tokens += int(output_tokens or 0)

    def estimate_usd(self) -> float:
        r = _rate(self.model)
        return self.input_tokens / 1e6 * r["input"] + self.output_tokens / 1e6 * r["output"]

    def report(self) -> str:
        if self.calls == 0:
            return "[cost] no LLM calls (mock mode) -> $0.00"
        return (f"[cost] {self.calls} call(s) | in {self.input_tokens} tok | "
                f"out {self.output_tokens} tok | est ${self.estimate_usd():.4f} "
                f"(model={self.model}; verify prices)")


# One tracker per process run. main.py resets it at the start of each run.
tracker = CostTracker()


def reset() -> None:
    global tracker
    tracker = CostTracker()
