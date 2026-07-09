"""costs.py  -  Cost meter. Prices corrected to current rates (per MTok)."""
from __future__ import annotations
import logging

log = logging.getLogger("agent.costs")

# USD per 1,000,000 tokens. Source: Anthropic pricing page.
#   Haiku 4.5:  $1 in / $5 out
#   Sonnet 4.5: $3 in / $15 out
#   Opus 4.x:   $5 in / $25 out
PRICES = {
    "haiku":   {"input": 1.00, "output": 5.00},
    "sonnet":  {"input": 3.00, "output": 15.00},  # Sonnet 4.5
    "opus":    {"input": 5.00, "output": 25.00},
    "default": {"input": 1.00, "output": 5.00},
}


def _rate(model_name: str) -> dict:
    m = (model_name or "").lower()
    for key in ("haiku", "sonnet", "opus"):
        if key in m:
            return PRICES[key]
    return PRICES["default"]


class CostTracker:
    """Meters per-model so a run that mixes Haiku (planner) and Sonnet (judge)
    is priced correctly for each."""
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.calls = 0
        self.by_model: dict[str, dict] = {}   # model -> {calls,input,output}

    def add(self, model_name: str, input_tokens: int, output_tokens: int) -> None:
        self.calls += 1
        m = self.by_model.setdefault(model_name, {"calls": 0, "input": 0, "output": 0})
        m["calls"] += 1
        m["input"] += int(input_tokens or 0)
        m["output"] += int(output_tokens or 0)
        log.debug("cost.add model=%s in=%s out=%s (total $%.4f)", model_name,
                  input_tokens, output_tokens, self.estimate_usd())

    def estimate_usd(self) -> float:
        total = 0.0
        for name, m in self.by_model.items():
            r = _rate(name)
            total += m["input"] / 1e6 * r["input"] + m["output"] / 1e6 * r["output"]
        return total

    def report(self) -> str:
        if self.calls == 0:
            return "[cost] no LLM calls recorded -> $0.00"
        parts = [f"[cost] {self.calls} call(s), est ${self.estimate_usd():.4f} (verify prices)"]
        for name, m in self.by_model.items():
            parts.append(f"    {name}: {m['calls']} call(s), in {m['input']} / out {m['output']} tok")
        return "\n".join(parts)


tracker = CostTracker()


def reset() -> None:
    tracker.reset()
