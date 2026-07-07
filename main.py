"""
main.py  -  Run the Stage 0 agent skeleton.

Usage:
    python main.py
    python main.py "Find me two hours of focus time before my 1:1"

No API key needed for Stage 0. This exists to prove the graph runs end to end
and the critique loop terminates. When you run it you should see the trace go:
intake -> plan -> critique(FAIL) -> revise -> critique(PASS) -> finalize.
"""

from __future__ import annotations
import sys

from src.state import AgentState
from src.calendar_mock import seed_calendar
from src.graph import build_graph


def run(goal: str) -> AgentState:
    app = build_graph()

    initial: AgentState = {
        "goal": goal,
        "calendar": seed_calendar(),
        "proposed_plan": [],
        "critique": {},
        "passed": False,
        "revision_count": 0,
        "max_revisions": 3,   # hard cap: the loop can never run more than this
        "log": [],
    }

    final = app.invoke(initial)
    return final


def main() -> None:
    goal = sys.argv[1] if len(sys.argv) > 1 else "Schedule two hours of deep work today"
    final = run(goal)

    print("\n=== TRACE ===")
    for line in final["log"]:
        print(" ", line)

    print("\n=== RESULT ===")
    print("  passed:        ", final["passed"])
    print("  revisions used:", final["revision_count"], "/", final["max_revisions"])
    print("  proposals:")
    for p in final["proposed_plan"]:
        print(f"    - {p['action']}: {p['title']}  [{p['start']} -> {p['end']}]")


if __name__ == "__main__":
    main()
