"""
main.py  -  Run the Stage 2 agent (async).

Mock vs live via USE_MOCK_LLM in .env. In MOCK mode the first plan is
deliberately bad (overlaps an event) so you can watch the rule-checker catch it
and the agent revise to a clean plan, all at $0.
"""

from __future__ import annotations
import sys
import asyncio

from src.state import AgentState
from src.calendar_mock import seed_calendar
from src.graph import build_graph
from src import costs
from src.config import USE_MOCK_LLM


async def run(goal: str) -> AgentState:
    costs.reset()
    app = build_graph()
    initial: AgentState = {
        "goal": goal,
        "calendar": seed_calendar(),
        "proposed_plan": [],
        "findings": [],
        "revision_feedback": None,
        "critique": {},
        "passed": False,
        "revision_count": 0,
        "max_revisions": 3,
        "log": [],
    }
    return await app.ainvoke(initial)


def main() -> None:
    goal = sys.argv[1] if len(sys.argv) > 1 else "Schedule two hours of deep work today"
    final = asyncio.run(run(goal))

    print(f"\n=== MODE: {'MOCK ($0)' if USE_MOCK_LLM else 'LIVE'} ===")
    print("=== TRACE ===")
    for line in final["log"]:
        print(" ", line)

    print("\n=== FINAL PLAN ===")
    print("  status:", "PASSED" if final["passed"] else "STOPPED (cap)")
    for p in final["proposed_plan"]:
        print(f"  - {p.get('action')}: {p.get('title')}  [{p.get('start')} -> {p.get('end')}]")
        print(f"      reason: {p.get('reason')}")

    if final["findings"]:
        print("\n=== LAST CRITIQUE FINDINGS ===")
        for f in final["findings"]:
            print(f"  [{f['source']}/{f['severity']}] {f['message']}")

    print("\n" + costs.tracker.report())


if __name__ == "__main__":
    main()
