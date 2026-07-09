"""
main.py  -  Run the conversational agent (async, human-in-the-loop).

The graph can PAUSE to ask you a question (clarification or high-stakes
confirmation) and RESUME with your answer. This loop handles that: invoke,
and while the result carries an interrupt, ask you and resume with Command.
"""

from __future__ import annotations
import sys
import asyncio

from src.logging_setup import setup
setup()

from langgraph.types import Command
from src.calendar_mock import seed_calendar
from src.graph import build_graph
from src import costs
from src.config import USE_MOCK_LLM

THREAD = {"configurable": {"thread_id": "cli-session"}}


def _initial(goal: str) -> dict:
    return {
        "goal": goal, "calendar": seed_calendar(),
        "proposed_plan": [], "findings": [], "revision_feedback": None,
        "critique": {}, "passed": False, "revision_count": 0, "max_revisions": 3,
        "prev_findings_sig": None, "needs_info": False, "stall": False,
        "plan_summary": None, "pending_question": None, "gathered_info": [],
        "clarify_count": 0, "max_clarifications": 2, "confirmed": False,
        "move_blocked": False, "log": [],
    }


async def run(goal: str, answer_fn=None) -> dict:
    """answer_fn(payload)->str supplies answers to interrupts. Defaults to input().
    Tests pass a scripted answer_fn so no human is needed."""
    if answer_fn is None:
        answer_fn = lambda p: input(f"\nAGENT ASKS: {p['question']}\n> ")

    costs.reset()
    app = build_graph()
    result = await app.ainvoke(_initial(goal), THREAD)
    while "__interrupt__" in result:
        payload = result["__interrupt__"][0].value
        answer = answer_fn(payload)
        result = await app.ainvoke(Command(resume=answer), THREAD)
    return result


def main() -> None:
    goal = sys.argv[1] if len(sys.argv) > 1 else "Schedule two hours of deep work today"
    final = asyncio.run(run(goal))

    print(f"\n=== MODE: {'MOCK ($0)' if USE_MOCK_LLM else 'LIVE'} ===")
    print("=== TRACE ===")
    for line in final["log"]:
        print(" ", line)
    print("\n=== FINAL PLAN ===  passed:", final["passed"])
    for p in final["proposed_plan"]:
        when = "" if p.get("action") == "remove" else f"  [{p.get('start')} -> {p.get('end')}]"
        print(f"  - {p.get('action')}: {p.get('title')}{when}")
        print(f"      reason: {p.get('reason')}")
    print("\n" + costs.tracker.report())


if __name__ == "__main__":
    main()
