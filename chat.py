"""
chat.py  -  Conversational front end (REPL) over the same agent graph.

Why this exists separately from main.py:
  main.py runs ONE goal and exits. The agent's human-in-the-loop machinery
  (LangGraph `interrupt`) already lets it pause mid-plan to ask you something,
  but it only fires when it needs to:
      - clarify : a critical detail is missing
      - confirm : the plan would MOVE an existing commitment (high stakes)
  A fully-specified, low-stakes goal produces zero interrupts. That is the
  act-vs-ask policy working, not a missing feature.

This REPL adds the missing piece: a SESSION. One `thread_id` is reused across
turns, so the checkpointer carries state forward and anything you told the
agent earlier (`gathered_info`) informs later turns.

    python chat.py            # mock mode if USE_MOCK_LLM=true ($0)

Commands:  :new   start a fresh session (forget gathered info)
           :info  show what the agent has learned this session
           :quit  exit

Stage 3.5 note: the ONLY thing separating this from a FastAPI service is that
`MemorySaver` is in-process. Swap it for a persistent checkpointer and map
`thread_id` to an HTTP session and these same calls become endpoints.
"""

from __future__ import annotations
import asyncio
import uuid

from src.logging_setup import setup
setup()

from langgraph.types import Command
from src.calendar_mock import seed_calendar
from src.graph import build_graph
from src import costs
from src.config import USE_MOCK_LLM


def _fresh_state(goal: str, carried: list[str]) -> dict:
    """A new turn. `carried` is what the user has already told us this session,
    replayed so the planner does not re-ask for it."""
    return {
        "goal": goal,
        "calendar": seed_calendar(),
        "proposed_plan": [], "findings": [], "revision_feedback": None,
        "critique": {}, "passed": False, "revision_count": 0, "max_revisions": 3,
        "prev_findings_sig": None, "needs_info": False, "stall": False,
        "plan_summary": None, "pending_question": None, "gathered_info": list(carried),
        "clarify_count": 0, "max_clarifications": 2,
        "confirmed": False, "move_blocked": False, "log": [],
    }


async def one_turn(app, cfg: dict, goal: str, carried: list[str], ask) -> dict:
    """Run a single goal to completion, pausing for the user whenever the agent
    raises an interrupt. Returns the final state."""
    costs.reset()
    result = await app.ainvoke(_fresh_state(goal, carried), cfg)

    # The graph suspends by returning an __interrupt__ payload. Answer it and
    # resume the SAME execution with Command(resume=...). Loop because a single
    # turn can pause more than once (e.g. clarify, then confirm).
    while "__interrupt__" in result:
        payload = result["__interrupt__"][0].value
        answer = ask(payload)
        result = await app.ainvoke(Command(resume=answer), cfg)
    return result


def _print_result(final: dict) -> None:
    print()
    if final["passed"]:
        print("  Plan:")
        for p in final["proposed_plan"]:
            if p.get("action") == "remove":
                when = ""                     # a removal occupies no time slot
            else:
                when = f"  [{p.get('start')} -> {p.get('end')}]"
            print(f"    - {p.get('action')}: {p.get('title')}{when}")
            print(f"        {p.get('reason')}")
    else:
        print("  No plan delivered. Reason:", final["log"][-1] if final["log"] else "unknown")
        for f in final.get("findings", []):
            print(f"    [{f['source']}/{f['severity']}] {f['message']}")
    print("\n  " + costs.tracker.report())


class AbortTurn(Exception):
    """User typed :abort / :quit while the agent was waiting on a question."""


def _ask(payload: dict) -> str:
    label = "?" if payload.get("type") == "clarification" else "!"
    answer = input(f"\n  agent {label} {payload['question']}\n  you > ").strip()
    # Commands must escape the interrupt, not be swallowed as an answer.
    # (Previously ':quit' was fed to the planner as a clarification.)
    if answer in (":quit", ":abort"):
        raise AbortTurn(answer)
    return answer


async def repl() -> None:
    app = build_graph()
    thread = str(uuid.uuid4())
    cfg = {"configurable": {"thread_id": thread}}
    carried: list[str] = []

    print(f"\nReflective Planning Agent  [{'MOCK ($0)' if USE_MOCK_LLM else 'LIVE'}]")
    print("Type a goal, or :new / :info / :quit\n")

    while True:
        try:
            goal = input("you > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            return

        if not goal:
            continue
        if goal == ":quit":
            print("bye")
            return
        if goal == ":new":
            thread = str(uuid.uuid4())
            cfg = {"configurable": {"thread_id": thread}}
            carried = []
            print("  (new session; gathered info cleared)\n")
            continue
        if goal == ":info":
            print("  gathered so far:", carried or "(nothing yet)", "\n")
            continue

        try:
            final = await one_turn(app, cfg, goal, carried, _ask)
        except AbortTurn as e:
            # Abandon this turn. The thread is mid-interrupt, so start a clean one.
            thread = str(uuid.uuid4())
            cfg = {"configurable": {"thread_id": thread}}
            print("\n  (turn abandoned)  " + costs.tracker.report() + "\n")
            if str(e) == ":quit":
                print("bye")
                return
            continue
        # Anything the agent asked and you answered persists into later turns.
        carried = list(final.get("gathered_info", carried))
        _print_result(final)
        print()


if __name__ == "__main__":
    asyncio.run(repl())
