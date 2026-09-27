"""
cli.py  -  Command-line front ends for the reflective planning agent.

Two entry points over the same agent graph:

    rpa "Find me two hours of deep work"   # one goal: print trace + plan, exit
    rpa-chat                               # conversational session (REPL)

Both run in mock mode ($0) unless USE_MOCK_LLM=false.

The graph can PAUSE to ask a question and RESUME with the answer. It only
does so when it needs to:
    - clarify : a critical detail is missing
    - confirm : the plan would MOVE or REMOVE an existing commitment
A fully-specified, low-stakes goal produces zero interrupts. That is the
act-vs-ask policy working, not a missing feature. `one_turn` handles the
loop: invoke, and while the result carries an interrupt, ask and resume.

`rpa-chat` adds a SESSION on top: one `thread_id` is reused across turns, so
anything you told the agent earlier (`gathered_info`) informs later turns.

    Commands:  :new    start a fresh session (forget gathered info)
               :info   show what the agent has learned this session
               :quit   exit (also works while the agent is asking)
               :abort  abandon the current question (while the agent is asking)

Stage 3.5 note: the ONLY thing separating this from a FastAPI service is that
`MemorySaver` is in-process. Swap it for a persistent checkpointer and map
`thread_id` to an HTTP session and these same calls become endpoints.
"""

from __future__ import annotations

import asyncio
import sys
import uuid
from collections.abc import Callable

from langgraph.types import Command

from src import costs
from src.calendar_mock import seed_calendar
from src.config import USE_MOCK_LLM
from src.graph import build_graph
from src.logging_setup import setup

DEFAULT_GOAL = "Schedule two hours of deep work today"

Ask = Callable[[dict], str]   # receives an interrupt payload, returns the answer


class AbortTurn(Exception):
    """The user typed :abort or :quit while the agent was waiting on a question."""

    def __init__(self, command: str) -> None:
        super().__init__(command)
        self.command = command


# --------------------------------------------------------------------------- #
# Core loop, shared by both entry points
# --------------------------------------------------------------------------- #

def _new_thread() -> dict:
    return {"configurable": {"thread_id": str(uuid.uuid4())}}


def _fresh_state(goal: str, carried: list[str]) -> dict:
    """State for a new turn. `carried` is what the user has already told us this
    session, replayed so the planner does not re-ask for it."""
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


async def one_turn(app, cfg: dict, goal: str, carried: list[str], ask: Ask) -> dict:
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


async def run(goal: str, answer_fn: Ask | None = None) -> dict:
    """Run one goal on a fresh graph and thread. answer_fn(payload) -> str answers
    interrupts; it defaults to asking on stdin. Tests pass a scripted answer_fn
    so no human is needed."""
    return await one_turn(build_graph(), _new_thread(), goal, [], answer_fn or _ask)


# --------------------------------------------------------------------------- #
# Terminal I/O
# --------------------------------------------------------------------------- #

def _ask(payload: dict) -> str:
    label = "?" if payload.get("type") == "clarification" else "!"
    answer = input(f"\n  agent {label} {payload['question']}\n  you > ").strip()
    # Commands must escape the interrupt, not be swallowed as an answer.
    if answer in (":quit", ":abort"):
        raise AbortTurn(answer)
    return answer


def _mode() -> str:
    return "MOCK ($0)" if USE_MOCK_LLM else "LIVE"


def _describe(step: dict) -> str:
    """'action: title  [start -> end]'. A removal occupies no time slot."""
    if step.get("action") == "remove":
        when = ""
    else:
        when = f"  [{step.get('start')} -> {step.get('end')}]"
    return f"{step.get('action')}: {step.get('title')}{when}"


def _print_result(final: dict) -> None:
    print()
    if final["passed"]:
        print("  Plan:")
        for step in final["proposed_plan"]:
            print(f"    - {_describe(step)}")
            print(f"        {step.get('reason')}")
    else:
        reason = final["log"][-1] if final["log"] else "unknown"
        print("  No plan delivered. Reason:", reason)
        for f in final.get("findings", []):
            print(f"    [{f['source']}/{f['severity']}] {f['message']}")
    print("\n  " + costs.tracker.report())


# --------------------------------------------------------------------------- #
# Conversational session
# --------------------------------------------------------------------------- #

async def repl() -> None:
    app = build_graph()
    cfg = _new_thread()
    carried: list[str] = []

    print(f"\nReflective Planning Agent  [{_mode()}]")
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
            cfg = _new_thread()
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
            cfg = _new_thread()
            print("\n  (turn abandoned)  " + costs.tracker.report() + "\n")
            if e.command == ":quit":
                print("bye")
                return
            continue
        # Anything the agent asked and you answered persists into later turns.
        carried = list(final.get("gathered_info", carried))
        _print_result(final)
        print()


# --------------------------------------------------------------------------- #
# Entry points (see [project.scripts] in pyproject.toml)
# --------------------------------------------------------------------------- #

def main() -> int:
    """`rpa [GOAL]` - run one goal, print the trace and final plan, exit."""
    setup()
    goal = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_GOAL
    try:
        final = asyncio.run(run(goal))
    except AbortTurn:
        print("\n  (aborted)")
        return 1

    print(f"\n=== MODE: {_mode()} ===")
    print("=== TRACE ===")
    for line in final["log"]:
        print(" ", line)
    print("\n=== FINAL PLAN ===  passed:", final["passed"])
    for step in final["proposed_plan"]:
        print(f"  - {_describe(step)}")
        print(f"      reason: {step.get('reason')}")
    print("\n" + costs.tracker.report())
    return 0


def chat() -> int:
    """`rpa-chat` - conversational session."""
    setup()
    asyncio.run(repl())
    return 0


if __name__ == "__main__":
    sys.exit(main())
