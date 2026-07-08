"""
planner.py  -  The planning node. Hand-wired (no create_react_agent).

LIVE path, two phases:
  1. run_tool_loop: the model uses tools (get_calendar, find_free_slots,
     check_conflicts, reschedule_event) to gather context. This is the ReAct
     part you own.
  2. structured_call(Plan): a final call with NO tools that returns a validated
     Plan object. Separating tool-use from final structured extraction is what
     makes the output reliable (Stage 1's fragile text-JSON parse is gone).

On a revision, `revision_feedback` from the critique is injected so the planner
fixes the specific findings rather than replanning blind.

MOCK path: returns a canned Plan. To exercise the critique loop for real at $0,
the first mock plan is deliberately BAD (overlaps an existing event) so the
rule-checker catches it; the revised mock plan is clean.
"""

from __future__ import annotations
import json

from langchain_core.messages import SystemMessage, HumanMessage

from src.state import AgentState
from src.config import USE_MOCK_LLM, get_model
from src.schemas import Plan
from src.agent_loop import run_tool_loop, structured_call

PLANNER_SYS = (
    "You are a scheduling planner. Use the tools to inspect the user's calendar, "
    "find free slots, and check conflicts before proposing anything. Prefer free "
    "slots inside working hours; only propose rescheduling an existing event when "
    "a higher-priority request genuinely requires it, and always give a reason."
)


def _mock_plan(state: AgentState) -> dict:
    if state["revision_count"] == 0:
        # deliberately overlaps 'Design review' (10:00-11:00) to trigger a rule finding
        proposals = [{
            "action": "add",
            "title": "Deep work (bad, mock)",
            "start": "2026-07-08T10:00",
            "end": "2026-07-08T12:00",
            "reason": "(mock) intentionally overlaps an existing meeting to exercise the critique",
        }]
        note = "MOCK plan rev 0 -> intentionally bad (overlap)"
    else:
        proposals = [{
            "action": "add",
            "title": "Deep work",
            "start": "2026-07-08T14:00",
            "end": "2026-07-08T16:00",
            "reason": "(mock) 14:00-16:00 is free and within working hours; no conflicts",
        }]
        note = f"MOCK plan rev {state['revision_count']} -> clean"
    return {"proposed_plan": proposals, "log": [f"[plan] {note}"]}


def _user_prompt(state: AgentState) -> str:
    base = f"Goal: {state['goal']}"
    fb = state.get("revision_feedback")
    if fb:
        base += f"\n\nYour previous plan was rejected. Fix these issues specifically:\n{fb}"
    return base


async def plan_node(state: AgentState) -> dict:
    if USE_MOCK_LLM:
        return _mock_plan(state)

    model = get_model("planner")
    messages = [SystemMessage(content=PLANNER_SYS), HumanMessage(content=_user_prompt(state))]

    # Phase 1: tool-use loop (owned, capped)
    messages = await run_tool_loop(model, messages, role="planner")

    # Phase 2: structured final plan (no tools, validated)
    messages.append(HumanMessage(content="Now output the final plan."))
    plan: Plan = await structured_call(model, Plan, messages, role="planner")

    proposals = [p.model_dump() for p in plan.proposals] if plan else []
    summary = plan.summary if plan else "(no plan returned)"
    return {"proposed_plan": proposals, "log": [f"[plan] LIVE plan -> {len(proposals)} proposal(s); {summary}"]}
