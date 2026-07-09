"""
planner.py  -  Planning node. Stage 3: can ask for missing info (via
pending_question) instead of thrashing, and uses gathered_info from prior turns.
"""

from __future__ import annotations
import logging

from langchain_core.messages import SystemMessage, HumanMessage

from src.state import AgentState
from src.config import USE_MOCK_LLM, get_model, REFERENCE_DATE
from src.calendar_mock import seed_calendar
from src.schemas import Plan
from src.agent_loop import run_tool_loop, structured_call

log = logging.getLogger("agent.planner")

PLANNER_SYS = (
    "You are a scheduling planner. Today is {today}. Use the tools to inspect the "
    "calendar, find free slots, and check conflicts before proposing.\n\n"
    "CAPABILITIES: add a new block, move an existing one, or remove an existing one. "
    "You cannot rename an event in place, and you cannot invite people or book venues. "
    "To REPLACE an existing event with a new one, emit TWO proposals: remove the old "
    "event (by event_id), and add the new one. To make room, move the old event instead "
    "of removing it. If the goal needs something outside these actions, say so and ask.\n\n"
    "FACTS: never assert that an event exists unless it appears in the calendar you "
    "were given. An event titled 'Lunch' is not a client lunch.\n\n"
    "IDS: for action='move' or 'remove' you MUST set event_id from get_calendar. "
    "'remove' needs no start/end. Always write a one-sentence `summary`.\n\n"
    "OUTPUT exactly one of:\n"
    "  1. proposals: concrete ISO-timed changes inside working hours (09:00-18:00);\n"
    "  2. needs_input=true with a single specific `question`, if a critical detail is "
    "missing or the goal needs a capability you lack;\n"
    "  3. no_change_needed=true, ONLY if the existing calendar already satisfies the "
    "goal exactly. Do not use this to avoid work."
)


def _day_summary() -> str:
    evs = [e for e in seed_calendar() if e["start"].startswith(REFERENCE_DATE)]
    return "; ".join(f"{e['title']} {e['start'][11:]}-{e['end'][11:]}" for e in evs) or "(no events)"


def _mock_plan(state: AgentState) -> dict:
    goal = state["goal"].lower()
    info = state.get("gathered_info") or []

    # Scenario 1 (clarify demo): a 'meeting' with no duration/time and nothing gathered yet
    if "meeting" in goal and not info:
        return {"proposed_plan": [], "needs_info": True,
                "pending_question": "What duration should the meeting be, and any preferred time?",
                "log": ["[plan] MOCK needs info -> will ask"]}

    # Scenario 2 (confirm demo): plan that moves an existing commitment (high-stakes).
    # If the user already declined a move, we must NOT propose one again (termination).
    if ("move" in goal or "reschedule" in goal) and not state.get("move_blocked"):
        return {"proposed_plan": [{"action": "move", "title": "Team sync",
                                   "start": f"{REFERENCE_DATE}T11:00", "end": f"{REFERENCE_DATE}T12:00",
                                   "reason": "(mock) illustrative move to demo the confirm step"}],
                "log": ["[plan] MOCK plan with a move -> will need confirmation"]}

    # Scenario 3: meeting after we gathered info -> concrete add
    if "meeting" in goal and info:
        return {"proposed_plan": [{"action": "add", "title": "Design team meeting",
                                   "start": f"{REFERENCE_DATE}T11:00", "end": f"{REFERENCE_DATE}T12:30",
                                   "reason": "(mock) 90 min per your input; free slot, no conflicts"}],
                "log": ["[plan] MOCK plan from gathered info"]}

    # Default: first attempt is deliberately BAD (overlaps Design review) so the
    # deterministic rule-checker catches it and the revise loop is visible at $0.
    if state["revision_count"] == 0:
        return {"proposed_plan": [{"action": "add", "title": "Deep work (bad, mock)",
                                   "start": f"{REFERENCE_DATE}T10:00", "end": f"{REFERENCE_DATE}T12:00",
                                   "reason": "(mock) intentionally overlaps an existing meeting"}],
                "log": ["[plan] MOCK rev0 -> intentionally bad (overlap)"]}
    return {"proposed_plan": [{"action": "add", "title": "Deep work",
                               "start": f"{REFERENCE_DATE}T14:00", "end": f"{REFERENCE_DATE}T16:00",
                               "reason": "(mock) free 14:00-16:00, no conflicts"}],
            "log": [f"[plan] MOCK rev{state['revision_count']} -> clean"]}


def _user_prompt(state: AgentState) -> str:
    base = (f"Goal: {state['goal']}\nToday is {REFERENCE_DATE}. Today's events: {_day_summary()}.")
    info = state.get("gathered_info") or []
    if info:
        base += "\n\nInformation gathered from the user so far:\n" + "\n".join(f"- {i}" for i in info)
    fb = state.get("revision_feedback")
    if fb:
        base += f"\n\nYour previous plan was rejected. Fix specifically:\n{fb}"
    return base


async def plan_node(state: AgentState) -> dict:
    log.info("NODE plan (revision=%d, gathered=%d)", state["revision_count"], len(state.get("gathered_info") or []))
    if USE_MOCK_LLM:
        return _mock_plan(state)

    model = get_model("planner")
    messages = [SystemMessage(content=PLANNER_SYS.format(today=REFERENCE_DATE)),
                HumanMessage(content=_user_prompt(state))]
    messages = await run_tool_loop(model, messages, role="planner")
    messages.append(HumanMessage(content="Now output the final plan (or an ASK: question if a critical detail is missing)."))
    plan: Plan = await structured_call(model, Plan, messages, role="planner")

    proposals = [p.model_dump() for p in plan.proposals] if plan else []
    summary = (plan.summary if plan else "") or ""
    if not summary and proposals:
        summary = "; ".join(f"{q['action']} {q['title']} {q['start'][11:]}-{q['end'][11:]}"
                            for q in proposals)
    if not summary:
        summary = "I could not produce a plan."

    # The planner now signals its outcome EXPLICITLY. `len(proposals) == 0` no
    # longer means "ask me" by accident.
    if plan and plan.needs_input and not proposals:
        q = plan.question.strip() or "What exactly should I schedule, and when?"
        log.info("planner asks: %s", q)
        return {"proposed_plan": [], "needs_info": True, "pending_question": q,
                "plan_summary": summary,
                "log": [f"[plan] LIVE needs info -> ask: {q}"]}

    if plan and plan.no_change_needed and not proposals:
        log.info("planner says no change needed: %s", summary)
        return {"proposed_plan": [], "needs_info": True, "plan_summary": summary,
                "pending_question": (f"I think your calendar already covers this "
                                     f"({summary}). Is that right, or what exactly "
                                     f"should I schedule?"),
                "log": [f"[plan] LIVE -> no change needed; {summary}"]}

    log.info("plan produced %d proposal(s): %s", len(proposals), summary)
    return {"proposed_plan": proposals, "pending_question": None, "plan_summary": summary,
            "log": [f"[plan] LIVE -> {len(proposals)} proposal(s); {summary}"]}
