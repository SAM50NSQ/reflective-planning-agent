"""
graph.py  -  Async graph with human-in-the-loop (Stage 3).

Two interrupt points (real LangGraph `interrupt`, resumed via `Command`):
  - clarify : planner lacks a critical detail -> graph PAUSES, asks, resumes, replans.
  - confirm : a passing plan MOVES an existing commitment (high stakes) -> graph
              PAUSES for approval before finalizing.

Low-stakes plans (a plain 'add' into a free slot) skip confirm entirely. That
ask-vs-proceed split IS the act-vs-ask policy.

TERMINATION (bug fixed in this stage): every branch out of `critique` is capped.
Previously a passing plan containing a `move` went straight to `confirm` without
consulting `max_revisions`; on decline it looped plan -> critique -> confirm ->
plan forever. Now:
  - decline sets `move_blocked`, so the planner must stop proposing moves, and
  - the confirm branch is guarded by the revision cap.

Requires a checkpointer (MemorySaver here; a persistent one in Stage 3.5) and a
thread_id in the run config, or `interrupt` cannot suspend/resume.
"""

from __future__ import annotations
import logging
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt
from langgraph.checkpoint.memory import MemorySaver

from src.state import AgentState
from src.planner import plan_node
from src.critique import critique_node

log = logging.getLogger("agent.graph")


async def intake_node(state: AgentState) -> dict:
    log.info("NODE intake: goal=%r, %d events", state["goal"], len(state["calendar"]))
    return {"log": [f"[intake] goal={state['goal']!r}; {len(state['calendar'])} events loaded"]}


async def clarify_node(state: AgentState) -> dict:
    q = state.get("pending_question")
    if not q:
        # No explicit question. Surface the planner's OWN reasoning so the user can
        # correct it, rather than asking a generic line that wastes a turn.
        why = state.get("plan_summary") or "I could not produce a plan."
        q = f"{why} What exactly should I schedule, and when?"
    n = state["clarify_count"] + 1
    log.info("NODE clarify (%d/%d) -> interrupt asking: %s", n, state["max_clarifications"], q)
    answer = interrupt({"type": "clarification", "question": q})     # PAUSES here
    log.info("clarify resumed with answer: %r", answer)
    # NOTE: we deliberately do NOT clear `revision_feedback` here. The critique
    # node put the judge's findings there, and the planner needs them on the next
    # attempt. Clearing it meant the planner replanned blind: the judge would say
    # "you must propose adding or moving the lunch block", the planner never saw
    # it, took the `no_change_needed` escape hatch, and the turn died. The system
    # was throwing away its own critic's catch.
    return {"gathered_info": [f"{q} -> {answer}"], "pending_question": None,
            "needs_info": False, "clarify_count": n,
            "log": [f"[clarify] asked {q!r}, got {answer!r}"]}


def _high_stakes(plan: list[dict]) -> list[dict]:
    """Moving or removing an existing commitment is destructive: always ask first."""
    return [p for p in plan if p.get("action") in ("move", "remove")]


async def confirm_node(state: AgentState) -> dict:
    risky = _high_stakes(state["proposed_plan"])
    desc = "; ".join(
        (f"remove {p['title']}" if p.get("action") == "remove"
         else f"move {p['title']} -> {p['start']}..{p['end']}")
        for p in risky)
    log.info("NODE confirm -> interrupt for high-stakes change: %s", desc)
    decision = interrupt({"type": "confirmation",
                          "question": f"This plan changes existing commitments ({desc}). Approve? (yes/no)"})
    approved = str(decision).strip().lower() in ("y", "yes", "approve", "ok", "sure")
    log.info("confirm resumed: approved=%s", approved)
    if approved:
        return {"confirmed": True, "log": ["[confirm] user approved the reschedule"]}
    # Decline: block moves for good, mark not-passed, send back with feedback.
    return {"passed": False, "move_blocked": True, "confirmed": False,
            "revision_feedback": ("User declined changing an existing event. Propose a plan "
                                  "that does NOT move or remove any existing commitment."),
            "log": ["[confirm] user declined -> replan without touching existing events"]}


async def revise_node(state: AgentState) -> dict:
    n = state["revision_count"] + 1
    log.info("NODE revise -> revision #%d", n)
    return {"revision_count": n,
            "log": [f"[revise] revision #{n}; {len(state.get('findings', []))} finding(s) fed back"]}


async def finalize_node(state: AgentState) -> dict:
    if state["passed"]:
        reason = "PASSED"
    elif state.get("needs_info"):
        reason = "STOPPED (needs more info)"
    elif state.get("stall"):
        reason = "STOPPED (no further progress)"
    elif state["clarify_count"] >= state["max_clarifications"]:
        reason = "STOPPED (asked as much as I usefully can)"
    else:
        reason = "STOPPED (revision cap reached)"
    log.info("NODE finalize: %s after %d revision(s)", reason, state["revision_count"])
    return {"log": [f"[finalize] {reason} after {state['revision_count']} revision(s)"]}


def route_after_critique(state: AgentState) -> str:
    capped = state["revision_count"] >= state["max_revisions"]

    asked_out = state["clarify_count"] >= state["max_clarifications"]

    if state["passed"]:
        needs_confirm = (bool(_high_stakes(state["proposed_plan"]))
                         and not state.get("confirmed"))
        # GUARD: never enter confirm once capped (was an infinite-loop path).
        decision = "confirm" if (needs_confirm and not capped) else "finalize"
    else:
        wants_info = bool(state.get("pending_question") or state.get("needs_info"))
        # GUARD: clarify is ALSO capped. Previously `needs_info` was checked before
        # the revision cap, so clarify -> plan -> critique -> clarify could spin
        # forever and burn tokens. Every edge out of a reflection node needs its
        # own termination argument; this one did not have it.
        if wants_info and not asked_out and not capped:
            decision = "clarify"
        elif capped or asked_out or state.get("stall"):
            decision = "finalize"
        else:
            decision = "revise"

    log.info("route_after_critique -> %s (passed=%s capped=%s asked_out=%s stall=%s needs_info=%s)",
             decision, state["passed"], capped, asked_out, state.get("stall"),
             state.get("needs_info"))
    return decision


def route_after_confirm(state: AgentState) -> str:
    if state["passed"]:
        return "finalize"
    return "finalize" if state["revision_count"] >= state["max_revisions"] else "revise"


def build_graph():
    g = StateGraph(AgentState)
    for name, fn in [("intake", intake_node), ("plan", plan_node), ("critique", critique_node),
                     ("clarify", clarify_node), ("confirm", confirm_node),
                     ("revise", revise_node), ("finalize", finalize_node)]:
        g.add_node(name, fn)

    g.add_edge(START, "intake")
    g.add_edge("intake", "plan")
    g.add_edge("plan", "critique")
    g.add_conditional_edges("critique", route_after_critique,
                            {"clarify": "clarify", "confirm": "confirm",
                             "revise": "revise", "finalize": "finalize"})
    g.add_edge("clarify", "plan")
    g.add_conditional_edges("confirm", route_after_confirm,
                            {"finalize": "finalize", "revise": "revise"})
    g.add_edge("revise", "plan")
    g.add_edge("finalize", END)
    return g.compile(checkpointer=MemorySaver())
