"""
graph.py  -  The LangGraph wiring: nodes + edges + the critique loop.

Stage 0 goal: prove the SHAPE works. Every node is a stub that just logs and
returns a small state update. No LLM, no cost. What matters here is the
control flow, especially the loop:

    intake -> plan -> critique -> (revise -> critique)* -> finalize

The conditional edge after `critique` is the heart of a self-correcting agent:
if the plan passed OR we hit the revision cap, go to `finalize`; otherwise go
to `revise` and try again. In Stage 0 the critique stub deliberately FAILS the
first pass and PASSES the second, so you can watch the loop run exactly once
and then terminate. That termination guarantee (the cap) is what keeps a
reflection loop from spinning forever and burning tokens.

Nodes return only the fields they change. LangGraph merges them into AgentState
(replacing plain fields, appending reducer fields like `log`/`proposed_plan`).
"""

from __future__ import annotations
from langgraph.graph import StateGraph, START, END

from src.state import AgentState


# --- nodes (Stage 0 stubs) ---

def intake_node(state: AgentState) -> dict:
    """Read the goal + calendar. Later: normalise/parse the request."""
    goal = state["goal"]
    n = len(state["calendar"])
    return {"log": [f"[intake] goal = {goal!r}; {n} existing events loaded"]}


def plan_node(state: AgentState) -> dict:
    """Stub planner. Later (Stage 1): a ReAct node that reasons + calls tools
    to propose real calendar changes. For now it emits one fixed proposal so
    downstream nodes have something to critique."""
    proposal = {
        "action": "add",
        "title": "Focus block: " + state["goal"][:40],
        "start": "2026-07-08T11:15",
        "end": "2026-07-08T12:45",
        "reason": "(stub) placeholder proposal so the loop has input",
    }
    return {
        "proposed_plan": [proposal],
        "log": ["[plan] produced 1 stub proposal"],
    }


def critique_node(state: AgentState) -> dict:
    """Stub critique. Later (Stage 2): deterministic rule-checks (overlaps,
    buffers, deadlines, working hours) PLUS one LLM-judge pass.

    Stage 0 behaviour: fail on the first pass, pass on the second, so you can
    see the revise loop execute once and then stop."""
    revision = state["revision_count"]
    if revision == 0:
        return {
            "critique": {"passed": False, "findings": ["(stub) first-pass fail to exercise the loop"]},
            "passed": False,
            "log": ["[critique] pass 0 -> FAIL (stub)"],
        }
    return {
        "critique": {"passed": True, "findings": []},
        "passed": True,
        "log": [f"[critique] pass {revision} -> PASS (stub)"],
    }


def revise_node(state: AgentState) -> dict:
    """Stub reviser. Later: take the critique findings and fix only the failing
    parts of the plan. For now it just bumps the revision counter and logs."""
    new_count = state["revision_count"] + 1
    return {
        "revision_count": new_count,
        "log": [f"[revise] applying revision #{new_count} (stub)"],
    }


def finalize_node(state: AgentState) -> dict:
    """Stub finaliser. Later: present the confirmed plan / emit output."""
    status = "PASSED" if state["passed"] else "STOPPED (hit revision cap)"
    return {"log": [f"[finalize] plan {status} after {state['revision_count']} revision(s)"]}


# --- conditional edge: the loop decision ---

def route_after_critique(state: AgentState) -> str:
    """Decide where to go after a critique pass.
    -> 'finalize' if the plan passed or we've hit the cap
    -> 'revise'   otherwise
    This one function is what makes the agent self-correcting AND guaranteed
    to terminate."""
    if state["passed"] or state["revision_count"] >= state["max_revisions"]:
        return "finalize"
    return "revise"


# --- graph assembly ---

def build_graph():
    g = StateGraph(AgentState)

    g.add_node("intake", intake_node)
    g.add_node("plan", plan_node)
    g.add_node("critique", critique_node)
    g.add_node("revise", revise_node)
    g.add_node("finalize", finalize_node)

    g.add_edge(START, "intake")
    g.add_edge("intake", "plan")
    g.add_edge("plan", "critique")
    g.add_conditional_edges("critique", route_after_critique, {
        "revise": "revise",
        "finalize": "finalize",
    })
    g.add_edge("revise", "critique")   # loop back
    g.add_edge("finalize", END)

    return g.compile()
