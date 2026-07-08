"""
graph.py  -  Async graph wiring for Stage 2.

The loop is now REAL and regenerates on failure:

    START -> intake -> plan -> critique -> (revise -> plan)* -> finalize -> END
                                  |______________________________^
             route_after_critique: pass OR revision cap -> finalize ; else -> revise

Key change from Stage 1: on failure the graph goes revise -> PLAN (not straight
back to critique), so the planner actually regenerates using the critique
feedback. `revise` bumps the counter; `plan` reads `revision_feedback`.
"""

from __future__ import annotations
from langgraph.graph import StateGraph, START, END

from src.state import AgentState
from src.planner import plan_node
from src.critique import critique_node


async def intake_node(state: AgentState) -> dict:
    return {"log": [f"[intake] goal={state['goal']!r}; {len(state['calendar'])} events loaded"]}


async def revise_node(state: AgentState) -> dict:
    n = state["revision_count"] + 1
    return {"revision_count": n,
            "log": [f"[revise] revision #{n}; feeding {len(state.get('findings', []))} finding(s) back to planner"]}


async def finalize_node(state: AgentState) -> dict:
    status = "PASSED" if state["passed"] else "STOPPED (revision cap reached)"
    return {"log": [f"[finalize] plan {status} after {state['revision_count']} revision(s)"]}


def route_after_critique(state: AgentState) -> str:
    if state["passed"] or state["revision_count"] >= state["max_revisions"]:
        return "finalize"
    return "revise"


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
    g.add_conditional_edges("critique", route_after_critique,
                            {"revise": "revise", "finalize": "finalize"})
    g.add_edge("revise", "plan")     # regenerate with feedback
    g.add_edge("finalize", END)
    return g.compile()
