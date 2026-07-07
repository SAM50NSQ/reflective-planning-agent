"""
state.py  -  The shared state for the planning agent.

Every node in the LangGraph graph receives this state, may read any field,
and returns a partial dict of fields to update. Keeping all agent state in
one typed object (rather than passing ad-hoc arguments) is what lets the
graph route, loop, and be inspected. This is the single most important file
to understand before anything else.

Stage 0: no LLM involved. These fields are populated by stub nodes so you can
watch the graph flow end to end before we add a single model call.
"""

from __future__ import annotations
from typing import TypedDict, Annotated
import operator


class CalendarEvent(TypedDict):
    """One existing event on the user's (mock) calendar."""
    id: str
    title: str
    start: str   # ISO 8601, e.g. "2026-07-08T10:00"
    end: str     # ISO 8601


class ProposedItem(TypedDict):
    """One change the agent proposes: a new block, a move, etc.

    In v1 these are *proposals* the user confirms (human-in-the-loop).
    """
    action: str        # "add" | "move" | "remove"  (Stage 0: only "add" used by stub)
    title: str
    start: str
    end: str
    reason: str        # why the agent proposed this (fills in from Stage 1+)


class AgentState(TypedDict):
    """The full agent state threaded through the graph.

    Notes on the two 'Annotated ... operator.add' fields:
    LangGraph merges each node's returned dict into the state. For plain fields,
    the new value REPLACES the old. For fields annotated with a reducer
    (operator.add here), the new value is APPENDED. We use that for `log` and
    `proposed_plan` so nodes can accumulate entries instead of overwriting.
    """
    # --- inputs ---
    goal: str                                  # what the user wants scheduled
    calendar: list[CalendarEvent]              # existing events (mock in Stage 0)

    # --- working state ---
    proposed_plan: Annotated[list[ProposedItem], operator.add]
    critique: dict                             # findings from the critique node
    passed: bool                               # did the plan clear the critique?
    revision_count: int                        # how many revise passes so far
    max_revisions: int                         # hard cap so the loop always ends

    # --- trace ---
    log: Annotated[list[str], operator.add]    # human-readable trace of the run
