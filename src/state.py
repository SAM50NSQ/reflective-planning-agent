"""
state.py  -  Shared state for the planning agent.

Stage 2 change (IMPORTANT): `proposed_plan` is now a plain field that REPLACES
on each update. In Stage 0/1 it had an `operator.add` reducer (append), but once
the agent revises a plan, appending would stack the new proposals on top of the
old broken ones. Revision must replace. Only `log` keeps the append reducer,
because a trace genuinely should accumulate. This is the reducer gotcha flagged
back in Stage 0, now resolved deliberately.
"""

from __future__ import annotations
from typing import TypedDict, Annotated, Optional
import operator


class CalendarEvent(TypedDict):
    id: str
    title: str
    start: str
    end: str


class AgentState(TypedDict):
    # --- inputs ---
    goal: str
    calendar: list[CalendarEvent]

    # --- working state ---
    proposed_plan: list[dict]              # REPLACED each plan/revision (no reducer)
    findings: list[dict]                   # critique output (rule + judge), replaced
    revision_feedback: Optional[str]       # what the reviser tells the planner to fix
    critique: dict
    passed: bool
    revision_count: int
    max_revisions: int

    # --- trace (accumulates) ---
    log: Annotated[list[str], operator.add]
