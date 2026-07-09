"""state.py  -  Shared state. Stage 3 adds conversational fields."""
from __future__ import annotations
from typing import TypedDict, Annotated, Optional
import operator


class CalendarEvent(TypedDict):
    id: str
    title: str
    start: str
    end: str


class AgentState(TypedDict):
    goal: str
    calendar: list[CalendarEvent]

    proposed_plan: list[dict]              # REPLACED each plan/revision
    findings: list[dict]
    revision_feedback: Optional[str]
    critique: dict
    passed: bool
    revision_count: int
    max_revisions: int

    # early-stop signals
    prev_findings_sig: Optional[str]
    needs_info: bool
    stall: bool

    # Stage 3: human-in-the-loop
    clarify_count: int                     # how many times we've asked the user
    max_clarifications: int                # hard cap on questions per turn
    confirmed: bool                        # user approved a high-stakes move
    move_blocked: bool                     # user declined; planner must not move events

    # Stage 3: conversation / human-in-the-loop
    plan_summary: Optional[str]            # planner's own words, surfaced when asking
    pending_question: Optional[str]                     # planner asks -> clarify node surfaces it
    gathered_info: Annotated[list[str], operator.add]   # answers accumulate across turns

    log: Annotated[list[str], operator.add]
