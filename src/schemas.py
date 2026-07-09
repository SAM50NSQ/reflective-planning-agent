"""
schemas.py  -  Pydantic models that force the LLM into structured output.

These replace the fragile "ask for JSON and parse the text" approach from
Stage 1. `with_structured_output(Plan)` makes the model return a validated Plan
object; no more parse-failures on stray prose. Same for the judge's verdict.
"""

from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, Field


class Proposal(BaseModel):
    action: Literal["add", "move", "remove"] = Field(
        description="add a new block, move an existing one, or remove an existing one")
    event_id: str = Field(default="", description="REQUIRED for 'move' and 'remove': id of the existing event")
    title: str
    start: str = Field(default="", description="ISO datetime, e.g. 2026-07-08T14:00 (ignored for 'remove')")
    end: str = Field(default="", description="ISO datetime, e.g. 2026-07-08T16:00 (ignored for 'remove')")
    reason: str = Field(description="why this proposal, in one sentence")


class Plan(BaseModel):
    """The planner's output.

    Three distinct outcomes, deliberately NOT collapsed into "proposals is empty":
      - proposals non-empty            -> a real plan to critique
      - needs_input=True + question    -> a critical detail is missing; ASK the user
      - no_change_needed=True          -> the calendar already satisfies the goal

    Earlier this was overloaded: `len(proposals) == 0` meant BOTH "ask me" and
    "nothing to do", so the agent asked a useless generic question instead of
    surfacing its own reasoning. An agent with no way to express the right answer
    will confabulate one.
    """
    proposals: list[Proposal] = Field(default_factory=list)
    summary: str = ""
    needs_input: bool = Field(default=False, description="true if a critical detail is missing")
    question: str = Field(default="", description="the single question to ask the user, if needs_input")
    no_change_needed: bool = Field(default=False, description="true only if the calendar already satisfies the goal")


class JudgeVerdict(BaseModel):
    """The LLM-judge's soft assessment. Deterministic rule-checks are handled
    separately in code; the judge only weighs in on things rules cannot, like
    whether the plan is sensible and well prioritised for the stated goal."""
    acceptable: bool = Field(description="is the plan sensible and good enough to deliver?")
    findings: list[str] = Field(default_factory=list, description="soft issues, if any")
    rationale: str = ""
