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
    action: Literal["add", "move"] = Field(description="add a new block or move an existing one")
    title: str
    start: str = Field(description="ISO datetime, e.g. 2026-07-08T14:00")
    end: str = Field(description="ISO datetime, e.g. 2026-07-08T16:00")
    reason: str = Field(description="why this proposal, in one sentence")


class Plan(BaseModel):
    proposals: list[Proposal] = Field(default_factory=list)
    summary: str = ""


class JudgeVerdict(BaseModel):
    """The LLM-judge's soft assessment. Deterministic rule-checks are handled
    separately in code; the judge only weighs in on things rules cannot, like
    whether the plan is sensible and well prioritised for the stated goal."""
    acceptable: bool = Field(description="is the plan sensible and good enough to deliver?")
    findings: list[str] = Field(default_factory=list, description="soft issues, if any")
    rationale: str = ""
