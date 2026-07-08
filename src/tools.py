"""
tools.py  -  Async tools the planner can call. All operate on the MOCK calendar.

Four tools:
  - get_calendar(day)            read existing events for a day
  - find_free_slots(day, mins)   compute open gaps within working hours
  - check_conflicts(start, end)  list events that overlap a proposed slot
  - reschedule_event(...)        PROPOSE-ONLY: describe moving an event, with a
                                 rationale. It NEVER mutates the calendar.

The propose-only rule matters: moving someone's existing meeting is a
consequential action. In this design the *authority* to act lives in the
critique / act-vs-ask policy (later stages), not inside a tool the planner can
fire on its own. So this tool returns a proposal object; nothing is applied.
"""

from __future__ import annotations
from datetime import datetime, timedelta

from langchain_core.tools import tool
from src.calendar_mock import seed_calendar, WORKING_HOURS

_FMT = "%Y-%m-%dT%H:%M"


def _parse(ts: str) -> datetime:
    return datetime.strptime(ts, _FMT)


@tool
async def get_calendar(day: str) -> list[dict]:
    """Return existing calendar events for the given day (format YYYY-MM-DD)."""
    return [e for e in seed_calendar() if e["start"].startswith(day)]


@tool
async def find_free_slots(day: str, duration_minutes: int) -> list[dict]:
    """Find open time slots of at least duration_minutes on the given day
    (YYYY-MM-DD), within working hours. Returns a list of {start, end}."""
    events = sorted(
        (e for e in seed_calendar() if e["start"].startswith(day)),
        key=lambda e: e["start"],
    )
    work_start = _parse(f"{day}T{WORKING_HOURS['start']}")
    work_end = _parse(f"{day}T{WORKING_HOURS['end']}")

    free: list[dict] = []
    cursor = work_start
    for e in events:
        s, en = _parse(e["start"]), _parse(e["end"])
        if s - cursor >= timedelta(minutes=duration_minutes):
            free.append({"start": cursor.strftime(_FMT), "end": s.strftime(_FMT)})
        cursor = max(cursor, en)
    if work_end - cursor >= timedelta(minutes=duration_minutes):
        free.append({"start": cursor.strftime(_FMT), "end": work_end.strftime(_FMT)})
    return free


@tool
async def check_conflicts(start: str, end: str) -> list[dict]:
    """Return existing events that overlap the proposed [start, end] window.
    Times are ISO like 2026-07-08T14:00. Empty list means no conflict."""
    ps, pe = _parse(start), _parse(end)
    day = start[:10]
    hits = []
    for e in seed_calendar():
        if not e["start"].startswith(day):
            continue
        es, ee = _parse(e["start"]), _parse(e["end"])
        if ps < ee and es < pe:  # overlap
            hits.append(e)
    return hits


@tool
async def reschedule_event(event_id: str, new_start: str, new_end: str, reason: str) -> dict:
    """PROPOSE moving an existing event to a new time. Does NOT apply the change.
    Returns a proposal object for later review/confirmation. Use this only when
    a new high-priority item genuinely requires displacing an existing one, and
    always give a clear reason."""
    match = next((e for e in seed_calendar() if e["id"] == event_id), None)
    return {
        "kind": "reschedule_proposal",
        "applied": False,               # explicit: nothing changed
        "event_id": event_id,
        "event_title": match["title"] if match else "(unknown)",
        "from": {"start": match["start"], "end": match["end"]} if match else None,
        "to": {"start": new_start, "end": new_end},
        "reason": reason,
    }


ALL_TOOLS = [get_calendar, find_free_slots, check_conflicts, reschedule_event]
