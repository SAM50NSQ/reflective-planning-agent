"""
calendar_mock.py  -  A fake calendar the agent reads and proposes changes to.

This deliberately replaces any real Google Calendar integration. It is a plain
in-memory list of events. Everything the agent needs to reason about
(overlaps, free slots, working hours) can be checked against this. Real
Google Calendar (OAuth, token refresh, live writes) is intentionally deferred
to 'future work' - it demonstrates integration, not agent design, and writing
to a real calendar during testing is a foot-gun.

Later stages will add helper functions here (find_free_slots, has_overlap,
etc.) that the deterministic rule-checks call. Stage 0 just provides the data.
"""

from __future__ import annotations
from src.state import CalendarEvent


def seed_calendar() -> list[CalendarEvent]:
    """A small, deliberately messy week so later critique rules have something
    real to catch (a tight gap, a busy morning, etc.)."""
    return [
        {"id": "e1",
         "title": "Standup",
         "start": "2026-07-08T09:30",
         "end": "2026-07-08T09:45"},
        {"id": "e2",
         "title": "Design review",
         "start": "2026-07-08T10:00",
         "end": "2026-07-08T11:00"},
        {"id": "e3",
         "title": "Lunch",
         "start": "2026-07-08T13:00",
         "end": "2026-07-08T14:00"},
        {"id": "e4",
         "title": "1:1 with lead",
         "start": "2026-07-08T16:00",
         "end": "2026-07-08T16:30"},
    ]


# Working-hours bounds the rule-checker will enforce from Stage 2 onward.
WORKING_HOURS = {"start": "09:00", "end": "18:00"}
