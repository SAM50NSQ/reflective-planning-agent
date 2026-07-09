"""
critique.py  -  Hybrid self-critique: deterministic rules + LLM-judge.
Stage 2.1 adds: stall detection (no progress across revisions) and needs-info
detection (planner returned no usable proposal), so the loop stops early instead
of burning the revision cap re-failing identically. Plus per-step logging.
"""

from __future__ import annotations
import json
import logging
from datetime import datetime

from langchain_core.messages import SystemMessage, HumanMessage

from src.state import AgentState
from src.config import USE_MOCK_LLM, get_model, REFERENCE_DATE
from src.calendar_mock import WORKING_HOURS
from src.schemas import JudgeVerdict
from src.agent_loop import structured_call

log = logging.getLogger("agent.critique")
_FMT = "%Y-%m-%dT%H:%M"


def _p(ts: str) -> datetime:
    return datetime.strptime(ts, _FMT)


def apply_plan(proposals: list[dict], calendar: list[dict]) -> list[dict]:
    """Return the calendar AS IT WOULD BE if the plan were applied.

    This is the fix for a real bug: the checker used to compare proposals against
    the ORIGINAL calendar, so a plan that moved 'Lunch' out of the way and then
    added 'Client Lunch' in the freed slot was rejected for "overlapping Lunch".
    A deterministic checker is only trustworthy if it evaluates the POST-plan
    world. Its confidence makes a false positive worse than an LLM's hedging.

    Moves are matched by `event_id` when present, else by title (fragile; the
    schema now asks the planner for the id).
    """
    gone_ids, gone_titles = set(), set()
    for p in proposals:
        if p.get("action") in ("move", "remove"):
            if p.get("event_id"):
                gone_ids.add(p["event_id"])
            else:
                gone_titles.add((p.get("title") or "").strip().lower())

    survivors = [e for e in calendar
                 if e["id"] not in gone_ids
                 and e["title"].strip().lower() not in gone_titles]
    return survivors


def rule_checks(proposals: list[dict], calendar: list[dict]) -> list[dict]:
    """Deterministic checks against the calendar as the plan would leave it."""
    out: list[dict] = []

    def err(msg: str) -> None:
        out.append({"source": "rule", "severity": "error", "message": msg})

    remaining = apply_plan(proposals, calendar)

    ids = {e["id"] for e in calendar}
    titles = {e["title"].strip().lower() for e in calendar}

    for p in proposals:
        title = p.get("title", "?")

        if p.get("action") == "remove":
            # A remove occupies no time; validate only that it names a real event.
            eid, t = p.get("event_id", ""), (p.get("title") or "").strip().lower()
            if eid not in ids and t not in titles:
                err(f"cannot remove {title!r}: no such event on the calendar")
            continue

        try:
            s, e = _p(p["start"]), _p(p["end"])
        except Exception:
            err(f"{title!r}: unparseable start/end time")
            continue
        if e <= s:
            err(f"{title!r}: end is not after start")
        day = p["start"][:10]
        ws, we = _p(f"{day}T{WORKING_HOURS['start']}"), _p(f"{day}T{WORKING_HOURS['end']}")
        if s < ws or e > we:
            err(f"{title!r}: outside working hours ({WORKING_HOURS['start']}-{WORKING_HOURS['end']})")
        for ev in remaining:              # <-- post-plan calendar, not the original
            if not ev["start"].startswith(day):
                continue
            es, ee = _p(ev["start"]), _p(ev["end"])
            if s < ee and es < e:
                err(f"{title!r} overlaps existing event {ev['title']!r}")

    # proposals must not collide with each other either
    timed = [q for q in proposals if q.get("action") != "remove"]
    for i in range(len(timed)):
        for j in range(i + 1, len(timed)):
            proposals_i, proposals_j = timed[i], timed[j]
            try:
                s1, e1 = _p(proposals_i["start"]), _p(proposals_i["end"])
                s2, e2 = _p(proposals_j["start"]), _p(proposals_j["end"])
                if s1 < e2 and s2 < e1:
                    err(f"proposals {proposals_i.get('title')!r} and {proposals_j.get('title')!r} overlap")
            except Exception:
                pass
    return out


JUDGE_SYS = """You are a scheduling critic. You judge ONLY whether a proposed set of
calendar changes is sensible for the goal.

Context you can rely on:
- Today's date is {today}. Any proposal dated {today} IS today. Never object that
  a date "might not be today".
- The user's existing events for today are listed below.
- The agent can ONLY add a new block or move an existing one. It CANNOT invite
  people, email anyone, book venues, or confirm third-party availability. Do NOT
  fail a plan for omitting those; they are outside its scope.
- Hard constraints (overlaps, working hours, end-after-start) are ALREADY checked
  by deterministic code against the post-plan calendar. Do not re-check them.
  Judge only soft quality: is the timing sensible, is the priority reasonable,
  does the plan actually serve the goal?

If the proposal list is empty, say so plainly and set acceptable=false.
Be brief. Do not invent facts about the user.
"""


async def _judge(goal: str, proposals: list[dict], calendar: list[dict]) -> JudgeVerdict | None:
    model = get_model("judge")
    today_events = "; ".join(f"{e['title']} {e['start'][11:]}-{e['end'][11:]}"
                             for e in calendar if e["start"].startswith(REFERENCE_DATE)) or "(none)"
    msgs = [
        SystemMessage(content=JUDGE_SYS.format(today=REFERENCE_DATE)),
        HumanMessage(content=(f"Goal: {goal}\n"
                              f"Today's existing events: {today_events}\n"
                              f"Proposals: {json.dumps(proposals)}")),
    ]
    return await structured_call(model, JudgeVerdict, msgs, role="judge")


def _feedback(findings: list[dict]) -> str:
    return "\n".join(f"- [{f['severity']}] {f['message']}" for f in findings)


async def critique_node(state: AgentState) -> dict:
    log.info("NODE critique (revision=%d)", state["revision_count"])
    proposals = state["proposed_plan"]
    rule_findings = rule_checks(proposals, state["calendar"])

    if USE_MOCK_LLM:
        verdict = JudgeVerdict(acceptable=True, findings=[], rationale="(mock) auto-accept")
    else:
        verdict = await _judge(state["goal"], proposals, state["calendar"])

    if verdict:
        log.info("judge verdict: acceptable=%s | rationale: %s", verdict.acceptable, verdict.rationale)
        for f in verdict.findings:
            log.info("judge finding: %s", f)
    else:
        log.warning("judge returned nothing")

    judge_findings = [{"source": "judge", "severity": "warning", "message": m}
                      for m in (verdict.findings if verdict else ["judge returned nothing"])]
    all_findings = rule_findings + judge_findings
    has_rule_error = any(f["severity"] == "error" for f in rule_findings)
    judge_ok = verdict.acceptable if verdict else False
    passed = (not has_rule_error) and judge_ok and len(proposals) > 0   # empty plan never passes

    # --- Stage 2.1: early-stop signals ---
    sig = "|".join(sorted(f["message"] for f in all_findings))
    prev = state.get("prev_findings_sig")
    no_progress = (state["revision_count"] > 0 and sig == prev and not passed)
    needs_info = (len(proposals) == 0)   # planner refused / asked for info
    if no_progress:
        log.warning("critique: no progress since last revision -> will stop early")
    if needs_info:
        log.info("critique: planner returned no proposals -> will ask the user (clarify)")

    log.info("critique rules=%d (err=%s) judge_ok=%s -> %s",
             len(rule_findings), has_rule_error, judge_ok, "PASS" if passed else "FAIL")

    return {
        "findings": all_findings,
        "passed": passed,
        "prev_findings_sig": sig,
        "needs_info": needs_info,
        "stall": no_progress,
        "critique": {"passed": passed, "rule_errors": has_rule_error, "judge_ok": judge_ok},
        "revision_feedback": None if passed else _feedback(all_findings),
        "log": [f"[critique] rules={len(rule_findings)} (errors={has_rule_error}) "
                f"judge_ok={judge_ok} -> {'PASS' if passed else 'FAIL'}"],
    }
