"""
critique.py  -  The centrepiece: hybrid self-critique.

Two independent checks, each used where it is strong:

  1. rule_checks(...)  - DETERMINISTIC Python. Catches hard, objective errors:
     end-before-start, outside working hours, overlaps with existing events,
     overlaps among the proposals themselves. Reliable, free, explainable.

  2. judge(...)        - the LLM judging SOFT qualities rules cannot: is the plan
     sensible and well-prioritised for the goal? Structured output (JudgeVerdict).

A plan PASSES only if there are no rule errors AND the judge finds it acceptable.
That split - hard rules for hard constraints, an LLM for judgement - is the
"where each excels" opinion this project is meant to demonstrate.
"""

from __future__ import annotations
import json
from datetime import datetime

from langchain_core.messages import SystemMessage, HumanMessage

from src.state import AgentState
from src.config import USE_MOCK_LLM, get_model
from src.calendar_mock import WORKING_HOURS
from src.schemas import JudgeVerdict
from src.agent_loop import structured_call

_FMT = "%Y-%m-%dT%H:%M"


def _p(ts: str) -> datetime:
    return datetime.strptime(ts, _FMT)


def rule_checks(proposals: list[dict], calendar: list[dict]) -> list[dict]:
    """Return a list of finding dicts: {source, severity, message}."""
    out: list[dict] = []

    def err(msg: str) -> None:
        out.append({"source": "rule", "severity": "error", "message": msg})

    for p in proposals:
        title = p.get("title", "?")
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
            err(f"{title!r}: falls outside working hours ({WORKING_HOURS['start']}-{WORKING_HOURS['end']})")
        for ev in calendar:
            if not ev["start"].startswith(day):
                continue
            es, ee = _p(ev["start"]), _p(ev["end"])
            if s < ee and es < e:
                err(f"{title!r} overlaps existing event {ev['title']!r}")

    # overlaps among the proposals themselves
    for i in range(len(proposals)):
        for j in range(i + 1, len(proposals)):
            try:
                s1, e1 = _p(proposals[i]["start"]), _p(proposals[i]["end"])
                s2, e2 = _p(proposals[j]["start"]), _p(proposals[j]["end"])
                if s1 < e2 and s2 < e1:
                    err(f"proposals {proposals[i].get('title')!r} and {proposals[j].get('title')!r} overlap each other")
            except Exception:
                pass
    return out


async def _judge(goal: str, proposals: list[dict]) -> JudgeVerdict | None:
    model = get_model("judge")
    msgs = [
        SystemMessage(content="You are a scheduling critic. Judge whether the plan is sensible "
                              "and well-prioritised for the user's goal. Rules already checked "
                              "hard constraints; focus on judgement."),
        HumanMessage(content=f"Goal: {goal}\nProposals: {json.dumps(proposals)}"),
    ]
    return await structured_call(model, JudgeVerdict, msgs, role="judge")


def _feedback(findings: list[dict]) -> str:
    return "\n".join(f"- [{f['severity']}] {f['message']}" for f in findings)


async def critique_node(state: AgentState) -> dict:
    proposals = state["proposed_plan"]
    rule_findings = rule_checks(proposals, state["calendar"])

    if USE_MOCK_LLM:
        verdict = JudgeVerdict(acceptable=True, findings=[], rationale="(mock) judge auto-accepts")
    else:
        verdict = await _judge(state["goal"], proposals)

    judge_findings = [{"source": "judge", "severity": "warning", "message": m}
                      for m in (verdict.findings if verdict else ["judge returned nothing"])]
    all_findings = rule_findings + judge_findings

    has_rule_error = any(f["severity"] == "error" for f in rule_findings)
    judge_ok = verdict.acceptable if verdict else False
    passed = (not has_rule_error) and judge_ok

    return {
        "findings": all_findings,
        "passed": passed,
        "critique": {"passed": passed, "rule_errors": has_rule_error, "judge_ok": judge_ok},
        "revision_feedback": None if passed else _feedback(all_findings),
        "log": [f"[critique] rules={len(rule_findings)} (errors={has_rule_error}) "
                f"judge_ok={judge_ok} -> {'PASS' if passed else 'FAIL'}"],
    }
