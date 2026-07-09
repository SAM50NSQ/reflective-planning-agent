"""
tests/smoke.py  -  Run this before every push. Costs nothing, hits no API.

    python -m tests.smoke

Why it exists: a refactor once deleted the judge's system prompt (JUDGE_SYS) and
nothing caught it, because the mock path never calls the judge. The error only
appeared on a LIVE run, after real money had been spent. Tests that only cover
the mock path do not cover the code that costs money.

Covers:
  1. every module imports (catches missing module-level names)
  2. the live judge path, with a STUB model (catches prompt/format errors)
  3. the deterministic rule engine, including the post-plan simulation
  4. every edge out of `critique` terminates
"""

from __future__ import annotations
import asyncio
import importlib
import sys

MODULES = ["src.schemas", "src.state", "src.config", "src.costs", "src.tools",
           "src.agent_loop", "src.calendar_mock", "src.critique", "src.planner",
           "src.graph", "src.logging_setup"]

failures: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        print(f"  ok    {name}")
    else:
        failures.append(name)
        print(f"  FAIL  {name}  {detail}")


def test_imports() -> None:
    print("\n[1] modules import")
    for m in MODULES:
        try:
            importlib.import_module(m)
            check(m, True)
        except Exception as e:  # noqa: BLE001
            check(m, False, repr(e))


def test_judge_path() -> None:
    """Exercise the LIVE judge path without an API key or a cent of spend."""
    print("\n[2] live judge path (stub model)")
    from src import critique
    from src.schemas import JudgeVerdict
    from src.calendar_mock import seed_calendar

    captured: dict = {}

    async def fake_structured_call(model, schema, messages, role):
        captured["system"] = messages[0].content
        captured["human"] = messages[1].content
        return JudgeVerdict(acceptable=True, findings=[], rationale="stub")

    orig_sc, orig_gm = critique.structured_call, critique.get_model
    critique.structured_call = fake_structured_call
    critique.get_model = lambda role: object()
    try:
        v = asyncio.run(critique._judge(
            "lunch with a client",
            [{"action": "add", "title": "Client Lunch",
              "start": "2026-07-08T13:00", "end": "2026-07-08T14:00", "reason": "r"}],
            seed_calendar(),
        ))
        check("judge returns a verdict", v is not None and v.acceptable is True)
        check("judge prompt has the date", "2026-07-08" in captured.get("system", ""))
        check("judge prompt states scope", "CANNOT" in captured.get("system", ""))
        check("judge sees today's events", "Design review" in captured.get("human", ""))
    except Exception as e:  # noqa: BLE001
        check("judge path executes", False, repr(e))
    finally:
        critique.structured_call, critique.get_model = orig_sc, orig_gm


def test_rules() -> None:
    print("\n[3] rule engine (post-plan simulation)")
    from src.critique import rule_checks
    from src.calendar_mock import seed_calendar
    cal = seed_calendar()
    D = "2026-07-08"

    def n(props):
        return len(rule_checks(props, cal))

    # the bug that vetoed a correct plan: move Lunch aside, add in the freed slot
    check("move+add in freed slot passes", n([
        {"action": "move", "event_id": "e3", "title": "Lunch",
         "start": f"{D}T14:00", "end": f"{D}T15:00", "reason": "r"},
        {"action": "add", "title": "Client Lunch",
         "start": f"{D}T13:00", "end": f"{D}T14:00", "reason": "r"},
    ]) == 0)

    check("bare add onto Lunch fails", n([
        {"action": "add", "title": "Client Lunch",
         "start": f"{D}T13:00", "end": f"{D}T14:00", "reason": "r"}]) == 1)
    check("out of working hours fails", n([
        {"action": "add", "title": "DW", "start": f"{D}T08:00", "end": f"{D}T09:00", "reason": "r"}]) == 1)
    check("end before start fails", n([
        {"action": "add", "title": "DW", "start": f"{D}T15:00", "end": f"{D}T14:00", "reason": "r"}]) == 1)
    check("move onto the 1:1 fails", n([
        {"action": "move", "event_id": "e3", "title": "Lunch",
         "start": f"{D}T16:00", "end": f"{D}T16:30", "reason": "r"}]) == 1)
    check("self-collision fails", n([
        {"action": "add", "title": "A", "start": f"{D}T14:00", "end": f"{D}T15:00", "reason": "r"},
        {"action": "add", "title": "B", "start": f"{D}T14:30", "end": f"{D}T15:30", "reason": "r"}]) == 1)
    check("clean add passes", n([
        {"action": "add", "title": "DW", "start": f"{D}T14:00", "end": f"{D}T16:00", "reason": "r"}]) == 0)

    # replace = remove the old event + add the new one in the freed slot
    check("remove+add (replace) passes", n([
        {"action": "remove", "event_id": "e3", "title": "Lunch", "reason": "r"},
        {"action": "add", "title": "Client Lunch",
         "start": f"{D}T13:00", "end": f"{D}T14:00", "reason": "r"},
    ]) == 0)
    check("remove of a non-existent event fails", n([
        {"action": "remove", "event_id": "zzz", "title": "Ghost", "reason": "r"}]) == 1)
    check("remove alone passes", n([
        {"action": "remove", "event_id": "e3", "title": "Lunch", "reason": "r"}]) == 0)


def test_termination() -> None:
    """Every exit edge from `critique` must terminate. Two of them once did not."""
    print("\n[4] loop termination")
    from src.graph import route_after_critique
    base = dict(passed=False, proposed_plan=[], revision_count=0, max_revisions=3,
                clarify_count=0, max_clarifications=2, stall=False,
                needs_info=True, pending_question="q", confirmed=False)

    check("asks when it needs info", route_after_critique({**base}) == "clarify")
    check("clarify is capped", route_after_critique({**base, "clarify_count": 2}) == "finalize")
    check("clarify yields to revision cap",
          route_after_critique({**base, "revision_count": 3}) == "finalize")

    passed_move = {**base, "passed": True, "needs_info": False, "pending_question": None,
                   "proposed_plan": [{"action": "move"}]}
    passed_remove = {**base, "passed": True, "needs_info": False, "pending_question": None,
                     "proposed_plan": [{"action": "remove"}]}
    check("high-stakes move asks to confirm", route_after_critique(passed_move) == "confirm")
    check("confirm is capped",
          route_after_critique({**passed_move, "revision_count": 3}) == "finalize")
    check("already-confirmed does not re-ask",
          route_after_critique({**passed_move, "confirmed": True}) == "finalize")
    check("removal also asks to confirm", route_after_critique(passed_remove) == "confirm")
    check("low-stakes plan finalizes silently",
          route_after_critique({**base, "passed": True, "needs_info": False,
                                "pending_question": None,
                                "proposed_plan": [{"action": "add"}]}) == "finalize")
    check("fixable failure revises",
          route_after_critique({**base, "needs_info": False, "pending_question": None}) == "revise")


def test_feedback_survives_clarify() -> None:
    """The judge's findings must reach the planner after a clarification.

    Regression: clarify_node used to return revision_feedback=None, wiping the
    critic's findings. The planner then replanned blind and took the
    `no_change_needed` escape hatch. The system threw away its own critic's catch.
    """
    print("\n[5] clarify preserves critic feedback")
    from src import graph

    orig = graph.interrupt
    graph.interrupt = lambda payload: "the user's answer"
    try:
        state = {"pending_question": "when?", "clarify_count": 0,
                 "max_clarifications": 2, "revision_feedback": "- [judge] propose a change"}
        out = asyncio.run(graph.clarify_node(state))
        check("clarify does not null revision_feedback", "revision_feedback" not in out)
        check("clarify records the answer", "the user's answer" in out["gathered_info"][0])
        check("clarify increments its counter", out["clarify_count"] == 1)
    except Exception as e:  # noqa: BLE001
        check("clarify_node executes", False, repr(e))
    finally:
        graph.interrupt = orig


def main() -> int:
    test_imports()
    test_judge_path()
    test_rules()
    test_termination()
    test_feedback_survives_clarify()
    print("\n" + ("ALL PASS" if not failures else f"{len(failures)} FAILURE(S): {failures}"))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
