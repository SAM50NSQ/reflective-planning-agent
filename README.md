# Reflective Planning Agent

A planning agent that proposes a constraint-satisfying schedule and then
**critiques and repairs its own plan**. Calendar is the domain; the point is the
self-correction loop and the split between deterministic rule-checks and LLM
judgement.

> Status: **Stage 2 - the self-correction loop is real.** Hand-wired ReAct loop,
> Pydantic structured output, deterministic rules + LLM-judge, regenerate-on-fail.

## What the agent does

```
intake -> plan -> critique -> (revise -> plan)* -> finalize
                     |                 ^
   rules (code) + judge (LLM) ---------+  fail -> feed findings back, replan
                                          pass or hit cap -> finalize
```

- **plan** (`src/planner.py`): a hand-wired reason/act/observe loop (`agent_loop.py`)
  uses tools to inspect the calendar, then a structured-output call returns a
  validated `Plan`. No prebuilt agent; the loop is explicit and capped.
- **critique** (`src/critique.py`): deterministic `rule_checks` (overlaps, buffers,
  working hours, end-after-start, self-overlap) PLUS one LLM `judge` call for soft
  quality. Passes only if no rule errors AND the judge accepts.
- **revise -> plan**: on failure the findings are fed back and the planner
  regenerates, capped by `max_revisions` (guaranteed termination).

## Design opinions (the point)

- **Hard rules for hard constraints; an LLM for judgement.** Overlaps are code;
  "is this sensible?" is the model. Using each where it is strong is the thesis.
- **Structured output, not text parsing.** `with_structured_output(schema)` gives
  validated Pydantic objects, killing Stage 1's fragile JSON parsing.
- **You own the loop.** Hand-wired (not `create_react_agent`, which is deprecated
  in LangGraph 1.0), so its termination is explicit and explainable.
- **Truthful cost.** Every model call's tokens are metered (`costs.py`).
- **Propose, don't act.** `reschedule_event` only proposes; nothing is applied.

## Run it

```bash
python -m venv venv
source venv/Scripts/activate            # Windows Git Bash
pip install -r requirements.txt
cp .env.example .env                     # then set a real model id + key for live

# MOCK (free): first plan is deliberately bad; watch it get caught and revised
python main.py "Find me two hours of deep work before my 1:1"

# LIVE: set USE_MOCK_LLM=false + PLANNER_MODEL/JUDGE_MODEL + ANTHROPIC_API_KEY
python main.py "Find me two hours of deep work before my 1:1"
```

## Known limits / next

- Live path (tool loop + judge) needs a valid model id in `.env`; verify the
  cost line shows non-zero tokens on the first live run.
- **Stage 3**: conversational input + a structured user profile that informs
  planning, plus an act-vs-ask policy for when to auto-apply vs confirm.
- **Future**: real Google Calendar, richer rules (travel buffers, deadlines).
