# Reflective Planning Agent

A planning agent that proposes a constraint-satisfying schedule and then
critiques and repairs its own plan before returning it. The calendar is the domain;
the point is the self-correction loop and the split between
deterministic rule-checks and LLM judgement.

> Status: 
Stage 0 - skeleton
Runnable, no LLM calls yet.

## Why

A reflection loop (produce -> critique -> revise, capped)
is the part of an agent that is easy to get wrong.
Two design choices:

> Hard rules + soft judgement.
Overlaps, buffers, deadline ordering and working-hours bounds
are checked by deterministic code (reliable, free,  explainable). 
"Is this schedule *sensible*?" is judged by an LLM. Neither alone is enough.
Using each where it is strong is the whole point.

> Guaranteed termination.
The critique loop is capped, so it can never spin
or burn tokens indefinitely.
(See `max_revisions` and `route_after_critique`.)

## Architecture (Stage 0)

```
START -> intake -> plan -> critique -> (revise -> critique)*  -> finalize -> END
                              |__________________________________^
                              route_after_critique: pass OR cap reached -> finalize
                                                    otherwise            -> revise
```

All nodes are currently stubs that log and return small state updates, so you
can watch the graph flow and the loop terminate before any model is added.

## Files

- `src/state.py` - the typed `AgentState` threaded through every node.
- `src/calendar_mock.py` - fake calendar the agent reads (no Google Calendar).
- `src/graph.py` - nodes, edges, and the critique-loop wiring.
- `main.py` - runs it and prints the trace.

## Run it

```bash
python -m venv venv
source venv/Scripts/activate      # Windows Git Bash;  use source venv/bin/activate on macOS/Linux
pip install -r requirements.txt
python main.py "Find me two hours of focus time before my 1:1"
```

Expected trace: `intake -> plan -> critique(FAIL) -> revise -> critique(PASS) -> finalize`.

## Roadmap

- Stage 1 - real ReAct planning node (LLM + mock tools).
- Stage 2 - real critique loop: deterministic rules + one LLM-judge pass.
- Stage 3 - conversational input + a structured user profile that informs planning, plus an explicit act-vs-ask policy.
- Future work - real Google Calendar integration, additional tools.
