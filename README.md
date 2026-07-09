# Reflective Planning Agent

A planning agent that proposes calendar changes, **critiques its own plan with two
independent critics**, revises, and asks a human before it touches anything that
matters.

The calendar is just a domain where constraints are legible. The point is the
control flow: a hand-wired ReAct loop, a hybrid critique (deterministic rules +
an LLM judge), bounded self-correction, and human-in-the-loop on high-stakes
actions.

```
START -> intake -> plan -> critique -> [clarify] -> plan
                            |  |          (ask the user, capped)
                            |  +-------> [revise] -> plan
                            |               (regenerate, capped)
                            +----------> [confirm] -> finalize -> END
                                         (approve destructive changes)
```

---

## Run it

```bash
python -m venv venv && source venv/Scripts/activate   # Git Bash on Windows
pip install -r requirements.txt
cp .env.example .env            # set a real model id + key for live mode

python -m tests.smoke           # free, no API. Must print ALL PASS.
python chat.py                  # conversational session
python main.py "Find me two hours of deep work"   # single goal
```

`USE_MOCK_LLM=true` (the default) runs the whole graph with canned model
responses at **$0**. Set it to `false` for live runs. A live turn costs roughly
$0.02 to $0.05.

---

## A real session

Verbatim, trimmed to the decisions. The calendar has `Lunch 13:00-14:00`.

```
you > Need to have lunch with a client today

planner  says no change needed: "The calendar already has a Lunch block from
         13:00-14:00 today, which satisfies the goal."
judge    acceptable=False: "The proposal list is empty ... the agent should
         either propose moving/removing that block, or propose a different slot."
         -> the planner's claim was false. The goal is lunch WITH A CLIENT.
route    -> clarify

  agent ? I think your calendar already covers this (...). Is that right, or
          what exactly should I schedule?
  you   > Replace my personal lunch with the client lunch

  agent ? What time should the client lunch be scheduled?
  you   > 1 to 2 in the afternoon

planner  2 proposals: remove Lunch (e3); add Client lunch 13:00-14:00
rules    0 errors        (checked against the POST-plan calendar)
judge    acceptable=True (timing sensible, serves the goal)
route    -> confirm      (a removal is destructive)

  agent ! This plan changes existing commitments (remove Lunch). Approve? (yes/no)
  you   > yes

  Plan:
    - remove: Lunch
        Removing personal lunch to make way for client lunch in the same slot.
    - add: Client lunch  [2026-07-08T13:00 -> 2026-07-08T14:00]
        Scheduling client lunch from 1 to 2 PM as requested.

  [cost] 11 call(s), est $0.0369
    claude-haiku-4-5: 8 calls    claude-sonnet-4-5: 3 calls
```

---

## Design decisions

**Two critics, because each is blind where the other sees.**
`rule_checks()` is plain Python: overlaps, working hours, end-after-start,
self-collision, removal of events that do not exist. Fast, free, explainable. The
LLM judge assesses what code cannot: is this timing sensible, does the plan
actually serve the goal.

**Rules evaluate the post-plan world, not the current one.**
`apply_plan()` simulates the proposals first. Without it, a correct two-step plan
(move `Lunch` aside, add `Client Lunch` in the freed slot) is rejected for
"overlapping Lunch". A checker that cannot simulate the change it is validating
will veto correct plans, and its confidence makes that worse than an LLM's
hedging.

**Every exit edge from `critique` has a termination argument.**
`max_revisions` bounds regeneration, `max_clarifications` bounds questions, and
`confirm` cannot be re-entered once capped. Loop safety is a property of every
edge, not of the node. Both loops here were once unbounded.

**Destructive actions require a human.**
`move` and `remove` route through `confirm`. A plain `add` into a free slot fires
no interrupt at all. Ask when it matters, act when it does not.

**Model choice is per-role and lives in `.env`.**
A cheap planner (`claude-haiku-4-5`) and a stronger judge (`claude-sonnet-4-5`)
are one config line each, via LangChain's `init_chat_model`. Swapping provider
changes no code. The judge is where quality is worth paying for.

**Structured output, not string parsing.**
The planner returns a validated Pydantic `Plan` with three distinct outcomes:
proposals, `needs_input` + `question`, or `no_change_needed`. Earlier,
`len(proposals) == 0` meant both "ask me" and "nothing to do", so the agent asked
a generic question instead of surfacing its own reasoning.

**The ReAct loop is hand-written** (`agent_loop.py`), not `create_react_agent`
(deprecated in LangGraph 1.0). About forty lines, and its termination is explicit.

---

## What actually broke

Every one of these was found by running the agent, not by reading the code.

**The rules produced a confident false positive.** They vetoed a correct
move-then-add plan that the judge had approved. The judge was right; the checker
was buggy. So much for "deterministic checks are always the authority".

**The judge caught planner fabrications the rules were structurally blind to.**
With zero proposals there is nothing for a rule to check. Across four runs the
planner asserted a "client lunch" already existed. The judge caught it every time.

**The judge is not deterministic, and it confabulates.** On one goal it rejected a
14:00-16:00 block for leaving no buffer before a 1:1. On a differently-phrased
goal it accepted the same block, praising how it "avoids early morning fatigue"
and allows "time to enter flow state", about a user it knows nothing about. Moving
from Haiku to Sonnet reduced the noise. It did not remove it. The deterministic
rules exist precisely because the judge cannot be trusted alone.

**An agent with no way to express the right answer will invent one.** The action
schema was `add | move` only. Asked to *replace* an event, the planner attempted a
rename, was rejected, then retreated to "no change needed" and hallucinated that
the personal lunch was the client lunch. Adding `remove` fixed the behaviour.
Telling a model what it cannot do, without telling it how to do the thing, just
makes it retreat.

**Mock mode saved money and hid bugs.** Every failure surfaced on the paid path: a
404 model id, a cost meter that silently under-reported, two unbounded loops, a
judge with no date context, a deleted system prompt. The mock path never called
the judge, so the code that cost money was the code with no coverage.
`tests/smoke.py` now stubs the model and exercises the live path for free.

---

## Files

| file | role |
| --- | --- |
| `src/graph.py` | nodes, edges, interrupts, termination guards |
| `src/agent_loop.py` | hand-wired reason/act/observe loop + metered structured calls |
| `src/planner.py` | the planning node: asks, proposes, or declares no-change |
| `src/critique.py` | `rule_checks` (deterministic) + `_judge` (LLM) |
| `src/schemas.py` | Pydantic contracts for the plan and the verdict |
| `src/costs.py` | per-model token and USD meter |
| `src/tools.py` | mock calendar tools, all proposal-only |
| `tests/smoke.py` | free pre-push guard: imports, judge path, rules, termination |

---

## Limitations

- The calendar is a mock. Nothing is mutated; every action is a proposal.
- No free-form repair. Each turn is a fresh goal; you cannot say "make it 90
  minutes instead". Answers to the agent's questions do carry across turns.
- `tests/smoke.py` is a pre-push guard, not an evaluation harness. It stubs the
  model, so it proves plumbing, not model quality.
- Prompting a small model not to confabulate is unreliable. The structural fix is
  to make the false claim unrepresentable, not to ask nicely.
- Moves and removes fall back to matching by title when `event_id` is absent.
  Fragile.

## Next

- FastAPI over the same graph: swap `MemorySaver` for a persistent checkpointer
  and map `thread_id` to a session. `one_turn()` is already shaped like a handler.
- A structured user profile so the agent stops re-asking what it has been told.
- Real Google Calendar behind the existing tool interface.
