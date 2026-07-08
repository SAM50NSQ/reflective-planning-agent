"""
agent_loop.py  -  The hand-wired reason/act/observe loop you OWN.

This replaces the deprecated `create_react_agent`. The loop is deliberately
small and explicit so you can explain exactly how it decides to stop:

    call model (with tools bound)
    -> if the model requested tools: execute them, append results, repeat
    -> if the model requested no tools: it is done, stop
    -> hard cap `max_steps` guarantees termination

Both helpers here record token usage into the cost meter on every model call,
so spend is tracked truthfully (the thing that was invisible in Stage 1).

`structured_call` uses `with_structured_output(schema, include_raw=True)`: the
`raw` message carries usage_metadata (so we can meter it) while `parsed` is the
validated Pydantic object (so no fragile text parsing).
"""

from __future__ import annotations
from typing import Any

from langchain_core.messages import AIMessage, ToolMessage
from src.tools import ALL_TOOLS
from src.costs import tracker
from src.config import model_name_for

_TOOLS_BY_NAME = {t.name: t for t in ALL_TOOLS}


def _record(message: Any, role: str) -> None:
    um = getattr(message, "usage_metadata", None)
    if um:
        tracker.add(model_name_for(role), um.get("input_tokens", 0), um.get("output_tokens", 0))


async def run_tool_loop(model, messages: list, role: str = "planner", max_steps: int = 5) -> list:
    """Reason/act/observe until the model stops calling tools or the cap is hit.
    Returns the full message list (including tool results)."""
    bound = model.bind_tools(ALL_TOOLS)
    for _ in range(max_steps):
        ai: AIMessage = await bound.ainvoke(messages)
        _record(ai, role)
        messages.append(ai)

        if not ai.tool_calls:      # model chose not to call a tool -> it is done
            break

        for tc in ai.tool_calls:   # execute each requested tool, observe results
            tool = _TOOLS_BY_NAME.get(tc["name"])
            try:
                result = await tool.ainvoke(tc["args"]) if tool else f"unknown tool {tc['name']}"
            except Exception as e:  # noqa: BLE001 - surface tool errors back to the model
                result = f"tool error: {e}"
            messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
    return messages


async def structured_call(model, schema, messages: list, role: str):
    """One model call that returns a validated Pydantic object AND is metered."""
    runnable = model.with_structured_output(schema, include_raw=True)
    out = await runnable.ainvoke(messages)
    if isinstance(out, dict):
        if out.get("raw") is not None:
            _record(out["raw"], role)
        return out.get("parsed")
    return out
