"""
agent_loop.py  -  Hand-wired reason/act/observe loop + structured-call helper.

Both helpers meter tokens via `costs.tracker` (referenced through the module so
the reset-aliasing bug can't recur) and log every model call and tool call so
failures are traceable.
"""

from __future__ import annotations
import logging
from typing import Any

from langchain_core.messages import AIMessage, ToolMessage
from src.tools import ALL_TOOLS
from src import costs                       # module reference (NOT `from costs import tracker`)
from src.config import model_name_for

log = logging.getLogger("agent.loop")
_TOOLS_BY_NAME = {t.name: t for t in ALL_TOOLS}


def _record(message: Any, role: str) -> None:
    um = getattr(message, "usage_metadata", None)
    if um:
        costs.tracker.add(model_name_for(role), um.get("input_tokens", 0), um.get("output_tokens", 0))
    else:
        log.debug("no usage_metadata on %s response", role)


async def run_tool_loop(model, messages: list, role: str = "planner", max_steps: int = 5) -> list:
    """Reason/act/observe until the model stops calling tools or the cap is hit."""
    bound = model.bind_tools(ALL_TOOLS)
    for step in range(max_steps):
        log.info("tool-loop step %d/%d (role=%s)", step + 1, max_steps, role)
        ai: AIMessage = await bound.ainvoke(messages)
        _record(ai, role)
        messages.append(ai)

        if not ai.tool_calls:
            log.info("tool-loop: model made no tool call -> done")
            break

        for tc in ai.tool_calls:
            log.info("tool-call: %s args=%s", tc["name"], tc["args"])
            tool = _TOOLS_BY_NAME.get(tc["name"])
            try:
                result = await tool.ainvoke(tc["args"]) if tool else f"unknown tool {tc['name']}"
            except Exception as e:  # noqa: BLE001
                result = f"tool error: {e}"
                log.warning("tool %s raised: %s", tc["name"], e)
            log.debug("tool-result: %s -> %s", tc["name"], str(result)[:200])
            messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
    return messages


async def structured_call(model, schema, messages: list, role: str):
    """One model call returning a validated Pydantic object; metered + logged."""
    log.info("structured_call role=%s schema=%s", role, schema.__name__)
    runnable = model.with_structured_output(schema, include_raw=True)
    out = await runnable.ainvoke(messages)
    if isinstance(out, dict):
        if out.get("raw") is not None:
            _record(out["raw"], role)
        if out.get("parsing_error"):
            log.warning("structured_call parsing_error: %s", out["parsing_error"])
        return out.get("parsed")
    return out
