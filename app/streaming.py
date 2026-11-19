"""Server-sent events for /ask/stream: one event per graph step, so the page can animate it."""

import json
from collections.abc import AsyncIterator
from typing import Any

from pinecone import PineconeError

from app import llm
from app.graph import AgentState
from app.logs import for_tenant

# The parts of each node's output the visualizer needs. Full chunk lists stay on the server.
STEP_FIELDS = ("retrieval", "grades", "search_query", "answer", "found_in_corpus", "citations")


def sse(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload)}\n\n"


def step_event(node: str, output: dict[str, Any]) -> dict[str, Any]:
    trace = output.get("trace") or [{"detail": ""}]
    event = {"type": "step", "node": node, "detail": trace[-1]["detail"]}
    event.update({field: output[field] for field in STEP_FIELDS if field in output})
    return event


def error_event(exc: Exception) -> dict[str, Any]:
    if isinstance(exc, llm.LLMTimeoutError):
        error = "llm_timeout"
    elif isinstance(exc, llm.LLMError):
        error = "llm_error"
    else:
        error = "vector_db_error"
    return {"type": "error", "error": error, "detail": str(exc) or type(exc).__name__}


async def ask_events(graph: Any, state: AgentState) -> AsyncIterator[str]:
    """Headers are already sent once streaming starts, so errors become an event, not a status."""
    try:
        async for update in graph.astream(state, stream_mode="updates"):
            for node, output in update.items():
                yield sse(step_event(node, output))
    except (llm.LLMError, PineconeError) as exc:
        for_tenant(state["tenant_id"]).error("ask stream failed: %s", exc)
        yield sse(error_event(exc))
    yield sse({"type": "done"})
