from __future__ import annotations

import asyncio
from typing import Literal

from langchain.agents.middleware import AgentMiddleware
from langchain.tools import ToolRuntime
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import StructuredTool

from core.trajectory_bank.retrieval import BankRetrieval
from core.utilities.messages import system_message_with_appended_text


class TrajectoryBankMiddleware(AgentMiddleware):
    """Return historical evidence after reasoning, inside SessionDump's wrapper."""

    def __init__(self, retrieval: BankRetrieval) -> None:
        self.retrieval = retrieval

        def retrieve(runtime: ToolRuntime, query: str = "", ref: str = "",
                     scope: Literal["session", "workspace"] = "session", offset: int = 0) -> ToolMessage:
            try:
                text, ids = retrieval.retrieve(query=query, ref=ref, scope=scope, offset=offset)
                return ToolMessage(content=text, tool_call_id=runtime.tool_call_id, name="trajectory_bank",
                                   additional_kwargs={"trajectory_record_ids": ids})
            except Exception as error:
                return ToolMessage(content=f"Trajectory retrieval unavailable: {error}", status="error",
                                   tool_call_id=runtime.tool_call_id, name="trajectory_bank")

        async def aretrieve(runtime: ToolRuntime, query: str = "", ref: str = "",
                            scope: Literal["session", "workspace"] = "session", offset: int = 0) -> ToolMessage:
            return await asyncio.to_thread(retrieve, runtime, query, ref, scope, offset)

        self.tools = [StructuredTool.from_function(
            func=retrieve, coroutine=aretrieve, name="trajectory_bank",
            description=("Recall historical trajectory evidence by query OR exact ref. Defaults to this session; "
                         "scope='workspace' searches other local sessions. For ref pagination, offset is a record "
                         "index for arc refs or character position for record refs. Do not batch retrieval with dependent actions."),
        )]

    def wrap_model_call(self, request, handler):
        return handler(self._prepare(request))

    async def awrap_model_call(self, request, handler):
        return await handler(self._prepare(request))

    def _prepare(self, request):
        return request.override(system_message=system_message_with_appended_text(
            request.system_message,
            "The reasoning tool may return relevant historical trajectory evidence. Reassess it before acting; "
            "use trajectory_bank for deliberate recall. Call reasoning or retrieval separately from dependent actions.",
        ))

    def _deferred(self, request) -> ToolMessage | None:
        if request.tool_call["name"] in {"reasoning", "trajectory_bank"}:
            return None
        last = next((message for message in reversed(request.state.get("messages", [])) if isinstance(message, AIMessage)), None)
        if last is not None and any(call["name"] in {"reasoning", "trajectory_bank"} for call in last.tool_calls):
            return ToolMessage(content="Action deferred: inspect the reasoning/retrieval result, then call this tool in the next model step.",
                               status="error", tool_call_id=request.tool_call["id"], name=request.tool_call["name"])
        return None

    def _recall(self, request, result):
        if request.tool_call["name"] != "reasoning" or not isinstance(result, ToolMessage) or result.status == "error":
            return result
        query = request.tool_call["args"].get("reasoning", "").strip()
        if not query:
            return result
        loaded = {id for message in request.state.get("messages", [])
                  for id in (getattr(message, "additional_kwargs", {}) or {}).get("trajectory_record_ids", [])}
        try:
            text, ids = self.retrieval.retrieve(query=query, automatic=True, exclude=loaded)
        except Exception as error:
            return result.model_copy(update={"content": f"{result.content}\nHistorical recall unavailable: {error}"})
        if not ids:
            return result
        return result.model_copy(update={"content": f"{result.content}\n\n{text}",
                                         "additional_kwargs": {**result.additional_kwargs, "trajectory_record_ids": ids}})

    def wrap_tool_call(self, request, handler):
        deferred = self._deferred(request)
        return deferred if deferred is not None else self._recall(request, handler(request))

    async def awrap_tool_call(self, request, handler):
        deferred = self._deferred(request)
        if deferred is not None:
            return deferred
        result = await handler(request)
        return await asyncio.to_thread(self._recall, request, result)
