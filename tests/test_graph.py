"""Prueba del ciclo completo sin consumir tokens ni necesitar una API key."""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from react_agent.graph import build_graph


class ScriptedModel:
    """Modelo falso que obliga a recorrer agent -> tools -> agent dos veces."""

    def bind_tools(self, tools: Sequence[Any]) -> "ScriptedModel":
        self.tools = tools
        return self

    async def ainvoke(self, messages: list[Any]) -> AIMessage:
        observations = [message for message in messages if isinstance(message, ToolMessage)]

        if len(observations) == 0:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "search_knowledge_base",
                        "args": {"query": "cantidad de pedidos del cliente 102", "limit": 1},
                        "id": "call-cantidad",
                        "type": "tool_call",
                    }
                ],
            )

        if len(observations) == 1:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "search_knowledge_base",
                        "args": {"query": "total acumulado cliente 102", "limit": 1},
                        "id": "call-total",
                        "type": "tool_call",
                    }
                ],
            )

        return AIMessage(content="El cliente 102 tiene 3 pedidos por un total de 14500 ARS.")


@pytest.mark.asyncio
async def test_graph_executes_two_tool_cycles_and_persists_state(tmp_path: Path) -> None:
    database_path = tmp_path / "test-checkpoints.sqlite"
    config = {"configurable": {"thread_id": "test-thread"}, "recursion_limit": 10}

    async with AsyncSqliteSaver.from_conn_string(str(database_path)) as checkpointer:
        await checkpointer.setup()
        app = build_graph(model=ScriptedModel(), checkpointer=checkpointer)

        result = await app.ainvoke(
            {"messages": [HumanMessage(content="¿Cuántos pedidos y cuál es el total?")]},
            config=config,
        )
        saved_state = await app.aget_state(config)

    tool_messages = [
        message for message in result["messages"] if isinstance(message, ToolMessage)
    ]
    assert len(tool_messages) == 2
    assert result["messages"][-1].content.startswith("El cliente 102 tiene 3 pedidos")
    assert len(saved_state.values["messages"]) == len(result["messages"])
