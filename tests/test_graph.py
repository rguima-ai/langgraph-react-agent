"""Prueba del ciclo completo sin consumir tokens ni necesitar una API key."""

import json
from pathlib import Path
from typing import Any

import pytest
from conftest import FIRST_QUESTION, FOLLOW_UP_QUESTION, ScriptedModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.errors import GraphRecursionError

from react_agent.graph import build_graph


def _config(thread_id: str = "test-thread") -> Any:
    return {"configurable": {"thread_id": thread_id}, "recursion_limit": 10}


def _shape(messages: list[Any]) -> list[str]:
    """Resume el historial como secuencia ReAct: human, tool_call, tool, answer."""

    shape = []
    for message in messages:
        if isinstance(message, HumanMessage):
            shape.append("human")
        elif isinstance(message, AIMessage) and message.tool_calls:
            shape.append("tool_call")
        elif isinstance(message, ToolMessage):
            shape.append("tool")
        else:
            shape.append("answer")
    return shape


@pytest.mark.asyncio
async def test_graph_executes_two_tool_cycles_and_persists_state(
    tmp_path: Path, scripted_model: ScriptedModel
) -> None:
    database_path = tmp_path / "test-checkpoints.sqlite"

    async with AsyncSqliteSaver.from_conn_string(str(database_path)) as checkpointer:
        await checkpointer.setup()
        app = build_graph(model=scripted_model, checkpointer=checkpointer)

        result = await app.ainvoke(
            {"messages": [HumanMessage(content=FIRST_QUESTION)]}, config=_config()
        )
        saved_state = await app.aget_state(_config())

    messages = result["messages"]
    # Dos ciclos completos agent -> tools -> agent antes de la respuesta final.
    assert _shape(messages) == ["human", "tool_call", "tool", "tool_call", "tool", "answer"]

    observations = [json.loads(m.content) for m in messages if isinstance(m, ToolMessage)]
    assert [o["status"] for o in observations] == ["success", "success"]
    assert [o["results"][0]["id"] for o in observations] == [
        "cliente-102-cantidad",
        "cliente-102-total",
    ]
    assert messages[-1].content.startswith("El cliente 102 tiene 3 pedidos")
    assert saved_state.values["messages"] == messages
    assert scripted_model.bind_kwargs == {"parallel_tool_calls": False}


@pytest.mark.asyncio
async def test_same_thread_id_recovers_context_after_reopening_sqlite(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "memory.sqlite"

    async with AsyncSqliteSaver.from_conn_string(str(database_path)) as checkpointer:
        await checkpointer.setup()
        app = build_graph(model=ScriptedModel(), checkpointer=checkpointer)
        await app.ainvoke({"messages": [HumanMessage(content=FIRST_QUESTION)]}, _config())

    # Nueva conexión y nuevo modelo: lo único compartido es el archivo SQLite.
    second_model = ScriptedModel()
    async with AsyncSqliteSaver.from_conn_string(str(database_path)) as checkpointer:
        app = build_graph(model=second_model, checkpointer=checkpointer)
        result = await app.ainvoke(
            {"messages": [HumanMessage(content=FOLLOW_UP_QUESTION)]}, _config()
        )
        other_thread = await app.ainvoke(
            {"messages": [HumanMessage(content=FOLLOW_UP_QUESTION)]},
            _config("otro-thread"),
        )

    assert second_model.received[0][1].content == FIRST_QUESTION
    assert _shape(result["messages"]) == [
        "human", "tool_call", "tool", "tool_call", "tool", "answer",
        "human", "tool_call", "tool", "answer",
    ]
    assert "PED-9003" in result["messages"][-1].content
    # Otro thread_id no comparte historial.
    assert other_thread["messages"][-1].content == "¿De qué cliente me hablás?"


class InvalidArgsModel:
    """Pide la herramienta con argumentos que Pydantic rechaza y luego responde."""

    def bind_tools(self, tools: Any, **kwargs: Any) -> "InvalidArgsModel":
        return self

    async def ainvoke(self, messages: list[Any]) -> AIMessage:
        if any(isinstance(m, ToolMessage) for m in messages):
            return AIMessage(content="No pude consultar la base.")
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "search_knowledge_base",
                    "args": {"query": "x", "limit": 10},
                    "id": "call-invalida",
                    "type": "tool_call",
                }
            ],
        )


@pytest.mark.asyncio
async def test_tool_node_turns_invalid_arguments_into_an_observation() -> None:
    async with AsyncSqliteSaver.from_conn_string(":memory:") as checkpointer:
        app = build_graph(model=InvalidArgsModel(), checkpointer=checkpointer)
        result = await app.ainvoke({"messages": [HumanMessage(content="hola")]}, _config())

    observation = result["messages"][2]
    assert isinstance(observation, ToolMessage)
    assert observation.status == "error"
    assert "limit" in str(observation.content)
    assert result["messages"][-1].content == "No pude consultar la base."


class EndlessModel:
    """Nunca deja de pedir herramientas: simula un agente en bucle."""

    def bind_tools(self, tools: Any, **kwargs: Any) -> "EndlessModel":
        return self

    async def ainvoke(self, messages: list[Any]) -> AIMessage:
        call_id = f"call-{len(messages)}"
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "search_knowledge_base",
                    "args": {"query": "cliente 102", "limit": 1},
                    "id": call_id,
                    "type": "tool_call",
                }
            ],
        )


@pytest.mark.asyncio
async def test_recursion_limit_stops_an_endless_cycle() -> None:
    async with AsyncSqliteSaver.from_conn_string(":memory:") as checkpointer:
        app = build_graph(model=EndlessModel(), checkpointer=checkpointer)
        with pytest.raises(GraphRecursionError):
            await app.ainvoke({"messages": [HumanMessage(content="hola")]}, _config())
