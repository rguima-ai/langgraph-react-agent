"""Modelo guionado compartido: decide como un LLM, pero de forma determinista."""

from collections.abc import Sequence
from typing import Any

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

FIRST_QUESTION = "¿Cuántos pedidos tiene el cliente 102 y cuál es el total acumulado?"
FOLLOW_UP_QUESTION = "¿Y cuál fue el último?"


def _tool_call(query: str, call_id: str) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": "search_knowledge_base",
                "args": {"query": query, "limit": 1},
                "id": call_id,
                "type": "tool_call",
            }
        ],
    )


class ScriptedModel:
    """Recorre dos ciclos de herramienta en la primera pregunta y uno en la segunda.

    La segunda pregunta no nombra al cliente: el modelo solo sabe de quién se habla si el
    historial recuperado del checkpoint contiene la primera pregunta.
    """

    def __init__(self) -> None:
        self.bind_kwargs: dict[str, Any] = {}
        self.received: list[list[BaseMessage]] = []

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> "ScriptedModel":
        self.tools = tools
        self.bind_kwargs = kwargs
        return self

    async def ainvoke(self, messages: list[BaseMessage]) -> AIMessage:
        self.received.append(list(messages))
        human_positions = [i for i, m in enumerate(messages) if isinstance(m, HumanMessage)]
        last_human = messages[human_positions[-1]]
        observations = [
            m for m in messages[human_positions[-1] :] if isinstance(m, ToolMessage)
        ]

        if last_human.content == FOLLOW_UP_QUESTION:
            earlier = [messages[i].content for i in human_positions[:-1]]
            if not any("cliente 102" in str(text) for text in earlier):
                return AIMessage(content="¿De qué cliente me hablás?")
            if not observations:
                return _tool_call("último pedido del cliente 102", "call-ultimo")
            return AIMessage(content=f"El último pedido fue: {observations[-1].content}")

        if len(observations) == 0:
            return _tool_call("cantidad de pedidos del cliente 102", "call-cantidad")
        if len(observations) == 1:
            return _tool_call("total acumulado cliente 102", "call-total")
        return AIMessage(content="El cliente 102 tiene 3 pedidos por un total de 14500 ARS.")


@pytest.fixture
def scripted_model() -> ScriptedModel:
    return ScriptedModel()
