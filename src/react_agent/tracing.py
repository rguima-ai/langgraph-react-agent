"""Conversión del historial de LangChain a una traza JSON legible."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage


def message_to_event(message: BaseMessage, position: int) -> dict[str, Any]:
    """Serializa únicamente los campos útiles para entender el ciclo ReAct."""

    event: dict[str, Any] = {
        "position": position,
        "message_type": message.type,
        "content": message.content,
    }

    if isinstance(message, HumanMessage):
        event["phase"] = "user_input"
    elif isinstance(message, AIMessage) and message.tool_calls:
        event["phase"] = "reason_and_tool_call"
        event["tool_calls"] = message.tool_calls
    elif isinstance(message, ToolMessage):
        event["phase"] = "tool_observation"
        event["tool_name"] = message.name
        event["tool_call_id"] = message.tool_call_id
    else:
        event["phase"] = "final_answer"

    return event


def save_trace(
    messages: list[BaseMessage],
    output_path: Path,
    thread_id: str,
) -> None:
    """Guarda la traza completa sin incluir prompts internos ni secretos."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "thread_id": thread_id,
        "generated_at": datetime.now(UTC).isoformat(),
        "event_count": len(messages),
        "events": [
            message_to_event(message, position)
            for position, message in enumerate(messages, start=1)
        ],
    }
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def print_new_messages(messages: list[BaseMessage], start_at: int = 0) -> None:
    """Muestra solo lo ocurrido en la interacción actual."""

    for message in messages[start_at:]:
        if isinstance(message, HumanMessage):
            print(f"\n[USUARIO] {message.content}")
        elif isinstance(message, AIMessage) and message.tool_calls:
            for tool_call in message.tool_calls:
                print(f"[AGENTE -> TOOL] {tool_call['name']}({tool_call['args']})")
        elif isinstance(message, ToolMessage):
            print(f"[OBSERVACIÓN] {message.content}")
        elif isinstance(message, AIMessage):
            print(f"[RESPUESTA] {message.content}")
