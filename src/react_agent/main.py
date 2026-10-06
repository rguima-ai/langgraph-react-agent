"""Demostración ejecutable del agente con dos turnos y memoria persistente."""

from __future__ import annotations

import asyncio
from typing import cast

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.errors import GraphRecursionError

from react_agent.config import ConfigurationError, Settings
from react_agent.graph import build_graph
from react_agent.tracing import print_new_messages, save_trace


async def main() -> None:
    """Ejecuta una búsqueda multi-paso y luego demuestra la memoria de sesión."""

    load_dotenv()
    settings = Settings.from_environment()
    settings.checkpoint_db_path.parent.mkdir(parents=True, exist_ok=True)

    model = ChatOpenAI(
        model=settings.openai_model,
        temperature=0,
        api_key=settings.openai_api_key,
    )
    config: RunnableConfig = {
        "configurable": {"thread_id": settings.thread_id},
        "recursion_limit": settings.recursion_limit,
    }

    async with AsyncSqliteSaver.from_conn_string(
        str(settings.checkpoint_db_path)
    ) as checkpointer:
        await checkpointer.setup()
        app = build_graph(model=model, checkpointer=checkpointer)

        # Si el thread ya tiene historial de ejecuciones anteriores, solo mostramos y
        # trazamos lo que ocurre en esta ejecución.
        previous_state = await app.aget_state(config)
        run_start = len(previous_state.values.get("messages", []))

        first_question = "¿Cuántos pedidos tiene el cliente 102 y cuál es el total acumulado?"
        first_result = await app.ainvoke(
            {"messages": [HumanMessage(content=first_question)]},
            config=config,
        )
        first_messages = cast(list[BaseMessage], first_result["messages"])
        print("\n--- INTERACCIÓN 1: razonamiento multi-paso ---")
        print_new_messages(first_messages, start_at=run_start)

        first_message_count = len(first_messages)
        second_question = "¿Y cuál fue el último?"
        second_result = await app.ainvoke(
            {"messages": [HumanMessage(content=second_question)]},
            config=config,
        )
        second_messages = cast(list[BaseMessage], second_result["messages"])
        print("\n--- INTERACCIÓN 2: mismo thread_id ---")
        print_new_messages(second_messages, start_at=first_message_count)

        save_trace(
            messages=second_messages[run_start:],
            output_path=settings.trace_output_path,
            thread_id=settings.thread_id,
        )
        print(f"\nTraza guardada en: {settings.trace_output_path}")
        print(f"Checkpoints guardados en: {settings.checkpoint_db_path}")


def run() -> None:
    """Punto de entrada síncrono que inicia el event loop una sola vez."""

    try:
        asyncio.run(main())
    except ConfigurationError as exc:
        print(f"Error de configuración: {exc}")
        raise SystemExit(1) from exc
    except GraphRecursionError as exc:
        print(f"El agente superó recursion_limit sin llegar a una respuesta: {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    run()
