"""Demostración completa de main.py sin red: el LLM real se reemplaza por ScriptedModel."""

import json
from pathlib import Path
from typing import Any

import pytest
from conftest import FIRST_QUESTION, ScriptedModel
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from react_agent import main as main_module
from react_agent.graph import build_graph

FAKE_KEY = "sk-test-no-es-una-clave-real"


def _isolate_from_dotenv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Evita que un .env real del desarrollador active llamadas pagas durante los tests."""

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(main_module, "load_dotenv", lambda *args, **kwargs: False)
    for name in ("OPENAI_MODEL", "CHECKPOINT_DB_PATH", "TRACE_OUTPUT_PATH", "THREAD_ID"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def demo_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    _isolate_from_dotenv(tmp_path, monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", FAKE_KEY)
    monkeypatch.setenv("THREAD_ID", "demo-test")
    monkeypatch.setattr(main_module, "ChatOpenAI", lambda **_: ScriptedModel())
    return tmp_path


@pytest.mark.asyncio
async def test_main_runs_twice_and_traces_only_the_current_execution(
    demo_env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    await main_module.main()
    await main_module.main()

    output = capsys.readouterr().out
    trace = json.loads((demo_env / "traces/latest_trace.json").read_text(encoding="utf-8"))
    phases = [event["phase"] for event in trace["events"]]

    # La traza cubre solo la última ejecución: 2 preguntas, 3 tool calls, 2 respuestas.
    assert phases == [
        "user_input", "reason_and_tool_call", "tool_observation",
        "reason_and_tool_call", "tool_observation", "final_answer",
        "user_input", "reason_and_tool_call", "tool_observation", "final_answer",
    ]
    assert trace["events"][0]["content"] == FIRST_QUESTION
    assert FAKE_KEY not in output
    assert FAKE_KEY not in json.dumps(trace)

    # El checkpoint sí conserva el historial de ambas ejecuciones del mismo thread.
    config: Any = {"configurable": {"thread_id": "demo-test"}}
    async with AsyncSqliteSaver.from_conn_string("data/checkpoints.sqlite") as checkpointer:
        app = build_graph(model=ScriptedModel(), checkpointer=checkpointer)
        state = await app.aget_state(config)
    humans = [m for m in state.values["messages"] if isinstance(m, HumanMessage)]
    assert len(humans) == 4


@pytest.mark.parametrize("api_key", [None, "replace_with_your_openai_api_key"])
def test_run_reports_missing_api_key(
    api_key: str | None,
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _isolate_from_dotenv(tmp_path, monkeypatch)
    if api_key is None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    else:
        monkeypatch.setenv("OPENAI_API_KEY", api_key)

    with pytest.raises(SystemExit) as exit_info:
        main_module.run()

    assert exit_info.value.code == 1
    assert "Falta OPENAI_API_KEY" in capsys.readouterr().out
