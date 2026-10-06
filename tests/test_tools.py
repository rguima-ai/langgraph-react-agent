"""Pruebas unitarias del contrato y de la herramienta."""

import json

import pytest
from pydantic import ValidationError

from react_agent.tools import SearchInput, search_knowledge_base


def test_search_input_rejects_an_invalid_limit() -> None:
    with pytest.raises(ValidationError):
        SearchInput(query="cliente 102", limit=10)


@pytest.mark.asyncio
async def test_search_tool_returns_a_limited_result() -> None:
    raw_result = await search_knowledge_base.ainvoke(
        {"query": "cantidad de pedidos del cliente 102", "limit": 1}
    )
    result = json.loads(raw_result)

    assert result["status"] == "success"
    assert len(result["results"]) == 1
    assert result["results"][0]["id"] == "cliente-102-cantidad"


@pytest.mark.asyncio
async def test_search_tool_converts_timeout_into_a_controlled_error() -> None:
    raw_result = await search_knowledge_base.ainvoke(
        {"query": "__simulate_timeout__", "limit": 1}
    )
    result = json.loads(raw_result)

    assert result["status"] == "error"
    assert result["error_type"] == "TimeoutError"
