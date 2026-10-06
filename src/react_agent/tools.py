"""Herramientas que el modelo puede decidir invocar."""

from __future__ import annotations

import asyncio
import json
import re
import unicodedata
from typing import Any

from langchain_core.tools import tool
from pydantic import BaseModel, Field


class SearchInput(BaseModel):
    """Contrato de entrada validado antes de consultar la base."""

    query: str = Field(
        min_length=3,
        max_length=200,
        description=(
            "Consulta técnica específica. Debe incluir la entidad y el dato buscado, "
            "por ejemplo: 'total acumulado de pedidos del cliente 102'."
        ),
    )
    limit: int = Field(
        default=1,
        ge=1,
        le=3,
        description="Cantidad máxima de fragmentos a recuperar, entre 1 y 3.",
    )


class MockVectorDB:
    """Base vectorial simulada para ejecutar la entrega sin infraestructura externa."""

    def __init__(self) -> None:
        self._documents: list[dict[str, Any]] = [
            {
                "id": "cliente-102-cantidad",
                "content": "El cliente 102 tiene 3 pedidos confirmados.",
                "tags": "cliente 102 cantidad cuantos pedidos confirmados",
            },
            {
                "id": "cliente-102-total",
                "content": "El total acumulado de los pedidos del cliente 102 es 14500 ARS.",
                "tags": "cliente 102 total acumulado importe monto pedidos",
            },
            {
                "id": "cliente-102-ultimo",
                "content": (
                    "El último pedido del cliente 102 es PED-9003, del 15/09/2026, "
                    "por 6500 ARS."
                ),
                "tags": "cliente 102 ultimo reciente pedido fecha importe",
            },
            {
                "id": "politica-devoluciones",
                "content": "Las devoluciones se aceptan hasta 30 días después de la compra.",
                "tags": "politica devoluciones plazo compra soporte",
            },
        ]

    @staticmethod
    def _tokens(text: str) -> set[str]:
        stop_words = {"a", "de", "del", "el", "la", "las", "los", "por", "y"}
        normalized = unicodedata.normalize("NFKD", text.lower())
        without_accents = "".join(char for char in normalized if not unicodedata.combining(char))
        return {
            token
            for token in re.findall(r"[a-z0-9]+", without_accents)
            if token not in stop_words
        }

    async def similarity_search(self, query: str, k: int) -> list[dict[str, Any]]:
        """Simula una consulta I/O y ordena documentos por palabras coincidentes."""

        await asyncio.sleep(0)
        if query == "__simulate_timeout__":
            raise TimeoutError("La base simulada no respondió a tiempo.")

        query_tokens = self._tokens(query)
        scored: list[tuple[int, dict[str, Any]]] = []

        for document in self._documents:
            searchable_text = f"{document['content']} {document['tags']}"
            score = len(query_tokens & self._tokens(searchable_text))
            if score > 0:
                scored.append((score, document))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            {"id": document["id"], "content": document["content"], "score": score}
            for score, document in scored[:k]
        ]


db = MockVectorDB()


@tool(args_schema=SearchInput)
async def search_knowledge_base(query: str, limit: int = 1) -> str:
    """Busca un dato concreto en la base interna de pedidos y soporte.

    Usá esta herramienta cuando la respuesta dependa de información interna que no debe
    inventarse. Hacé una consulta específica por cada dato independiente y, si el resultado
    es insuficiente, volvé a invocarla con una consulta más precisa. La herramienta devuelve
    JSON con los fragmentos encontrados o un error controlado.
    """

    try:
        results = await db.similarity_search(query=query, k=limit)
        if not results:
            return json.dumps(
                {
                    "status": "not_found",
                    "query": query,
                    "results": [],
                    "message": "No hubo coincidencias. Reformulá la consulta o pedí aclaración.",
                },
                ensure_ascii=False,
            )

        return json.dumps(
            {
                "status": "success",
                "query": query,
                "results": results,
                "message": "Usá solo estos datos. Si falta otro dato, hacé otra búsqueda.",
            },
            ensure_ascii=False,
        )
    except (TimeoutError, ConnectionError) as exc:
        return json.dumps(
            {
                "status": "error",
                "query": query,
                "error_type": type(exc).__name__,
                "message": "La búsqueda falló de forma controlada. Podés reintentar.",
            },
            ensure_ascii=False,
        )


TOOLS = [search_knowledge_base]
