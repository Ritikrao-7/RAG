from typing import Any

import pytest

from app import llm


class FakeStore:
    """In-memory stand-in for Pinecone, with one dict per tenant namespace."""

    def __init__(self) -> None:
        self.namespaces: dict[str, dict[str, dict[str, Any]]] = {}
        self.query_count = 0
        self.rerank_count = 0

    async def upsert(self, tenant_id: str, vectors: list[dict[str, Any]]) -> None:
        namespace = self.namespaces.setdefault(tenant_id, {})
        for vector in vectors:
            namespace[vector["id"]] = vector

    async def query(self, tenant_id: str, vector: list[float], top_k: int) -> list[dict[str, Any]]:
        self.query_count += 1
        stored = list(self.namespaces.get(tenant_id, {}).values())
        return [{"id": item["id"], **item["metadata"]} for item in stored[:top_k]]

    async def list_ids(self, tenant_id: str, prefix: str) -> list[str]:
        return [key for key in self.namespaces.get(tenant_id, {}) if key.startswith(prefix)]

    async def list_chunks(self, tenant_id: str, limit: int) -> list[dict[str, Any]]:
        stored = list(self.namespaces.get(tenant_id, {}).values())[:limit]
        return [{"id": item["id"], **item["metadata"]} for item in stored]

    async def delete(self, tenant_id: str, ids: list[str]) -> None:
        for vector_id in ids:
            self.namespaces[tenant_id].pop(vector_id, None)

    async def rerank(
        self, query: str, chunks: list[dict[str, Any]], top_k: int
    ) -> list[dict[str, Any]]:
        self.rerank_count += 1
        return list(reversed(chunks))[:top_k]

    async def ping(self) -> None:
        return None


@pytest.fixture
def store() -> FakeStore:
    return FakeStore()


@pytest.fixture
def mock_llm(monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    """Mock every Gemini call. Tests change `replies` to steer grading."""
    replies = {"grade": "yes"}

    async def fake_ask(prompt: str) -> str:
        if "Does this passage help" in prompt:
            return replies["grade"]
        if "Rewrite this question" in prompt:
            return "a clearer question"
        return "An answer from the passages."

    async def fake_embed_query(text: str) -> list[float]:
        return [0.1, 0.2]

    async def fake_embed_documents(texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2] for _ in texts]

    monkeypatch.setattr(llm, "ask", fake_ask)
    monkeypatch.setattr(llm, "embed_query", fake_embed_query)
    monkeypatch.setattr(llm, "embed_documents", fake_embed_documents)
    return replies
