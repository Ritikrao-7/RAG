"""Pinecone access. Every method takes tenant_id, so no call can skip tenant scoping."""

import asyncio
from typing import Any, Protocol

from pinecone import Pinecone

from app.config import get_settings

FETCH_BATCH_SIZE = 100


class VectorStore(Protocol):
    async def upsert(self, tenant_id: str, vectors: list[dict[str, Any]]) -> None: ...

    async def query(self, tenant_id: str, vector: list[float], top_k: int) -> list[dict[str, Any]]: ...

    async def list_ids(self, tenant_id: str, prefix: str) -> list[str]: ...

    async def list_chunks(self, tenant_id: str, limit: int) -> list[dict[str, Any]]: ...

    async def delete(self, tenant_id: str, ids: list[str]) -> None: ...

    async def rerank(
        self, query: str, chunks: list[dict[str, Any]], top_k: int
    ) -> list[dict[str, Any]]: ...

    async def ping(self) -> None: ...


class PineconeStore:
    """One namespace per tenant, plus a tenant_id metadata filter on every query."""

    def __init__(self) -> None:
        self._client: Pinecone | None = None
        self._host: str | None = None

    async def _open_index(self) -> Any:
        settings = get_settings()
        if self._client is None:
            self._client = Pinecone(api_key=settings.pinecone_api_key)
        if self._host is None:
            description = await asyncio.to_thread(
                self._client.describe_index, settings.pinecone_index_name
            )
            self._host = description.host
        return self._client.IndexAsyncio(host=self._host)

    async def upsert(self, tenant_id: str, vectors: list[dict[str, Any]]) -> None:
        async with await self._open_index() as index:
            await index.upsert(
                vectors=vectors, namespace=tenant_id, batch_size=100, show_progress=False
            )

    async def query(self, tenant_id: str, vector: list[float], top_k: int) -> list[dict[str, Any]]:
        async with await self._open_index() as index:
            response = await index.query(
                vector=vector,
                top_k=top_k,
                namespace=tenant_id,
                filter={"tenant_id": {"$eq": tenant_id}},
                include_metadata=True,
            )
        return [
            {"id": match.id, "score": match.score, **(match.metadata or {})}
            for match in response.matches
        ]

    async def list_ids(self, tenant_id: str, prefix: str) -> list[str]:
        async with await self._open_index() as index:
            return [
                item.id
                async for page in index.list(prefix=prefix, namespace=tenant_id)
                for item in page.vectors
                if item.id
            ]

    async def list_chunks(self, tenant_id: str, limit: int) -> list[dict[str, Any]]:
        ids = (await self.list_ids(tenant_id, ""))[:limit]
        chunks: list[dict[str, Any]] = []
        async with await self._open_index() as index:
            for start in range(0, len(ids), FETCH_BATCH_SIZE):
                response = await index.fetch(
                    ids=ids[start : start + FETCH_BATCH_SIZE], namespace=tenant_id
                )
                chunks += [
                    {"id": vector_id, **(vector.metadata or {})}
                    for vector_id, vector in response.vectors.items()
                    if (vector.metadata or {}).get("tenant_id") == tenant_id
                ]
        return chunks

    async def delete(self, tenant_id: str, ids: list[str]) -> None:
        async with await self._open_index() as index:
            await index.delete(ids=ids, namespace=tenant_id)

    async def rerank(
        self, query: str, chunks: list[dict[str, Any]], top_k: int
    ) -> list[dict[str, Any]]:
        settings = get_settings()
        if self._client is None:
            self._client = Pinecone(api_key=settings.pinecone_api_key)
        result = await asyncio.to_thread(
            self._client.inference.rerank,
            model=settings.rerank_model,
            query=query,
            documents=[{"text": chunk["chunk_text"]} for chunk in chunks],
            top_n=top_k,
            return_documents=False,
            parameters={"truncate": "END"},
        )
        return [{**chunks[ranked.index], "rerank_score": ranked.score} for ranked in result.data]

    async def ping(self) -> None:
        async with await self._open_index() as index:
            await index.describe_index_stats()
