"""Finding chunks. Semantic search is the default; hybrid and reranking are optional extras.

Hybrid: runs BM25 keyword search in the app and merges it with Pinecone's semantic results.
That keeps the existing cosine index, at the cost of loading the tenant's chunks per search.
Reranking: Pinecone's reranker reads each candidate with the question and reorders them.

Pass a `stages` list to search() to get a snapshot of the ranking after each step,
which the visualizer replays.
"""

import re
from typing import Any

from rank_bm25 import BM25Okapi

from app.logs import preview
from app.vectorstore import VectorStore

CANDIDATES = 20
MAX_KEYWORD_CHUNKS = 500
RRF_K = 60
STAGE_PREVIEW_LENGTH = 220


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:[-.][a-z0-9]+)*", text.lower())


def rank_keywords(
    chunks: list[dict[str, Any]], query: str, limit: int
) -> list[tuple[float, dict[str, Any]]]:
    if not chunks:
        return []
    scores = BM25Okapi([tokenize(chunk["chunk_text"]) for chunk in chunks]).get_scores(
        tokenize(query)
    )
    ranked = sorted(zip(scores, chunks), key=lambda pair: pair[0], reverse=True)
    return [(float(score), chunk) for score, chunk in ranked[:limit] if score > 0]


def keyword_search(chunks: list[dict[str, Any]], query: str, limit: int) -> list[dict[str, Any]]:
    return [chunk for _, chunk in rank_keywords(chunks, query, limit)]


def fuse_rankings(*rankings: list[dict[str, Any]]) -> list[tuple[float, dict[str, Any]]]:
    """Reciprocal rank fusion: a chunk scores well if it ranks high in either list."""
    scores: dict[str, float] = {}
    chunks: dict[str, dict[str, Any]] = {}
    for ranking in rankings:
        for position, chunk in enumerate(ranking, start=1):
            scores[chunk["id"]] = scores.get(chunk["id"], 0) + 1 / (RRF_K + position)
            chunks[chunk["id"]] = chunk
    return [(scores[key], chunks[key]) for key in sorted(scores, key=scores.get, reverse=True)]


def merge_rankings(*rankings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [chunk for _, chunk in fuse_rankings(*rankings)]


def snapshot(stage: str, scored: list[tuple[float | None, dict[str, Any]]]) -> dict[str, Any]:
    return {
        "stage": stage,
        "chunks": [
            {
                "id": chunk["id"],
                "source": chunk.get("source", ""),
                "chunk_index": chunk.get("chunk_index", 0),
                "score": score,
                "text": preview(chunk.get("chunk_text", ""), STAGE_PREVIEW_LENGTH),
            }
            for score, chunk in scored
        ],
    }


def with_scores(chunks: list[dict[str, Any]], key: str) -> list[tuple[float | None, dict[str, Any]]]:
    return [(chunk.get(key), chunk) for chunk in chunks]


async def search(
    store: VectorStore,
    tenant_id: str,
    query: str,
    vector: list[float],
    top_k: int,
    *,
    hybrid: bool = False,
    rerank: bool = False,
    stored_chunks: list[dict[str, Any]] | None = None,
    stages: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    record = stages.append if stages is not None else lambda _: None

    if not hybrid and not rerank:
        found = await store.query(tenant_id, vector, top_k)
        record(snapshot("semantic", with_scores(found, "score")))
        return found

    found = await store.query(tenant_id, vector, CANDIDATES)
    scored = with_scores(found, "score")
    record(snapshot("semantic", scored))
    if hybrid:
        if stored_chunks is None:
            stored_chunks = await store.list_chunks(tenant_id, MAX_KEYWORD_CHUNKS)
        keyword_hits = rank_keywords(stored_chunks, query, CANDIDATES)
        record(snapshot("keyword", keyword_hits))
        fused = fuse_rankings(found, [chunk for _, chunk in keyword_hits])[:CANDIDATES]
        record(snapshot("fused", fused))
        scored = fused
        found = [chunk for _, chunk in fused]

    if rerank and found:
        reranked = await store.rerank(query, found, top_k)
        record(snapshot("reranked", with_scores(reranked, "rerank_score")))
        return reranked
    record(snapshot("top_k", scored[:top_k]))
    return found[:top_k]


def mode_label(hybrid: bool, rerank: bool) -> str:
    return {
        (False, False): "semantic only",
        (True, False): "hybrid",
        (False, True): "rerank",
        (True, True): "hybrid + rerank",
    }[(hybrid, rerank)]
