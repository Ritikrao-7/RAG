"""Compares the four search modes on where the known correct chunk lands.

It only tests finding chunks, so it uses embeddings and the reranker, not the chat model.
"""

import json
from pathlib import Path
from typing import Any

from app import llm
from app.config import get_settings
from app.logs import for_tenant
from app.retrieval import MAX_KEYWORD_CHUNKS, mode_label, search
from app.vectorstore import VectorStore

TESTS_DIR = Path(__file__).parent.parent / "data" / "retrieval_tests"
DATASETS = {
    "three_urls": TESTS_DIR / "three_urls_questions.json",
    "catalogue": TESTS_DIR / "catalogue_questions.json",
    "shoes": TESTS_DIR / "shoes_questions.json",
}
MODES = [(False, False), (True, False), (False, True), (True, True)]


def rank_of(chunks: list[dict[str, Any]], gold: str) -> int | None:
    for position, chunk in enumerate(chunks, start=1):
        if gold.lower() in chunk["chunk_text"].lower():
            return position
    return None


def summarize(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary = []
    for hybrid, rerank in MODES:
        label = mode_label(hybrid, rerank)
        ranks = [result["ranks"][label] for result in results]
        summary.append(
            {
                "mode": label,
                "found_first": sum(rank == 1 for rank in ranks) / len(ranks),
                "found_in_top_5": sum(rank is not None for rank in ranks) / len(ranks),
            }
        )
    return summary


async def compare_modes(tenant_id: str, dataset: str, store: VectorStore) -> dict[str, Any]:
    log = for_tenant(tenant_id)
    top_k = get_settings().top_k
    questions = json.loads(DATASETS[dataset].read_text())["questions"]
    stored_chunks = await store.list_chunks(tenant_id, MAX_KEYWORD_CHUNKS)
    log.info("comparison started: %s, %d questions", dataset, len(questions))

    results = []
    for item in questions:
        vector = await llm.embed_query(item["question"])
        ranks: dict[str, int | None] = {}
        for hybrid, rerank in MODES:
            chunks = await search(
                store,
                tenant_id,
                item["question"],
                vector,
                top_k,
                hybrid=hybrid,
                rerank=rerank,
                stored_chunks=stored_chunks,
            )
            ranks[mode_label(hybrid, rerank)] = rank_of(chunks, item["gold"])
        log.info("compare: %s | %s", item["question"], ranks)
        results.append({"question": item["question"], "kind": item["kind"], "ranks": ranks})

    summary = summarize(results)
    log.info("comparison finished: %s", summary)
    return {"dataset": dataset, "chunks_searched": len(stored_chunks), "summary": summary, "results": results}
