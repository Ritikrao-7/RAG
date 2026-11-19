import asyncio
import hashlib
import ipaddress
import logging
import socket
from functools import lru_cache

import httpx
import trafilatura
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app import llm
from app.config import get_settings
from app.logs import for_tenant
from app.models import Document, IngestResponse
from app.vectorstore import VectorStore

USER_AGENT = "Mozilla/5.0 (compatible; BrandQABot/1.0)"


class UrlFetchError(Exception):
    def __init__(self, url: str, reason: str) -> None:
        super().__init__(f"Could not read {url}: {reason}")


@lru_cache
def _splitter() -> RecursiveCharacterTextSplitter:
    settings = get_settings()
    return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )


def make_doc_id(document: Document) -> str:
    """Stable ID: the same url, or the same source, always maps to the same document."""
    key = str(document.url) if document.url else document.source or document.text
    return hashlib.sha256(key.encode()).hexdigest()


async def refuse_private_hosts(request: httpx.Request) -> None:
    """Runs before every request, redirects included, so a public URL cannot bounce
    the server onto localhost, the cloud metadata address or a private network."""
    url = str(request.url)
    try:
        addresses = await asyncio.get_running_loop().getaddrinfo(
            request.url.host, request.url.port or 443, type=socket.SOCK_STREAM
        )
    except socket.gaierror as exc:
        raise UrlFetchError(url, "host not found") from exc
    for *_, sockaddr in addresses:
        if not ipaddress.ip_address(sockaddr[0].split("%")[0]).is_global:
            raise UrlFetchError(url, "it points to a private or internal address")


async def fetch_text(client: httpx.AsyncClient, url: str, log: logging.LoggerAdapter) -> str:
    log.info("fetching %s", url)
    try:
        response = await client.get(url)
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise UrlFetchError(url, f"HTTP {exc.response.status_code}") from exc
    except httpx.HTTPError as exc:
        raise UrlFetchError(url, type(exc).__name__) from exc

    text = await asyncio.to_thread(trafilatura.extract, response.text, include_tables=True)
    if not text:
        raise UrlFetchError(url, "no readable text on the page")
    log.info("fetched %s: %d words", url, len(text.split()))
    return text


async def read_documents(tenant_id: str, documents: list[Document]) -> list[str]:
    """Fetch everything before storing anything, so one bad URL stores nothing."""
    settings = get_settings()
    log = for_tenant(tenant_id)
    async with httpx.AsyncClient(
        follow_redirects=True,
        timeout=settings.url_timeout_seconds,
        headers={"User-Agent": USER_AGENT},
        event_hooks={"request": [refuse_private_hosts]},
    ) as client:

        async def read(document: Document) -> str:
            if document.url:
                return await fetch_text(client, str(document.url), log)
            return document.text

        return await asyncio.gather(*(read(document) for document in documents))


async def store_document(tenant_id: str, document: Document, text: str, store: VectorStore) -> int:
    doc_id = make_doc_id(document)
    chunks = _splitter().split_text(text)
    source = str(document.url) if document.url else document.source or f"text-{doc_id[:12]}"
    log = for_tenant(tenant_id)
    log.info("doc %s from %s: split into %d chunks", doc_id[:10], source, len(chunks))

    embeddings = await llm.embed_documents(chunks)
    log.info("doc %s: embedded %d chunks", doc_id[:10], len(embeddings))
    vectors = [
        {
            "id": f"{doc_id}-{chunk_index}",
            "values": embedding,
            "metadata": {
                "tenant_id": tenant_id,
                "doc_id": doc_id,
                "chunk_index": chunk_index,
                "source": source,
                "brand": document.brand,
                "chunk_text": chunk,
            },
        }
        for chunk_index, (chunk, embedding) in enumerate(zip(chunks, embeddings, strict=True))
    ]
    await store.upsert(tenant_id, vectors)
    log.info("doc %s: upserted %d vectors", doc_id[:10], len(vectors))

    # A shorter re-ingest leaves old high-index chunks behind. Remove them after the
    # upsert so the document stays searchable throughout.
    new_ids = {vector["id"] for vector in vectors}
    existing_ids = await store.list_ids(tenant_id, f"{doc_id}-")
    stale_ids = [vector_id for vector_id in existing_ids if vector_id not in new_ids]
    if stale_ids:
        await store.delete(tenant_id, stale_ids)
        log.info("doc %s: removed %d stale chunks", doc_id[:10], len(stale_ids))

    return len(vectors)


async def ingest_documents(
    tenant_id: str, documents: list[Document], store: VectorStore
) -> IngestResponse:
    log = for_tenant(tenant_id)
    log.info("ingest started: %d documents", len(documents))
    texts = await read_documents(tenant_id, documents)
    chunks_stored = 0
    for document, text in zip(documents, texts, strict=True):
        chunks_stored += await store_document(tenant_id, document, text, store)
    log.info("ingest finished: %d documents, %d chunks", len(documents), chunks_stored)
    return IngestResponse(documents_stored=len(documents), chunks_stored=chunks_stored)
