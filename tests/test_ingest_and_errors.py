import httpx
import pytest
from fastapi.testclient import TestClient
from pinecone import PineconeError

from app import ingestion, llm
from app.main import create_app

LONG_TEXT = " ".join(f"Sentence number {i} about running shoes." for i in range(400))


def body(text, source="same-doc"):
    return {"documents": [{"text": text, "source": source, "brand": "Nike"}]}


def test_same_document_twice_does_not_grow(store, mock_llm):
    client = TestClient(create_app(store))

    first = client.post("/tenants/acme/ingest", json=body(LONG_TEXT)).json()
    count = len(store.namespaces["acme"])
    second = client.post("/tenants/acme/ingest", json=body(LONG_TEXT)).json()

    assert first == second
    assert len(store.namespaces["acme"]) == count


def test_shorter_reingest_removes_stale_chunks(store, mock_llm):
    client = TestClient(create_app(store))

    client.post("/tenants/acme/ingest", json=body(LONG_TEXT))
    assert len(store.namespaces["acme"]) > 1

    client.post("/tenants/acme/ingest", json=body("A short replacement."))
    chunks = list(store.namespaces["acme"].values())

    assert len(chunks) == 1
    assert chunks[0]["metadata"]["chunk_text"] == "A short replacement."


def test_unreachable_url_returns_422_and_stores_nothing(store, mock_llm, monkeypatch):
    async def refuse(self, url, **kwargs):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(httpx.AsyncClient, "get", refuse)
    client = TestClient(create_app(store))

    response = client.post(
        "/tenants/acme/ingest",
        json={"documents": [{"url": "https://example.com/gone", "brand": "Nike"}]},
    )

    assert response.status_code == 422
    assert response.json()["error"] == "url_unreachable"
    assert store.namespaces == {}


def test_llm_timeout_returns_504(store, mock_llm, monkeypatch):
    async def time_out(text):
        raise llm.LLMTimeoutError("Gemini did not respond in time")

    monkeypatch.setattr(llm, "embed_query", time_out)
    client = TestClient(create_app(store))

    response = client.post("/tenants/acme/ask", json={"question": "Anything?"})

    assert response.status_code == 504
    assert response.json()["error"] == "llm_timeout"


async def test_bad_gemini_setup_becomes_llm_error(monkeypatch):
    def missing_key():
        raise ValueError("API key required")

    monkeypatch.setattr(llm, "chat_model", missing_key)

    with pytest.raises(llm.LLMError):
        await llm.ask("Anything?")


def test_vector_db_failure_returns_503(store, mock_llm, monkeypatch):
    async def fail(tenant_id, vector, top_k):
        raise PineconeError("down")

    monkeypatch.setattr(store, "query", fail)
    client = TestClient(create_app(store))

    response = client.post("/tenants/acme/ask", json={"question": "Anything?"})

    assert response.status_code == 503
    assert response.json()["error"] == "vector_db_error"


def test_logs_and_chunks_are_kept_per_tenant(store, mock_llm):
    client = TestClient(create_app(store))
    question = "How does Nike pricing compare?"
    client.post("/tenants/logs-a/ingest", json=body("Nike costs $145."))
    client.post("/tenants/logs-a/ask", json={"question": question})

    messages = [e["message"] for e in client.get("/tenants/logs-a/logs").json()["entries"]]
    assert f"question: {question}" in messages
    assert any(message.startswith("grade yes") for message in messages)
    assert client.get("/tenants/logs-a/chunks").json()["total"] == 1

    assert client.get("/tenants/logs-b/logs").json()["entries"] == []
    assert client.get("/tenants/logs-b/chunks").json() == {"total": 0, "chunks": []}


def test_home_page_and_samples_load(store):
    client = TestClient(create_app(store))

    assert "<title>" in client.get("/").text
    assert "Brand Q&amp;A Agent" in client.get("/console").text
    assert len(client.get("/samples/ingest_urls.json").json()["documents"]) == 12


def test_document_needs_url_or_text(store, mock_llm):
    client = TestClient(create_app(store))

    response = client.post("/tenants/acme/ingest", json={"documents": [{"brand": "Nike"}]})

    assert response.status_code == 422


def test_doc_id_uses_source_so_edited_text_replaces_the_old_version():
    first = ingestion.Document(text="old", source="manual-1", brand="Nike")
    second = ingestion.Document(text="new", source="manual-1", brand="Nike")

    assert ingestion.make_doc_id(first) == ingestion.make_doc_id(second)
