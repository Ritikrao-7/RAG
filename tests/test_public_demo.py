import json

import httpx
import pytest
from fastapi.testclient import TestClient

from app import llm
from app.config import get_settings
from app.ingestion import UrlFetchError, refuse_private_hosts
from app.main import create_app
from app.retrieval import search

TEXT = {"documents": [{"text": "The Pegasus 41 costs $140.", "source": "notes", "brand": "Nike"}]}


@pytest.fixture
def settings(monkeypatch):
    """Settings are cached, so tests change the shared object and monkeypatch restores it."""

    def change(**values):
        for key, value in values.items():
            monkeypatch.setattr(get_settings(), key, value)

    return change


def events(response):
    return [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]


def test_create_and_read_user(store):
    client = TestClient(create_app(store))

    created = client.post("/users", json={"name": "Ada", "email": "Ada@Example.com "})
    user = created.json()

    assert created.status_code == 201
    assert user["id"].startswith("u_")
    assert user["email"] == "ada@example.com"
    assert client.get(f"/users/{user['id']}").json()["name"] == "Ada"
    assert client.get("/users/u_missing").status_code == 404


def test_user_needs_a_valid_email(store):
    client = TestClient(create_app(store))

    assert client.post("/users", json={"name": "Ada", "email": "not-an-email"}).status_code == 422


def test_registration_required_blocks_unknown_tenants(store, mock_llm, settings):
    settings(require_registration=True)
    client = TestClient(create_app(store))
    user_id = client.post("/users", json={"name": "Ada", "email": "a@b.co"}).json()["id"]

    rejected = client.post("/tenants/someone-else/ingest", json=TEXT)

    assert rejected.status_code == 404
    assert rejected.json()["error"] == "unknown_user"
    assert client.post(f"/tenants/{user_id}/ingest", json=TEXT).status_code == 200


def test_rate_limit_counts_writes_per_tenant(store, mock_llm, settings):
    settings(requests_per_minute=2)
    client = TestClient(create_app(store))

    codes = [client.post("/tenants/acme/ingest", json=TEXT).status_code for _ in range(3)]

    assert codes == [200, 200, 429]
    assert client.get("/tenants/acme/chunks").status_code == 200


def test_chunk_quota_stops_further_ingests(store, mock_llm, settings):
    settings(max_chunks_per_tenant=1)
    client = TestClient(create_app(store))

    client.post("/tenants/acme/ingest", json=TEXT)
    response = client.post("/tenants/acme/ingest", json=TEXT)

    assert response.status_code == 403
    assert response.json()["error"] == "quota_exceeded"


def test_delete_source_removes_only_that_source(store, mock_llm):
    client = TestClient(create_app(store))
    other = {"documents": [{"text": "Other text.", "source": "other", "brand": "Nike"}]}
    client.post("/tenants/acme/ingest", json=TEXT)
    client.post("/tenants/acme/ingest", json=other)

    response = client.delete("/tenants/acme/sources", params={"source": "notes"})

    assert response.json() == {"chunks_deleted": 1}
    assert [chunk["source"] for chunk in client.get("/tenants/acme/chunks").json()["chunks"]] == [
        "other"
    ]


def test_stream_sends_each_step_with_retrieval_stages(store, mock_llm):
    client = TestClient(create_app(store))
    client.post("/tenants/acme/ingest", json=TEXT)

    response = client.post(
        "/tenants/acme/ask/stream", json={"question": "price?", "hybrid": True, "rerank": True}
    )
    sent = events(response)

    assert response.headers["content-type"].startswith("text/event-stream")
    assert [event.get("node") for event in sent] == ["retrieve", "grade", "answer", None]
    assert sent[-1] == {"type": "done"}
    stages = [stage["stage"] for stage in sent[0]["retrieval"]["stages"]]
    assert stages == ["semantic", "keyword", "fused", "reranked"]
    assert sent[1]["grades"][0]["relevant"] is True
    assert sent[2]["found_in_corpus"] is True


def test_stream_shows_the_rewrite_loop(store, mock_llm):
    mock_llm["grade"] = "no"
    client = TestClient(create_app(store))
    client.post("/tenants/acme/ingest", json=TEXT)

    sent = events(client.post("/tenants/acme/ask/stream", json={"question": "price?"}))

    assert [event.get("node") for event in sent] == [
        "retrieve", "grade", "rewrite", "retrieve", "grade", "not_found", None,
    ]
    assert sent[2]["search_query"] == "a clearer question"
    assert sent[3]["retrieval"]["query"] == "a clearer question"


def test_stream_reports_llm_errors_as_an_event(store, mock_llm, monkeypatch):
    async def time_out(text):
        raise llm.LLMTimeoutError("Gemini did not respond in time")

    monkeypatch.setattr(llm, "embed_query", time_out)
    client = TestClient(create_app(store))

    sent = events(client.post("/tenants/acme/ask/stream", json={"question": "price?"}))

    assert sent[0]["type"] == "error"
    assert sent[0]["error"] == "llm_timeout"
    assert sent[-1] == {"type": "done"}


async def test_hybrid_without_rerank_records_fused_top_k(store):
    await store.upsert(
        "t",
        [
            {"id": "d-0", "metadata": {"chunk_text": "trail shoes", "score": 0.9}},
            {"id": "d-1", "metadata": {"chunk_text": "road shoes"}},
        ],
    )
    stages = []

    await search(store, "t", "road", [0.1], 1, hybrid=True, stages=stages)

    assert [stage["stage"] for stage in stages] == ["semantic", "keyword", "fused", "top_k"]
    assert stages[-1]["chunks"][0]["score"] == stages[2]["chunks"][0]["score"]


@pytest.mark.parametrize(
    "url", ["http://127.0.0.1/", "http://localhost:8000/", "http://169.254.169.254/latest/"]
)
async def test_private_addresses_are_refused(url):
    with pytest.raises(UrlFetchError, match="private or internal"):
        await refuse_private_hosts(httpx.Request("GET", url))
