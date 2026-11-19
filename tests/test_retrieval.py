from fastapi.testclient import TestClient

from app.comparison import rank_of
from app.main import create_app
from app.retrieval import keyword_search, merge_rankings, search, tokenize


def chunk(chunk_id, text):
    return {"id": chunk_id, "chunk_text": text, "source": "s", "brand": "b"}


def test_tokenize_keeps_model_codes_whole():
    assert tokenize("Which shoe has SKU SW-2306?") == ["which", "shoe", "has", "sku", "sw-2306"]


def test_keyword_search_finds_the_exact_code():
    chunks = [
        chunk("a", "The Aero 3 has SKU SW-1403 and a mesh upper."),
        chunk("b", "The Aero 4 has SKU SW-1404 and a mesh upper."),
        chunk("c", "The Aero 5 has SKU SW-1405 and a mesh upper."),
    ]

    assert keyword_search(chunks, "SW-1404", 5)[0]["id"] == "b"


def test_keyword_search_handles_no_chunks():
    assert keyword_search([], "anything", 5) == []


def test_merge_prefers_chunks_ranked_high_in_both_lists():
    first = [chunk("a", ""), chunk("b", "")]
    second = [chunk("b", ""), chunk("c", "")]

    assert [item["id"] for item in merge_rankings(first, second)] == ["b", "a", "c"]


def test_rank_of_gives_position_of_the_correct_chunk():
    chunks = [chunk("a", "nothing"), chunk("b", "the ANSWER is here")]

    assert rank_of(chunks, "answer") == 2
    assert rank_of(chunks, "missing") is None


async def test_modes_off_uses_semantic_search_only(store):
    await store.upsert("t", [{"id": "d-0", "metadata": {"chunk_text": "x"}}])

    await search(store, "t", "q", [0.1], 5)

    assert store.query_count == 1
    assert store.rerank_count == 0


async def test_rerank_mode_reorders_candidates(store):
    await store.upsert(
        "t",
        [
            {"id": "d-0", "metadata": {"chunk_text": "first"}},
            {"id": "d-1", "metadata": {"chunk_text": "second"}},
        ],
    )

    chunks = await search(store, "t", "q", [0.1], 5, rerank=True)

    assert store.rerank_count == 1
    assert [item["id"] for item in chunks] == ["d-1", "d-0"]


async def test_hybrid_finds_a_keyword_match_semantic_search_missed(store):
    await store.upsert(
        "t",
        [
            {"id": f"d-{i}", "metadata": {"chunk_text": f"filler number {i}"}}
            for i in range(25)
        ]
        + [{"id": "gold", "metadata": {"chunk_text": "code SW-9999 is here"}}],
    )

    plain = await search(store, "t", "SW-9999", [0.1], 5)
    hybrid = await search(store, "t", "SW-9999", [0.1], 5, hybrid=True)

    assert "gold" not in [item["id"] for item in plain]
    assert "gold" in [item["id"] for item in hybrid]


def test_ask_turns_modes_on_only_when_asked(store, mock_llm):
    client = TestClient(create_app(store))
    document = {"text": "Asana costs more.", "source": "s", "brand": "Asana"}
    client.post("/tenants/modes/ingest", json={"documents": [document]})

    off = client.post("/tenants/modes/ask", json={"question": "Price?"}).json()
    on = client.post(
        "/tenants/modes/ask", json={"question": "Price?", "hybrid": True, "rerank": True}
    ).json()

    assert off["trace"][0]["detail"] == "1 chunks retrieved"
    assert "hybrid + rerank" in on["trace"][0]["detail"]


def test_compare_needs_data_and_a_known_dataset(store, mock_llm):
    client = TestClient(create_app(store))

    assert client.post("/tenants/compare-empty/compare").status_code == 422
    assert client.post("/tenants/x/compare?dataset=unknown").status_code == 422
