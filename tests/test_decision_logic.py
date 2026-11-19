from app.graph import build_graph, decide, initial_state

CHUNK = {"chunk_text": "Asana costs more.", "source": "manual-1", "brand": "Asana"}


def state_with(relevant_chunks, retry_count):
    return {**initial_state("acme", "q"), "relevant_chunks": relevant_chunks, "retry_count": retry_count}


def test_relevant_chunks_go_to_answer():
    assert decide(state_with([CHUNK], retry_count=0)) == "answer"
    assert decide(state_with([CHUNK], retry_count=1)) == "answer"


def test_nothing_relevant_retries_once():
    assert decide(state_with([], retry_count=0)) == "rewrite"


def test_nothing_relevant_after_retry_is_not_found():
    assert decide(state_with([], retry_count=1)) == "not_found"


async def test_graph_stops_after_one_retry(store, mock_llm):
    await store.upsert("acme", [{"id": "d-0", "metadata": CHUNK}])
    mock_llm["grade"] = "no"

    result = await build_graph(store).ainvoke(initial_state("acme", "capital of France?"))

    assert [entry["node"] for entry in result["trace"]] == [
        "retrieve", "grade", "rewrite", "retrieve", "grade", "not_found",
    ]
    assert store.query_count == 2
    assert result["found_in_corpus"] is False
    assert result["citations"] == []


async def test_graph_answers_with_citations(store, mock_llm):
    await store.upsert("acme", [{"id": "d-0", "metadata": CHUNK}])

    result = await build_graph(store).ainvoke(initial_state("acme", "Asana pricing?"))

    assert [entry["node"] for entry in result["trace"]] == ["retrieve", "grade", "answer"]
    assert result["found_in_corpus"] is True
    assert result["citations"] == [CHUNK]
