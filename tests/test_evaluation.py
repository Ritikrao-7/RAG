import asyncio

from fastapi.testclient import TestClient

from app import evaluation
from app.evaluation import case_passed, load_cases, summarize
from app.main import create_app


def fake_scores(question, answer, contexts, reference):
    return {"faithfulness": 1.0, "answer_correctness": 0.8, "context_recall": 0.5}


def test_case_needs_right_found_flag_and_key_facts():
    case = {"expected_found": True, "must_contain": ["145", "150"]}

    assert case_passed(case, "It costs $145 vs $150.", found=True)
    assert not case_passed(case, "It costs $145.", found=True)
    assert not case_passed(case, "It costs $145 vs $150.", found=False)


def test_not_found_case_passes_when_nothing_is_found():
    case = {"expected_found": False, "must_contain": []}

    assert case_passed(case, "Not in corpus.", found=False)
    assert not case_passed(case, "Paris.", found=True)


def test_summary_averages_only_scored_cases_and_checks_targets():
    cases = [
        {"passed": True, "scores": {"faithfulness": 1.0, "answer_correctness": None}},
        {"passed": False, "scores": {}},
    ]

    metrics = {metric["name"]: metric for metric in summarize(cases)}

    assert metrics["accuracy"]["value"] == 0.5
    assert metrics["accuracy"]["met"] is False
    assert metrics["faithfulness"]["value"] == 1.0
    assert metrics["faithfulness"]["met"] is True
    assert metrics["answer_correctness"]["value"] is None
    assert metrics["answer_correctness"]["met"] is None


def test_provider_error_does_not_lower_accuracy():
    cases = [
        {"passed": True, "scores": {}, "error": None},
        {"passed": False, "scores": {}, "error": "LLMError"},
    ]

    metrics = {metric["name"]: metric for metric in summarize(cases)}

    assert metrics["accuracy"]["value"] == 1.0


def test_every_answerable_case_has_key_facts():
    for case in load_cases():
        assert case["expected_found"] is bool(case["must_contain"])


def test_evaluation_needs_data(store, mock_llm):
    client = TestClient(create_app(store))

    response = client.post("/tenants/eval-empty/evaluate")

    assert response.status_code == 422
    assert response.json()["error"] == "no_data"


def test_evaluation_runs_in_background_and_stays_per_tenant(store, mock_llm, monkeypatch):
    monkeypatch.setattr(evaluation, "score_answer", fake_scores)
    monkeypatch.setattr(evaluation.get_settings(), "eval_pause_seconds", 0)
    document = {"text": "Costs $145 vs $150.", "source": "s", "brand": "Nike"}

    with TestClient(create_app(store)) as client:
        client.post("/tenants/eval-a/ingest", json={"documents": [document]})
        assert client.post("/tenants/eval-a/evaluate?limit=1").status_code == 202

        status = {}
        for _ in range(200):
            status = client.get("/tenants/eval-a/evaluate").json()
            if status["state"] != "running":
                break
            client.portal.call(asyncio.sleep, 0.01)

        assert status["state"] == "finished"
        assert status["cases"][0]["scores"]["faithfulness"] == 1.0
        assert client.get("/tenants/eval-b/evaluate").json()["state"] == "idle"
