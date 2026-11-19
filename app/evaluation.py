"""RAGAS evaluation. It makes real Gemini calls, so it runs on demand and not in pytest.

Accuracy is our own check: did the service find the answer when it should, stay quiet
when it should, and include the key facts? RAGAS adds the quality scores on top.
"""

import asyncio
import json
import math
import os
from pathlib import Path
from typing import Any

from app import llm
from app.config import get_settings
from app.graph import initial_state
from app.logs import for_tenant, preview

os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")

DATASET_PATH = (
    Path(__file__).parent.parent / "data" / "running_shoes" / "samples" / "eval_dataset.json"
)
METRIC_NAMES = ("faithfulness", "answer_correctness", "context_recall")

# Our own goals for this project. The assignment does not set any.
TARGETS = {
    "accuracy": 0.8,
    "faithfulness": 0.8,
    "answer_correctness": 0.6,
    "context_recall": 0.7,
}


def load_cases(limit: int | None = None) -> list[dict[str, Any]]:
    cases = json.loads(DATASET_PATH.read_text())["cases"]
    return cases[:limit] if limit else cases


def case_passed(case: dict[str, Any], answer: str, found: bool) -> bool:
    if found != case["expected_found"]:
        return False
    return all(term.lower() in answer.lower() for term in case["must_contain"])


def score_answer(
    question: str, answer: str, contexts: list[str], reference: str
) -> dict[str, float | None]:
    """Blocking, because ragas.evaluate runs its own event loop. Call it in a thread."""
    from ragas import EvaluationDataset, RunConfig, SingleTurnSample, evaluate
    from ragas.metrics import answer_correctness, context_recall, faithfulness

    sample = SingleTurnSample(
        user_input=question, response=answer, retrieved_contexts=contexts, reference=reference
    )
    result = evaluate(
        EvaluationDataset([sample]),
        metrics=[faithfulness, answer_correctness, context_recall],
        llm=llm.chat_model(),
        embeddings=llm.embedding_model(),
        # Few workers, because the Gemini free tier has a low rate limit.
        run_config=RunConfig(timeout=120, max_retries=4, max_wait=30, max_workers=2),
        show_progress=False,
        allow_nest_asyncio=False,
    )
    row = result.to_pandas().iloc[0]
    scores: dict[str, float | None] = {}
    for name in METRIC_NAMES:
        value = float(row[name])
        scores[name] = None if math.isnan(value) else value
    return scores


def average(values: list[float | None]) -> float | None:
    scored = [value for value in values if value is not None]
    return sum(scored) / len(scored) if scored else None


def summarize(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # A provider error says nothing about answer quality, so it is left out of accuracy.
    answered = [case for case in cases if not case.get("error")]
    values: dict[str, float | None] = {
        "accuracy": average([1.0 if case["passed"] else 0.0 for case in answered]),
    }
    for name in METRIC_NAMES:
        values[name] = average([case["scores"].get(name) for case in cases])

    return [
        {
            "name": name,
            "value": value,
            "target": TARGETS[name],
            "met": None if value is None else value >= TARGETS[name],
        }
        for name, value in values.items()
    ]


class EvaluationRunner:
    """Runs one evaluation per tenant in the background and keeps the latest result."""

    def __init__(self, graph: Any) -> None:
        self._graph = graph
        self._jobs: dict[str, dict[str, Any]] = {}
        self._tasks: dict[str, asyncio.Task] = {}

    def status(self, tenant_id: str) -> dict[str, Any]:
        return self._jobs.get(tenant_id, {"state": "idle", "done": 0, "total": 0})

    def is_running(self, tenant_id: str) -> bool:
        return self.status(tenant_id)["state"] == "running"

    def start(self, tenant_id: str, limit: int | None) -> dict[str, Any]:
        cases = load_cases(limit)
        self._jobs[tenant_id] = {
            "state": "running",
            "done": 0,
            "total": len(cases),
            "metrics": [],
            "cases": [],
        }
        self._tasks[tenant_id] = asyncio.create_task(self._run(tenant_id, cases))
        return self._jobs[tenant_id]

    async def _run(self, tenant_id: str, cases: list[dict[str, Any]]) -> None:
        job = self._jobs[tenant_id]
        log = for_tenant(tenant_id)
        pause = get_settings().eval_pause_seconds
        log.info("evaluation started: %d questions", len(cases))
        try:
            for number, case in enumerate(cases, start=1):
                log.info("eval %d/%d: %s", number, len(cases), case["question"])
                job["cases"].append(await self._run_case(tenant_id, case))
                job["done"] = number
                job["metrics"] = summarize(job["cases"])
                if number < len(cases):
                    await asyncio.sleep(pause)
            job["state"] = "finished"
            log.info("evaluation finished: %s", summary_text(job["metrics"]))
        except Exception as exc:
            job["state"] = "failed"
            log.error("evaluation failed: %s", type(exc).__name__)

    async def _run_case(self, tenant_id: str, case: dict[str, Any]) -> dict[str, Any]:
        log = for_tenant(tenant_id)
        result = {
            "question": case["question"],
            "expected": case["expected"],
            "expected_found": case["expected_found"],
            "actual": "",
            "found_in_corpus": False,
            "chunks_used": 0,
            "passed": False,
            "scores": {},
            "error": None,
        }
        for attempt in (1, 2):
            try:
                await self._answer_case(tenant_id, case, result)
                result["error"] = None
                break
            except Exception as exc:
                result["error"] = type(exc).__name__
                log.error("eval attempt %d failed: %s", attempt, type(exc).__name__)
                if attempt == 1:
                    await asyncio.sleep(get_settings().eval_pause_seconds * 2)

        outcome = "ERROR" if result["error"] else "PASS" if result["passed"] else "FAIL"
        log.info("expected: %s", case["expected"])
        log.info("got: %s", preview(result["actual"], 240))
        log.info("%s | scores: %s", outcome, result["scores"])
        return result

    async def _answer_case(
        self, tenant_id: str, case: dict[str, Any], result: dict[str, Any]
    ) -> None:
        state = await self._graph.ainvoke(initial_state(tenant_id, case["question"]))
        contexts = [chunk["chunk_text"] for chunk in state["relevant_chunks"]]
        result["actual"] = state["answer"]
        result["found_in_corpus"] = state["found_in_corpus"]
        result["chunks_used"] = len(contexts)
        result["passed"] = case_passed(case, state["answer"], state["found_in_corpus"])

        # "Not in corpus" answers have nothing to score, so accuracy alone judges them.
        if case["expected_found"] and state["found_in_corpus"]:
            result["scores"] = await asyncio.to_thread(
                score_answer, case["question"], state["answer"], contexts, case["expected"]
            )


def summary_text(metrics: list[dict[str, Any]]) -> str:
    return ", ".join(
        f"{metric['name']} {'n/a' if metric['value'] is None else round(metric['value'], 2)}"
        for metric in metrics
    )
