"""Gemini access. Everything goes through ask() and the embed functions, so tests mock one place."""

import asyncio
from collections.abc import Awaitable, Callable
from functools import lru_cache
from typing import TypeVar

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from app.config import get_settings

T = TypeVar("T")

EMBED_BATCH_SIZE = 50
MAX_RETRIES = 3

# Grading sends 5 calls at once, which can trip Gemini's free-tier rate limit.
_chat_slots = asyncio.Semaphore(3)


class LLMError(Exception):
    """Gemini returned an error."""


class LLMTimeoutError(LLMError):
    """Gemini did not respond in time."""


@lru_cache
def chat_model() -> ChatGoogleGenerativeAI:
    settings = get_settings()
    return ChatGoogleGenerativeAI(
        model=settings.chat_model,
        google_api_key=settings.google_api_key,
        temperature=0,
        max_retries=MAX_RETRIES,
    )


@lru_cache
def embedding_model() -> GoogleGenerativeAIEmbeddings:
    settings = get_settings()
    return GoogleGenerativeAIEmbeddings(
        model=settings.embedding_model,
        google_api_key=settings.google_api_key,
        output_dimensionality=settings.embedding_dimensions,
    )


async def _guarded(call: Callable[[], Awaitable[T]]) -> T:
    """Apply the timeout and turn provider errors into our own two error types."""
    try:
        return await asyncio.wait_for(call(), timeout=get_settings().llm_timeout_seconds)
    except TimeoutError as exc:
        raise LLMTimeoutError("Gemini did not respond in time") from exc
    except Exception as exc:
        raise LLMError(f"Gemini request failed: {type(exc).__name__}") from exc


async def ask(prompt: str) -> str:
    async with _chat_slots:
        response = await _guarded(lambda: chat_model().ainvoke(prompt))
    return str(response.text).strip()


async def embed_query(text: str) -> list[float]:
    return await _guarded(
        lambda: embedding_model().aembed_query(text, task_type="RETRIEVAL_QUERY")
    )


async def embed_documents(texts: list[str]) -> list[list[float]]:
    # Batched so the timeout applies per batch, not to a whole corpus.
    vectors: list[list[float]] = []
    for start in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[start : start + EMBED_BATCH_SIZE]
        vectors += await _guarded(
            lambda: embedding_model().aembed_documents(batch, task_type="RETRIEVAL_DOCUMENT")
        )
    return vectors
