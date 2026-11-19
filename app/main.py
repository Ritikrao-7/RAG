import asyncio
import importlib.util
import time
from collections import defaultdict, deque
from pathlib import Path as FilePath
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, Path, Query, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pinecone import PineconeError

from app import llm
from app.comparison import TESTS_DIR, compare_modes
from app.config import get_settings
from app.evaluation import EvaluationRunner
from app.graph import build_graph, initial_state
from app.ingestion import UrlFetchError, ingest_documents
from app.logs import configure_logging, for_tenant, log_buffer
from app.models import (
    AskRequest,
    AskResponse,
    ChunksResponse,
    ComparisonResponse,
    CreateUserRequest,
    DeleteSourceResponse,
    ErrorResponse,
    EvalStatus,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    LogsResponse,
    UserResponse,
)
from app.streaming import ask_events
from app.users import UserStore, make_user_store
from app.vectorstore import PineconeStore, VectorStore

STATIC_DIR = FilePath(__file__).parent / "static"
SAMPLES_DIR = FilePath(__file__).parent.parent / "data" / "running_shoes" / "samples"
MAX_CHUNKS_SHOWN = 500

TenantId = Annotated[str, Path(pattern=r"^[A-Za-z0-9_-]{1,64}$")]


class Rejected(Exception):
    def __init__(self, status_code: int, error: str, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.error = error


class RateLimiter:
    """Sliding one-minute window per key. In memory, so each server instance counts alone."""

    def __init__(self) -> None:
        self._calls: dict[str, deque] = defaultdict(deque)

    def allow(self, key: str, limit: int) -> bool:
        now = time.monotonic()
        calls = self._calls[key]
        while calls and now - calls[0] > 60:
            calls.popleft()
        if len(calls) >= limit:
            return False
        calls.append(now)
        return True


def client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for", "")
    return forwarded.split(",")[0].strip() or (request.client.host if request.client else "-")


def error_response(status_code: int, error: str, detail: str) -> JSONResponse:
    body = ErrorResponse(error=error, detail=detail).model_dump()
    return JSONResponse(status_code=status_code, content=body)


def failed(request: Request, status_code: int, error: str, detail: str) -> JSONResponse:
    tenant_id = request.path_params.get("tenant_id", "-")
    for_tenant(tenant_id).error("%s failed: %s", error, detail)
    return error_response(status_code, error, detail)


def create_app(store: VectorStore | None = None, users: UserStore | None = None) -> FastAPI:
    configure_logging()
    vector_store = store or PineconeStore()
    user_store = users or make_user_store()
    graph = build_graph(vector_store)
    evaluation = EvaluationRunner(graph)
    limiter = RateLimiter()
    known_users: set[str] = set()

    def check_rate(key: str) -> None:
        limit = get_settings().requests_per_minute
        if limit and not limiter.allow(key, limit):
            raise Rejected(429, "rate_limited", f"Limit is {limit} requests a minute.")

    async def guard_tenant(request: Request) -> None:
        """Runs on every route; only routes with a tenant id are checked."""
        tenant_id = request.path_params.get("tenant_id")
        if tenant_id is None:
            return
        if get_settings().require_registration and tenant_id not in known_users:
            if await user_store.get(tenant_id) is None:
                raise Rejected(404, "unknown_user", "Create a user first with POST /users.")
            known_users.add(tenant_id)
        if request.method in ("POST", "DELETE"):
            check_rate(tenant_id)

    app = FastAPI(title="Brand Q&A Agent", dependencies=[Depends(guard_tenant)])

    @app.exception_handler(Rejected)
    async def handle_rejected(request: Request, exc: Rejected) -> JSONResponse:
        return failed(request, exc.status_code, exc.error, str(exc))

    @app.exception_handler(UrlFetchError)
    async def handle_url_error(request: Request, exc: UrlFetchError) -> JSONResponse:
        return failed(request, 422, "url_unreachable", str(exc))

    @app.exception_handler(llm.LLMTimeoutError)
    async def handle_llm_timeout(request: Request, exc: llm.LLMTimeoutError) -> JSONResponse:
        return failed(request, 504, "llm_timeout", str(exc))

    @app.exception_handler(llm.LLMError)
    async def handle_llm_error(request: Request, exc: llm.LLMError) -> JSONResponse:
        return failed(request, 502, "llm_error", str(exc))

    @app.exception_handler(PineconeError)
    async def handle_vector_db_error(request: Request, exc: PineconeError) -> JSONResponse:
        detail = f"Pinecone request failed: {type(exc).__name__}"
        return failed(request, 503, "vector_db_error", detail)

    @app.post("/users", response_model=UserResponse, status_code=201)
    async def create_user(body: CreateUserRequest, request: Request) -> UserResponse:
        check_rate(f"ip:{client_ip(request)}")
        user = await user_store.create(body.name, body.email.lower())
        known_users.add(user["id"])
        return UserResponse(**user)

    @app.get("/users/{user_id}", response_model=UserResponse)
    async def get_user(user_id: TenantId) -> UserResponse | JSONResponse:
        user = await user_store.get(user_id)
        if user is None:
            return error_response(404, "unknown_user", "No user with this id.")
        return UserResponse(**user)

    @app.post("/tenants/{tenant_id}/ingest", response_model=IngestResponse)
    async def ingest(tenant_id: TenantId, request: IngestRequest) -> IngestResponse:
        limit = get_settings().max_chunks_per_tenant
        if limit and len(await vector_store.list_ids(tenant_id, "")) >= limit:
            raise Rejected(403, "quota_exceeded", f"Each user can store up to {limit} chunks.")
        return await ingest_documents(tenant_id, request.documents, vector_store)

    @app.delete("/tenants/{tenant_id}/sources", response_model=DeleteSourceResponse)
    async def delete_source(tenant_id: TenantId, source: str) -> DeleteSourceResponse:
        stored = await vector_store.list_chunks(tenant_id, MAX_CHUNKS_SHOWN)
        ids = [chunk["id"] for chunk in stored if chunk["source"] == source]
        if ids:
            await vector_store.delete(tenant_id, ids)
        for_tenant(tenant_id).info("deleted %d chunks from %s", len(ids), source)
        return DeleteSourceResponse(chunks_deleted=len(ids))

    @app.post("/tenants/{tenant_id}/ask", response_model=AskResponse)
    async def ask(tenant_id: TenantId, request: AskRequest) -> AskResponse:
        for_tenant(tenant_id).info("question: %s", request.question)
        result = await graph.ainvoke(
            initial_state(tenant_id, request.question, request.hybrid, request.rerank)
        )
        return AskResponse(
            answer=result["answer"],
            found_in_corpus=result["found_in_corpus"],
            citations=result["citations"],
            trace=result["trace"],
        )

    @app.post("/tenants/{tenant_id}/ask/stream")
    async def ask_stream(tenant_id: TenantId, request: AskRequest) -> StreamingResponse:
        for_tenant(tenant_id).info("question (stream): %s", request.question)
        state = initial_state(tenant_id, request.question, request.hybrid, request.rerank)
        return StreamingResponse(
            ask_events(graph, state),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/tenants/{tenant_id}/chunks", response_model=ChunksResponse)
    async def chunks(tenant_id: TenantId) -> ChunksResponse:
        stored = await vector_store.list_chunks(tenant_id, MAX_CHUNKS_SHOWN)
        stored.sort(key=lambda chunk: (chunk["source"], chunk["chunk_index"]))
        return ChunksResponse(total=len(stored), chunks=stored)

    @app.get("/tenants/{tenant_id}/logs", response_model=LogsResponse)
    async def logs(tenant_id: TenantId, after: int = 0) -> LogsResponse:
        return LogsResponse(entries=log_buffer.read(tenant_id, after))

    @app.post("/tenants/{tenant_id}/compare", response_model=ComparisonResponse)
    async def compare(
        tenant_id: TenantId, dataset: Literal["three_urls", "shoes", "catalogue"] = "three_urls"
    ) -> dict | JSONResponse:
        if not await vector_store.list_ids(tenant_id, ""):
            return error_response(422, "no_data", "Ingest this dataset for the tenant first.")
        return await compare_modes(tenant_id, dataset, vector_store)

    @app.post("/tenants/{tenant_id}/evaluate", response_model=EvalStatus, status_code=202)
    async def start_evaluation(
        tenant_id: TenantId, limit: Annotated[int | None, Query(ge=1)] = None
    ) -> EvalStatus | JSONResponse:
        if importlib.util.find_spec("ragas") is None:
            return error_response(501, "evaluation_unavailable", "ragas is not installed here.")
        if evaluation.is_running(tenant_id):
            return error_response(409, "evaluation_running", "An evaluation is already running.")
        if not await vector_store.list_ids(tenant_id, ""):
            return error_response(422, "no_data", "Ingest the full sample set before evaluating.")
        return EvalStatus(**evaluation.start(tenant_id, limit))

    @app.get("/tenants/{tenant_id}/evaluate", response_model=EvalStatus)
    async def evaluation_status(tenant_id: TenantId) -> EvalStatus:
        return EvalStatus(**evaluation.status(tenant_id))

    @app.get("/", include_in_schema=False)
    async def home() -> FileResponse:
        return FileResponse(STATIC_DIR / "app.html")

    @app.get("/console", include_in_schema=False)
    async def console() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        vector_db, llm_result = await asyncio.gather(
            vector_store.ping(), llm.ask("Reply with ok."), return_exceptions=True
        )
        vector_db_ok = not isinstance(vector_db, BaseException)
        llm_ok = not isinstance(llm_result, BaseException)
        return HealthResponse(
            status="ok" if vector_db_ok and llm_ok else "degraded",
            vector_db=vector_db_ok,
            llm=llm_ok,
        )

    app.mount("/samples", StaticFiles(directory=SAMPLES_DIR), name="samples")
    app.mount("/retrieval-tests", StaticFiles(directory=TESTS_DIR), name="retrieval_tests")
    return app


app = create_app()
