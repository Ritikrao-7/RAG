from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator


class Document(BaseModel):
    url: HttpUrl | None = None
    text: str | None = Field(default=None, min_length=1, max_length=200_000)
    source: str | None = None
    brand: str = Field(min_length=1)

    @model_validator(mode="after")
    def require_url_or_text(self) -> "Document":
        if (self.url is None) == (self.text is None):
            raise ValueError("provide exactly one of url or text")
        return self


class IngestRequest(BaseModel):
    documents: list[Document] = Field(min_length=1)


class IngestResponse(BaseModel):
    documents_stored: int
    chunks_stored: int


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    hybrid: bool = False
    rerank: bool = False


class Citation(BaseModel):
    source: str
    brand: str
    chunk_text: str


class TraceEntry(BaseModel):
    node: str
    detail: str


class AskResponse(BaseModel):
    answer: str
    found_in_corpus: bool
    citations: list[Citation]
    trace: list[TraceEntry]


class LogEntry(BaseModel):
    id: int
    time: str
    level: str
    message: str


class LogsResponse(BaseModel):
    entries: list[LogEntry]


class StoredChunk(BaseModel):
    id: str
    source: str
    brand: str
    chunk_index: int
    chunk_text: str


class ChunksResponse(BaseModel):
    total: int
    chunks: list[StoredChunk]


class EvalMetric(BaseModel):
    name: str
    value: float | None
    target: float
    met: bool | None


class EvalCaseResult(BaseModel):
    question: str
    expected: str
    expected_found: bool
    actual: str
    found_in_corpus: bool
    chunks_used: int
    passed: bool
    scores: dict[str, float | None]
    error: str | None


class EvalStatus(BaseModel):
    state: str
    done: int
    total: int
    metrics: list[EvalMetric] = []
    cases: list[EvalCaseResult] = []


class ModeSummary(BaseModel):
    mode: str
    found_first: float
    found_in_top_5: float


class ComparisonResult(BaseModel):
    question: str
    kind: str
    ranks: dict[str, int | None]


class ComparisonResponse(BaseModel):
    dataset: str
    chunks_searched: int
    summary: list[ModeSummary]
    results: list[ComparisonResult]


class HealthResponse(BaseModel):
    status: str
    vector_db: bool
    llm: bool


class CreateUserRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    name: str = Field(min_length=1, max_length=100)
    email: str = Field(max_length=254, pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class UserResponse(BaseModel):
    id: str
    name: str
    email: str
    created_at: str


class DeleteSourceResponse(BaseModel):
    chunks_deleted: int


class ErrorResponse(BaseModel):
    error: str
    detail: str
