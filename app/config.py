from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All configuration comes from environment variables or .env."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    google_api_key: str = ""
    pinecone_api_key: str = ""
    pinecone_index_name: str = "brand-qa"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"

    chat_model: str = "gemini-3.1-flash-lite"
    embedding_model: str = "gemini-embedding-001"
    embedding_dimensions: int = 768
    rerank_model: str = "bge-reranker-v2-m3"

    llm_timeout_seconds: float = 60
    url_timeout_seconds: float = 20

    # 400 tokens keeps one idea per chunk; 50 overlap keeps sentences that span a boundary.
    chunk_size: int = 400
    chunk_overlap: int = 50
    top_k: int = 5

    # Public deployment. Postgres holds the demo users; without it they live in memory.
    database_url: str = ""
    # When true, /tenants/{id} only accepts ids created through POST /users.
    require_registration: bool = False
    # 0 turns a limit off. The rate limit is per server instance, so it is best effort.
    requests_per_minute: int = 0
    max_chunks_per_tenant: int = 0

    # Pause between evaluation questions, to stay under the Gemini free-tier rate limit.
    eval_pause_seconds: float = 15


@lru_cache
def get_settings() -> Settings:
    return Settings()
