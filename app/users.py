"""Users of the public demo. A user's id is also their tenant id.

There is no password or email check: the id lives in the browser, and the name and
email are kept so we know who is using the demo. Postgres when DATABASE_URL is set,
otherwise memory, which is enough for local runs and tests.
"""

import uuid
from datetime import datetime, timezone
from typing import Any, Protocol

from app.config import get_settings

CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS demo_users (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    email TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
)
"""


def new_user_id() -> str:
    # Matches the tenant id pattern, so it can be used in /tenants/{id} paths as is.
    return f"u_{uuid.uuid4().hex}"


class UserStore(Protocol):
    async def create(self, name: str, email: str) -> dict[str, Any]: ...

    async def get(self, user_id: str) -> dict[str, Any] | None: ...


class MemoryUserStore:
    def __init__(self) -> None:
        self._users: dict[str, dict[str, Any]] = {}

    async def create(self, name: str, email: str) -> dict[str, Any]:
        user = {
            "id": new_user_id(),
            "name": name,
            "email": email,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        self._users[user["id"]] = user
        return user

    async def get(self, user_id: str) -> dict[str, Any] | None:
        return self._users.get(user_id)


class PostgresUserStore:
    """Opens a connection per call, which suits serverless; use a pooled URL (e.g. Neon's)."""

    def __init__(self, url: str) -> None:
        self._url = url
        self._table_ready = False

    async def _connect(self) -> Any:
        import psycopg
        from psycopg.rows import dict_row

        connection = await psycopg.AsyncConnection.connect(
            self._url, autocommit=True, row_factory=dict_row
        )
        if not self._table_ready:
            await connection.execute(CREATE_TABLE)
            self._table_ready = True
        return connection

    async def create(self, name: str, email: str) -> dict[str, Any]:
        async with await self._connect() as connection:
            cursor = await connection.execute(
                "INSERT INTO demo_users (id, name, email) VALUES (%s, %s, %s) "
                "RETURNING id, name, email, created_at",
                (new_user_id(), name, email),
            )
            row = await cursor.fetchone()
        return {**row, "created_at": row["created_at"].isoformat()}

    async def get(self, user_id: str) -> dict[str, Any] | None:
        async with await self._connect() as connection:
            cursor = await connection.execute(
                "SELECT id, name, email, created_at FROM demo_users WHERE id = %s", (user_id,)
            )
            row = await cursor.fetchone()
        return {**row, "created_at": row["created_at"].isoformat()} if row else None


def make_user_store() -> UserStore:
    url = get_settings().database_url
    return PostgresUserStore(url) if url else MemoryUserStore()
