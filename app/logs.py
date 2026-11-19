"""Logging to the terminal, plus a per-tenant copy the test page can read.

Logs can hold questions and chunk previews, so they are stored per tenant, the same
way the data is. They live in memory and are lost when the server restarts.
"""

import logging
from collections import defaultdict, deque
from datetime import datetime, timezone
from itertools import count

MAX_ENTRIES_PER_TENANT = 500

logger = logging.getLogger("brand_qa")


class TenantLogBuffer(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self._entries: dict[str, deque] = defaultdict(lambda: deque(maxlen=MAX_ENTRIES_PER_TENANT))
        self._next_id = count(1)

    def emit(self, record: logging.LogRecord) -> None:
        tenant_id = getattr(record, "tenant_id", None)
        if tenant_id:
            self._entries[tenant_id].append(
                {
                    "id": next(self._next_id),
                    "time": datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
                    "level": record.levelname,
                    "message": record.getMessage(),
                }
            )

    def read(self, tenant_id: str, after: int = 0) -> list[dict]:
        return [entry for entry in self._entries.get(tenant_id, ()) if entry["id"] > after]


class HideLogPolling(logging.Filter):
    """The page polls for logs every second, which would flood the terminal."""

    def filter(self, record: logging.LogRecord) -> bool:
        return "/logs" not in record.getMessage()


log_buffer = TenantLogBuffer()


def configure_logging() -> None:
    if logger.handlers:
        return
    terminal = logging.StreamHandler()
    terminal.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)-7s [%(tenant_id)s] %(message)s",
            datefmt="%H:%M:%S",
            defaults={"tenant_id": "-"},
        )
    )
    logger.addHandler(terminal)
    logger.addHandler(log_buffer)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logging.getLogger("uvicorn.access").addFilter(HideLogPolling())


def for_tenant(tenant_id: str) -> logging.LoggerAdapter:
    return logging.LoggerAdapter(logger, {"tenant_id": tenant_id})


def preview(text: str, length: int = 100) -> str:
    return " ".join(text.split())[:length]
