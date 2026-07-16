"""Retry tự động cho lỗi 429 (quota exceeded) khi gọi LLM qua LangChain."""
from __future__ import annotations

import logging
import time
from typing import Callable, TypeVar

logger = logging.getLogger(__name__)
T = TypeVar("T")


def _is_quota_error(exc: Exception) -> bool:
    try:
        from google.api_core.exceptions import ResourceExhausted
        return isinstance(exc, ResourceExhausted)
    except ImportError:
        return "429" in str(exc) or "ResourceExhausted" in type(exc).__name__


def call_with_retry(fn: Callable[[], T], max_retries: int = 4, initial_wait_sec: float = 2.0) -> T:
    wait = initial_wait_sec
    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            if _is_quota_error(exc) and attempt < max_retries:
                logger.warning("429 quota exceeded — lần %d/%d, chờ %.0fs rồi thử lại.", attempt, max_retries, wait)
                time.sleep(wait)
                wait *= 2
                continue
            raise
    raise RuntimeError(f"Hết {max_retries} lần retry: {last_exc}")
