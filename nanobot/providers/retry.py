"""LLM call retry with exponential backoff."""

import asyncio
from collections.abc import Awaitable, Callable
from typing import TypeVar

from loguru import logger

T = TypeVar("T")

# HTTP status codes and error patterns that are safe to retry
_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
_RETRYABLE_PATTERNS = (
    "rate limit",
    "timeout",
    "timed out",
    "connection",
    "server error",
    "overloaded",
    "too many requests",
    "temporarily unavailable",
)


def _is_retryable(error: Exception) -> bool:
    """Check if an error is transient and safe to retry."""
    error_str = str(error).lower()

    # Check for known retryable patterns
    if any(pattern in error_str for pattern in _RETRYABLE_PATTERNS):
        return True

    # Check for HTTP status codes in the error message
    for code in _RETRYABLE_STATUS_CODES:
        if str(code) in error_str:
            return True

    return False


async def with_retry(
    fn: Callable[..., Awaitable[T]],
    *args: object,
    max_retries: int = 3,
    base_delay: float = 1.0,
    **kwargs: object,
) -> T:
    """Call an async function with exponential backoff retry on transient errors.

    Args:
        fn: Async function to call.
        *args: Positional arguments for fn.
        max_retries: Maximum number of retry attempts.
        base_delay: Base delay in seconds (doubles each retry).
        **kwargs: Keyword arguments for fn.

    Returns:
        The return value of fn.

    Raises:
        The last exception if all retries are exhausted or error is non-retryable.
    """
    last_error: Exception | None = None

    for attempt in range(max_retries + 1):
        try:
            return await fn(*args, **kwargs)
        except Exception as e:
            last_error = e

            if attempt >= max_retries or not _is_retryable(e):
                raise

            delay = base_delay * (2**attempt)
            logger.info(
                f"Retryable error (attempt {attempt + 1}/{max_retries}), retrying in {delay}s: {e}"
            )
            await asyncio.sleep(delay)

    # Should not reach here, but satisfy type checker
    raise last_error  # type: ignore[misc]
