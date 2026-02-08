"""Tests for LLM retry with exponential backoff."""

from unittest.mock import AsyncMock, patch

import pytest

from nanobot.providers.retry import _is_retryable, with_retry


def test_retryable_rate_limit():
    assert _is_retryable(Exception("Rate limit exceeded (429)"))


def test_retryable_server_error():
    assert _is_retryable(Exception("Internal server error 500"))


def test_retryable_timeout():
    assert _is_retryable(Exception("Connection timed out"))


def test_retryable_overloaded():
    assert _is_retryable(Exception("Server overloaded"))


def test_not_retryable_auth():
    assert not _is_retryable(Exception("Invalid API key (401)"))


def test_not_retryable_not_found():
    assert not _is_retryable(Exception("Model not found (404)"))


def test_not_retryable_bad_request():
    assert not _is_retryable(Exception("Bad request: missing field"))


@pytest.mark.asyncio
async def test_retry_succeeds_first_try():
    fn = AsyncMock(return_value="ok")
    result = await with_retry(fn, max_retries=3)
    assert result == "ok"
    assert fn.call_count == 1


@pytest.mark.asyncio
async def test_retry_succeeds_after_transient():
    fn = AsyncMock(side_effect=[Exception("Rate limit 429"), "ok"])
    with patch("nanobot.providers.retry.asyncio.sleep", new_callable=AsyncMock):
        result = await with_retry(fn, max_retries=3, base_delay=0.01)
    assert result == "ok"
    assert fn.call_count == 2


@pytest.mark.asyncio
async def test_retry_exhausted():
    fn = AsyncMock(side_effect=Exception("Server error 500"))
    with patch("nanobot.providers.retry.asyncio.sleep", new_callable=AsyncMock):
        with pytest.raises(Exception, match="500"):
            await with_retry(fn, max_retries=2, base_delay=0.01)
    assert fn.call_count == 3  # initial + 2 retries


@pytest.mark.asyncio
async def test_retry_non_retryable_immediate():
    fn = AsyncMock(side_effect=Exception("Invalid API key"))
    with pytest.raises(Exception, match="Invalid API key"):
        await with_retry(fn, max_retries=3)
    assert fn.call_count == 1
