"""Tests for streaming infrastructure."""

import pytest

from nanobot.providers.base import LLMProvider, LLMResponse, StreamChunk


def test_stream_chunk_defaults():
    chunk = StreamChunk()
    assert chunk.content == ""
    assert chunk.tool_calls == []
    assert chunk.finish_reason is None
    assert chunk.usage == {}


def test_stream_chunk_with_content():
    chunk = StreamChunk(content="hello", finish_reason="stop")
    assert chunk.content == "hello"
    assert chunk.finish_reason == "stop"


class MockProvider(LLMProvider):
    """Concrete provider for testing default stream() fallback."""

    async def chat(self, messages, **kwargs):
        return LLMResponse(
            content="test response",
            finish_reason="stop",
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        )

    def get_default_model(self):
        return "mock-model"


@pytest.mark.asyncio
async def test_stream_fallback():
    provider = MockProvider()
    chunks = []
    async for chunk in provider.stream(messages=[{"role": "user", "content": "hi"}]):
        chunks.append(chunk)
    assert len(chunks) == 1
    assert chunks[0].content == "test response"
    assert chunks[0].finish_reason == "stop"
    assert chunks[0].usage["total_tokens"] == 15
