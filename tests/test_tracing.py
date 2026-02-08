"""Tests for observability tracing."""

import pytest

from nanobot.agent.tracing import Span, Tracer


def test_span_creation():
    s = Span(name="test", trace_id="abc123")
    assert s.name == "test"
    assert s.trace_id == "abc123"
    assert len(s.span_id) == 12
    assert s.start_time > 0
    assert s.status == "ok"


def test_span_duration():
    s = Span(name="test", trace_id="t1", start_time=1000.0)
    s.end_time = 1000.5
    assert s.duration_ms == pytest.approx(500.0)


def test_span_duration_not_ended():
    s = Span(name="test", trace_id="t1")
    assert s.duration_ms == 0.0


def test_tracer_new_trace():
    tracer = Tracer()
    tid = tracer.new_trace()
    assert len(tid) == 16
    assert tracer._current_trace_id == tid


@pytest.mark.asyncio
async def test_tracer_span_records():
    tracer = Tracer()
    tracer.new_trace()
    async with tracer.span("test_op"):
        pass
    assert len(tracer.spans) == 1
    assert tracer.spans[0].name == "test_op"
    assert tracer.spans[0].status == "ok"
    assert tracer.spans[0].duration_ms > 0


@pytest.mark.asyncio
async def test_tracer_span_error():
    tracer = Tracer()
    tracer.new_trace()
    with pytest.raises(ValueError):
        async with tracer.span("failing"):
            raise ValueError("boom")
    assert tracer.spans[0].status == "error"
    assert "boom" in tracer.spans[0].attributes["error"]


@pytest.mark.asyncio
async def test_tracer_disabled():
    tracer = Tracer(enabled=False)
    tracer.new_trace()
    async with tracer.span("noop"):
        pass
    assert len(tracer.spans) == 0


def test_tracer_get_trace_spans():
    tracer = Tracer()
    s1 = Span(name="a", trace_id="t1")
    s2 = Span(name="b", trace_id="t2")
    s3 = Span(name="c", trace_id="t1")
    tracer.spans = [s1, s2, s3]
    result = tracer.get_trace_spans("t1")
    assert len(result) == 2


def test_tracer_clear():
    tracer = Tracer()
    tracer.spans = [Span(name="x", trace_id="t")]
    tracer.clear()
    assert len(tracer.spans) == 0
