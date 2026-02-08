"""Lightweight observability tracing for agent operations."""

import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

from loguru import logger


@dataclass
class Span:
    """A single traced operation."""

    name: str
    trace_id: str
    span_id: str = ""
    parent_id: str | None = None
    start_time: float = 0.0
    end_time: float = 0.0
    attributes: dict[str, Any] = field(default_factory=dict)
    status: str = "ok"

    def __post_init__(self) -> None:
        if not self.span_id:
            self.span_id = uuid.uuid4().hex[:12]
        if not self.start_time:
            self.start_time = time.time()

    @property
    def duration_ms(self) -> float:
        """Duration in milliseconds."""
        if self.end_time <= 0:
            return 0.0
        return (self.end_time - self.start_time) * 1000


class Tracer:
    """Lightweight tracer that logs spans and stores them for inspection."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.spans: list[Span] = []
        self._current_trace_id: str = ""

    def new_trace(self) -> str:
        """Start a new trace and return the trace ID."""
        self._current_trace_id = uuid.uuid4().hex[:16]
        return self._current_trace_id

    @asynccontextmanager
    async def span(
        self,
        name: str,
        parent_id: str | None = None,
        attributes: dict[str, Any] | None = None,
    ):
        """Async context manager that creates and records a span.

        Usage:
            async with tracer.span("llm_call", attributes={"model": "gpt-4"}):
                await provider.chat(...)
        """
        if not self.enabled:
            yield None
            return

        s = Span(
            name=name,
            trace_id=self._current_trace_id,
            parent_id=parent_id,
            attributes=attributes or {},
        )

        try:
            yield s
            s.status = "ok"
        except Exception as e:
            s.status = "error"
            s.attributes["error"] = str(e)
            raise
        finally:
            s.end_time = time.time()
            self.spans.append(s)

            logger.debug(f"[trace:{s.trace_id[:8]}] {s.name} {s.duration_ms:.1f}ms [{s.status}]")

    def get_trace_spans(self, trace_id: str) -> list[Span]:
        """Get all spans for a given trace."""
        return [s for s in self.spans if s.trace_id == trace_id]

    def clear(self) -> None:
        """Clear all recorded spans."""
        self.spans.clear()
