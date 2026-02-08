"""Progress event types for real-time feedback during agent processing."""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable


class ProgressKind(Enum):
    """Types of progress events emitted during agent processing."""

    THINKING = "thinking"
    TOOL_START = "tool_start"
    TOOL_COMPLETE = "tool_complete"
    STREAMING = "streaming"
    CLARIFICATION = "clarification"
    ERROR = "error"


@dataclass
class ProgressEvent:
    """A progress event emitted during agent message processing."""

    channel: str
    chat_id: str
    kind: ProgressKind
    detail: str = ""
    tool_name: str | None = None
    iteration: int = 0
    total_iterations: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


ProgressCallback = Callable[[ProgressEvent], Awaitable[None]]
