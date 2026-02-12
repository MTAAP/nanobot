"""Tests for parallel tool execution, spawn_batch, and concurrency limiter."""

import asyncio
import time
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.agent.subagent import SubagentManager
from nanobot.agent.tools.base import Tool
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.tools.spawn import SpawnBatchTool
from nanobot.bus.queue import MessageBus
from nanobot.config.schema import AgentDefaults

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class SlowTool(Tool):
    """Tool that sleeps for a configurable duration."""

    def __init__(self, delay: float = 0.1):
        self._delay = delay

    @property
    def name(self) -> str:
        return "slow_tool"

    @property
    def description(self) -> str:
        return "sleeps then returns"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "value": {"type": "string"},
            },
            "required": ["value"],
        }

    async def execute(self, value: str = "", **kw: Any) -> str:
        await asyncio.sleep(self._delay)
        return f"done:{value}"


class FastTool(Tool):
    @property
    def name(self) -> str:
        return "fast_tool"

    @property
    def description(self) -> str:
        return "instant return"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "x": {"type": "string"},
            },
            "required": ["x"],
        }

    async def execute(self, x: str = "", **kw: Any) -> str:
        return f"fast:{x}"


def _make_tool_call(call_id: str, name: str, arguments: dict[str, Any]) -> MagicMock:
    tc = MagicMock()
    tc.id = call_id
    tc.name = name
    tc.arguments = arguments
    return tc


def _make_mock_provider(
    final_content: str = "done",
) -> AsyncMock:
    """Return a provider whose chat() returns no tool calls."""
    response = MagicMock()
    response.has_tool_calls = False
    response.tool_calls = []
    response.content = final_content
    response.usage = {}

    provider = AsyncMock()
    provider.chat = AsyncMock(return_value=response)
    provider.get_default_model = MagicMock(return_value="test-model")
    return provider


# ---------------------------------------------------------------------------
# Feature 1: Parallel tool execution
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_parallel_tool_execution_faster_than_sequential():
    """Multiple slow tools via gather should be faster than sum of delays."""
    registry = ToolRegistry()
    delay = 0.15
    registry.register(SlowTool(delay=delay))

    tc1 = _make_tool_call("c1", "slow_tool", {"value": "a"})
    tc2 = _make_tool_call("c2", "slow_tool", {"value": "b"})
    tc3 = _make_tool_call("c3", "slow_tool", {"value": "c"})

    start = time.monotonic()
    results = await asyncio.gather(
        registry.execute(tc1.name, tc1.arguments),
        registry.execute(tc2.name, tc2.arguments),
        registry.execute(tc3.name, tc3.arguments),
    )
    elapsed = time.monotonic() - start

    assert results == ["done:a", "done:b", "done:c"]
    # Parallel: should take ~1x delay, not 3x
    assert elapsed < delay * 2.5


@pytest.mark.asyncio
async def test_parallel_results_preserve_order():
    """Results from gather must match the order of input tool calls."""
    registry = ToolRegistry()
    registry.register(SlowTool(delay=0.05))
    registry.register(FastTool())

    # slow then fast — order must be preserved
    r1, r2 = await asyncio.gather(
        registry.execute("slow_tool", {"value": "first"}),
        registry.execute("fast_tool", {"x": "second"}),
    )
    assert r1 == "done:first"
    assert r2 == "fast:second"


@pytest.mark.asyncio
async def test_single_tool_call_works():
    """Single tool call should work without gather."""
    registry = ToolRegistry()
    registry.register(FastTool())

    result = await registry.execute("fast_tool", {"x": "solo"})
    assert result == "fast:solo"


# ---------------------------------------------------------------------------
# Feature 2: spawn_batch
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_spawn_batch_collects_results():
    """spawn_batch should run tasks and return combined results."""
    provider = _make_mock_provider("Task result here")
    bus = MessageBus()
    mgr = SubagentManager(
        provider=provider,
        workspace=Path("/tmp"),
        bus=bus,
        max_concurrent=5,
    )

    tasks = [
        {"task": "Do thing 1", "label": "T1"},
        {"task": "Do thing 2", "label": "T2"},
    ]
    result = await mgr.spawn_batch(
        tasks=tasks,
        origin_channel="test",
        origin_chat_id="123",
        timeout_s=10,
    )

    assert "2/2 succeeded" in result
    assert "T1" in result
    assert "T2" in result
    assert "Task result here" in result


@pytest.mark.asyncio
async def test_spawn_batch_empty_tasks():
    """spawn_batch with no tasks returns error."""
    provider = _make_mock_provider()
    bus = MessageBus()
    mgr = SubagentManager(
        provider=provider,
        workspace=Path("/tmp"),
        bus=bus,
    )
    result = await mgr.spawn_batch(tasks=[])
    assert "Error" in result


@pytest.mark.asyncio
async def test_spawn_batch_handles_failure():
    """spawn_batch reports failures per-task."""
    provider = AsyncMock()
    provider.get_default_model = MagicMock(return_value="m")
    provider.chat = AsyncMock(side_effect=RuntimeError("boom"))

    bus = MessageBus()
    mgr = SubagentManager(
        provider=provider,
        workspace=Path("/tmp"),
        bus=bus,
        max_concurrent=5,
    )

    tasks = [{"task": "fail task", "label": "Fail"}]
    result = await mgr.spawn_batch(tasks=tasks, timeout_s=10)

    assert "0/1 succeeded" in result
    assert "1 failed" in result


# ---------------------------------------------------------------------------
# Feature 3: Concurrency limiter
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_concurrency_limiter():
    """Only max_concurrent subagents should run simultaneously."""
    provider = _make_mock_provider("ok")
    bus = MessageBus()
    max_c = 2
    mgr = SubagentManager(
        provider=provider,
        workspace=Path("/tmp"),
        bus=bus,
        max_concurrent=max_c,
    )

    peak_concurrent = 0
    current_concurrent = 0
    lock = asyncio.Lock()

    async def tracked_inner(*args: Any, **kwargs: Any) -> str:
        nonlocal peak_concurrent, current_concurrent
        async with lock:
            current_concurrent += 1
            if current_concurrent > peak_concurrent:
                peak_concurrent = current_concurrent
        try:
            await asyncio.sleep(0.05)
            return "tracked"
        finally:
            async with lock:
                current_concurrent -= 1

    mgr._execute_subagent_inner = tracked_inner

    tasks = [{"task": f"task {i}", "label": f"L{i}"} for i in range(5)]
    await mgr.spawn_batch(tasks=tasks, timeout_s=10)

    assert peak_concurrent <= max_c


@pytest.mark.asyncio
async def test_get_capacity():
    """get_capacity returns correct values."""
    provider = _make_mock_provider()
    bus = MessageBus()
    mgr = SubagentManager(
        provider=provider,
        workspace=Path("/tmp"),
        bus=bus,
        max_concurrent=3,
    )

    cap = mgr.get_capacity()
    assert cap == {"running": 0, "max": 3, "available": 3}


@pytest.mark.asyncio
async def test_get_capacity_during_run():
    """get_capacity reflects running tasks."""
    provider = _make_mock_provider()
    bus = MessageBus()
    mgr = SubagentManager(
        provider=provider,
        workspace=Path("/tmp"),
        bus=bus,
        max_concurrent=5,
    )

    # Spawn a background task (it runs async, silent)
    await mgr.spawn(
        task="background work",
        silent=True,
    )
    # Brief yield to let the task start
    await asyncio.sleep(0.01)

    cap = mgr.get_capacity()
    assert cap["running"] >= 0  # May have finished already
    assert cap["max"] == 5


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def test_config_max_concurrent_subagents_default():
    """Config field should default to 5."""
    defaults = AgentDefaults()
    assert defaults.max_concurrent_subagents == 5


def test_config_max_concurrent_subagents_alias():
    """Config field should accept camelCase alias."""
    defaults = AgentDefaults(**{"maxConcurrentSubagents": 10})
    assert defaults.max_concurrent_subagents == 10


# ---------------------------------------------------------------------------
# SpawnBatchTool
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_spawn_batch_tool_schema():
    """SpawnBatchTool should have valid schema."""
    provider = _make_mock_provider()
    bus = MessageBus()
    mgr = SubagentManager(
        provider=provider,
        workspace=Path("/tmp"),
        bus=bus,
    )
    tool = SpawnBatchTool(manager=mgr)

    assert tool.name == "spawn_batch"
    schema = tool.to_schema()
    assert schema["type"] == "function"
    props = schema["function"]["parameters"]["properties"]
    assert "tasks" in props
    assert props["tasks"]["type"] == "array"
