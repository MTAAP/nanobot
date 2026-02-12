"""Spawn tools for creating background subagents."""

import uuid
from typing import TYPE_CHECKING, Any

from nanobot.agent.tools.base import Tool

if TYPE_CHECKING:
    from nanobot.agent.subagent import SubagentManager
    from nanobot.registry.store import AgentRegistry


class SpawnTool(Tool):
    """
    Tool to spawn a subagent for background task execution.

    The subagent runs asynchronously and announces its result back
    to the main agent when complete.
    """

    def __init__(
        self,
        manager: "SubagentManager",
        registry: "AgentRegistry | None" = None,
    ):
        self._manager = manager
        self._registry = registry
        self._origin_channel = "cli"
        self._origin_chat_id = "direct"

    def set_context(self, channel: str, chat_id: str) -> None:
        """Set the origin context for subagent announcements."""
        self._origin_channel = channel
        self._origin_chat_id = chat_id

    @property
    def name(self) -> str:
        return "spawn"

    @property
    def description(self) -> str:
        return (
            "Spawn a background subagent to handle a complex task independently. "
            "Use this when a task will take >30 seconds, requires 5+ tool calls, "
            "or when the user wants something done 'in the background'. "
            "The subagent will complete the task and you will be notified with the result. "
            "Examples: debugging complex issues, analyzing multiple files/PRs, "
            "web research, checking multiple systems."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "The task for the subagent to complete",
                },
                "label": {
                    "type": "string",
                    "description": "Optional short label for the task (for display)",
                },
            },
            "required": ["task"],
        }

    async def execute(self, task: str, label: str | None = None, **kwargs: Any) -> str:
        """Spawn a subagent to execute the given task."""
        registry_task_id: str | None = None

        # If registry is enabled, create a task before spawning
        # If registry is enabled, create a task before spawning
        if self._registry:
            try:
                registry_task_id = str(uuid.uuid4())[:8]
                await self._registry.create_task(
                    task_id=registry_task_id,
                    description=task[:500],
                    priority="medium",
                    complexity="complex",
                )
            except Exception:
                registry_task_id = None

        return await self._manager.spawn(
            task=task,
            label=label,
            origin_channel=self._origin_channel,
            origin_chat_id=self._origin_chat_id,
            registry_task_id=registry_task_id,
        )


class SpawnBatchTool(Tool):
    """Spawn multiple subagents in parallel and collect results."""

    def __init__(self, manager: "SubagentManager"):
        self._manager = manager
        self._origin_channel = "cli"
        self._origin_chat_id = "direct"

    def set_context(self, channel: str, chat_id: str) -> None:
        """Set the origin context for progress events."""
        self._origin_channel = channel
        self._origin_chat_id = chat_id

    @property
    def name(self) -> str:
        return "spawn_batch"

    @property
    def description(self) -> str:
        return (
            "Spawn multiple subagents to work on tasks in parallel. "
            "Results are collected and returned together. Use for "
            "batch operations like checking multiple PRs, researching "
            "multiple topics, or processing multiple items."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "tasks": {
                    "type": "array",
                    "description": "List of tasks to execute in parallel",
                    "items": {
                        "type": "object",
                        "properties": {
                            "task": {
                                "type": "string",
                                "description": "The task description",
                            },
                            "label": {
                                "type": "string",
                                "description": "Short label for display",
                            },
                        },
                        "required": ["task"],
                    },
                    "minItems": 1,
                    "maxItems": 10,
                },
            },
            "required": ["tasks"],
        }

    async def execute(self, tasks: list[dict[str, str]], **kwargs: Any) -> str:
        """Execute batch spawn and return combined results."""
        try:
            return await self._manager.spawn_batch(
                tasks=tasks,
                origin_channel=self._origin_channel,
                origin_chat_id=self._origin_chat_id,
            )
        except Exception as e:
            return f"Error: batch spawn failed: {e}"
