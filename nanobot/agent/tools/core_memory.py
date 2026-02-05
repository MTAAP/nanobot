"""Core memory tools for reading and updating the agent's persistent scratchpad."""

from typing import Any

from nanobot.agent.tools.base import Tool
from nanobot.memory.core import CoreMemory


class CoreMemoryReadTool(Tool):
    """
    Read the agent's core memory.

    Core memory is a small persistent scratchpad always visible in
    the agent's context. This tool retrieves its contents.
    """

    def __init__(self, core_memory: CoreMemory):
        """
        Initialize the core memory read tool.

        Args:
            core_memory: CoreMemory instance to read from.
        """
        self._core_memory = core_memory

    @property
    def name(self) -> str:
        return "core_memory_read"

    @property
    def description(self) -> str:
        return (
            "Read the agent's core memory (persistent scratchpad). "
            "Returns all sections or a specific section. Core memory is "
            "always visible in your context - use this to review what "
            "you've stored about the user, preferences, and projects."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "section": {
                    "type": "string",
                    "description": ("Name of the section to read. Omit to read all sections."),
                },
            },
            "required": [],
        }

    async def execute(self, **kwargs: Any) -> str:
        """Execute the core memory read."""
        section = kwargs.get("section")

        if self._core_memory is None:
            return "Core memory is not available."

        try:
            return self._core_memory.read(section=section)
        except Exception as e:
            return f"Error reading core memory: {e}"


class CoreMemoryUpdateTool(Tool):
    """
    Update a section of the agent's core memory.

    Core memory is always visible in context. Use it for key user info,
    current projects, and important preferences.
    """

    def __init__(self, core_memory: CoreMemory):
        """
        Initialize the core memory update tool.

        Args:
            core_memory: CoreMemory instance to update.
        """
        self._core_memory = core_memory

    @property
    def name(self) -> str:
        return "core_memory_update"

    @property
    def description(self) -> str:
        return (
            "Update a section of core memory. Core memory is always "
            "visible in your context - use it for key user info, current "
            "projects, and important preferences. Creates the section if "
            "it does not exist."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "section": {
                    "type": "string",
                    "description": (
                        "Name of the section to update "
                        "(e.g. 'user', 'preferences', 'current_projects')."
                    ),
                },
                "content": {
                    "type": "string",
                    "description": (
                        "New content for the section. Replaces existing "
                        "content entirely. Keep concise - total core memory "
                        "is limited to 2000 characters."
                    ),
                },
            },
            "required": ["section", "content"],
        }

    async def execute(self, **kwargs: Any) -> str:
        """Execute the core memory update."""
        section = kwargs.get("section", "")
        content = kwargs.get("content", "")

        if self._core_memory is None:
            return "Core memory is not available."

        if not section:
            return "Error: section is required."

        try:
            return self._core_memory.update(section=section, content=content)
        except Exception as e:
            return f"Error updating core memory: {e}"
