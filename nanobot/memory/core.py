"""Core memory store — agent-writable persistent scratchpad always in context."""

import json
from pathlib import Path

from loguru import logger

from nanobot.utils.helpers import ensure_dir

# Maximum total characters across all sections
MAX_TOTAL_CHARS = 2000

# Default sections created on first initialization
DEFAULT_SECTIONS: dict[str, str] = {
    "user": "",
    "preferences": "",
    "current_projects": "",
}


class CoreMemory:
    """
    Agent-writable persistent memory blocks, always in context.

    A small (~500 token) JSON-backed scratchpad that is ALWAYS injected into
    the system prompt. Edited via dedicated core_memory_read/update tools,
    not file-editing tools.
    """

    def __init__(self, workspace: Path):
        """
        Initialize core memory.

        Args:
            workspace: Root workspace path. Core memory file is stored
                       at workspace/memory/core_memory.json.
        """
        memory_dir = ensure_dir(workspace / "memory")
        self.store_path = memory_dir / "core_memory.json"
        self._data: dict[str, str] = {}
        self._load()

    def _load(self) -> None:
        """Load core memory from disk, creating defaults if missing."""
        if not self.store_path.exists():
            logger.debug("Core memory file not found, creating defaults")
            self._data = dict(DEFAULT_SECTIONS)
            self._save()
            return

        try:
            raw = self.store_path.read_text(encoding="utf-8")
            loaded = json.loads(raw)
            if not isinstance(loaded, dict):
                logger.warning("Core memory file had invalid format, resetting")
                self._data = dict(DEFAULT_SECTIONS)
                self._save()
                return
            self._data = {str(k): str(v) for k, v in loaded.items()}
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Failed to load core memory, resetting: {e}")
            self._data = dict(DEFAULT_SECTIONS)
            self._save()

    def _save(self) -> None:
        """Persist core memory to disk."""
        try:
            self.store_path.write_text(
                json.dumps(self._data, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError as e:
            logger.error(f"Failed to save core memory: {e}")

    def _total_chars(self) -> int:
        """Return total character count across all sections."""
        return sum(len(v) for v in self._data.values())

    def read(self, section: str | None = None) -> str:
        """
        Read a section or all of core memory.

        Args:
            section: Section name to read, or None for all sections.

        Returns:
            Formatted content string.
        """
        if section is not None:
            if section not in self._data:
                return f"Section '{section}' does not exist."
            content = self._data[section]
            if not content:
                return f"Section '{section}' is empty."
            return f"### {section}\n{content}"

        if not self._data:
            return "Core memory is empty."

        parts: list[str] = []
        for key, value in self._data.items():
            label = f"### {key}"
            parts.append(f"{label}\n{value}" if value else f"{label}\n(empty)")
        return "\n\n".join(parts)

    def update(self, section: str, content: str) -> str:
        """
        Update a section of core memory. Creates the section if new.

        Args:
            section: Section name.
            content: New content for the section.

        Returns:
            Confirmation or error message.
        """
        # Calculate what the new total size would be
        old_len = len(self._data.get(section, ""))
        new_total = self._total_chars() - old_len + len(content)

        if new_total > MAX_TOTAL_CHARS:
            available = MAX_TOTAL_CHARS - (self._total_chars() - old_len)
            return (
                f"Error: Content too large. Total core memory would be "
                f"{new_total} chars (max {MAX_TOTAL_CHARS}). "
                f"Available space for this section: {max(0, available)} chars."
            )

        is_new = section not in self._data
        self._data[section] = content
        self._save()

        action = "Created" if is_new else "Updated"
        logger.debug(
            f"{action} core memory section '{section}' "
            f"({len(content)} chars, total: {self._total_chars()})"
        )
        return (
            f"{action} section '{section}'. "
            f"Total core memory: {self._total_chars()}/{MAX_TOTAL_CHARS} chars."
        )

    def delete_section(self, section: str) -> bool:
        """
        Delete a section from core memory.

        Args:
            section: Section name to delete.

        Returns:
            True if deleted, False if section didn't exist.
        """
        if section not in self._data:
            return False
        del self._data[section]
        self._save()
        logger.debug(f"Deleted core memory section '{section}'")
        return True

    def list_sections(self) -> list[str]:
        """List all section names."""
        return list(self._data.keys())

    def get_context(self) -> str:
        """
        Get formatted core memory for system prompt injection.

        Returns:
            Formatted string with all sections, or empty string if
            all sections are empty.
        """
        has_content = any(v for v in self._data.values())
        if not has_content:
            return ""

        parts = ["## Core Memory"]
        for section, content in self._data.items():
            if content:
                parts.append(f"### {section}")
                parts.append(content)
        return "\n".join(parts)
