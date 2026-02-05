"""Memory search tool for querying past conversations."""

import re
from datetime import datetime, timedelta
from typing import Any

from nanobot.agent.tools.base import Tool


def _parse_time_range(time_range: str) -> tuple[str | None, str | None]:
    """Parse a time_range string into (after, before) ISO date strings.

    Supports: "today", "this_week", "this_month", "last_N_days".

    Returns:
        Tuple of (after_iso, before_iso). before is always None (up to now).
    """
    now = datetime.now()
    lower = time_range.strip().lower()

    if lower == "today":
        after = now.replace(hour=0, minute=0, second=0, microsecond=0)
        return after.isoformat(), None

    if lower == "this_week":
        start = now - timedelta(days=now.weekday())
        after = start.replace(hour=0, minute=0, second=0, microsecond=0)
        return after.isoformat(), None

    if lower == "this_month":
        after = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        return after.isoformat(), None

    match = re.match(r"last_(\d+)_days?", lower)
    if match:
        days = int(match.group(1))
        after = now - timedelta(days=days)
        return after.isoformat(), None

    return None, None


class MemorySearchTool(Tool):
    """
    Search semantic memory from past conversations.

    Allows the agent to recall information from previous sessions,
    including facts, decisions, and conversation context.
    """

    def __init__(self, vector_store: Any):
        """
        Initialize the memory search tool.

        Args:
            vector_store: VectorStore instance for semantic search.
        """
        self._vector_store = vector_store

    @property
    def name(self) -> str:
        return "memory_search"

    @property
    def description(self) -> str:
        return (
            "Search past conversations and extracted facts from memory. "
            "Use this to recall information discussed in previous sessions, "
            "user preferences, decisions made, or any other context from "
            "the past. Supports time-filtered and type-filtered search."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Search query describing what you're looking for. "
                        "Be specific about the topic or type of information."
                    ),
                },
                "limit": {
                    "type": "integer",
                    "description": ("Maximum number of results to return (default: 5)"),
                    "default": 5,
                    "minimum": 1,
                    "maximum": 20,
                },
                "time_range": {
                    "type": "string",
                    "description": (
                        "Optional time filter: 'today', 'this_week', "
                        "'this_month', or 'last_N_days' (e.g. 'last_7_days')"
                    ),
                },
                "type_filter": {
                    "type": "string",
                    "description": (
                        "Filter by memory type: 'fact', 'conversation', or 'all' (default: 'all')"
                    ),
                    "enum": ["fact", "conversation", "all"],
                    "default": "all",
                },
            },
            "required": ["query"],
        }

    async def execute(self, **kwargs: Any) -> str:
        """Execute the memory search."""
        query = kwargs.get("query", "")
        limit = kwargs.get("limit", 5)
        time_range = kwargs.get("time_range")
        type_filter = kwargs.get("type_filter", "all")

        if not query:
            return "Error: query is required"

        if self._vector_store is None:
            return "Memory search is not available (vector store not initialized)"

        try:
            # Parse time range into after/before dates
            after = None
            before = None
            if time_range:
                after, before = _parse_time_range(time_range)

            results = await self._vector_store.search(
                query=query,
                top_k=limit,
                after=after,
                before=before,
            )

            if not results:
                return f"No memories found matching: {query}"

            # Apply type filter on results if specified
            if type_filter and type_filter != "all":
                results = [r for r in results if r.get("metadata", {}).get("type") == type_filter]

            if not results:
                filters = []
                if time_range:
                    filters.append(f"time_range={time_range}")
                if type_filter != "all":
                    filters.append(f"type={type_filter}")
                filter_str = f" (filters: {', '.join(filters)})" if filters else ""
                return f"No memories found matching: {query}{filter_str}"

            # Format results
            output = [f"Found {len(results)} relevant memories:\n"]

            for i, result in enumerate(results, 1):
                similarity = result.get("similarity", 0)
                text = result.get("text", "")
                metadata = result.get("metadata", {})
                created_at = result.get("created_at", "")

                # Format metadata
                session_key = metadata.get("session_key", "unknown")
                entry_type = metadata.get("type", "conversation")

                output.append(f"--- Memory {i} (similarity: {similarity:.2f}) ---")
                output.append(f"Type: {entry_type}")
                output.append(f"Session: {session_key}")
                if created_at:
                    output.append(f"Date: {created_at[:10]}")
                output.append(f"Content: {text[:500]}{'...' if len(text) > 500 else ''}")
                output.append("")

            return "\n".join(output)

        except Exception as e:
            return f"Error searching memory: {e}"
