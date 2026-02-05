"""Memory forget tool for removing specific memories from the vector store."""

from typing import Any

from loguru import logger

from nanobot.agent.tools.base import Tool

# Maximum deletions allowed per tool instance (per conversation)
MAX_DELETIONS_PER_SESSION = 5


class MemoryForgetTool(Tool):
    """
    Remove specific memories from the vector store.

    Use when the user asks to forget something or when incorrect
    information was stored. Limited to 5 deletions per conversation.
    Supports a preview mode (confirm=False) to show what would be
    deleted before actually deleting.
    """

    def __init__(self, vector_store: Any):
        """
        Initialize the memory forget tool.

        Args:
            vector_store: VectorStore instance with search and delete methods.
        """
        self._vector_store = vector_store
        self._deletion_count = 0

    @property
    def name(self) -> str:
        return "memory_forget"

    @property
    def description(self) -> str:
        return (
            "Remove specific memories from the vector store. Use when "
            "the user asks you to forget something or when incorrect "
            "information was stored. Set confirm=false first to preview "
            "what would be deleted, then confirm=true to execute. "
            "Limited to 5 deletions per conversation."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Description of what to forget. Be specific "
                        "about the topic or information to remove."
                    ),
                },
                "confirm": {
                    "type": "boolean",
                    "description": (
                        "Set to false to preview what would be deleted. "
                        "Set to true to actually delete the memories."
                    ),
                },
            },
            "required": ["query", "confirm"],
        }

    async def execute(self, **kwargs: Any) -> str:
        """Execute the memory forget operation."""
        query = kwargs.get("query", "")
        confirm = kwargs.get("confirm", False)

        if not query:
            return "Error: query is required."

        if self._vector_store is None:
            return "Memory forget is not available (vector store not initialized)."

        if self._deletion_count >= MAX_DELETIONS_PER_SESSION:
            return (
                f"Deletion limit reached ({MAX_DELETIONS_PER_SESSION} "
                f"per conversation). Cannot delete more memories in "
                f"this session."
            )

        try:
            # Search for matching memories
            results = await self._vector_store.search(
                query=query,
                top_k=3,
                min_similarity=0.7,
            )

            if not results:
                return f"No memories found matching: {query}"

            # Preview mode: show what would be deleted
            if not confirm:
                lines = [f"Found {len(results)} memories that would be deleted:\n"]
                for i, result in enumerate(results, 1):
                    text = result.get("text", "")
                    similarity = result.get("similarity", 0)
                    created_at = result.get("created_at", "")
                    preview = text[:200] + "..." if len(text) > 200 else text
                    lines.append(f"{i}. (similarity: {similarity:.2f}) {preview}")
                    if created_at:
                        lines.append(f"   Date: {created_at[:10]}")
                    lines.append("")

                lines.append("Call again with confirm=true to delete these memories.")
                return "\n".join(lines)

            # Deletion mode
            deleted = 0
            for result in results:
                entry_id = result.get("id")
                if entry_id is None:
                    continue
                try:
                    if await self._vector_store.delete(entry_id):
                        deleted += 1
                except Exception as e:
                    logger.warning(f"Failed to delete memory entry: {e}")

            self._deletion_count += deleted
            remaining = MAX_DELETIONS_PER_SESSION - self._deletion_count

            logger.debug(
                f"Deleted {deleted} memories for query '{query}', "
                f"{remaining} deletions remaining this session"
            )
            return (
                f"Deleted {deleted} memories matching: {query}. "
                f"Remaining deletions this session: {remaining}."
            )

        except Exception as e:
            return f"Error during memory forget: {e}"
