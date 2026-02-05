"""Message compaction for context window management."""

from typing import Any

from loguru import logger

from nanobot.providers.base import LLMProvider
from nanobot.utils.tokens import count_messages_tokens

# Compaction prompt template
COMPACTION_PROMPT = """Summarize this conversation history concisely while preserving:
1. Key decisions made and their reasoning
2. Important facts, names, dates mentioned
3. User preferences and habits
4. Pending tasks or commitments
5. Technical details that may be needed later

Previous summary (if any):
{previous_summary}

Messages to summarize:
{messages}

Write a concise summary (max 500 words) that captures the essential context:"""


class MessageCompactor:
    """
    Compacts conversation history by summarizing older messages.

    Preserves recent messages intact while summarizing older ones,
    enabling rolling summaries for long-running conversations.
    """

    def __init__(
        self,
        provider: LLMProvider,
        model: str | None = None,
        keep_recent: int = 10,
    ):
        """
        Initialize the compactor.

        Args:
            provider: LLM provider for generating summaries.
            model: Model to use for compaction (None = use provider default).
            keep_recent: Number of recent messages to keep intact.
        """
        self.provider = provider
        self.model = model
        self.keep_recent = keep_recent

    async def compact_messages(
        self,
        messages: list[dict[str, Any]],
        target_tokens: int,
        previous_summary: str | None = None,
    ) -> tuple[list[dict[str, Any]], str | None]:
        """
        Compact messages to fit within target token budget.

        Strategy:
        1. Keep system message intact
        2. Keep last N messages intact
        3. Summarize older messages into a single summary message

        Args:
            messages: List of messages to compact.
            target_tokens: Target token budget.
            previous_summary: Optional summary from previous compaction.

        Returns:
            Tuple of (compacted messages, new rolling summary).
        """
        current_tokens = count_messages_tokens(messages)

        if current_tokens <= target_tokens:
            # No compaction needed
            return messages, previous_summary

        logger.info(
            f"Compacting messages: {current_tokens} tokens -> target {target_tokens} tokens"
        )

        # Separate system message (if any) from conversation
        system_msg = None
        conversation = messages
        if messages and messages[0].get("role") == "system":
            system_msg = messages[0]
            conversation = messages[1:]

        # Determine how many messages to keep
        keep_count = min(self.keep_recent, len(conversation))

        if keep_count >= len(conversation):
            # Not enough messages to compact, just truncate
            logger.warning("Not enough messages to compact, truncating instead")
            return self._truncate_messages(messages, target_tokens), previous_summary

        # Split into messages to summarize and messages to keep
        to_summarize = conversation[:-keep_count] if keep_count > 0 else conversation
        to_keep = conversation[-keep_count:] if keep_count > 0 else []

        # Generate summary of older messages
        try:
            summary = await self._generate_summary(to_summarize, previous_summary)
        except Exception as e:
            logger.error(f"Compaction failed: {e}, falling back to truncation")
            return self._truncate_messages(messages, target_tokens), previous_summary

        # Build compacted messages
        compacted = []

        if system_msg:
            compacted.append(system_msg)

        # Add summary as a system-injected context message
        if summary:
            compacted.append(
                {
                    "role": "user",
                    "content": f"[Previous conversation summary]\n{summary}",
                }
            )
            compacted.append(
                {
                    "role": "assistant",
                    "content": "I understand the context from our previous conversation. How can I help?",
                }
            )

        # Add recent messages
        compacted.extend(to_keep)

        # Verify we're under budget
        compacted_tokens = count_messages_tokens(compacted)
        if compacted_tokens > target_tokens:
            logger.warning(
                f"Compaction still over budget: {compacted_tokens} > {target_tokens}, truncating"
            )
            return self._truncate_messages(compacted, target_tokens), summary

        logger.info(
            f"Compaction complete: {current_tokens} -> {compacted_tokens} tokens "
            f"({len(messages)} -> {len(compacted)} messages)"
        )

        return compacted, summary

    async def _generate_summary(
        self,
        messages: list[dict[str, Any]],
        previous_summary: str | None,
    ) -> str:
        """Generate a summary of the given messages."""
        # Format messages for the prompt
        formatted = []
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if content:
                formatted.append(f"{role.upper()}: {content}")

        messages_text = "\n\n".join(formatted)
        previous_text = previous_summary or "(No previous summary)"

        prompt = COMPACTION_PROMPT.format(
            previous_summary=previous_text,
            messages=messages_text,
        )

        response = await self.provider.chat(
            messages=[{"role": "user", "content": prompt}],
            model=self.model,
            max_tokens=1024,
            temperature=0.3,  # Lower temperature for more consistent summaries
        )

        return response.content or ""

    def _truncate_messages(
        self,
        messages: list[dict[str, Any]],
        target_tokens: int,
    ) -> list[dict[str, Any]]:
        """Truncate messages to fit within token budget (fallback)."""
        from nanobot.utils.tokens import count_message_tokens

        # Keep system message if present
        result = []
        total_tokens = 0

        if messages and messages[0].get("role") == "system":
            system_msg = messages[0]
            system_tokens = count_message_tokens(system_msg)
            result.append(system_msg)
            total_tokens = system_tokens
            messages = messages[1:]

        # Work backwards, keeping most recent messages
        kept = []
        for msg in reversed(messages):
            msg_tokens = count_message_tokens(msg)
            if total_tokens + msg_tokens > target_tokens:
                break
            kept.append(msg)
            total_tokens += msg_tokens

        # Reverse to restore order
        kept.reverse()
        result.extend(kept)

        return result


async def should_compact(
    messages: list[dict[str, Any]],
    max_tokens: int,
    threshold: float = 0.8,
) -> bool:
    """
    Check if compaction should be triggered.

    Args:
        messages: Current message list.
        max_tokens: Maximum context tokens.
        threshold: Percentage threshold to trigger compaction.

    Returns:
        True if compaction should be triggered.
    """
    current_tokens = count_messages_tokens(messages)
    trigger_at = int(max_tokens * threshold)
    return current_tokens > trigger_at
