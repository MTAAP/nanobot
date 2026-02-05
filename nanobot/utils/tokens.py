"""Token counting utilities using tiktoken."""

from functools import lru_cache
from typing import Any

import tiktoken


@lru_cache(maxsize=1)
def get_tokenizer() -> tiktoken.Encoding:
    """Get the tiktoken encoder (cached for performance)."""
    return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """
    Count tokens in a text string.

    Args:
        text: The text to count tokens for.

    Returns:
        Number of tokens.
    """
    if not text:
        return 0
    enc = get_tokenizer()
    return len(enc.encode(text))


def count_message_tokens(message: dict[str, Any]) -> int:
    """
    Count tokens in a single message.

    Follows OpenAI's message token counting format:
    - 4 tokens overhead per message
    - Plus tokens for content

    Args:
        message: A message dict with 'role' and 'content'.

    Returns:
        Number of tokens including overhead.
    """
    # Base overhead per message (role, separators)
    tokens = 4

    content = message.get("content", "")
    if isinstance(content, str):
        tokens += count_tokens(content)
    elif isinstance(content, list):
        # Handle multimodal content (text + images)
        for part in content:
            if isinstance(part, dict):
                if part.get("type") == "text":
                    tokens += count_tokens(part.get("text", ""))
                elif part.get("type") == "image_url":
                    # Estimate: images use roughly 85-170 tokens depending on detail
                    tokens += 128
            elif isinstance(part, str):
                tokens += count_tokens(part)

    # Tool calls in assistant messages
    if message.get("tool_calls"):
        for tc in message["tool_calls"]:
            tokens += 4  # Tool call overhead
            if isinstance(tc, dict):
                func = tc.get("function", {})
                tokens += count_tokens(func.get("name", ""))
                tokens += count_tokens(func.get("arguments", ""))

    return tokens


def count_messages_tokens(messages: list[dict[str, Any]]) -> int:
    """
    Count total tokens in a list of messages.

    Args:
        messages: List of message dicts.

    Returns:
        Total token count.
    """
    # Base overhead for the messages array
    total = 3

    for msg in messages:
        total += count_message_tokens(msg)

    return total


def truncate_to_token_limit(text: str, max_tokens: int) -> str:
    """
    Truncate text to fit within a token limit.

    Args:
        text: The text to truncate.
        max_tokens: Maximum tokens allowed.

    Returns:
        Truncated text that fits within the limit.
    """
    if not text:
        return text

    enc = get_tokenizer()
    tokens = enc.encode(text)

    if len(tokens) <= max_tokens:
        return text

    # Truncate and decode
    truncated = enc.decode(tokens[:max_tokens])

    # Add truncation indicator
    return truncated + "... [truncated]"
