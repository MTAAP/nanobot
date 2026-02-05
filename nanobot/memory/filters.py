"""Filters for memory content before indexing."""

import re

from loguru import logger

# Imperative verb prefixes that signal instructions
_IMPERATIVE_PREFIXES = (
    "always ",
    "never ",
    "must ",
    "should ",
    "remember to ",
    "make sure ",
    "ensure ",
    "do not ",
    "don't ",
)

# System-prompt-like phrases
_SYSTEM_PHRASES = (
    "you are ",
    "your role is",
    "ignore previous",
    "disregard",
    "override",
)

# Tool / function reference phrases
_TOOL_PHRASES = (
    "call memory_search",
    "use tool",
    "execute",
    "run command",
)

# Manipulation phrases
_MANIPULATION_PHRASES = (
    "from now on",
    "going forward always",
    "in all future",
)

# PII regex patterns
_PASSWORD_RE = re.compile(r"(?:password|passwd|pwd)\s*[:=]\s*\S+", re.IGNORECASE)
_API_KEY_RE = re.compile(r"api[_\-]?key\s*[:=]\s*\S+", re.IGNORECASE)
_TOKEN_RE = re.compile(r"token\s*[:=]\s*\S+", re.IGNORECASE)
_SECRET_RE = re.compile(r"secret\s*[:=]\s*\S+", re.IGNORECASE)
# Common credential prefixes: OpenAI sk-, GitHub ghp_, Slack xoxb-
_CREDENTIAL_RE = re.compile(
    r"\b(?:sk-[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{36,}"
    r"|xoxb-[A-Za-z0-9\-]{20,}|xoxp-[A-Za-z0-9\-]{20,})\b"
)
# Credit card: 4 groups of 4 digits
_CREDIT_CARD_RE = re.compile(r"\b\d{4}[\s\-]\d{4}[\s\-]\d{4}[\s\-]\d{4}\b")
# SSN: XXX-XX-XXXX
_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")


def is_instruction(text: str) -> bool:
    """Detect if text looks like a behavioral instruction.

    Returns True if the text appears to be a directive,
    command, or behavioral instruction rather than a factual
    statement.
    """
    lower = text.lower().strip()

    # Check imperative prefixes
    for prefix in _IMPERATIVE_PREFIXES:
        if lower.startswith(prefix):
            return True

    # Check system-prompt-like language
    for phrase in _SYSTEM_PHRASES:
        if phrase in lower:
            return True

    # Check tool/function references
    for phrase in _TOOL_PHRASES:
        if phrase in lower:
            return True

    # Check manipulation phrases
    for phrase in _MANIPULATION_PHRASES:
        if phrase in lower:
            return True

    return False


def detect_pii(text: str) -> list[str]:
    """Scan text for PII patterns.

    Returns a list of PII types found (empty list means clean).
    """
    found: list[str] = []

    if _PASSWORD_RE.search(text):
        found.append("password")
    if _API_KEY_RE.search(text):
        found.append("api_key")
    if _TOKEN_RE.search(text):
        found.append("token")
    if _SECRET_RE.search(text):
        found.append("secret")
    if _CREDENTIAL_RE.search(text):
        found.append("credential")
    if _CREDIT_CARD_RE.search(text):
        found.append("credit_card")
    if _SSN_RE.search(text):
        found.append("ssn")

    return found


def sanitize_for_memory(text: str) -> str | None:
    """Main entry point for memory content filtering.

    Returns None if the text should be skipped (e.g. it is an
    instruction). Returns the text as-is if clean. Logs warnings
    for PII detections but does not redact.
    """
    if is_instruction(text):
        logger.debug(f"Skipping instruction-like content: {text[:80]}...")
        return None

    pii_types = detect_pii(text)
    if pii_types:
        logger.warning(f"PII detected in memory content ({', '.join(pii_types)}): {text[:60]}...")

    return text
