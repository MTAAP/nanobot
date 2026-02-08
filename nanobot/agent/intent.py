"""Intent classification for incoming user queries.

Two-tier classification: fast regex patterns first, LLM fallback for ambiguous cases.
"""

import json
import re
from enum import Enum
from typing import Any

from loguru import logger

from nanobot.providers.base import LLMProvider


class QueryIntent(Enum):
    """Classified intent of a user query."""

    FACTUAL = "factual"
    VERIFY_STATE = "verify_state"
    MEMORY = "memory"
    ACTION = "action"
    CONVERSATIONAL = "conversational"


# Tier-1 regex patterns mapped to intents.
# Each entry is (compiled_regex, intent, confidence).
# Patterns are tested in order; first match above the confidence threshold wins.
_TIER1_PATTERNS: list[tuple[re.Pattern[str], QueryIntent, float]] = [
    # FACTUAL -- verifiable facts, calculations, live data
    (
        re.compile(
            r"\b(what\s+time|what\'?s?\s+the\s+time|current\s+time"
            r"|what\s+day\s+is\s+it|what\'?s?\s+the\s+date|today\'?s?\s+date"
            r"|what\s+date|what\s+year)\b",
            re.IGNORECASE,
        ),
        QueryIntent.FACTUAL,
        0.95,
    ),
    (
        re.compile(
            r"\b(weather|temperature|forecast|humidity|wind\s+speed"
            r"|what\'?s?\s+the\s+temperature)\b",
            re.IGNORECASE,
        ),
        QueryIntent.FACTUAL,
        0.90,
    ),
    (
        re.compile(
            r"\b(calculate|compute|how\s+many|how\s+much\s+is"
            r"|what\s+is\s+\d|convert\s+\d|sum\s+of|average\s+of)\b",
            re.IGNORECASE,
        ),
        QueryIntent.FACTUAL,
        0.90,
    ),
    (
        re.compile(
            r"\b(system\s+status|uptime|cpu\s+usage|disk\s+space"
            r"|memory\s+usage|server\s+status)\b",
            re.IGNORECASE,
        ),
        QueryIntent.FACTUAL,
        0.85,
    ),
    # VERIFY_STATE -- questions about mutable state the bot manages
    (
        re.compile(
            r"\b(reminder|reminders|scheduled|schedule"
            r"|cron\s+job|cron\s+jobs|timer|timers|alarm|alarms)\b",
            re.IGNORECASE,
        ),
        QueryIntent.VERIFY_STATE,
        0.90,
    ),
    (
        re.compile(
            r"\b(is\s+.+?\s+still|do\s+i\s+have|are\s+there\s+any"
            r"|what\s+about\s+the\s+.+?\s+reminder"
            r"|what\s+about\s+my\s+.+?\s+schedule)\b",
            re.IGNORECASE,
        ),
        QueryIntent.VERIFY_STATE,
        0.80,
    ),
    # ACTION -- imperative commands to perform file/system operations
    (
        re.compile(
            r"(?:^|[.!?]\s+)(?:please\s+)?"
            r"(?:write|create|update|edit|add|set\s+up|initialize|populate)"
            r"\s+(?:the\s+|my\s+|a\s+|in\s+|to\s+)?\w",
            re.IGNORECASE,
        ),
        QueryIntent.ACTION,
        0.85,
    ),
    (
        re.compile(
            r"(?:^|[.!?]\s+)(?:please\s+)?"
            r"(?:install|configure|schedule|run|execute|deploy|remove|delete)"
            r"\s+\w",
            re.IGNORECASE,
        ),
        QueryIntent.ACTION,
        0.85,
    ),
]

_CONFIDENCE_THRESHOLD = 0.75

_LLM_CLASSIFICATION_PROMPT = """\
Classify the user message into exactly one intent category.

Categories:
- factual: Verifiable facts (time, date, weather, calculations, system status)
- verify_state: Questions about mutable state (reminders, schedules, cron jobs, timers)
- memory: Questions about past conversations or decisions
- action: Commands to do something
- conversational: General chat, greetings, opinions

Respond with ONLY a JSON object: {"intent": "<category>"}

User message: """


class IntentClassifier:
    """Two-tier intent classifier: regex patterns then optional LLM fallback."""

    def __init__(
        self,
        enabled: bool = True,
        llm_fallback: bool = True,
        provider: LLMProvider | None = None,
        model: str | None = None,
    ) -> None:
        self.enabled = enabled
        self.llm_fallback = llm_fallback
        self.provider = provider
        self.model = model

    async def classify(self, message: str) -> QueryIntent:
        """Classify a user message into a QueryIntent.

        Tier 1: regex pattern matching (fast, high-precision).
        Tier 2: LLM fallback for ambiguous messages.
        Falls back to CONVERSATIONAL if both tiers fail or classifier is disabled.
        """
        if not self.enabled:
            return QueryIntent.CONVERSATIONAL

        text = message.strip()
        if not text:
            return QueryIntent.CONVERSATIONAL

        # -- Tier 1: rule-based patterns --
        intent, confidence = self._match_patterns(text)
        if intent is not None and confidence >= _CONFIDENCE_THRESHOLD:
            logger.debug(
                f"Intent classified via rules: {intent.value} (confidence={confidence:.2f})"
            )
            return intent

        # -- Tier 2: LLM fallback --
        if self.llm_fallback and self.provider is not None:
            llm_intent = await self._classify_with_llm(text)
            if llm_intent is not None:
                logger.debug(f"Intent classified via LLM: {llm_intent.value}")
                return llm_intent

        # Default
        logger.debug("Intent defaulting to CONVERSATIONAL")
        return QueryIntent.CONVERSATIONAL

    def _match_patterns(self, text: str) -> tuple[QueryIntent | None, float]:
        """Run tier-1 regex patterns and return best match."""
        best_intent: QueryIntent | None = None
        best_confidence = 0.0

        for pattern, intent, confidence in _TIER1_PATTERNS:
            if pattern.search(text) and confidence > best_confidence:
                best_intent = intent
                best_confidence = confidence

        return best_intent, best_confidence

    async def _classify_with_llm(self, text: str) -> QueryIntent | None:
        """Use a cheap LLM call to classify intent."""
        assert self.provider is not None
        messages: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": _LLM_CLASSIFICATION_PROMPT + text,
            },
        ]
        try:
            response = await self.provider.chat(
                messages=messages,
                model=self.model,
                max_tokens=64,
                temperature=0.0,
            )
            if not response.content:
                logger.debug("LLM intent classification returned empty")
                return None
            return self._parse_llm_response(response.content)
        except Exception as e:
            logger.warning(f"LLM intent classification failed: {e}")
            return None

    def _parse_llm_response(self, content: str) -> QueryIntent | None:
        """Parse the JSON response from the LLM into a QueryIntent."""
        try:
            # Strip markdown fences if present
            cleaned = content.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[-1]
                cleaned = cleaned.rsplit("```", 1)[0]
                cleaned = cleaned.strip()

            data = json.loads(cleaned)
            raw_intent = data.get("intent", "").lower().strip()

            # Map string values to enum
            mapping = {e.value: e for e in QueryIntent}
            intent = mapping.get(raw_intent)
            if intent is None:
                logger.debug(f"LLM returned unknown intent: {raw_intent!r}")
            return intent
        except (json.JSONDecodeError, AttributeError) as e:
            logger.debug(f"Failed to parse LLM intent response: {e}")
            return None
