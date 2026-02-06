"""Session compaction for managing conversation history size."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from nanobot.agent.memory.extractor import MemoryExtractor


@dataclass
class CompactionConfig:
    """Configuration for session compaction.

    Note: Compaction only reduces message count when len(messages) > threshold.
    At exactly threshold, all messages are preserved in the 3-layer structure.
    """

    threshold: int = 50
    max_messages: int = 20
    recent_turns_keep: int = 8
    summary_max_turns: int = 15
    max_facts: int = 10


# Inline keywords for fallback heuristic extraction
_FACT_KEYWORDS = [
    "my name is",
    "i am",
    "i'm",
    "i work",
    "i live",
    "i prefer",
    "remember that",
    "note that",
    "important:",
    "email:",
    "phone:",
    "address:",
    "birthday:",
    "project uses",
    "using",
    "configured to",
]


class SessionCompactor:
    """Compacts session history using layered summarization."""

    MIN_QUESTION_LENGTH = 20
    MIN_CONTENT_LENGTH = 50
    MIN_SENTENCE_LENGTH = 30
    MAX_EXTRACT_LENGTH = 150

    def __init__(
        self,
        config: CompactionConfig | None = None,
        extractor: "MemoryExtractor | None" = None,
    ):
        self.config = config or CompactionConfig()
        self._extractor = extractor

    def compact(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Compact message history to reduce size."""
        if not isinstance(messages, list):
            raise TypeError(f"messages must be a list, got {type(messages)}")
        if len(messages) < self.config.threshold:
            logger.debug(
                "Skipping compaction: %d < %d",
                len(messages),
                self.config.threshold,
            )
            return messages

        recent_count = self.config.recent_turns_keep * 2
        recent_start = max(0, len(messages) - recent_count)
        recent = messages[recent_start:]

        middle_count = self.config.summary_max_turns * 2
        middle_end = recent_start
        middle_start = max(0, middle_end - middle_count)
        middle = messages[middle_start:middle_end]

        old = messages[:middle_start]

        compacted: list[dict[str, Any]] = []
        recall_parts: list[str] = []

        if old:
            facts = self._extract_facts(old)
            if facts:
                recall_parts.append(f"Key facts:\n{facts}")

        if middle:
            summary = self._summarize(middle)
            if summary:
                recall_parts.append(f"Recent discussion summary:\n{summary}")

        if recall_parts:
            recall_content = "[Recalling from earlier in our conversation]\n\n" + "\n\n".join(
                recall_parts
            )
            compacted.append({"role": "assistant", "content": recall_content})

        compacted.extend(recent)

        logger.info(
            "Compacted %d -> %d (old: %d, middle: %d, recent: %d)",
            len(messages),
            len(compacted),
            len(old),
            len(middle),
            len(recent),
        )

        return compacted

    def _extract_facts(self, messages: list[dict[str, Any]]) -> str:
        """Extract key facts from old messages."""
        if self._extractor:
            facts = self._extractor.extract_heuristic(messages, max_facts=self.config.max_facts)
        else:
            facts = self._inline_extract(messages, max_facts=self.config.max_facts)
        return "\n".join(f"- {fact}" for fact in facts)

    def _inline_extract(
        self,
        messages: list[dict[str, Any]],
        max_facts: int = 10,
    ) -> list[str]:
        """Inline heuristic fact extraction (fallback when no extractor)."""
        facts: list[str] = []
        seen: set[str] = set()

        for msg in messages:
            content = msg.get("content", "")
            if not content or msg.get("role") == "system":
                continue

            for line in content.split("\n"):
                line = line.strip()
                if len(line) < 10:
                    continue

                if any(kw in line.lower() for kw in _FACT_KEYWORDS):
                    fact = line[:200]
                    if fact not in seen:
                        facts.append(fact)
                        seen.add(fact)
                        if len(facts) >= max_facts:
                            return facts

        return facts

    def _summarize(self, messages: list[dict[str, Any]]) -> str:
        """Summarize middle-section messages using heuristics."""
        user_questions: list[str] = []
        seen_questions: set[str] = set()
        assistant_conclusions: list[str] = []
        seen_conclusions: set[str] = set()

        for msg in messages:
            content = msg.get("content", "")
            if not content:
                continue

            role = msg.get("role", "")

            if role == "user":
                for line in content.split("\n"):
                    line = line.strip()
                    if line.endswith("?") and len(line) > self.MIN_QUESTION_LENGTH:
                        extracted = line[: self.MAX_EXTRACT_LENGTH]
                        if extracted not in seen_questions:
                            user_questions.append(extracted)
                            seen_questions.add(extracted)

            if role == "assistant" and len(content) > self.MIN_CONTENT_LENGTH:
                for sentence in content.split(".")[:3]:
                    sentence = sentence.strip()
                    if len(sentence) > self.MIN_SENTENCE_LENGTH:
                        extracted = sentence[: self.MAX_EXTRACT_LENGTH]
                        if extracted not in seen_conclusions:
                            assistant_conclusions.append(extracted)
                            seen_conclusions.add(extracted)
                        break

        parts: list[str] = []
        if user_questions:
            parts.append("User asked about:")
            for q in user_questions[:3]:
                parts.append(f"  - {q}")

        if assistant_conclusions:
            parts.append("Assistant responses:")
            for c in assistant_conclusions[:3]:
                parts.append(f"  - {c}")

        return "\n".join(parts) if parts else "General discussion continued"
