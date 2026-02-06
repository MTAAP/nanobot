"""Memory extractor for automatically extracting facts from conversations."""

import json
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Literal

from loguru import logger
from pydantic import BaseModel, Field, ValidationError

# Approximate token budget for extraction (~500 tokens at ~4 chars/token)
MAX_CONVERSATION_CHARS = 2000
MAX_FACT_CONTENT_LENGTH = 500
MIN_FACT_CONTENT_LENGTH = 5
# Patterns that suggest prompt injection; reject facts matching these
REJECT_PATTERNS = [
    r"ignore\s+(previous|all)\s+instructions",
    r"disregard\s+(previous|all)",
    r"you\s+are\s+now\s+",
    r"new\s+instructions\s*:",
]

# Shared keywords for heuristic fact extraction
FACT_KEYWORDS = [
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


def sanitize_for_memory(text: str, max_length: int = MAX_FACT_CONTENT_LENGTH) -> str:
    """Sanitize text for memory storage: strip controls, normalize, escape HTML, cap length."""
    if not text:
        return ""
    text = "".join(c for c in text if unicodedata.category(c) != "Cc")
    text = re.sub(r"\s+", " ", text).strip()
    text = text.replace("<", "&lt;").replace(">", "&gt;")
    return text[:max_length] if len(text) > max_length else text


def is_valid_memory(text: str) -> bool:
    """Reject empty, too-short, too-long, or injection-looking text."""
    if not text or len(text.strip()) < MIN_FACT_CONTENT_LENGTH:
        return False
    if len(text) > MAX_FACT_CONTENT_LENGTH:
        return False
    lowered = text.lower()
    for pat in REJECT_PATTERNS:
        if re.search(pat, lowered):
            return False
    return True


@dataclass
class ExtractionMetrics:
    """Metrics for a single extraction run."""

    facts_extracted: int = 0
    lessons_extracted: int = 0
    tool_lessons_extracted: int = 0
    facts_by_type: dict[str, int] = field(default_factory=dict)
    llm_calls: int = 0
    llm_failures: int = 0
    heuristic_fallbacks: int = 0


FactType = Literal["user", "project", "technical", "preference", "tool", "generic"]
ImportanceLevel = Literal["high", "medium", "low"]

EXTRACTION_PROMPT = """Analyze the conversation and extract key facts by type.

<conversation>
{conversation}
</conversation>

Extract facts in these categories:
- user: Personal info (name, job, location, preferences)
- project: Decisions, requirements, relationships, project context
- technical: Technical preferences (tools, languages, configurations)
- preference: Communication style, response preferences
- tool: Tool usage patterns, configurations, preferences

Rules:
- Facts only, no opinions or temporary context
- Self-contained statements
- Skip greetings and small talk
- Include a "type" field for each fact

Return JSON array: [{{"type": "user", "fact": "...", "importance": "high"}}]

Facts:"""


LESSON_EXTRACTION_PROMPT = """Analyze this correction and extract a reusable lesson.

<previous_answer>
{previous_answer}
</previous_answer>

<user_correction>
{user_correction}
</user_correction>

Extract a generalized lesson that:
- Is timeless and third-person ("When doing X, prefer Y instead of Z")
- Avoids specific context that won't apply later
- Focuses on the principle, not the instance
- Is actionable for future similar situations

Return JSON: {{"lesson": "...", "category": "reasoning|tool_usage|communication|other", "importance": "high|medium|low"}}"""


PRE_COMPACTION_PROMPT = """Before we compact this conversation, list important facts, decisions, or lessons that should persist beyond this session.

<conversation>
{conversation}
</conversation>

Reply with only a JSON array of objects with "fact" and "importance" (high|medium|low). No other text.
Example: [{{"fact": "User prefers Python for scripts", "importance": "high"}}]

Facts:"""


class LessonExtractionSchema(BaseModel):
    """Validated lesson from LLM."""

    lesson: str = Field(..., min_length=1, max_length=500)
    category: Literal["reasoning", "tool_usage", "communication", "other"] = "other"
    importance: ImportanceLevel = "medium"


TRIVIAL_PATTERNS = [
    r"^(ok|okay|yes|no|thanks|sure|got it|cool|nice|great|hmm|ah|oh|lol|yep|yeah)[\.\!\?]?\s*$",
    r"^[\s\W]*$",
]


@dataclass
class ExtractedFact:
    """A fact extracted from conversation with metadata."""

    content: str
    importance: float  # 0.0 to 1.0
    source: str  # "llm" or "heuristic" or "lesson"
    fact_type: str = "generic"
    metadata: dict[str, Any] = field(default_factory=dict)


class ExtractedFactSchema(BaseModel):
    """Pydantic schema for validating LLM-extracted facts."""

    fact: str = Field(..., min_length=1, max_length=500)
    importance: ImportanceLevel = "medium"
    type: FactType = "generic"


class UserFactSchema(BaseModel):
    """Schema for user/personal facts."""

    fact: str = Field(..., min_length=1, max_length=500)
    importance: ImportanceLevel = "medium"
    type: Literal["user"] = "user"


class ProjectFactSchema(BaseModel):
    """Schema for project-specific facts."""

    fact: str = Field(..., min_length=1, max_length=500)
    importance: ImportanceLevel = "medium"
    type: Literal["project"] = "project"
    project_name: str | None = None


class TechnicalFactSchema(BaseModel):
    """Schema for technical preference facts."""

    fact: str = Field(..., min_length=1, max_length=500)
    importance: ImportanceLevel = "medium"
    type: Literal["technical"] = "technical"


class PreferenceFactSchema(BaseModel):
    """Schema for communication/preference facts."""

    fact: str = Field(..., min_length=1, max_length=500)
    importance: ImportanceLevel = "medium"
    type: Literal["preference"] = "preference"


class ToolFactSchema(BaseModel):
    """Schema for tool usage facts."""

    fact: str = Field(..., min_length=1, max_length=500)
    importance: ImportanceLevel = "medium"
    type: Literal["tool"] = "tool"
    tool_name: str | None = None


TYPED_SCHEMAS: dict[str, type[BaseModel]] = {
    "user": UserFactSchema,
    "project": ProjectFactSchema,
    "technical": TechnicalFactSchema,
    "preference": PreferenceFactSchema,
    "tool": ToolFactSchema,
}


class MemoryExtractor:
    """Extracts memorable facts and lessons from conversations."""

    def __init__(self, model: str = "gpt-4o-mini", max_facts: int = 5):
        self.model = model
        self.max_facts = max_facts
        self._trivial_patterns = [re.compile(p, re.IGNORECASE) for p in TRIVIAL_PATTERNS]
        self._last_metrics = ExtractionMetrics()

    def get_metrics(self) -> ExtractionMetrics:
        """Return metrics from the last extraction run."""
        return self._last_metrics

    async def extract(
        self,
        messages: list[dict[str, Any]],
        max_facts: int = 5,
    ) -> list[ExtractedFact]:
        """Extract facts from a conversation using LLM with heuristic fallback."""
        self._last_metrics = ExtractionMetrics()
        if not messages:
            return []

        user_messages = [m for m in messages if m.get("role") == "user"]
        if len(user_messages) < 3:
            return []

        if user_messages:
            last_msg = user_messages[-1].get("content", "").strip()
            if not last_msg or any(p.match(last_msg) for p in self._trivial_patterns):
                logger.debug("Skipping trivial message: %s", last_msg[:50])
                return []

        conversation = self._format_conversation(messages)
        if len(conversation) < 50:
            return []

        try:
            facts = await self._llm_extract(conversation)
            facts = facts[:max_facts]
            self._last_metrics.llm_calls += 1
            self._last_metrics.facts_extracted = len(facts)
            for f in facts:
                t = getattr(f, "fact_type", "generic")
                self._last_metrics.facts_by_type[t] = self._last_metrics.facts_by_type.get(t, 0) + 1
            logger.info(
                "extraction_complete facts=%d by_type=%s",
                len(facts),
                dict(self._last_metrics.facts_by_type),
            )
            return facts
        except Exception as e:
            self._last_metrics.llm_failures += 1
            self._last_metrics.heuristic_fallbacks += 1
            logger.warning("LLM extraction failed: %s", e)
            facts = self._heuristic_extract(messages)[:max_facts]
            self._last_metrics.facts_extracted = len(facts)
            for f in facts:
                t = getattr(f, "fact_type", "generic")
                self._last_metrics.facts_by_type[t] = self._last_metrics.facts_by_type.get(t, 0) + 1
            logger.info(
                "extraction_complete method=heuristic facts=%d",
                len(facts),
            )
            return facts

    async def extract_lessons(
        self,
        messages: list[dict[str, Any]],
        max_facts: int = 5,
    ) -> list[ExtractedFact]:
        """Extract lesson-style facts from user corrections (LLM generalization)."""
        if not messages:
            return []

        correction_patterns = [
            "that was wrong",
            "you were wrong",
            "no, that's wrong",
            "no that's wrong",
            "this is incorrect",
            "actually,",
            "actually ",
            "so next time",
            "in the future",
            "don't do that",
            "that's not right",
            "i meant",
            "you misunderstood",
        ]

        lessons: list[ExtractedFact] = []
        seen: set[str] = set()
        window = messages[-10:]

        for i, msg in enumerate(window):
            if msg.get("role") != "user":
                continue
            content = msg.get("content", "")
            lowered = content.lower().strip()
            if not content or len(lowered) < 10:
                continue
            if not any(pat in lowered for pat in correction_patterns):
                continue

            previous_answer = ""
            for j in range(i - 1, -1, -1):
                if window[j].get("role") == "assistant":
                    prev = window[j].get("content", "")
                    if isinstance(prev, str) and prev.strip():
                        previous_answer = prev.strip()[:1500]
                    break

            lesson_content: str | None = None
            importance = 0.9
            source = "lesson"
            meta: dict[str, Any] = {}

            try:
                generalized = await self._llm_extract_lesson(
                    previous_answer=previous_answer or "(no prior answer)",
                    user_correction=content.strip()[:1000],
                )
                if generalized:
                    lesson_content = self._normalize_lesson(generalized.lesson)
                    importance = {
                        "high": 0.9,
                        "medium": 0.7,
                        "low": 0.3,
                    }.get(generalized.importance, 0.7)
                    source = "llm_lesson"
                    meta = {"category": generalized.category}
            except Exception as e:
                logger.debug("LLM lesson extraction failed: %s", e)

            if not lesson_content:
                lesson_content = self._normalize_lesson(content.strip()[:500])

            # Sanitize and validate lesson content
            if lesson_content:
                lesson_content = sanitize_for_memory(lesson_content)
            if not lesson_content or not is_valid_memory(lesson_content):
                continue

            if lesson_content not in seen:
                seen.add(lesson_content)
                lessons.append(
                    ExtractedFact(
                        content=lesson_content[:500],
                        importance=importance,
                        source=source,
                        fact_type="lesson",
                        metadata=meta,
                    )
                )
                if len(lessons) >= max_facts:
                    break

        self._last_metrics.lessons_extracted = len(lessons)
        return lessons

    async def extract_for_pre_compaction(
        self,
        messages: list[dict[str, Any]],
        max_facts: int = 10,
    ) -> list[ExtractedFact]:
        """Extract facts before compaction via single LLM turn."""
        if not messages:
            return []
        conversation = self._format_conversation(messages[-30:])
        if len(conversation) < 50:
            return []
        try:
            import litellm

            response = await litellm.acompletion(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": PRE_COMPACTION_PROMPT.format(conversation=conversation),
                    }
                ],
                max_tokens=500,
                temperature=0.1,
            )
            raw = response.choices[0].message.content.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()
            data = json.loads(raw)
            if not isinstance(data, list):
                return []
            importance_map: dict[str, float] = {
                "high": 0.9,
                "medium": 0.7,
                "low": 0.3,
            }
            out: list[ExtractedFact] = []
            for item in data[:max_facts]:
                if isinstance(item, dict) and item.get("fact"):
                    fact = sanitize_for_memory(str(item["fact"]))
                    if not is_valid_memory(fact):
                        continue
                    imp = importance_map.get(
                        str(item.get("importance", "medium")).lower(),
                        0.7,
                    )
                    out.append(
                        ExtractedFact(
                            content=fact,
                            importance=imp,
                            source="pre_compaction",
                            fact_type="generic",
                            metadata={},
                        )
                    )
            return out
        except json.JSONDecodeError as e:
            logger.warning("Pre-compaction JSON failed: %s", e)
            return []
        except Exception as e:
            logger.warning("Pre-compaction extraction failed: %s", e)
            return []

    def extract_tool_lessons(
        self,
        messages: list[dict[str, Any]],
        max_lessons: int = 3,
    ) -> list[ExtractedFact]:
        """Extract tool-specific lessons from tool execution failures."""
        lessons: list[ExtractedFact] = []
        seen: set[str] = set()
        failure_indicators = (
            "error:",
            "failed",
            "exit code:",
            "exception",
            "traceback",
        )

        for msg in messages:
            if msg.get("role") != "tool":
                continue
            content = msg.get("content", "")
            if not content or not isinstance(content, str):
                continue
            lowered = content.lower()
            if not any(ind in lowered for ind in failure_indicators):
                continue

            tool_name = msg.get("name", "unknown")
            first_line = content.strip().split("\n")[0][:200]
            lesson_content = f"When using {tool_name}: avoid outcomes that lead to: {first_line}"

            # Sanitize and validate
            lesson_content = sanitize_for_memory(lesson_content)
            if not is_valid_memory(lesson_content):
                continue

            if lesson_content in seen:
                continue
            seen.add(lesson_content)

            lessons.append(
                ExtractedFact(
                    content=lesson_content[:500],
                    importance=0.7,
                    source="tool_failure",
                    fact_type="tool_lesson",
                    metadata={
                        "tool_name": tool_name,
                        "error_type": "execution",
                    },
                )
            )
            if len(lessons) >= max_lessons:
                break

        self._last_metrics.tool_lessons_extracted = len(lessons)
        return lessons

    def extract_heuristic(
        self,
        messages: list[dict[str, Any]],
        max_facts: int = 10,
    ) -> list[str]:
        """Extract facts using keyword heuristics. Returns plain strings for compaction."""
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

                if any(kw in line.lower() for kw in FACT_KEYWORDS):
                    fact = line[:200]
                    if fact not in seen:
                        facts.append(fact)
                        seen.add(fact)
                        if len(facts) >= max_facts:
                            return facts

        return facts

    def _sanitize_for_prompt(self, text: str) -> str:
        """Sanitize user content before embedding in prompts."""
        if not text:
            return ""
        text = "".join(c for c in text if unicodedata.category(c) != "Cc")
        text = re.sub(r"\s+", " ", text).strip()
        text = text.replace("```", "'''")
        text = text.replace("</", "&lt;/")
        text = text.replace("<", "&lt;").replace(">", "&gt;")
        if len(text) > MAX_CONVERSATION_CHARS:
            return text[:MAX_CONVERSATION_CHARS] + "..."
        return text

    def _format_conversation(self, messages: list[dict[str, Any]]) -> str:
        """Format messages for extraction prompt with sanitization."""
        parts: list[str] = []
        total = 0
        for msg in reversed(messages[-20:]):
            if total >= MAX_CONVERSATION_CHARS:
                break
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if content and role in ("user", "assistant"):
                content = self._sanitize_for_prompt(str(content))[:500]
            if content and role in ("user", "assistant"):
                line = f"{role.upper()}: {content}"
                parts.append(line)
                total += len(line)
        parts.reverse()
        return "\n".join(parts)

    async def _llm_extract_lesson(
        self, previous_answer: str, user_correction: str
    ) -> LessonExtractionSchema | None:
        """Use LLM to generalize a correction into a reusable lesson."""
        import litellm

        prompt = LESSON_EXTRACTION_PROMPT.format(
            previous_answer=self._sanitize_for_prompt(previous_answer),
            user_correction=self._sanitize_for_prompt(user_correction),
        )
        response = await litellm.acompletion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.1,
        )
        raw = response.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()
        data = json.loads(raw)
        return LessonExtractionSchema(**data)

    def _normalize_lesson(self, text: str) -> str:
        """Make lesson third-person and timeless."""
        if not text or len(text.strip()) < 10:
            return ""
        text = text.strip()
        for phrase in [
            "yesterday",
            "earlier",
            "that time",
            "last time",
            "just now",
        ]:
            text = re.sub(
                rf"\b{re.escape(phrase)}\b",
                "",
                text,
                flags=re.IGNORECASE,
            )
        text = re.sub(r"\s+", " ", text).strip()
        if text.lower().startswith(("you should", "you must", "when ")):
            return text[0].upper() + text[1:]
        return text[0].upper() + text[1:] if text else ""

    async def _llm_extract(self, conversation: str) -> list[ExtractedFact]:
        """Extract facts using LLM with Pydantic validation."""
        import litellm

        response = await litellm.acompletion(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": EXTRACTION_PROMPT.format(conversation=conversation),
                }
            ],
            max_tokens=300,
            temperature=0.1,
        )

        content = response.choices[0].message.content.strip()

        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()

        try:
            raw_data = json.loads(content)
            if not isinstance(raw_data, list):
                raise ValueError("Expected JSON array")

            extracted: list[ExtractedFact] = []
            importance_map: dict[str, float] = {
                "high": 0.9,
                "medium": 0.7,
                "low": 0.3,
            }

            for item in raw_data[: self.max_facts]:
                try:
                    if not isinstance(item, dict):
                        validated = ExtractedFactSchema(fact=str(item))
                        fact_type = "generic"
                        meta: dict[str, Any] = {}
                    else:
                        fact_type = item.get("type", "generic")
                        if fact_type not in TYPED_SCHEMAS:
                            fact_type = "generic"
                        if fact_type in TYPED_SCHEMAS:
                            validated = TYPED_SCHEMAS[fact_type](**item)
                        else:
                            validated = ExtractedFactSchema(**item)

                        meta = {}
                        if hasattr(validated, "project_name"):
                            pn = getattr(validated, "project_name")
                            if pn:
                                meta["project_name"] = pn
                        if hasattr(validated, "tool_name"):
                            tn = getattr(validated, "tool_name")
                            if tn:
                                meta["tool_name"] = tn

                    fact_text = sanitize_for_memory(validated.fact)
                    if not is_valid_memory(fact_text):
                        continue
                    extracted.append(
                        ExtractedFact(
                            content=fact_text,
                            importance=importance_map.get(
                                getattr(
                                    validated,
                                    "importance",
                                    "medium",
                                ),
                                0.5,
                            ),
                            source="llm",
                            fact_type=fact_type,
                            metadata=meta,
                        )
                    )
                except ValidationError as ve:
                    logger.debug("Fact validation skipped: %s", ve)
                    continue
            return extracted
        except json.JSONDecodeError as e:
            logger.warning("LLM extraction JSON decode failed: %s", e)
            return []
        except ValueError as e:
            logger.warning("LLM extraction validation failed: %s", e)
            return []

    def _heuristic_extract(self, messages: list[dict[str, Any]]) -> list[ExtractedFact]:
        """Extract facts using simple heuristics (fallback)."""
        facts: list[ExtractedFact] = []
        seen: set[str] = set()

        patterns: list[tuple[str, float, str]] = [
            ("my name is", 0.9, "user"),
            ("call me", 0.8, "user"),
            ("i am a", 0.7, "user"),
            ("i work", 0.8, "user"),
            ("i live", 0.8, "user"),
            ("i prefer", 0.7, "preference"),
            ("i like", 0.6, "preference"),
            ("i use", 0.6, "technical"),
            ("we decided", 0.8, "project"),
            ("project uses", 0.7, "project"),
            ("configured to", 0.6, "technical"),
        ]

        for msg in messages:
            if msg.get("role") != "user":
                continue

            content = msg.get("content", "").lower()

            for indicator, importance, fact_type in patterns:
                if indicator in content:
                    start = content.find(indicator)
                    end = next(
                        (
                            content.find(sep, start)
                            for sep in [".", "!", "?", "\n"]
                            if content.find(sep, start) != -1
                        ),
                        len(content),
                    )

                    fact_text = content[start:end].strip()
                    if fact_text and len(fact_text) > 5:
                        fact = self._to_third_person(fact_text)
                        if fact:
                            fact = fact[0].upper() + fact[1:]

                        sanitized = sanitize_for_memory(fact)
                        if sanitized and is_valid_memory(sanitized) and sanitized not in seen:
                            facts.append(
                                ExtractedFact(
                                    content=sanitized,
                                    importance=importance,
                                    source="heuristic",
                                    fact_type=fact_type,
                                    metadata={},
                                )
                            )
                            seen.add(sanitized)

        return facts

    def _to_third_person(self, text: str) -> str:
        """Convert first-person text to third-person."""
        replacements = [
            (r"\bmy\b", "User's"),
            (r"\bi am\b", "User is"),
            (r"\bi'm\b", "User is"),
            (r"\bi have\b", "User has"),
            (r"\bi've\b", "User has"),
            (r"\bi will\b", "User will"),
            (r"\bi'll\b", "User will"),
            (r"\bi\b", "User"),
        ]
        for pattern, replacement in replacements:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        return re.sub(r"\bUser User\b", "User", text)
