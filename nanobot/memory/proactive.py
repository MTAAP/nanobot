"""Proactive memory surfacing and interaction pattern learning."""

import hashlib
import json
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from loguru import logger

from nanobot.utils.atomic import atomic_write_json
from nanobot.utils.helpers import ensure_dir


class ProactiveMemory:
    """Surfaces relevant context proactively and learns usage patterns."""

    # Patterns that indicate commitments or deadlines
    _COMMITMENT_PATTERNS = [
        r"\[COMMITMENT\]",
        r"deadline",
        r"due\s+(date|by|on)",
        r"remind\s+me",
        r"don't\s+forget",
        r"need\s+to\s+finish",
        r"by\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)",
        r"by\s+\d{4}-\d{2}-\d{2}",
        r"before\s+\d{4}-\d{2}-\d{2}",
        r"scheduled\s+for",
    ]

    def __init__(
        self,
        vector_store: Any,
        entity_store: Any | None = None,
        data_dir: str | Path | None = None,
    ):
        """
        Initialize proactive memory.

        Args:
            vector_store: Vector store instance (duck typed).
            entity_store: Optional entity store for relational queries.
            data_dir: Directory for pattern data. Defaults to
                ~/.nanobot/memory.
        """
        self.vector_store = vector_store
        self.entity_store = entity_store

        if data_dir:
            self._data_dir = Path(data_dir).expanduser()
        else:
            self._data_dir = Path.home() / ".nanobot" / "memory"
        ensure_dir(self._data_dir)

        self._patterns_path = self._data_dir / "patterns.json"
        self._dismissed_path = self._data_dir / "dismissed.json"
        self._commitment_re = re.compile(
            "|".join(self._COMMITMENT_PATTERNS),
            re.IGNORECASE,
        )

    async def get_reminders(self) -> list[str]:
        """
        Check for memories that should be proactively surfaced.

        Looks for commitments and deadlines within the next 3 days.

        Returns:
            List of reminder strings.
        """
        queries = [
            "commitment deadline due soon",
            "remind me upcoming task scheduled",
            "promise to do need to finish",
        ]

        candidates: list[dict[str, Any]] = []
        seen_texts: set[str] = set()

        for query in queries:
            try:
                results = await self.vector_store.search(query, top_k=10, min_similarity=0.3)
                for r in results:
                    text = r.get("text", "")
                    if text not in seen_texts:
                        seen_texts.add(text)
                        candidates.append(r)
            except Exception as e:
                logger.error(f"Reminder search failed: {e}")

        reminders: list[str] = []
        now = datetime.now()
        horizon = now + timedelta(days=3)
        dismissed = self._load_dismissed()

        for entry in candidates:
            text = entry.get("text", "")
            if not self._commitment_re.search(text):
                continue

            # Skip dismissed reminders
            if self._text_hash(text) in dismissed:
                continue

            # Check if the memory references a date in the near future
            if self._is_within_horizon(text, now, horizon):
                reminders.append(text)

        logger.debug(f"Found {len(reminders)} proactive reminders")
        return reminders

    def dismiss_reminder(self, text: str) -> None:
        """Dismiss a reminder so it won't be surfaced again.

        Args:
            text: The reminder text to dismiss.
        """
        dismissed = self._load_dismissed()
        dismissed.add(self._text_hash(text))
        self._save_dismissed(dismissed)
        logger.debug(f"Dismissed reminder: {text[:80]}")

    def _is_within_horizon(
        self,
        text: str,
        now: datetime,
        horizon: datetime,
    ) -> bool:
        """
        Check if a memory text references a date within the horizon.

        If no parseable date is found, the memory is included anyway
        (commitments without dates are still relevant).

        Args:
            text: Memory text to check.
            now: Current datetime.
            horizon: Future cutoff datetime.

        Returns:
            True if the memory should be surfaced.
        """
        date_pattern = re.compile(r"\d{4}-\d{2}-\d{2}")
        matches = date_pattern.findall(text)

        if not matches:
            # No date found -- include commitment-tagged memories
            return True

        for date_str in matches:
            try:
                dt = datetime.strptime(date_str, "%Y-%m-%d")
                if now - timedelta(days=1) <= dt <= horizon:
                    return True
            except ValueError:
                continue

        return False

    def record_interaction_pattern(
        self,
        session_key: str,
        topic: str,
        timestamp: str,
    ) -> None:
        """
        Record an interaction for pattern analysis.

        Stores patterns in a JSON file at data_dir/patterns.json.

        Args:
            session_key: Session identifier (e.g. "telegram:12345").
            topic: Topic or category of the interaction.
            timestamp: ISO format timestamp.
        """
        patterns = self._load_patterns()

        if session_key not in patterns:
            patterns[session_key] = []

        patterns[session_key].append({"topic": topic, "timestamp": timestamp})

        # Keep at most 1000 entries per session to prevent bloat
        if len(patterns[session_key]) > 1000:
            patterns[session_key] = patterns[session_key][-1000:]

        self._save_patterns(patterns)
        logger.debug(f"Recorded interaction pattern: session={session_key}, topic={topic}")

    def get_patterns(self, session_key: str | None = None) -> dict[str, Any]:
        """
        Analyze interaction patterns.

        Args:
            session_key: Optional session key to filter by.

        Returns:
            Dict with common_topics, active_hours, and
            frequent_sessions.
        """
        patterns = self._load_patterns()

        if session_key:
            entries = patterns.get(session_key, [])
        else:
            entries = []
            for records in patterns.values():
                entries.extend(records)

        if not entries:
            return {
                "common_topics": [],
                "active_hours": [],
                "frequent_sessions": [],
            }

        # Count topics
        topic_counts: dict[str, int] = {}
        for entry in entries:
            topic = entry.get("topic", "")
            if topic:
                topic_counts[topic] = topic_counts.get(topic, 0) + 1

        common_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)[:10]

        # Analyze active hours
        hour_counts: dict[int, int] = {}
        for entry in entries:
            ts = entry.get("timestamp", "")
            try:
                dt = datetime.fromisoformat(ts)
                hour_counts[dt.hour] = hour_counts.get(dt.hour, 0) + 1
            except (ValueError, TypeError):
                continue

        active_hours = sorted(hour_counts.items(), key=lambda x: x[1], reverse=True)[:5]

        # Count sessions
        session_counts: dict[str, int] = {}
        for key, records in patterns.items():
            session_counts[key] = len(records)

        frequent_sessions = sorted(
            session_counts.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:5]

        return {
            "common_topics": [{"topic": t, "count": c} for t, c in common_topics],
            "active_hours": [{"hour": h, "count": c} for h, c in active_hours],
            "frequent_sessions": [{"session": s, "count": c} for s, c in frequent_sessions],
        }

    def _load_patterns(self) -> dict[str, list[dict[str, str]]]:
        """Load patterns from the JSON file."""
        if not self._patterns_path.exists():
            return {}
        try:
            data = self._patterns_path.read_text(encoding="utf-8")
            return json.loads(data) if data.strip() else {}
        except (json.JSONDecodeError, OSError) as e:
            logger.error(f"Failed to load patterns: {e}")
            return {}

    def _save_patterns(self, patterns: dict[str, list[dict[str, str]]]) -> None:
        """Save patterns to the JSON file."""
        try:
            atomic_write_json(self._patterns_path, patterns)
        except OSError as e:
            logger.error(f"Failed to save patterns: {e}")

    @staticmethod
    def _text_hash(text: str) -> str:
        """Return a stable hash for reminder text."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

    def _load_dismissed(self) -> set[str]:
        """Load dismissed reminder hashes from disk."""
        if not self._dismissed_path.exists():
            return set()
        try:
            data = self._dismissed_path.read_text(encoding="utf-8")
            items = json.loads(data) if data.strip() else []
            return set(items)
        except (json.JSONDecodeError, OSError) as e:
            logger.error(f"Failed to load dismissed reminders: {e}")
            return set()

    def _save_dismissed(self, dismissed: set[str]) -> None:
        """Save dismissed reminder hashes to disk."""
        try:
            atomic_write_json(self._dismissed_path, sorted(dismissed), indent=0)
        except OSError as e:
            logger.error(f"Failed to save dismissed reminders: {e}")
