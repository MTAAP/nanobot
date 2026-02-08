"""Cost and token usage tracking with SQLite persistence."""

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

from loguru import logger


@dataclass
class UsageRecord:
    """A single LLM usage record."""

    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    session_key: str
    subsystem: str = "main"
    timestamp: float = 0.0

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = time.time()


class UsageTracker:
    """SQLite-backed token and cost tracker."""

    def __init__(self, db_path: Path | None = None):
        if db_path is None:
            db_path = Path.home() / ".nanobot" / "data" / "usage.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(db_path))
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("""
            CREATE TABLE IF NOT EXISTS usage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model TEXT NOT NULL,
                prompt_tokens INTEGER NOT NULL,
                completion_tokens INTEGER NOT NULL,
                total_tokens INTEGER NOT NULL,
                session_key TEXT NOT NULL,
                subsystem TEXT DEFAULT 'main',
                timestamp REAL NOT NULL
            )
        """)
        self._db.commit()

    def record(self, rec: UsageRecord) -> None:
        """Record a usage entry."""
        self._db.execute(
            "INSERT INTO usage"
            " (model, prompt_tokens, completion_tokens, total_tokens,"
            "  session_key, subsystem, timestamp)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                rec.model,
                rec.prompt_tokens,
                rec.completion_tokens,
                rec.total_tokens,
                rec.session_key,
                rec.subsystem,
                rec.timestamp,
            ),
        )
        self._db.commit()

    def get_session_total(self, session_key: str) -> dict[str, int]:
        """Get total token usage for a session."""
        row = self._db.execute(
            "SELECT COALESCE(SUM(prompt_tokens), 0),"
            "       COALESCE(SUM(completion_tokens), 0),"
            "       COALESCE(SUM(total_tokens), 0)"
            " FROM usage WHERE session_key = ?",
            (session_key,),
        ).fetchone()
        return {
            "prompt_tokens": row[0],
            "completion_tokens": row[1],
            "total_tokens": row[2],
        }

    def get_daily_total(self, days: int = 1) -> dict[str, int]:
        """Get total token usage for the last N days."""
        cutoff = time.time() - (days * 86400)
        row = self._db.execute(
            "SELECT COALESCE(SUM(prompt_tokens), 0),"
            "       COALESCE(SUM(completion_tokens), 0),"
            "       COALESCE(SUM(total_tokens), 0)"
            " FROM usage WHERE timestamp >= ?",
            (cutoff,),
        ).fetchone()
        return {
            "prompt_tokens": row[0],
            "completion_tokens": row[1],
            "total_tokens": row[2],
        }

    def get_model_breakdown(self, days: int = 7) -> list[dict]:
        """Get per-model token breakdown for the last N days."""
        cutoff = time.time() - (days * 86400)
        rows = self._db.execute(
            "SELECT model,"
            "       SUM(prompt_tokens), SUM(completion_tokens),"
            "       SUM(total_tokens), COUNT(*)"
            " FROM usage WHERE timestamp >= ?"
            " GROUP BY model ORDER BY SUM(total_tokens) DESC",
            (cutoff,),
        ).fetchall()
        return [
            {
                "model": r[0],
                "prompt_tokens": r[1],
                "completion_tokens": r[2],
                "total_tokens": r[3],
                "calls": r[4],
            }
            for r in rows
        ]

    def log_session_summary(self, session_key: str) -> None:
        """Log a summary of session token usage."""
        totals = self.get_session_total(session_key)
        if totals["total_tokens"] > 0:
            logger.info(
                f"Session {session_key} usage: "
                f"{totals['prompt_tokens']} prompt + "
                f"{totals['completion_tokens']} completion = "
                f"{totals['total_tokens']} total tokens"
            )

    def close(self) -> None:
        """Close database connection."""
        self._db.close()
