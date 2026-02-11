"""Lightweight SQLite-backed notification store."""

import sqlite3
from pathlib import Path


class NotificationStore:
    """Persists notifications across restarts.

    Categories: cron_ok, cron_error, scan_complete
    """

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(db_path))
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("""
            CREATE TABLE IF NOT EXISTS notifications (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                body TEXT,
                category TEXT,
                link TEXT,
                read INTEGER DEFAULT 0,
                created_at TEXT DEFAULT (datetime('now'))
            )
        """)
        self._db.commit()

    def add(
        self,
        title: str,
        body: str = "",
        category: str = "",
        link: str = "",
    ) -> int:
        """Insert a notification and return its ID."""
        cursor = self._db.execute(
            "INSERT INTO notifications (title, body, category, link) VALUES (?, ?, ?, ?)",
            (title, body, category, link),
        )
        self._db.commit()
        return cursor.lastrowid or 0

    def list_recent(self, limit: int = 20) -> list[dict]:
        """Return the most recent notifications."""
        rows = self._db.execute(
            "SELECT id, title, body, category, link, read, created_at "
            "FROM notifications ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [
            {
                "id": r[0],
                "title": r[1],
                "body": r[2],
                "category": r[3],
                "link": r[4],
                "read": bool(r[5]),
                "created_at": r[6],
            }
            for r in rows
        ]

    def count_unread(self) -> int:
        """Return the number of unread notifications."""
        row = self._db.execute("SELECT COUNT(*) FROM notifications WHERE read = 0").fetchone()
        return row[0] if row else 0

    def mark_read(self, notification_id: int) -> None:
        """Mark a single notification as read."""
        self._db.execute(
            "UPDATE notifications SET read = 1 WHERE id = ?",
            (notification_id,),
        )
        self._db.commit()

    def mark_all_read(self) -> None:
        """Mark every notification as read."""
        self._db.execute("UPDATE notifications SET read = 1 WHERE read = 0")
        self._db.commit()
