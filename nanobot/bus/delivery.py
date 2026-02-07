"""Persistent outbound message delivery queue with retry."""

import json
import sqlite3
import time
import uuid
from pathlib import Path

from loguru import logger

from nanobot.bus.events import OutboundMessage


class DeliveryQueue:
    """SQLite-backed outbound message queue with retry and dead-letter logging.

    Flow: enqueue() -> process() loop dequeues -> send_fn() -> mark_delivered()
    On failure: retry with exponential backoff. After max_retries: dead letter.
    """

    MAX_RETRIES = 3
    BASE_DELAY_S = 5  # 5s, 10s, 20s backoff

    def __init__(self, db_path: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(db_path))
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("""
            CREATE TABLE IF NOT EXISTS outbound_queue (
                id TEXT PRIMARY KEY,
                channel TEXT NOT NULL,
                chat_id TEXT NOT NULL,
                content TEXT NOT NULL,
                reply_to TEXT,
                media_json TEXT,
                metadata_json TEXT,
                status TEXT DEFAULT 'pending',
                retry_count INTEGER DEFAULT 0,
                next_retry_at REAL DEFAULT 0,
                created_at REAL NOT NULL,
                error TEXT
            )
        """)
        self._db.commit()

    def enqueue(self, msg: OutboundMessage) -> str:
        """Store message for delivery. Returns delivery ID."""
        delivery_id = str(uuid.uuid4())[:12]
        self._db.execute(
            "INSERT INTO outbound_queue"
            " (id, channel, chat_id, content, reply_to,"
            "  media_json, metadata_json, status, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', ?)",
            (
                delivery_id,
                msg.channel,
                msg.chat_id,
                msg.content,
                msg.reply_to,
                json.dumps(msg.media) if msg.media else None,
                json.dumps(msg.metadata) if msg.metadata else None,
                time.time(),
            ),
        )
        self._db.commit()
        return delivery_id

    def dequeue_ready(self) -> list[tuple[str, OutboundMessage]]:
        """Get all messages ready for delivery."""
        now = time.time()
        rows = self._db.execute(
            "SELECT id, channel, chat_id, content, reply_to,"
            "       media_json, metadata_json"
            " FROM outbound_queue"
            " WHERE status IN ('pending', 'retry')"
            "   AND next_retry_at <= ?"
            " ORDER BY created_at"
            " LIMIT 10",
            (now,),
        ).fetchall()

        results = []
        for row in rows:
            did, channel, chat_id, content, reply_to, med_json, meta_json = row
            media = json.loads(med_json) if med_json else []
            metadata = json.loads(meta_json) if meta_json else {}
            msg = OutboundMessage(
                channel=channel,
                chat_id=chat_id,
                content=content,
                reply_to=reply_to,
                media=media,
                metadata=metadata,
            )
            self._db.execute(
                "UPDATE outbound_queue SET status = 'in_flight' WHERE id = ?",
                (did,),
            )
            results.append((did, msg))

        if results:
            self._db.commit()
        return results

    def mark_delivered(self, delivery_id: str) -> None:
        """Mark message as successfully delivered."""
        self._db.execute(
            "DELETE FROM outbound_queue WHERE id = ?",
            (delivery_id,),
        )
        self._db.commit()

    def mark_failed(self, delivery_id: str, error: str) -> None:
        """Mark delivery attempt as failed."""
        row = self._db.execute(
            "SELECT retry_count FROM outbound_queue WHERE id = ?",
            (delivery_id,),
        ).fetchone()
        if not row:
            return

        retry_count = row[0] + 1
        if retry_count >= self.MAX_RETRIES:
            self._db.execute(
                "UPDATE outbound_queue"
                " SET status = 'dead', retry_count = ?, error = ?"
                " WHERE id = ?",
                (retry_count, error, delivery_id),
            )
            logger.warning(
                f"Message {delivery_id} dead-lettered after {retry_count} attempts: {error}"
            )
        else:
            delay = self.BASE_DELAY_S * (2**retry_count)
            next_retry = time.time() + delay
            self._db.execute(
                "UPDATE outbound_queue"
                " SET status = 'retry', retry_count = ?,"
                "     next_retry_at = ?, error = ?"
                " WHERE id = ?",
                (retry_count, next_retry, error, delivery_id),
            )
            logger.info(f"Message {delivery_id} retry {retry_count} in {delay}s")

        self._db.commit()

    def recover_in_flight(self) -> None:
        """On startup, reset any in-flight messages back to pending."""
        count = self._db.execute(
            "UPDATE outbound_queue SET status = 'pending' WHERE status = 'in_flight'"
        ).rowcount
        self._db.commit()
        if count:
            logger.info(f"Recovered {count} in-flight messages from previous run")

    def cleanup_old(self, max_age_hours: int = 24) -> None:
        """Remove old dead-letter messages."""
        cutoff = time.time() - (max_age_hours * 3600)
        self._db.execute(
            "DELETE FROM outbound_queue WHERE status = 'dead' AND created_at < ?",
            (cutoff,),
        )
        self._db.commit()

    def close(self) -> None:
        """Close database connection."""
        self._db.close()
