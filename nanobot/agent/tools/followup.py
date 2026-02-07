"""Follow-up task tracking tool."""

import sqlite3
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from nanobot.agent.tools.base import Tool


class FollowUpTool(Tool):
    """Track tasks the agent has promised to follow up on."""

    def __init__(self, db_path: Path):
        self._db_path = db_path
        self._db: sqlite3.Connection | None = None
        self._context_channel = ""
        self._context_chat_id = ""
        self._init_db()

    def _init_db(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(self._db_path))
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("""
            CREATE TABLE IF NOT EXISTS followups (
                id TEXT PRIMARY KEY,
                description TEXT NOT NULL,
                deadline_ms INTEGER,
                channel TEXT,
                chat_id TEXT,
                status TEXT DEFAULT 'pending',
                created_at_ms INTEGER,
                completed_at_ms INTEGER
            )
        """)
        self._db.commit()

    def set_context(self, channel: str, chat_id: str) -> None:
        """Set the current conversation context."""
        self._context_channel = channel
        self._context_chat_id = chat_id

    @property
    def name(self) -> str:
        return "follow_up"

    @property
    def description(self) -> str:
        return (
            "Track follow-up tasks you've promised. Create follow-ups "
            "with optional deadlines. The daemon will remind you of "
            "overdue items.\nActions: create, list, complete, dismiss"
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["create", "list", "complete", "dismiss"],
                },
                "description": {
                    "type": "string",
                    "description": ("What to follow up on (for create)"),
                },
                "deadline": {
                    "type": "string",
                    "description": ("ISO datetime deadline (optional, for create)"),
                },
                "followup_id": {
                    "type": "string",
                    "description": ("ID of follow-up (for complete/dismiss)"),
                },
            },
            "required": ["action"],
        }

    async def execute(
        self,
        action: str,
        description: str | None = None,
        deadline: str | None = None,
        followup_id: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Execute the follow-up action."""
        try:
            if action == "create":
                return self._create(description=description, deadline=deadline)
            elif action == "list":
                return self._list()
            elif action == "complete":
                return self._update_status(followup_id, "completed")
            elif action == "dismiss":
                return self._update_status(followup_id, "dismissed")
            else:
                return f"Error: unknown action '{action}'"
        except Exception as e:
            return f"Error: {e}"

    def _create(
        self,
        description: str | None,
        deadline: str | None,
    ) -> str:
        if not description:
            return "Error: description is required"

        fid = str(uuid.uuid4())[:8]
        deadline_ms = None
        if deadline:
            try:
                dt = datetime.fromisoformat(deadline)
                deadline_ms = int(dt.timestamp() * 1000)
            except ValueError:
                return f"Error: invalid deadline format: {deadline}"

        assert self._db
        self._db.execute(
            "INSERT INTO followups "
            "(id, description, deadline_ms, channel, chat_id, "
            "status, created_at_ms) "
            "VALUES (?, ?, ?, ?, ?, 'pending', ?)",
            (
                fid,
                description,
                deadline_ms,
                self._context_channel,
                self._context_chat_id,
                int(time.time() * 1000),
            ),
        )
        self._db.commit()
        return f"Follow-up created: {fid} - {description}"

    def _list(self) -> str:
        assert self._db
        rows = self._db.execute(
            "SELECT id, description, deadline_ms, status "
            "FROM followups WHERE status = 'pending' "
            "ORDER BY deadline_ms NULLS LAST"
        ).fetchall()
        if not rows:
            return "No pending follow-ups."
        lines = []
        for fid, desc, deadline_ms, status in rows:
            deadline_str = ""
            if deadline_ms:
                dt = datetime.fromtimestamp(deadline_ms / 1000, tz=timezone.utc)
                deadline_str = f" (due: {dt.strftime('%Y-%m-%d %H:%M')})"
            lines.append(f"- [{fid}] {desc}{deadline_str}")
        return "\n".join(lines)

    def _update_status(self, fid: str | None, status: str) -> str:
        if not fid:
            return "Error: followup_id is required"
        assert self._db
        now_ms = int(time.time() * 1000)
        result = self._db.execute(
            "UPDATE followups SET status = ?, completed_at_ms = ? WHERE id = ?",
            (status, now_ms, fid),
        )
        self._db.commit()
        if result.rowcount == 0:
            return f"Error: follow-up {fid} not found"
        return f"Follow-up {fid} marked as {status}"

    def get_overdue(self) -> list[dict]:
        """Called by heartbeat to check for overdue items."""
        assert self._db
        now_ms = int(time.time() * 1000)
        rows = self._db.execute(
            "SELECT id, description, deadline_ms, channel, chat_id "
            "FROM followups "
            "WHERE status = 'pending' AND deadline_ms IS NOT NULL "
            "AND deadline_ms <= ?",
            (now_ms,),
        ).fetchall()
        return [
            {
                "id": r[0],
                "description": r[1],
                "deadline_ms": r[2],
                "channel": r[3],
                "chat_id": r[4],
            }
            for r in rows
        ]
