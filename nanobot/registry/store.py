"""SQLite-backed agent and task registry for ACP."""

import asyncio
import json
import os
import sqlite3
import time
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from loguru import logger

CURRENT_SCHEMA_VERSION = 1


class AgentState(str, Enum):
    """Agent lifecycle states."""

    IDLE = "idle"
    INIT = "init"
    WORKING = "working"
    VERIFYING = "verifying"
    COMPLETED = "completed"
    FAILED = "failed"
    INIT_FAILURE = "init_failure"


class TaskState(str, Enum):
    """Task lifecycle states."""

    PENDING = "pending"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    VERIFYING = "verifying"
    COMPLETED = "completed"
    FAILED = "failed"


# Valid state transitions for agents
AGENT_TRANSITIONS: dict[AgentState, set[AgentState]] = {
    AgentState.IDLE: {AgentState.INIT},
    AgentState.INIT: {AgentState.WORKING, AgentState.INIT_FAILURE},
    AgentState.WORKING: {AgentState.VERIFYING, AgentState.FAILED},
    AgentState.VERIFYING: {AgentState.COMPLETED, AgentState.FAILED},
    AgentState.COMPLETED: {AgentState.IDLE},
    AgentState.FAILED: {AgentState.IDLE},
    AgentState.INIT_FAILURE: set(),  # terminal
}

# Valid state transitions for tasks
TASK_TRANSITIONS: dict[TaskState, set[TaskState]] = {
    TaskState.PENDING: {TaskState.ASSIGNED},
    TaskState.ASSIGNED: {TaskState.IN_PROGRESS},
    TaskState.IN_PROGRESS: {TaskState.VERIFYING, TaskState.FAILED},
    TaskState.VERIFYING: {TaskState.COMPLETED, TaskState.FAILED},
    TaskState.COMPLETED: set(),
    TaskState.FAILED: {TaskState.PENDING},
}


class AgentRegistry:
    """SQLite-backed registry for tracking agents and tasks.

    Follows the same patterns as nanobot.memory.vectors.VectorStore:
    WAL mode, busy_timeout, schema versioning, asyncio.Lock, chmod 0o600.
    """

    def __init__(self, workspace: Path):
        self._db_dir = workspace / ".agents"
        self._db_dir.mkdir(parents=True, exist_ok=True)
        self._db_path = self._db_dir / "registry.db"
        self._write_lock = asyncio.Lock()
        logger.debug(f"Registry store: db_path={self._db_path}")
        try:
            self._init_db()
        except Exception as e:
            parent = self._db_path.parent
            logger.error(
                f"Registry DB init failed: path={self._db_path}, "
                f"parent_exists={parent.exists()}, "
                f"parent_writable={os.access(parent, os.W_OK)}, "
                f"error={e}"
            )
            raise

    @property
    def db_path(self) -> Path:
        return self._db_path

    def _init_db(self) -> None:
        """Initialize database schema."""
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=30000")

            conn.execute("CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL)")

            cursor = conn.execute("SELECT version FROM schema_version")
            row = cursor.fetchone()

            if row is None:
                self._create_tables(conn)
                conn.execute(
                    "INSERT INTO schema_version (version) VALUES (?)",
                    (CURRENT_SCHEMA_VERSION,),
                )
            elif row[0] < CURRENT_SCHEMA_VERSION:
                self._run_migrations(conn, row[0])

        try:
            os.chmod(self._db_path, 0o600)
        except OSError:
            pass

    def _create_tables(self, conn: sqlite3.Connection) -> None:
        """Create all tables for a fresh database."""
        conn.execute("""
            CREATE TABLE agents (
                agent_id TEXT PRIMARY KEY,
                agent_type TEXT NOT NULL,
                capabilities TEXT,
                task_id TEXT,
                state TEXT NOT NULL DEFAULT 'idle',
                state_reason TEXT DEFAULT '',
                last_pulse_at REAL,
                registered_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)

        conn.execute("""
            CREATE TABLE tasks (
                task_id TEXT PRIMARY KEY,
                description TEXT NOT NULL,
                priority TEXT NOT NULL DEFAULT 'medium',
                complexity TEXT NOT NULL DEFAULT 'simple',
                assigned_agent_id TEXT,
                state TEXT NOT NULL DEFAULT 'pending',
                state_reason TEXT DEFAULT '',
                proof_of_work TEXT,
                metadata TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)

        conn.execute("""
            CREATE TABLE state_transitions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                entity_type TEXT NOT NULL,
                entity_id TEXT NOT NULL,
                from_state TEXT NOT NULL,
                to_state TEXT NOT NULL,
                reason TEXT DEFAULT '',
                timestamp TEXT NOT NULL
            )
        """)

        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_transitions_entity "
            "ON state_transitions (entity_type, entity_id)"
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_agents_state ON agents (state)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_tasks_state ON tasks (state)")

    def _run_migrations(self, conn: sqlite3.Connection, from_version: int) -> None:
        """Run schema migrations."""
        # Future migrations go here
        conn.execute("UPDATE schema_version SET version = ?", (CURRENT_SCHEMA_VERSION,))

    def _now_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _record_transition(
        self,
        conn: sqlite3.Connection,
        entity_type: str,
        entity_id: str,
        from_state: str,
        to_state: str,
        reason: str = "",
    ) -> None:
        """Record a state transition in the audit log."""
        conn.execute(
            "INSERT INTO state_transitions "
            "(entity_type, entity_id, from_state, to_state, reason, timestamp) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (entity_type, entity_id, from_state, to_state, reason, self._now_iso()),
        )

    # ------------------------------------------------------------------
    # Agent operations
    # ------------------------------------------------------------------

    async def register_agent(
        self,
        agent_id: str,
        agent_type: str,
        capabilities: list[str] | None = None,
        task_id: str | None = None,
    ) -> dict[str, Any]:
        """Register a new agent."""
        now = self._now_iso()
        caps_json = json.dumps(capabilities or [])

        async with self._write_lock:
            with sqlite3.connect(str(self._db_path)) as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO agents "
                    "(agent_id, agent_type, capabilities, task_id, state, "
                    "state_reason, last_pulse_at, registered_at, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        agent_id,
                        agent_type,
                        caps_json,
                        task_id,
                        AgentState.IDLE.value,
                        "",
                        time.time(),
                        now,
                        now,
                    ),
                )
                conn.commit()

        logger.debug(f"Registered agent {agent_id} (type={agent_type})")
        return await self.get_agent(agent_id)

    async def update_agent_state(
        self,
        agent_id: str,
        new_state: AgentState,
        reason: str = "",
    ) -> None:
        """Transition an agent to a new state with validation."""
        agent = await self.get_agent(agent_id)
        if not agent:
            raise ValueError(f"Agent {agent_id} not found")

        current = AgentState(agent["state"])
        if new_state not in AGENT_TRANSITIONS.get(current, set()):
            raise ValueError(f"Invalid agent transition: {current.value} -> {new_state.value}")

        async with self._write_lock:
            with sqlite3.connect(str(self._db_path)) as conn:
                conn.execute(
                    "UPDATE agents SET state = ?, state_reason = ?, updated_at = ? "
                    "WHERE agent_id = ?",
                    (new_state.value, reason, self._now_iso(), agent_id),
                )
                self._record_transition(
                    conn, "agent", agent_id, current.value, new_state.value, reason
                )
                conn.commit()

    async def record_pulse(self, agent_id: str) -> None:
        """Update the last pulse timestamp for an agent."""
        async with self._write_lock:
            with sqlite3.connect(str(self._db_path)) as conn:
                conn.execute(
                    "UPDATE agents SET last_pulse_at = ?, updated_at = ? WHERE agent_id = ?",
                    (time.time(), self._now_iso(), agent_id),
                )
                conn.commit()

    async def mark_stale_agents(self, stale_threshold_s: int = 180) -> list[str]:
        """Mark agents as FAILED if they haven't pulsed within the threshold."""
        cutoff = time.time() - stale_threshold_s
        stale_ids: list[str] = []

        async with self._write_lock:
            with sqlite3.connect(str(self._db_path)) as conn:
                rows = conn.execute(
                    "SELECT agent_id, state FROM agents "
                    "WHERE last_pulse_at < ? AND state IN (?, ?, ?)",
                    (
                        cutoff,
                        AgentState.INIT.value,
                        AgentState.WORKING.value,
                        AgentState.VERIFYING.value,
                    ),
                ).fetchall()

                now = self._now_iso()
                for agent_id, current_state in rows:
                    conn.execute(
                        "UPDATE agents SET state = ?, state_reason = ?, updated_at = ? "
                        "WHERE agent_id = ?",
                        (AgentState.FAILED.value, "stale - no pulse", now, agent_id),
                    )
                    self._record_transition(
                        conn,
                        "agent",
                        agent_id,
                        current_state,
                        AgentState.FAILED.value,
                        "stale - no pulse",
                    )
                    stale_ids.append(agent_id)

                conn.commit()

        if stale_ids:
            logger.warning(f"Marked {len(stale_ids)} stale agents as FAILED: {stale_ids}")
        return stale_ids

    async def get_agent(self, agent_id: str) -> dict[str, Any] | None:
        """Get agent details."""
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute("SELECT * FROM agents WHERE agent_id = ?", (agent_id,)).fetchone()

        if not row:
            return None
        result = dict(row)
        if result.get("capabilities"):
            result["capabilities"] = json.loads(result["capabilities"])
        return result

    async def list_agents(self, state: AgentState | None = None) -> list[dict[str, Any]]:
        """List agents, optionally filtered by state."""
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.row_factory = sqlite3.Row
            if state:
                rows = conn.execute(
                    "SELECT * FROM agents WHERE state = ? ORDER BY updated_at DESC",
                    (state.value,),
                ).fetchall()
            else:
                rows = conn.execute("SELECT * FROM agents ORDER BY updated_at DESC").fetchall()

        results = []
        for row in rows:
            d = dict(row)
            if d.get("capabilities"):
                d["capabilities"] = json.loads(d["capabilities"])
            results.append(d)
        return results

    # ------------------------------------------------------------------
    # Task operations
    # ------------------------------------------------------------------

    async def create_task(
        self,
        task_id: str,
        description: str,
        priority: str = "medium",
        complexity: str = "simple",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create a new task."""
        now = self._now_iso()
        meta_json = json.dumps(metadata or {})

        async with self._write_lock:
            with sqlite3.connect(str(self._db_path)) as conn:
                conn.execute(
                    "INSERT INTO tasks "
                    "(task_id, description, priority, complexity, state, "
                    "state_reason, metadata, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        task_id,
                        description,
                        priority,
                        complexity,
                        TaskState.PENDING.value,
                        "",
                        meta_json,
                        now,
                        now,
                    ),
                )
                conn.commit()

        logger.debug(f"Created task {task_id}: {description[:50]}")
        return await self.get_task(task_id)

    async def assign_task(self, task_id: str, agent_id: str) -> None:
        """Assign a task to an agent. Transitions task to ASSIGNED."""
        task = await self.get_task(task_id)
        if not task:
            raise ValueError(f"Task {task_id} not found")

        current = TaskState(task["state"])
        if TaskState.ASSIGNED not in TASK_TRANSITIONS.get(current, set()):
            raise ValueError(f"Cannot assign task in state {current.value}")

        async with self._write_lock:
            with sqlite3.connect(str(self._db_path)) as conn:
                conn.execute(
                    "UPDATE tasks SET assigned_agent_id = ?, state = ?, updated_at = ? "
                    "WHERE task_id = ?",
                    (agent_id, TaskState.ASSIGNED.value, self._now_iso(), task_id),
                )
                conn.execute(
                    "UPDATE agents SET task_id = ?, updated_at = ? WHERE agent_id = ?",
                    (task_id, self._now_iso(), agent_id),
                )
                self._record_transition(
                    conn,
                    "task",
                    task_id,
                    current.value,
                    TaskState.ASSIGNED.value,
                    f"assigned to {agent_id}",
                )
                conn.commit()

    async def update_task_state(
        self,
        task_id: str,
        new_state: TaskState,
        reason: str = "",
    ) -> None:
        """Transition a task to a new state with validation."""
        task = await self.get_task(task_id)
        if not task:
            raise ValueError(f"Task {task_id} not found")

        current = TaskState(task["state"])
        if new_state not in TASK_TRANSITIONS.get(current, set()):
            raise ValueError(f"Invalid task transition: {current.value} -> {new_state.value}")

        async with self._write_lock:
            with sqlite3.connect(str(self._db_path)) as conn:
                conn.execute(
                    "UPDATE tasks SET state = ?, state_reason = ?, updated_at = ? "
                    "WHERE task_id = ?",
                    (new_state.value, reason, self._now_iso(), task_id),
                )
                self._record_transition(
                    conn, "task", task_id, current.value, new_state.value, reason
                )
                conn.commit()

    async def submit_proof(self, task_id: str, proof_json: str) -> None:
        """Attach proof of work to a task."""
        async with self._write_lock:
            with sqlite3.connect(str(self._db_path)) as conn:
                conn.execute(
                    "UPDATE tasks SET proof_of_work = ?, updated_at = ? WHERE task_id = ?",
                    (proof_json, self._now_iso(), task_id),
                )
                conn.commit()

    async def get_task(self, task_id: str) -> dict[str, Any] | None:
        """Get task details."""
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute("SELECT * FROM tasks WHERE task_id = ?", (task_id,)).fetchone()

        if not row:
            return None
        result = dict(row)
        if result.get("metadata"):
            result["metadata"] = json.loads(result["metadata"])
        if result.get("proof_of_work"):
            result["proof_of_work"] = json.loads(result["proof_of_work"])
        return result

    async def list_tasks(self, state: TaskState | None = None) -> list[dict[str, Any]]:
        """List tasks, optionally filtered by state."""
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.row_factory = sqlite3.Row
            if state:
                rows = conn.execute(
                    "SELECT * FROM tasks WHERE state = ? ORDER BY created_at DESC",
                    (state.value,),
                ).fetchall()
            else:
                rows = conn.execute("SELECT * FROM tasks ORDER BY created_at DESC").fetchall()

        results = []
        for row in rows:
            d = dict(row)
            if d.get("metadata"):
                d["metadata"] = json.loads(d["metadata"])
            if d.get("proof_of_work"):
                d["proof_of_work"] = json.loads(d["proof_of_work"])
            results.append(d)
        return results

    async def get_verifying_tasks(self) -> list[dict[str, Any]]:
        """Get all tasks in VERIFYING state."""
        return await self.list_tasks(TaskState.VERIFYING)

    # ------------------------------------------------------------------
    # Stats and lifecycle
    # ------------------------------------------------------------------

    async def get_stats(self) -> dict[str, Any]:
        """Get registry statistics."""
        with sqlite3.connect(str(self._db_path)) as conn:
            agent_count = conn.execute("SELECT COUNT(*) FROM agents").fetchone()[0]
            task_count = conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]

            agent_states: dict[str, int] = {}
            for row in conn.execute("SELECT state, COUNT(*) FROM agents GROUP BY state").fetchall():
                agent_states[row[0]] = row[1]

            task_states: dict[str, int] = {}
            for row in conn.execute("SELECT state, COUNT(*) FROM tasks GROUP BY state").fetchall():
                task_states[row[0]] = row[1]

            transition_count = conn.execute("SELECT COUNT(*) FROM state_transitions").fetchone()[0]

        return {
            "agents": agent_count,
            "tasks": task_count,
            "agent_states": agent_states,
            "task_states": task_states,
            "transitions": transition_count,
        }

    async def get_transitions(self, entity_type: str, entity_id: str) -> list[dict[str, Any]]:
        """Get transition history for an entity."""
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM state_transitions "
                "WHERE entity_type = ? AND entity_id = ? "
                "ORDER BY timestamp ASC",
                (entity_type, entity_id),
            ).fetchall()

        return [dict(r) for r in rows]

    def close(self) -> None:
        """Close the registry (no-op for connection-per-call pattern)."""
        pass
