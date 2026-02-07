"""Tests for the agent registry (ACP core state machine)."""

import asyncio
import time

import pytest

from nanobot.registry.store import (
    AGENT_TRANSITIONS,
    TASK_TRANSITIONS,
    AgentRegistry,
    AgentState,
    TaskState,
)


@pytest.fixture
def registry(tmp_path):
    """Create a fresh AgentRegistry in a temp directory."""
    return AgentRegistry(tmp_path)


@pytest.fixture
def event_loop():
    """Provide a fresh event loop for each test."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


def run(coro):
    """Helper to run async code in tests."""
    return asyncio.get_event_loop().run_until_complete(coro)


# ------------------------------------------------------------------
# Agent CRUD
# ------------------------------------------------------------------


class TestAgentCRUD:
    def test_register_agent(self, registry):
        agent = run(registry.register_agent(
            "agent-1", "subagent", capabilities=["read_file", "exec"]
        ))
        assert agent["agent_id"] == "agent-1"
        assert agent["agent_type"] == "subagent"
        assert agent["capabilities"] == ["read_file", "exec"]
        assert agent["state"] == AgentState.IDLE.value

    def test_get_agent_not_found(self, registry):
        result = run(registry.get_agent("nonexistent"))
        assert result is None

    def test_list_agents(self, registry):
        run(registry.register_agent("a1", "sub"))
        run(registry.register_agent("a2", "sub"))
        agents = run(registry.list_agents())
        assert len(agents) == 2

    def test_list_agents_by_state(self, registry):
        run(registry.register_agent("a1", "sub"))
        run(registry.register_agent("a2", "sub"))
        run(registry.update_agent_state("a1", AgentState.INIT))
        idle = run(registry.list_agents(AgentState.IDLE))
        assert len(idle) == 1
        assert idle[0]["agent_id"] == "a2"


# ------------------------------------------------------------------
# Agent state transitions
# ------------------------------------------------------------------


class TestAgentStateTransitions:
    def test_valid_transition_idle_to_init(self, registry):
        run(registry.register_agent("a1", "sub"))
        run(registry.update_agent_state("a1", AgentState.INIT))
        agent = run(registry.get_agent("a1"))
        assert agent["state"] == AgentState.INIT.value

    def test_valid_full_lifecycle(self, registry):
        run(registry.register_agent("a1", "sub"))
        run(registry.update_agent_state("a1", AgentState.INIT))
        run(registry.update_agent_state("a1", AgentState.WORKING))
        run(registry.update_agent_state("a1", AgentState.VERIFYING))
        run(registry.update_agent_state("a1", AgentState.COMPLETED))
        run(registry.update_agent_state("a1", AgentState.IDLE))
        agent = run(registry.get_agent("a1"))
        assert agent["state"] == AgentState.IDLE.value

    def test_invalid_transition_raises(self, registry):
        run(registry.register_agent("a1", "sub"))
        with pytest.raises(ValueError, match="Invalid agent transition"):
            run(registry.update_agent_state("a1", AgentState.WORKING))

    def test_init_failure_is_terminal(self, registry):
        run(registry.register_agent("a1", "sub"))
        run(registry.update_agent_state("a1", AgentState.INIT))
        run(registry.update_agent_state("a1", AgentState.INIT_FAILURE, reason="bad env"))
        # INIT_FAILURE has no valid transitions
        with pytest.raises(ValueError, match="Invalid agent transition"):
            run(registry.update_agent_state("a1", AgentState.IDLE))

    def test_agent_not_found_raises(self, registry):
        with pytest.raises(ValueError, match="not found"):
            run(registry.update_agent_state("nonexistent", AgentState.INIT))

    def test_transition_with_reason(self, registry):
        run(registry.register_agent("a1", "sub"))
        run(registry.update_agent_state("a1", AgentState.INIT, reason="starting up"))
        agent = run(registry.get_agent("a1"))
        assert agent["state_reason"] == "starting up"


# ------------------------------------------------------------------
# Task CRUD
# ------------------------------------------------------------------


class TestTaskCRUD:
    def test_create_task(self, registry):
        task = run(registry.create_task("t1", "Do something", priority="high"))
        assert task["task_id"] == "t1"
        assert task["description"] == "Do something"
        assert task["priority"] == "high"
        assert task["state"] == TaskState.PENDING.value

    def test_get_task_not_found(self, registry):
        assert run(registry.get_task("nonexistent")) is None

    def test_list_tasks(self, registry):
        run(registry.create_task("t1", "Task 1"))
        run(registry.create_task("t2", "Task 2"))
        tasks = run(registry.list_tasks())
        assert len(tasks) == 2

    def test_create_task_with_metadata(self, registry):
        meta = {"source": "daemon", "triage_reason": "bug fix"}
        task = run(registry.create_task("t1", "Fix bug", metadata=meta))
        assert task["metadata"] == meta


# ------------------------------------------------------------------
# Task state transitions
# ------------------------------------------------------------------


class TestTaskStateTransitions:
    def test_valid_full_lifecycle(self, registry):
        run(registry.create_task("t1", "Do work"))
        run(registry.register_agent("a1", "sub"))
        run(registry.assign_task("t1", "a1"))
        task = run(registry.get_task("t1"))
        assert task["state"] == TaskState.ASSIGNED.value
        assert task["assigned_agent_id"] == "a1"

        run(registry.update_task_state("t1", TaskState.IN_PROGRESS))
        run(registry.update_task_state("t1", TaskState.VERIFYING))
        run(registry.update_task_state("t1", TaskState.COMPLETED))
        task = run(registry.get_task("t1"))
        assert task["state"] == TaskState.COMPLETED.value

    def test_invalid_task_transition(self, registry):
        run(registry.create_task("t1", "Do work"))
        with pytest.raises(ValueError, match="Invalid task transition"):
            run(registry.update_task_state("t1", TaskState.IN_PROGRESS))

    def test_failed_to_pending_retry(self, registry):
        run(registry.create_task("t1", "Do work"))
        run(registry.register_agent("a1", "sub"))
        run(registry.assign_task("t1", "a1"))
        run(registry.update_task_state("t1", TaskState.IN_PROGRESS))
        run(registry.update_task_state("t1", TaskState.FAILED, reason="crash"))
        run(registry.update_task_state("t1", TaskState.PENDING, reason="retry"))
        task = run(registry.get_task("t1"))
        assert task["state"] == TaskState.PENDING.value

    def test_task_not_found_raises(self, registry):
        with pytest.raises(ValueError, match="not found"):
            run(registry.update_task_state("nonexistent", TaskState.ASSIGNED))


# ------------------------------------------------------------------
# Pulse and stale detection
# ------------------------------------------------------------------


class TestPulseAndStale:
    def test_record_pulse(self, registry):
        run(registry.register_agent("a1", "sub"))
        before = time.time()
        run(registry.record_pulse("a1"))
        agent = run(registry.get_agent("a1"))
        assert agent["last_pulse_at"] >= before

    def test_mark_stale_agents(self, registry):
        run(registry.register_agent("a1", "sub"))
        run(registry.update_agent_state("a1", AgentState.INIT))
        run(registry.update_agent_state("a1", AgentState.WORKING))
        # Manually set pulse to old time
        import sqlite3
        with sqlite3.connect(str(registry.db_path)) as conn:
            conn.execute(
                "UPDATE agents SET last_pulse_at = ? WHERE agent_id = ?",
                (time.time() - 300, "a1"),
            )
            conn.commit()

        stale = run(registry.mark_stale_agents(stale_threshold_s=180))
        assert "a1" in stale
        agent = run(registry.get_agent("a1"))
        assert agent["state"] == AgentState.FAILED.value

    def test_non_stale_agents_untouched(self, registry):
        run(registry.register_agent("a1", "sub"))
        run(registry.update_agent_state("a1", AgentState.INIT))
        run(registry.update_agent_state("a1", AgentState.WORKING))
        run(registry.record_pulse("a1"))

        stale = run(registry.mark_stale_agents(stale_threshold_s=180))
        assert len(stale) == 0


# ------------------------------------------------------------------
# State transition audit log
# ------------------------------------------------------------------


class TestTransitionLog:
    def test_transitions_recorded(self, registry):
        run(registry.register_agent("a1", "sub"))
        run(registry.update_agent_state("a1", AgentState.INIT))
        run(registry.update_agent_state("a1", AgentState.WORKING))

        transitions = run(registry.get_transitions("agent", "a1"))
        assert len(transitions) == 2
        assert transitions[0]["from_state"] == "idle"
        assert transitions[0]["to_state"] == "init"
        assert transitions[1]["from_state"] == "init"
        assert transitions[1]["to_state"] == "working"

    def test_task_transitions_recorded(self, registry):
        run(registry.create_task("t1", "Work"))
        run(registry.register_agent("a1", "sub"))
        run(registry.assign_task("t1", "a1"))

        transitions = run(registry.get_transitions("task", "t1"))
        assert len(transitions) == 1
        assert transitions[0]["to_state"] == "assigned"


# ------------------------------------------------------------------
# Stats
# ------------------------------------------------------------------


class TestStats:
    def test_get_stats(self, registry):
        run(registry.register_agent("a1", "sub"))
        run(registry.create_task("t1", "Task 1"))
        run(registry.create_task("t2", "Task 2"))

        stats = run(registry.get_stats())
        assert stats["agents"] == 1
        assert stats["tasks"] == 2
        assert stats["agent_states"]["idle"] == 1
        assert stats["task_states"]["pending"] == 2


# ------------------------------------------------------------------
# Schema versioning
# ------------------------------------------------------------------


class TestSchemaVersioning:
    def test_db_created_with_version(self, registry):
        import sqlite3
        with sqlite3.connect(str(registry.db_path)) as conn:
            version = conn.execute(
                "SELECT version FROM schema_version"
            ).fetchone()[0]
        assert version == 1

    def test_reopen_db_preserves_data(self, tmp_path):
        reg1 = AgentRegistry(tmp_path)
        run(reg1.register_agent("a1", "sub"))

        reg2 = AgentRegistry(tmp_path)
        agent = run(reg2.get_agent("a1"))
        assert agent is not None
        assert agent["agent_id"] == "a1"


# ------------------------------------------------------------------
# Proof submission
# ------------------------------------------------------------------


class TestProofSubmission:
    def test_submit_proof(self, registry):
        run(registry.create_task("t1", "Do work"))
        run(registry.submit_proof("t1", '{"items": [{"type": "test"}]}'))
        task = run(registry.get_task("t1"))
        assert task["proof_of_work"] == {"items": [{"type": "test"}]}

    def test_get_verifying_tasks(self, registry):
        run(registry.create_task("t1", "Work"))
        run(registry.register_agent("a1", "sub"))
        run(registry.assign_task("t1", "a1"))
        run(registry.update_task_state("t1", TaskState.IN_PROGRESS))
        run(registry.update_task_state("t1", TaskState.VERIFYING))

        tasks = run(registry.get_verifying_tasks())
        assert len(tasks) == 1
        assert tasks[0]["task_id"] == "t1"
