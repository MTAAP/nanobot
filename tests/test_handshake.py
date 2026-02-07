"""Tests for the agent handshake protocol."""

import asyncio
import os

import pytest

from nanobot.registry.handshake import AgentHandshake, HandshakeError
from nanobot.registry.store import AgentRegistry, AgentState, TaskState


@pytest.fixture
def workspace(tmp_path):
    """Create a writable workspace."""
    ws = tmp_path / "workspace"
    ws.mkdir()
    return ws


@pytest.fixture
def registry(workspace):
    """Create a registry in the workspace."""
    return AgentRegistry(workspace)


@pytest.fixture
def handshake(registry, workspace):
    """Create a handshake instance."""
    return AgentHandshake(registry, workspace)


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


class TestSuccessfulHandshake:
    def test_basic_handshake(self, registry, handshake):
        run(registry.create_task("t1", "Do work"))
        result = run(handshake.perform(
            agent_id="agent-1",
            task_id="t1",
            capabilities=["read_file"],
        ))
        assert result["ok"] is True
        agent = run(registry.get_agent("agent-1"))
        assert agent["state"] == AgentState.WORKING.value

    def test_handshake_assigns_task(self, registry, handshake):
        run(registry.create_task("t1", "Do work"))
        run(handshake.perform(agent_id="agent-1", task_id="t1"))
        task = run(registry.get_task("t1"))
        assert task["state"] == TaskState.ASSIGNED.value
        assert task["assigned_agent_id"] == "agent-1"

    def test_handshake_with_tool_validation(self, registry, handshake):
        run(registry.create_task("t1", "Do work"))
        result = run(handshake.perform(
            agent_id="agent-1",
            task_id="t1",
            required_tools=["read_file", "exec"],
            available_tool_names=["read_file", "write_file", "exec"],
        ))
        assert result["ok"] is True
        assert result["checks"]["tools"]["ok"] is True

    def test_handshake_with_credential_validation(self, registry, handshake):
        # Set env var for test
        os.environ["TEST_API_KEY"] = "test-value"
        try:
            run(registry.create_task("t1", "Do work"))
            result = run(handshake.perform(
                agent_id="agent-1",
                task_id="t1",
                required_credentials=["TEST_API_KEY"],
            ))
            assert result["ok"] is True
            assert result["checks"]["credentials"]["ok"] is True
        finally:
            del os.environ["TEST_API_KEY"]


class TestWorkspaceValidation:
    def test_nonexistent_workspace(self, registry, tmp_path):
        bad_ws = tmp_path / "nonexistent"
        hs = AgentHandshake(registry, bad_ws)
        run(registry.create_task("t1", "Do work"))
        with pytest.raises(HandshakeError, match="does not exist"):
            run(hs.perform(agent_id="agent-1", task_id="t1"))

    def test_workspace_is_file(self, registry, tmp_path):
        file_ws = tmp_path / "not_a_dir"
        file_ws.write_text("nope", encoding="utf-8")
        hs = AgentHandshake(registry, file_ws)
        run(registry.create_task("t1", "Do work"))
        with pytest.raises(HandshakeError, match="not a directory"):
            run(hs.perform(agent_id="agent-1", task_id="t1"))


class TestToolValidation:
    def test_missing_tools(self, registry, handshake):
        run(registry.create_task("t1", "Do work"))
        with pytest.raises(HandshakeError, match="missing tools"):
            run(handshake.perform(
                agent_id="agent-1",
                task_id="t1",
                required_tools=["nonexistent_tool"],
                available_tool_names=["read_file"],
            ))

    def test_no_required_tools_passes(self, registry, handshake):
        run(registry.create_task("t1", "Do work"))
        result = run(handshake.perform(
            agent_id="agent-1",
            task_id="t1",
            required_tools=[],
        ))
        assert result["ok"] is True


class TestCredentialValidation:
    def test_missing_credentials(self, registry, handshake):
        run(registry.create_task("t1", "Do work"))
        with pytest.raises(HandshakeError, match="missing env vars"):
            run(handshake.perform(
                agent_id="agent-1",
                task_id="t1",
                required_credentials=["NONEXISTENT_VAR_12345"],
            ))


class TestFailureState:
    def test_failure_sets_init_failure(self, registry, tmp_path):
        bad_ws = tmp_path / "nonexistent"
        hs = AgentHandshake(registry, bad_ws)
        run(registry.create_task("t1", "Do work"))
        with pytest.raises(HandshakeError):
            run(hs.perform(agent_id="agent-1", task_id="t1"))

        agent = run(registry.get_agent("agent-1"))
        assert agent["state"] == AgentState.INIT_FAILURE.value

    def test_handshake_error_includes_checks(self, registry, tmp_path):
        bad_ws = tmp_path / "nonexistent"
        hs = AgentHandshake(registry, bad_ws)
        run(registry.create_task("t1", "Do work"))
        with pytest.raises(HandshakeError) as exc_info:
            run(hs.perform(agent_id="agent-1", task_id="t1"))

        assert "workspace" in exc_info.value.checks
        assert exc_info.value.checks["workspace"]["ok"] is False
