"""Agent handshake protocol for environment validation."""

import os
from pathlib import Path
from typing import Any

from loguru import logger

from nanobot.registry.store import AgentRegistry, AgentState


class HandshakeError(Exception):
    """Raised when an agent handshake fails."""

    def __init__(self, message: str, checks: dict[str, Any] | None = None):
        super().__init__(message)
        self.checks = checks or {}


class AgentHandshake:
    """Validates agent environment before task execution.

    The handshake protocol:
    1. Register agent in registry (state -> INIT)
    2. Validate workspace exists and is writable
    3. Validate required tools are available
    4. Validate required credentials (env vars)
    5. On success: transition state -> WORKING, assign task
    6. On failure: transition state -> INIT_FAILURE, raise HandshakeError
    """

    def __init__(self, registry: AgentRegistry, workspace: Path):
        self._registry = registry
        self._workspace = workspace

    async def perform(
        self,
        agent_id: str,
        task_id: str,
        capabilities: list[str] | None = None,
        required_tools: list[str] | None = None,
        required_credentials: list[str] | None = None,
        available_tool_names: list[str] | None = None,
    ) -> dict[str, Any]:
        """Execute the full handshake protocol.

        Args:
            agent_id: Unique identifier for the agent.
            task_id: Task to assign on successful handshake.
            capabilities: Agent capabilities to register.
            required_tools: Tool names that must be available.
            required_credentials: Environment variable names that must be set.
            available_tool_names: Currently registered tool names for validation.

        Returns:
            Dict with handshake results including checks performed.

        Raises:
            HandshakeError: If any validation step fails.
        """
        checks: dict[str, Any] = {}

        # Step 1: Register agent
        await self._registry.register_agent(
            agent_id=agent_id,
            agent_type="subagent",
            capabilities=capabilities,
            task_id=task_id,
        )
        await self._registry.update_agent_state(agent_id, AgentState.INIT)

        try:
            # Step 2: Validate workspace
            checks["workspace"] = self._validate_workspace()

            # Step 3: Validate tools
            checks["tools"] = self._validate_tools(required_tools or [], available_tool_names or [])

            # Step 4: Validate credentials
            checks["credentials"] = self._validate_credentials(required_credentials or [])

            # Check for any failures
            failures = {k: v for k, v in checks.items() if not v.get("ok")}
            if failures:
                reasons = [f"{k}: {v.get('error', 'unknown')}" for k, v in failures.items()]
                reason = "; ".join(reasons)
                await self._registry.update_agent_state(
                    agent_id, AgentState.INIT_FAILURE, reason=reason
                )
                raise HandshakeError(f"Handshake failed: {reason}", checks)

            # Step 5: Success - transition to WORKING and assign task
            await self._registry.update_agent_state(
                agent_id, AgentState.WORKING, reason="handshake passed"
            )
            await self._registry.assign_task(task_id, agent_id)

            logger.debug(f"Handshake passed for agent {agent_id}, assigned task {task_id}")
            return {"ok": True, "checks": checks}

        except HandshakeError:
            raise
        except Exception as e:
            await self._registry.update_agent_state(
                agent_id, AgentState.INIT_FAILURE, reason=str(e)
            )
            raise HandshakeError(f"Handshake error: {e}", checks) from e

    def _validate_workspace(self) -> dict[str, Any]:
        """Check that workspace exists and is writable."""
        if not self._workspace.exists():
            return {"ok": False, "error": f"workspace {self._workspace} does not exist"}

        if not self._workspace.is_dir():
            return {"ok": False, "error": f"{self._workspace} is not a directory"}

        # Check writability by testing os.access
        if not os.access(self._workspace, os.W_OK):
            return {"ok": False, "error": f"{self._workspace} is not writable"}

        return {"ok": True}

    def _validate_tools(self, required: list[str], available: list[str]) -> dict[str, Any]:
        """Check that required tools are registered."""
        if not required:
            return {"ok": True, "checked": []}

        missing = [t for t in required if t not in available]
        if missing:
            return {
                "ok": False,
                "error": f"missing tools: {', '.join(missing)}",
                "missing": missing,
            }
        return {"ok": True, "checked": required}

    def _validate_credentials(self, required: list[str]) -> dict[str, Any]:
        """Check that required environment variables are set."""
        if not required:
            return {"ok": True, "checked": []}

        missing = [v for v in required if not os.environ.get(v)]
        if missing:
            return {
                "ok": False,
                "error": f"missing env vars: {', '.join(missing)}",
                "missing": missing,
            }
        return {"ok": True, "checked": required}
