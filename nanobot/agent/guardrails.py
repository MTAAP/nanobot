"""Tool execution guardrails for dangerous operations."""

import asyncio
import json
import re
import uuid
from dataclasses import dataclass
from typing import Any

from loguru import logger


@dataclass
class GuardrailRule:
    """A rule that gates tool execution on user approval."""

    tool_name: str
    condition: str  # "always" or "pattern"
    pattern: str | None = None
    description: str = ""


DEFAULT_RULES: list[GuardrailRule] = [
    GuardrailRule(
        tool_name="exec",
        condition="pattern",
        pattern=r"rm\s+-rf|sudo\s+|shutdown|reboot|mkfs|dd\s+if=|chmod\s+777|>\s*/dev/",
        description="Dangerous shell command",
    ),
    GuardrailRule(
        tool_name="write_file",
        condition="pattern",
        pattern=r"\.env|\.ssh|secret|password|credential|private_key|authorized_keys",
        description="Write to sensitive file",
    ),
    GuardrailRule(
        tool_name="edit_file",
        condition="pattern",
        pattern=r"\.env|\.ssh|secret|password|credential|private_key|authorized_keys",
        description="Edit sensitive file",
    ),
]


class GuardrailEngine:
    """Checks tool calls against safety rules and manages approval flow."""

    def __init__(
        self,
        enabled: bool = False,
        timeout_s: int = 60,
        custom_rules: list[GuardrailRule] | None = None,
    ):
        self.enabled = enabled
        self.timeout_s = timeout_s
        self.rules = list(DEFAULT_RULES)
        if custom_rules:
            self.rules.extend(custom_rules)
        self._pending: dict[str, asyncio.Event] = {}
        self._approvals: dict[str, bool] = {}

    def check(self, tool_name: str, arguments: dict[str, Any]) -> GuardrailRule | None:
        """Check if a tool call matches any guardrail rule.

        Returns the matching rule, or None if no rule applies.
        """
        if not self.enabled:
            return None

        args_str = json.dumps(arguments).lower()

        for rule in self.rules:
            if rule.tool_name != tool_name:
                continue
            if rule.condition == "always":
                return rule
            if rule.condition == "pattern" and rule.pattern:
                if re.search(rule.pattern, args_str, re.IGNORECASE):
                    return rule
        return None

    def request_approval(
        self,
        rule: GuardrailRule,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> str:
        """Create a pending approval request. Returns approval_id."""
        approval_id = uuid.uuid4().hex[:12]
        self._pending[approval_id] = asyncio.Event()
        self._approvals[approval_id] = False
        logger.info(
            f"Guardrail triggered: {rule.description} for {tool_name} (approval: {approval_id})"
        )
        return approval_id

    def approve(self, approval_id: str) -> None:
        """Approve a pending guardrail request."""
        if approval_id in self._pending:
            self._approvals[approval_id] = True
            self._pending[approval_id].set()

    def deny(self, approval_id: str) -> None:
        """Deny a pending guardrail request."""
        if approval_id in self._pending:
            self._approvals[approval_id] = False
            self._pending[approval_id].set()

    async def wait_for_approval(
        self,
        approval_id: str,
        timeout: float | None = None,
    ) -> bool:
        """Wait for user to approve or deny. Returns True if approved."""
        if approval_id not in self._pending:
            return False

        effective_timeout = timeout or self.timeout_s
        event = self._pending[approval_id]

        try:
            await asyncio.wait_for(event.wait(), timeout=effective_timeout)
            approved = self._approvals.get(approval_id, False)
        except asyncio.TimeoutError:
            logger.info(f"Guardrail approval {approval_id} timed out after {effective_timeout}s")
            approved = False
        finally:
            self._pending.pop(approval_id, None)
            self._approvals.pop(approval_id, None)

        return approved
