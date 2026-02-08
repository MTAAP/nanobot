"""Heartbeat service - periodic agent wake-up to check for tasks."""

import asyncio
import hashlib
import json
import tempfile
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Coroutine

from loguru import logger

from nanobot.providers.base import LLMProvider

if TYPE_CHECKING:
    from nanobot.registry.store import AgentRegistry

# Default interval: 30 minutes
DEFAULT_HEARTBEAT_INTERVAL_S = 30 * 60

# Default HEARTBEAT.md content — headers + HTML comments only so
# _is_heartbeat_empty() treats it as "no actionable tasks".
DEFAULT_STRATEGY_CONTENT = """\
# Heartbeat Strategy

<!-- This file is checked by the daemon every ~30 minutes (or ~5 min during active chat).
     Add tasks below as markdown checkboxes. The daemon will read and execute them.
     Mark completed tasks with [x] — they will be skipped on future ticks.

     When to add tasks here:
     - Recurring checks you want performed periodically
     - Background maintenance tasks
     - Self-improvement goals based on TOOLS.md lessons

     Example:
     - [ ] Check for new messages in all channels
     - [ ] Review TOOLS.md for recurring failures and create fix PRs
-->
"""

# The prompt sent to agent during heartbeat
HEARTBEAT_PROMPT = """Read HEARTBEAT.md in your workspace (if it exists).
Follow any instructions or tasks listed there.
If nothing needs attention, reply with just: HEARTBEAT_OK"""

# Token that indicates "nothing to do"
HEARTBEAT_OK_TOKEN = "HEARTBEAT_OK"

TRIAGE_PROMPT_TEMPLATE = """\
You are a task triage agent. Analyze this context and decide if the main \
agent should act.

## Strategy File ({strategy_file})
{strategy_content}

## Git Status
{git_status}

## Active Sessions
{tmux_sessions}

## Overdue Follow-ups
{overdue_followups}

## Tool Lessons / Known Issues
{tools_lessons}

Respond with JSON only: {{"act": true/false, "reason": "...", \
"priority": "low/medium/high", "complexity": "simple/complex"}}
Rules:
- act=true only if there are clear, actionable tasks
- Checked/completed items (- [x]) are NOT actionable
- Empty/missing strategy file means act=false
- complexity=simple for quick tasks (< 5 min), complex for longer tasks
- Overdue follow-ups are high priority and should always trigger action
- If tool lessons contain recurring failures or "FIX NEEDED" markers, \
suggest a self-improvement task with complexity=complex"""

EXECUTION_PROMPT_TEMPLATE = """\
[Daemon Mode - Priority: {priority}]

Triage reason: {reason}

## Strategy
{strategy_content}

## Current State
Git: {git_status}
Sessions: {tmux_sessions}

Execute the highest priority task from the strategy file. \
Use available tools including tmux for persistent shells. \
Mark completed tasks with [x] in the strategy file."""


def _is_heartbeat_empty(content: str | None) -> bool:
    """Check if HEARTBEAT.md has no actionable content.

    Skips blank lines, markdown headers, HTML comments (including multi-line),
    and checked/unchecked checkboxes without text beyond the marker.
    """
    if not content:
        return True

    skip_patterns = {"- [ ]", "* [ ]", "- [x]", "* [x]"}
    in_comment = False

    for line in content.split("\n"):
        stripped = line.strip()

        # Track multi-line HTML comments
        if in_comment:
            if "-->" in stripped:
                in_comment = False
            continue

        if stripped.startswith("<!--"):
            if "-->" not in stripped:
                in_comment = True
            continue

        if not stripped or stripped.startswith("#") or stripped in skip_patterns:
            continue
        return False  # Found actionable content

    return True


class HeartbeatService:
    """
    Periodic heartbeat service that wakes the agent to check for tasks.

    The agent reads HEARTBEAT.md from the workspace and executes any
    tasks listed there. If nothing needs attention, it replies HEARTBEAT_OK.

    In daemon mode (when triage_provider is set), uses a three-tier pipeline:
    - Tier 0: Gather context (strategy file, git status, tmux sessions)
    - Tier 1: Triage via cheap LLM call to decide whether to act
    - Tier 2: Execute via the main agent loop
    """

    def __init__(
        self,
        workspace: Path,
        on_heartbeat: Callable[[str], Coroutine[Any, Any, str]] | None = None,
        interval_s: int = DEFAULT_HEARTBEAT_INTERVAL_S,
        enabled: bool = True,
        triage_provider: LLMProvider | None = None,
        triage_model: str | None = None,
        execution_model: str | None = None,
        strategy_file: str = "HEARTBEAT.md",
        max_iterations: int = 25,
        cooldown_after_action: int = 600,
        cooldown_high: int = 60,
        cooldown_medium: int = 300,
        cooldown_low: int = 600,
        registry: "AgentRegistry | None" = None,
        registry_config: Any = None,
        on_notify: Callable[[str], Coroutine[Any, Any, None]] | None = None,
        on_spawn: Callable[[str, str], Coroutine[Any, Any, str]] | None = None,
    ):
        self.workspace = workspace
        self.on_heartbeat = on_heartbeat
        self.interval_s = interval_s
        self.enabled = enabled
        self._running = False
        self._task: asyncio.Task | None = None

        # Daemon mode params
        self._triage_provider = triage_provider
        self._triage_model = triage_model
        self._execution_model = execution_model
        self._strategy_file = strategy_file
        self._max_iterations = max_iterations
        self._daemon_mode = triage_provider is not None

        # Per-priority cooldowns
        self._cooldowns = {
            "high": cooldown_high,
            "medium": cooldown_medium,
            "low": cooldown_low,
        }
        self._last_action_by_priority: dict[str, float] = {
            "high": 0.0,
            "medium": 0.0,
            "low": 0.0,
        }

        # Dynamic interval: shorter during active conversations
        self._base_interval = interval_s
        self._active_interval = max(60, interval_s // 6)
        self._last_user_activity: float = 0.0

        # Registry params
        self._registry = registry
        self._registry_config = registry_config
        self._on_notify = on_notify
        self._on_spawn = on_spawn
        self._monitor_task: asyncio.Task | None = None

        # Content-hash tracking: skip triage when nothing changed since last no-action
        self._last_noaction_hash: str | None = None

        # Ensure strategy file exists for new/existing deployments
        self._ensure_strategy_file()

    @property
    def heartbeat_file(self) -> Path:
        return self.workspace / self._strategy_file

    def _read_heartbeat_file(self) -> str | None:
        """Read strategy file content."""
        if self.heartbeat_file.exists():
            try:
                return self.heartbeat_file.read_text(encoding="utf-8")
            except Exception:
                return None
        return None

    def _ensure_strategy_file(self) -> None:
        """Create the default strategy file if it doesn't exist."""
        if self.heartbeat_file.exists():
            return
        try:
            self.heartbeat_file.parent.mkdir(parents=True, exist_ok=True)
            self.heartbeat_file.write_text(DEFAULT_STRATEGY_CONTENT, encoding="utf-8")
            logger.info(f"Created default strategy file: {self.heartbeat_file}")
        except Exception as e:
            logger.warning(f"Failed to create strategy file: {e}")

    async def start(self) -> None:
        """Start the heartbeat service."""
        if not self.enabled:
            logger.info("Heartbeat disabled")
            return

        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        mode = "daemon" if self._daemon_mode else "legacy"
        logger.info(f"Heartbeat started (every {self.interval_s}s, mode={mode})")

        # Start registry monitor if registry is enabled
        if self._registry and self._registry_config:
            interval = getattr(self._registry_config, "monitor_interval", 30)
            self._monitor_task = asyncio.create_task(self._run_monitor_loop(interval))
            logger.info(f"Registry monitor started (every {interval}s)")

    def stop(self) -> None:
        """Stop the heartbeat service."""
        self._running = False
        if self._task:
            self._task.cancel()
            self._task = None
        if self._monitor_task:
            self._monitor_task.cancel()
            self._monitor_task = None

    def notify_user_activity(self) -> None:
        """Called by AgentLoop when a user message is processed."""
        self._last_user_activity = time.time()

    @property
    def _current_interval(self) -> int:
        """Shorter interval during active conversation periods."""
        if time.time() - self._last_user_activity < 600:
            return self._active_interval
        return self._base_interval

    async def _run_loop(self) -> None:
        """Main heartbeat loop with dynamic intervals."""
        while self._running:
            try:
                await asyncio.sleep(self._current_interval)
                if self._running:
                    await self._tick()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")

    async def _tick(self) -> None:
        """Execute a single heartbeat tick."""
        if self._daemon_mode:
            await self._daemon_tick()
            return

        # Legacy path
        content = self._read_heartbeat_file()

        if _is_heartbeat_empty(content):
            logger.debug("Heartbeat: no tasks (strategy file empty)")
            return

        logger.info("Heartbeat: checking for tasks...")

        if self.on_heartbeat:
            try:
                response = await self.on_heartbeat(HEARTBEAT_PROMPT)

                if HEARTBEAT_OK_TOKEN.replace("_", "") in response.upper().replace("_", ""):
                    logger.info("Heartbeat: OK (no action needed)")
                else:
                    logger.info("Heartbeat: completed task")

            except Exception as e:
                logger.error(f"Heartbeat execution failed: {e}")

    # ------------------------------------------------------------------
    # Daemon three-tier pipeline
    # ------------------------------------------------------------------

    async def _gather_context(self) -> dict[str, Any]:
        """Tier 0: Gather context from strategy file, git, and tmux."""
        strategy_content: str | None = None
        git_status: str | None = None
        tmux_sessions: str | None = None

        # Read strategy file (treat template-only content as empty)
        try:
            sf = self.workspace / self._strategy_file
            if sf.exists():
                raw = sf.read_text(encoding="utf-8")
                strategy_content = None if _is_heartbeat_empty(raw) else raw
        except Exception as e:
            logger.warning(f"Tier 0: failed to read strategy file: {e}")

        # Git status
        try:
            proc = await asyncio.create_subprocess_exec(
                "git",
                "status",
                "--porcelain",
                "--branch",
                cwd=str(self.workspace),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await proc.communicate()
            if proc.returncode == 0:
                git_status = stdout.decode("utf-8").strip()
        except Exception as e:
            logger.warning(f"Tier 0: git status failed: {e}")

        # Tmux sessions
        try:
            socket_path = Path(tempfile.gettempdir()) / "nanobot-tmux-sockets" / "nanobot.sock"
            if socket_path.exists():
                proc = await asyncio.create_subprocess_exec(
                    "tmux",
                    "-S",
                    str(socket_path),
                    "list-sessions",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, _ = await proc.communicate()
                if proc.returncode == 0:
                    tmux_sessions = stdout.decode("utf-8").strip()
        except Exception as e:
            logger.warning(f"Tier 0: tmux check failed: {e}")

        # Check for overdue follow-ups
        overdue_followups: list[str] = []
        try:
            followup_db = Path.home() / ".nanobot" / "data" / "followups.db"
            if followup_db.exists():
                import sqlite3

                db = sqlite3.connect(str(followup_db))
                now_ms = int(time.time() * 1000)
                rows = db.execute(
                    "SELECT description, deadline_ms FROM followups "
                    "WHERE status = 'pending' "
                    "AND deadline_ms IS NOT NULL "
                    "AND deadline_ms <= ?",
                    (now_ms,),
                ).fetchall()
                db.close()
                overdue_followups = [r[0] for r in rows]
        except Exception as e:
            logger.warning(f"Tier 0: follow-up check failed: {e}")

        # Check TOOLS.md for recorded tool failures/lessons
        tools_lessons: str | None = None
        try:
            tools_md = self.workspace / "TOOLS.md"
            if tools_md.exists():
                content = tools_md.read_text(encoding="utf-8")
                markers = [
                    "FAILURE",
                    "LESSON",
                    "ERROR",
                    "MISTAKE",
                    "BUG",
                    "FIX NEEDED",
                ]
                if any(m in content.upper() for m in markers):
                    tools_lessons = content[-2000:]
        except Exception as e:
            logger.warning(f"Tier 0: TOOLS.md check failed: {e}")

        has_signals = bool(strategy_content) or bool(overdue_followups) or bool(tools_lessons)

        return {
            "strategy_content": strategy_content,
            "git_status": git_status,
            "tmux_sessions": tmux_sessions,
            "overdue_followups": overdue_followups,
            "tools_lessons": tools_lessons,
            "has_signals": has_signals,
        }

    async def _triage(self, context: dict[str, Any]) -> dict[str, Any]:
        """Tier 1: Triage context with a cheap LLM call."""
        try:
            overdue = context.get("overdue_followups", [])
            overdue_str = "\n".join(f"- {f}" for f in overdue) if overdue else "None."
            prompt = TRIAGE_PROMPT_TEMPLATE.format(
                strategy_file=self._strategy_file,
                strategy_content=(context["strategy_content"] or "No strategy file found."),
                git_status=(context["git_status"] or "Not a git repository."),
                tmux_sessions=(context["tmux_sessions"] or "No active sessions."),
                overdue_followups=overdue_str,
                tools_lessons=(context.get("tools_lessons") or "None."),
            )

            assert self._triage_provider is not None
            response = await self._triage_provider.chat(
                messages=[{"role": "user", "content": prompt}],
                model=self._triage_model,
                temperature=0.0,
                max_tokens=256,
            )

            raw = (response.content or "").strip()
            # Strip markdown code fences if present
            if raw.startswith("```"):
                lines = raw.split("\n")
                raw = "\n".join(lines[1:-1]) if len(lines) > 2 else raw

            result = json.loads(raw)
            return {
                "act": bool(result.get("act", False)),
                "reason": str(result.get("reason", "")),
                "priority": str(result.get("priority", "low")),
                "complexity": str(result.get("complexity", "simple")),
            }
        except Exception as e:
            logger.warning(f"Tier 1: triage failed: {e}")
            return {"act": False, "reason": "parse error"}

    async def _execute_daemon_action(
        self,
        context: dict[str, Any],
        triage: dict[str, Any],
    ) -> None:
        """Tier 2: Execute action via the main agent loop or spawn subagent."""
        priority = triage.get("priority", "low")
        cooldown = self._cooldowns.get(priority, 600)
        last_action = self._last_action_by_priority.get(priority, 0)
        elapsed = time.time() - last_action

        if elapsed < cooldown:
            # High priority can bypass if 60s since ANY action
            if priority == "high":
                min_elapsed = min(time.time() - t for t in self._last_action_by_priority.values())
                if min_elapsed >= self._cooldowns["high"]:
                    logger.info("Tier 2: high-priority task bypassing lower-priority cooldown")
                else:
                    remaining = int(self._cooldowns["high"] - min_elapsed)
                    logger.info(f"Tier 2: high-priority cooldown, {remaining}s remaining")
                    return
            else:
                remaining = int(cooldown - elapsed)
                logger.info(f"Tier 2: {priority}-priority cooldown active, {remaining}s remaining")
                return

        complexity = triage.get("complexity", "simple")
        reason = triage.get("reason", "")

        # Hybrid dispatch: complex tasks go to subagent via registry
        if complexity == "complex" and self._registry and self._on_spawn:
            try:
                # Dedup: skip if a pending task with similar description already exists
                from nanobot.registry.store import TaskState

                existing = await self._registry.list_tasks(state=TaskState.PENDING)
                for t in existing:
                    if reason and reason[:40] in t.get("description", ""):
                        logger.info(f"Tier 2: skipping duplicate task, existing={t.get('task_id')}")
                        return

                task_id = str(uuid.uuid4())[:8]
                await self._registry.create_task(
                    task_id=task_id,
                    description=reason,
                    priority=triage.get("priority", "medium"),
                    complexity="complex",
                )
                await self._on_spawn(reason, task_id)
                self._last_action_by_priority[priority] = time.time()
                logger.info(f"Tier 2: spawned subagent for complex task {task_id}")
                return
            except Exception as e:
                logger.warning(f"Tier 2: spawn failed, falling back to direct: {e}")

        # Simple tasks or fallback: direct execution
        prompt = EXECUTION_PROMPT_TEMPLATE.format(
            priority=triage.get("priority", "low"),
            reason=reason,
            strategy_content=(context["strategy_content"] or "No strategy file."),
            git_status=context["git_status"] or "N/A",
            tmux_sessions=context["tmux_sessions"] or "No active sessions.",
        )

        try:
            if self.on_heartbeat:
                await self.on_heartbeat(prompt)
            self._last_action_by_priority[priority] = time.time()
            logger.info("Tier 2: daemon action executed")
        except Exception as e:
            logger.warning(f"Tier 2: execution failed: {e}")

    async def _daemon_tick(self) -> None:
        """Orchestrator: run the three-tier daemon pipeline."""
        # Tier 0: gather context
        try:
            context = await self._gather_context()
        except Exception as e:
            logger.warning(f"Daemon tick: context gathering failed: {e}")
            return

        if not context["has_signals"]:
            logger.debug("Daemon tick: no signals detected")
            return

        # Content-hash check: skip triage if nothing changed since last no-action
        ctx_hash = hashlib.md5(
            json.dumps(context, sort_keys=True, default=str).encode()
        ).hexdigest()
        if ctx_hash == self._last_noaction_hash:
            logger.debug("Daemon tick: context unchanged since last no-action, skipping triage")
            return

        # Tier 1: triage
        try:
            triage = await self._triage(context)
        except Exception as e:
            logger.warning(f"Daemon tick: triage failed: {e}")
            return

        if not triage.get("act"):
            self._last_noaction_hash = ctx_hash
            logger.info(f"Daemon tick: no action needed - {triage.get('reason')}")
            return

        # Clear hash so next tick re-evaluates after action
        self._last_noaction_hash = None

        # Tier 2: execute
        try:
            await self._execute_daemon_action(context, triage)
        except Exception as e:
            logger.warning(f"Daemon tick: execution failed: {e}")

    async def trigger_now(self) -> str | None:
        """Manually trigger a heartbeat."""
        if self.on_heartbeat:
            return await self.on_heartbeat(HEARTBEAT_PROMPT)
        return None

    # ------------------------------------------------------------------
    # Registry monitor loop
    # ------------------------------------------------------------------

    async def _run_monitor_loop(self, interval: int = 30) -> None:
        """Periodic background loop for registry health monitoring."""
        while self._running:
            try:
                await asyncio.sleep(interval)
                if self._running and self._registry:
                    await self._monitor_registry()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Registry monitor error: {e}")

    async def _monitor_registry(self) -> None:
        """Run registry health checks: stale agents, PoW verification."""
        if not self._registry:
            return

        # Mark stale agents as FAILED
        threshold = 180
        if self._registry_config:
            threshold = getattr(self._registry_config, "stale_threshold", 180)

        stale_ids = await self._registry.mark_stale_agents(threshold)

        # Requeue tasks from stale agents
        if stale_ids:
            from nanobot.registry.store import TaskState

            for agent in stale_ids:
                agent_data = await self._registry.get_agent(agent)
                if agent_data and agent_data.get("task_id"):
                    task_id = agent_data["task_id"]
                    task = await self._registry.get_task(task_id)
                    if task and task["state"] not in (
                        TaskState.COMPLETED.value,
                        TaskState.PENDING.value,
                    ):
                        try:
                            await self._registry.update_task_state(
                                task_id,
                                TaskState.FAILED,
                                reason=f"agent {agent} went stale",
                            )
                        except ValueError:
                            pass

        # Verify proof of work for tasks in VERIFYING state
        await self._verify_proof_of_work()

    async def _verify_proof_of_work(self) -> None:
        """Verify PoW for tasks in VERIFYING state."""
        if not self._registry:
            return

        from nanobot.registry.proof import ProofOfWork, ProofVerifier
        from nanobot.registry.store import TaskState

        tasks = await self._registry.get_verifying_tasks()
        if not tasks:
            return

        verifier = ProofVerifier(self.workspace)

        for task in tasks:
            task_id = task["task_id"]
            proof_data = task.get("proof_of_work")

            if not proof_data:
                # No proof submitted yet, mark as failed
                try:
                    await self._registry.update_task_state(
                        task_id, TaskState.FAILED, reason="no proof submitted"
                    )
                except ValueError:
                    pass
                await self._notify_human(task, "Task verification failed: no proof submitted")
                continue

            try:
                proof = ProofOfWork.from_json(
                    json.dumps(proof_data) if isinstance(proof_data, dict) else proof_data
                )
                result = await verifier.verify(proof)

                if result["valid"]:
                    await self._registry.update_task_state(
                        task_id,
                        TaskState.COMPLETED,
                        reason=f"verified {result['verified_count']} proofs",
                    )
                    desc = task.get("description", "")[:100]
                    await self._notify_human(task, f"Task completed and verified: {desc}")
                else:
                    details = "; ".join(
                        f.get("error", "unknown") for f in result.get("failed_items", [])
                    )
                    await self._registry.update_task_state(
                        task_id,
                        TaskState.FAILED,
                        reason=f"verification failed: {details}",
                    )
                    await self._notify_human(task, f"Task verification failed: {details}")
            except Exception as e:
                logger.warning(f"PoW verification error for task {task_id}: {e}")

    async def _notify_human(self, task: dict[str, Any], message: str) -> None:
        """Send notification via the on_notify callback."""
        if self._on_notify:
            try:
                full_msg = f"[Agent Registry] {message}"
                await self._on_notify(full_msg)
            except Exception as e:
                logger.warning(f"Notification failed: {e}")
