"""Cron tool for agent self-scheduling."""

import time
from datetime import datetime
from typing import TYPE_CHECKING, Any
from zoneinfo import ZoneInfo

from loguru import logger

from nanobot.agent.tools.base import Tool

if TYPE_CHECKING:
    from nanobot.cron.service import CronService


class CronTool(Tool):
    """
    Tool to create, manage, and execute scheduled tasks.

    The agent can use this to schedule reminders, recurring checks,
    and other time-based tasks.
    """

    MAX_JOBS = 50
    MIN_INTERVAL_SECONDS = 60

    def __init__(
        self,
        cron_service: "CronService",
        vector_store: Any = None,
        default_timezone: str = "UTC",
    ):
        self._service = cron_service
        self._vector_store = vector_store
        self._default_timezone = default_timezone
        self._context_channel: str | None = None
        self._context_chat_id: str | None = None

    def set_context(self, channel: str, chat_id: str) -> None:
        """Set the current conversation context for default delivery target."""
        self._context_channel = channel
        self._context_chat_id = chat_id

    @property
    def name(self) -> str:
        return "cron"

    @property
    def description(self) -> str:
        return (
            "Manage scheduled tasks. Actions: list (show jobs), add (create job), "
            "remove (delete job), enable/disable (toggle job), run (execute immediately). "
            "Use this to schedule reminders, recurring checks, and time-based tasks."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["list", "add", "remove", "enable", "disable", "run"],
                    "description": "Action to perform",
                },
                "job_id": {
                    "type": "string",
                    "description": "Job ID (for remove/enable/disable/run actions)",
                },
                "name": {
                    "type": "string",
                    "description": "Job name (for add action)",
                },
                "message": {
                    "type": "string",
                    "description": "The prompt/message to execute when job runs (for add action)",
                },
                "schedule_type": {
                    "type": "string",
                    "enum": ["at", "every", "cron"],
                    "description": (
                        "Schedule type: 'at' for one-time, 'every' for interval, "
                        "'cron' for cron expression"
                    ),
                },
                "at_time": {
                    "type": "string",
                    "description": "ISO datetime for one-time jobs (e.g., '2025-02-06T09:00:00')",
                },
                "every_seconds": {
                    "type": "integer",
                    "description": "Interval in seconds for recurring jobs (minimum 60)",
                },
                "cron_expr": {
                    "type": "string",
                    "description": "Cron expression (e.g., '0 9 * * *' for 9 AM daily)",
                },
                "timezone": {
                    "type": "string",
                    "description": "Timezone for scheduling (e.g., 'America/New_York')",
                },
                "deliver": {
                    "type": "boolean",
                    "description": (
                        "Whether to send the job's response back to the chat channel. "
                        "MUST be true for reminders, notifications, and any job where "
                        "the user expects to see the result."
                    ),
                },
                "channel": {
                    "type": "string",
                    "description": (
                        "Target channel for delivery (telegram/whatsapp/discord/feishu)"
                    ),
                },
                "to": {
                    "type": "string",
                    "description": "Target chat ID for delivery",
                },
            },
            "required": ["action"],
        }

    async def execute(
        self,
        action: str,
        job_id: str | None = None,
        name: str | None = None,
        message: str | None = None,
        schedule_type: str | None = None,
        at_time: str | None = None,
        every_seconds: int | None = None,
        cron_expr: str | None = None,
        timezone: str | None = None,
        deliver: bool = False,
        channel: str | None = None,
        to: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Execute the cron action."""
        try:
            if action == "list":
                return self._list_jobs()
            elif action == "add":
                return self._add_job(
                    name=name,
                    message=message,
                    schedule_type=schedule_type,
                    at_time=at_time,
                    every_seconds=every_seconds,
                    cron_expr=cron_expr,
                    timezone=timezone,
                    deliver=deliver,
                    channel=channel,
                    to=to,
                )
            elif action == "remove":
                return await self._remove_job(job_id)
            elif action == "enable":
                return self._toggle_job(job_id, enabled=True)
            elif action == "disable":
                return self._toggle_job(job_id, enabled=False)
            elif action == "run":
                return await self._run_job(job_id)
            else:
                return f"Error: Unknown action '{action}'"
        except Exception as e:
            return f"Error: {e}"

    def _list_jobs(self) -> str:
        """List all scheduled jobs."""
        jobs = self._service.list_jobs(include_disabled=True)

        if not jobs:
            return "No scheduled jobs."

        lines = ["Scheduled Jobs:", ""]
        for job in jobs:
            # Format schedule
            if job.schedule.kind == "every":
                sched = f"every {(job.schedule.every_ms or 0) // 1000}s"
            elif job.schedule.kind == "cron":
                sched = f"cron: {job.schedule.expr}"
            else:
                sched = "one-time"

            # Format next run in configured timezone
            next_run = "N/A"
            if job.state.next_run_at_ms:
                tz = ZoneInfo(self._default_timezone)
                next_run = datetime.fromtimestamp(job.state.next_run_at_ms / 1000, tz=tz).strftime(
                    "%Y-%m-%d %H:%M"
                )

            status = "enabled" if job.enabled else "disabled"
            delivery = ""
            if job.payload.deliver and job.payload.to:
                delivery = f" -> {job.payload.channel or '?'}:{job.payload.to}"

            lines.append(f"- [{job.id}] {job.name} ({status})")
            lines.append(f"  Schedule: {sched}")
            lines.append(f"  Next run: {next_run}{delivery}")
            lines.append(f"  Message: {job.payload.message[:50]}...")
            lines.append("")

        return "\n".join(lines)

    def _add_job(
        self,
        name: str | None,
        message: str | None,
        schedule_type: str | None,
        at_time: str | None,
        every_seconds: int | None,
        cron_expr: str | None,
        timezone: str | None,
        deliver: bool,
        channel: str | None,
        to: str | None,
    ) -> str:
        """Add a new scheduled job."""
        from nanobot.cron.types import CronSchedule

        # Validate required fields
        if not name:
            return "Error: 'name' is required for add action"
        if not message:
            return "Error: 'message' is required for add action"
        if not schedule_type:
            return "Error: 'schedule_type' is required (at, every, or cron)"

        # Check job limit
        existing_jobs = self._service.list_jobs(include_disabled=True)
        if len(existing_jobs) >= self.MAX_JOBS:
            return f"Error: Maximum job limit ({self.MAX_JOBS}) reached"

        # Check for duplicate name (case-insensitive)
        name_lower = name.lower()
        for job in existing_jobs:
            if job.name.lower() == name_lower:
                return f"Error: A job named '{job.name}' already exists (ID: {job.id})"

        # Build schedule
        schedule: CronSchedule
        now_ms = int(time.time() * 1000)

        if schedule_type == "at":
            if not at_time:
                return "Error: 'at_time' is required for schedule_type='at'"
            try:
                dt = datetime.fromisoformat(at_time)
                if dt.tzinfo is None:
                    # Interpret naive datetime in user's configured timezone
                    dt = dt.replace(tzinfo=ZoneInfo(timezone or self._default_timezone))
                at_ms = int(dt.timestamp() * 1000)
            except ValueError as e:
                return f"Error: Invalid at_time format: {e}"
            if at_ms <= now_ms:
                return "Error: at_time must be in the future"
            tz_name = timezone or self._default_timezone
            schedule = CronSchedule(kind="at", at_ms=at_ms, tz=tz_name)

        elif schedule_type == "every":
            if not every_seconds:
                return "Error: 'every_seconds' is required for schedule_type='every'"
            if every_seconds < self.MIN_INTERVAL_SECONDS:
                return f"Error: Minimum interval is {self.MIN_INTERVAL_SECONDS} seconds"
            schedule = CronSchedule(
                kind="every", every_ms=every_seconds * 1000, tz=timezone or self._default_timezone
            )

        elif schedule_type == "cron":
            if not cron_expr:
                return "Error: 'cron_expr' is required for schedule_type='cron'"
            # Validate cron expression
            try:
                from croniter import croniter

                croniter(cron_expr)  # Will raise if invalid
            except ImportError:
                return "Error: croniter package not installed for cron expressions"
            except Exception as e:
                return f"Error: Invalid cron expression: {e}"
            schedule = CronSchedule(
                kind="cron", expr=cron_expr, tz=timezone or self._default_timezone
            )

        else:
            return f"Error: Invalid schedule_type '{schedule_type}'"

        # Use context as default delivery target if deliver=True but no explicit target
        if deliver:
            if not channel and self._context_channel:
                channel = self._context_channel
            if not to and self._context_chat_id:
                to = self._context_chat_id

        # Create the job
        job = self._service.add_job(
            name=name,
            schedule=schedule,
            message=message,
            deliver=deliver,
            channel=channel,
            to=to,
            delete_after_run=(schedule_type == "at"),  # Auto-delete one-time jobs
        )

        # Format response
        if schedule_type == "at":
            when = datetime.fromisoformat(at_time).strftime("%Y-%m-%d %H:%M")
            sched_desc = f"at {when}"
        elif schedule_type == "every":
            sched_desc = f"every {every_seconds} seconds"
        else:
            sched_desc = f"cron '{cron_expr}'"

        delivery_info = ""
        if deliver and to:
            delivery_info = f" (will deliver to {channel or 'current channel'}:{to})"

        return f"Created job '{name}' (ID: {job.id}) scheduled {sched_desc}{delivery_info}"

    async def _remove_job(self, job_id: str | None) -> str:
        """Remove a job by ID and clean up associated vector store entries."""
        if not job_id:
            return "Error: 'job_id' is required for remove action"

        job = self._service.remove_job(job_id)
        if not job:
            return f"Error: Job '{job_id}' not found"

        # Clean up vector store entries referencing this job
        if self._vector_store and job.payload.message:
            try:
                await self._vector_store.delete_by_query(job.name, min_similarity=0.75)
                await self._vector_store.delete_by_query(job.payload.message, min_similarity=0.75)
            except Exception as e:
                logger.debug(f"Vector cleanup after cron removal failed: {e}")

        return f"Removed job {job_id}"

    def _toggle_job(self, job_id: str | None, enabled: bool) -> str:
        """Enable or disable a job."""
        if not job_id:
            action = "enable" if enabled else "disable"
            return f"Error: 'job_id' is required for {action} action"

        job = self._service.enable_job(job_id, enabled=enabled)
        if job:
            status = "enabled" if enabled else "disabled"
            return f"Job '{job.name}' ({job_id}) is now {status}"
        return f"Error: Job '{job_id}' not found"

    async def _run_job(self, job_id: str | None) -> str:
        """Run a job immediately."""
        if not job_id:
            return "Error: 'job_id' is required for run action"

        if await self._service.run_job(job_id, force=True):
            return f"Job {job_id} executed"
        return f"Error: Job '{job_id}' not found or failed to execute"
