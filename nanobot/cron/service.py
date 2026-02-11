"""Cron service backed by APScheduler with SQLite persistence."""

import asyncio
import json
import sqlite3
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Coroutine

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.date import DateTrigger
from apscheduler.triggers.interval import IntervalTrigger
from loguru import logger

from nanobot.cron.types import (
    CronJob,
    CronJobState,
    CronPayload,
    CronSchedule,
)


def _now_ms() -> int:
    return int(time.time() * 1000)


class CronService:
    """Service for managing and executing scheduled jobs.

    Uses APScheduler with SQLite for persistent, reliable scheduling.
    Metadata (name, payload, state history) stored in a separate SQLite table.
    """

    def __init__(
        self,
        store_path: Path,
        on_job: (Callable[[CronJob], Coroutine[Any, Any, str | None]] | None) = None,
    ):
        self.store_path = store_path  # Legacy JSON path (for migration)
        self.on_job = on_job
        self._running = False
        self._missed_jobs: list[str] = []
        self._executing: set[str] = set()  # job IDs currently running

        # SQLite paths
        data_dir = store_path.parent
        self._db_path = data_dir / "cron.db"

        # APScheduler
        self._scheduler: AsyncIOScheduler | None = None
        self._db: sqlite3.Connection | None = None

    def _init_db(self) -> None:
        """Initialize metadata SQLite database."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(self._db_path))
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("""
            CREATE TABLE IF NOT EXISTS cron_jobs (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                enabled INTEGER DEFAULT 1,
                schedule_json TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                last_run_at_ms INTEGER,
                last_status TEXT,
                last_error TEXT,
                created_at_ms INTEGER,
                updated_at_ms INTEGER,
                delete_after_run INTEGER DEFAULT 0,
                max_retries INTEGER DEFAULT 3,
                retry_count INTEGER DEFAULT 0
            )
        """)
        self._db.commit()

    def _init_scheduler(self) -> None:
        """Initialize APScheduler with in-memory job store.

        Our cron_jobs SQLite table is the source of truth. _sync_scheduler()
        rebuilds all APScheduler jobs from it on every startup, so a persistent
        job store is redundant and causes pickle errors with closures.
        """
        job_defaults = {
            "coalesce": True,
            "max_instances": 1,
            "misfire_grace_time": 3600,
        }
        self._scheduler = AsyncIOScheduler(
            job_defaults=job_defaults,
        )

    async def start(self) -> None:
        """Start the cron service."""
        self._running = True
        self._init_db()
        self._init_scheduler()

        # Migrate from legacy JSON if needed
        await self._maybe_migrate_json()

        # Re-register all enabled jobs with APScheduler
        self._sync_scheduler()

        self._scheduler.start()
        job_count = self._count_jobs()
        logger.info(f"Cron service started with {job_count} jobs (APScheduler + SQLite)")

    async def execute_missed(self) -> None:
        """Execute any 'at' jobs that were missed during downtime.

        Only runs jobs whose last_status is None (never executed).
        """
        for job_id in self._missed_jobs:
            job = self._load_job_metadata(job_id)
            if job and job.enabled and job.state.last_status is None:
                logger.info(f"Cron: executing missed job '{job.name}' ({job_id})")
                await self._execute_job(job)
        self._missed_jobs.clear()

    def stop(self) -> None:
        """Stop the cron service."""
        self._running = False
        if self._scheduler:
            self._scheduler.shutdown(wait=False)
        if self._db:
            self._db.close()

    async def _maybe_migrate_json(self) -> None:
        """Migrate from legacy jobs.json to SQLite on first run."""
        if not self.store_path.exists():
            return
        if self._count_jobs() > 0:
            return  # Already migrated

        logger.info("Migrating cron jobs from JSON to SQLite...")
        try:
            data = json.loads(self.store_path.read_text(encoding="utf-8"))
            for j in data.get("jobs", []):
                schedule = CronSchedule(
                    kind=j["schedule"]["kind"],
                    at_ms=j["schedule"].get("atMs"),
                    every_ms=j["schedule"].get("everyMs"),
                    expr=j["schedule"].get("expr"),
                    tz=j["schedule"].get("tz"),
                )
                payload = CronPayload(
                    kind=j["payload"].get("kind", "agent_turn"),
                    message=j["payload"].get("message", ""),
                    deliver=j["payload"].get("deliver", False),
                    channel=j["payload"].get("channel"),
                    to=j["payload"].get("to"),
                )
                job = CronJob(
                    id=j["id"],
                    name=j["name"],
                    enabled=j.get("enabled", True),
                    schedule=schedule,
                    payload=payload,
                    state=CronJobState(
                        last_run_at_ms=j.get("state", {}).get("lastRunAtMs"),
                        last_status=j.get("state", {}).get("lastStatus"),
                        last_error=j.get("state", {}).get("lastError"),
                    ),
                    created_at_ms=j.get("createdAtMs", 0),
                    updated_at_ms=j.get("updatedAtMs", 0),
                    delete_after_run=j.get("deleteAfterRun", False),
                )
                self._save_job_metadata(job)

            migrated = self.store_path.with_suffix(".json.migrated")
            self.store_path.rename(migrated)
            logger.info(f"Migrated {len(data.get('jobs', []))} jobs. Old file: {migrated}")
        except Exception as e:
            logger.error(f"Migration failed: {e}")

    def _sync_scheduler(self) -> None:
        """Register all enabled jobs with APScheduler.

        Past "at" jobs that never executed are collected into _missed_jobs
        for later recovery via execute_missed().
        """
        assert self._db and self._scheduler
        self._missed_jobs.clear()
        cursor = self._db.execute(
            "SELECT id, schedule_json, last_status FROM cron_jobs WHERE enabled = 1"
        )
        for row in cursor.fetchall():
            job_id, schedule_json, last_status = row
            schedule = self._parse_schedule(json.loads(schedule_json))
            trigger = self._make_trigger(schedule)
            if trigger:
                existing = self._scheduler.get_job(job_id)
                if existing:
                    existing.reschedule(trigger)
                else:
                    self._scheduler.add_job(
                        self._execute_wrapper,
                        trigger=trigger,
                        id=job_id,
                        args=[job_id],
                        replace_existing=True,
                    )
            elif schedule.kind == "at" and schedule.at_ms and last_status is None:
                # Past "at" job that never ran -- queue for recovery
                self._missed_jobs.append(job_id)

    def _make_trigger(
        self, schedule: CronSchedule
    ) -> DateTrigger | IntervalTrigger | CronTrigger | None:
        """Convert CronSchedule to APScheduler trigger."""
        if schedule.kind == "at" and schedule.at_ms:
            run_date = datetime.fromtimestamp(schedule.at_ms / 1000, tz=timezone.utc)
            if run_date <= datetime.now(tz=timezone.utc):
                return None  # Already past
            return DateTrigger(run_date=run_date)

        if schedule.kind == "every" and schedule.every_ms:
            return IntervalTrigger(seconds=schedule.every_ms / 1000)

        if schedule.kind == "cron" and schedule.expr:
            try:
                parts = schedule.expr.strip().split()
                if len(parts) == 5:
                    return CronTrigger(
                        minute=parts[0],
                        hour=parts[1],
                        day=parts[2],
                        month=parts[3],
                        day_of_week=parts[4],
                        timezone=schedule.tz or "UTC",
                    )
            except Exception as e:
                logger.error(f"Invalid cron expression '{schedule.expr}': {e}")

        return None

    def _get_next_run_ms(self, job_id: str) -> int | None:
        """Get next fire time for a job from APScheduler."""
        if not self._scheduler:
            return None
        aps_job = self._scheduler.get_job(job_id)
        if aps_job and aps_job.next_run_time:
            return int(aps_job.next_run_time.timestamp() * 1000)
        return None

    async def _execute_wrapper(self, job_id: str) -> None:
        """APScheduler callback -- loads job metadata and executes."""
        job = self._load_job_metadata(job_id)
        if not job:
            logger.warning(f"Cron: job {job_id} not found in metadata store")
            return
        await self._execute_job(job)

    async def _execute_job(self, job: CronJob) -> None:
        """Execute a single job with retry logic."""
        start_ms = _now_ms()
        self._executing.add(job.id)
        logger.info(f"Cron: executing job '{job.name}' ({job.id})")

        try:
            if self.on_job:
                await asyncio.wait_for(self.on_job(job), timeout=300)

            job.state.last_status = "ok"
            job.state.last_error = None
            job.retry_count = 0
            logger.info(f"Cron: job '{job.name}' completed")

        except asyncio.TimeoutError:
            job.state.last_status = "error"
            job.state.last_error = "execution timed out (300s)"
            logger.error(f"Cron: job '{job.name}' timed out")
            self._maybe_retry(job)

        except Exception as e:
            job.state.last_status = "error"
            job.state.last_error = str(e)[:500]
            logger.error(f"Cron: job '{job.name}' failed: {e}")
            self._maybe_retry(job)

        finally:
            self._executing.discard(job.id)

        job.state.last_run_at_ms = start_ms

        # Handle one-shot jobs -- only delete/disable on SUCCESS
        if job.schedule.kind == "at" and job.state.last_status == "ok":
            if job.delete_after_run:
                self._delete_job_metadata(job.id)
                if self._scheduler:
                    try:
                        self._scheduler.remove_job(job.id, jobstore="default")
                    except Exception:
                        pass
                return
            else:
                job.enabled = False
                job.state.next_run_at_ms = None
                if self._scheduler:
                    try:
                        self._scheduler.remove_job(job.id, jobstore="default")
                    except Exception:
                        pass

        self._save_job_metadata(job)

    def _maybe_retry(self, job: CronJob) -> None:
        """Schedule a retry for a failed job if retries remain."""
        if job.retry_count >= job.max_retries:
            logger.warning(f"Cron: job '{job.name}' exhausted {job.max_retries} retries, disabling")
            job.enabled = False
            if self._scheduler:
                try:
                    self._scheduler.remove_job(job.id, jobstore="default")
                except Exception:
                    pass
            return

        job.retry_count += 1
        delay_s = 60 * (2 ** (job.retry_count - 1))
        retry_time = datetime.fromtimestamp(time.time() + delay_s, tz=timezone.utc)
        logger.info(
            f"Cron: scheduling retry {job.retry_count}/"
            f"{job.max_retries} for '{job.name}' in {delay_s}s"
        )

        if self._scheduler:
            self._scheduler.add_job(
                self._execute_wrapper,
                trigger=DateTrigger(run_date=retry_time),
                id=f"{job.id}_retry_{job.retry_count}",
                args=[job.id],
                replace_existing=True,
            )

    # ========== Metadata Store (SQLite) ==========

    def _save_job_metadata(self, job: CronJob) -> None:
        """Upsert job metadata to SQLite."""
        assert self._db
        schedule_json = json.dumps(
            {
                "kind": job.schedule.kind,
                "atMs": job.schedule.at_ms,
                "everyMs": job.schedule.every_ms,
                "expr": job.schedule.expr,
                "tz": job.schedule.tz,
            }
        )
        payload_json = json.dumps(
            {
                "kind": job.payload.kind,
                "message": job.payload.message,
                "deliver": job.payload.deliver,
                "channel": job.payload.channel,
                "to": job.payload.to,
            }
        )
        self._db.execute(
            """
            INSERT OR REPLACE INTO cron_jobs
            (id, name, enabled, schedule_json, payload_json,
             last_run_at_ms, last_status, last_error,
             created_at_ms, updated_at_ms, delete_after_run,
             max_retries, retry_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                job.id,
                job.name,
                int(job.enabled),
                schedule_json,
                payload_json,
                job.state.last_run_at_ms,
                job.state.last_status,
                job.state.last_error,
                job.created_at_ms,
                _now_ms(),
                int(job.delete_after_run),
                job.max_retries,
                job.retry_count,
            ),
        )
        self._db.commit()

    def _load_job_metadata(self, job_id: str) -> CronJob | None:
        """Load a single job from SQLite."""
        assert self._db
        row = self._db.execute("SELECT * FROM cron_jobs WHERE id = ?", (job_id,)).fetchone()
        if not row:
            return None
        return self._row_to_job(row)

    def _delete_job_metadata(self, job_id: str) -> None:
        """Delete job from SQLite."""
        assert self._db
        self._db.execute("DELETE FROM cron_jobs WHERE id = ?", (job_id,))
        self._db.commit()

    def _count_jobs(self) -> int:
        """Count total jobs in metadata store."""
        if not self._db:
            return 0
        row = self._db.execute("SELECT COUNT(*) FROM cron_jobs").fetchone()
        return row[0] if row else 0

    def _row_to_job(self, row: tuple) -> CronJob:
        """Convert SQLite row to CronJob dataclass."""
        (
            job_id,
            name,
            enabled,
            schedule_json,
            payload_json,
            last_run_at_ms,
            last_status,
            last_error,
            created_at_ms,
            updated_at_ms,
            delete_after_run,
            max_retries,
            retry_count,
        ) = row

        sched = json.loads(schedule_json)
        pay = json.loads(payload_json)

        job = CronJob(
            id=job_id,
            name=name,
            enabled=bool(enabled),
            schedule=CronSchedule(
                kind=sched["kind"],
                at_ms=sched.get("atMs"),
                every_ms=sched.get("everyMs"),
                expr=sched.get("expr"),
                tz=sched.get("tz"),
            ),
            payload=CronPayload(
                kind=pay.get("kind", "agent_turn"),
                message=pay.get("message", ""),
                deliver=pay.get("deliver", False),
                channel=pay.get("channel"),
                to=pay.get("to"),
            ),
            state=CronJobState(
                next_run_at_ms=self._get_next_run_ms(job_id),
                last_run_at_ms=last_run_at_ms,
                last_status=last_status,
                last_error=last_error,
            ),
            created_at_ms=created_at_ms or 0,
            updated_at_ms=updated_at_ms or 0,
            delete_after_run=bool(delete_after_run),
            max_retries=max_retries or 3,
            retry_count=retry_count or 0,
        )
        return job

    @staticmethod
    def _parse_schedule(data: dict) -> CronSchedule:
        return CronSchedule(
            kind=data["kind"],
            at_ms=data.get("atMs"),
            every_ms=data.get("everyMs"),
            expr=data.get("expr"),
            tz=data.get("tz"),
        )

    # ========== Public API ==========

    def _ensure_db(self) -> None:
        """Lazily initialize the DB if not yet started."""
        if self._db is None:
            self._init_db()

    def list_jobs(self, include_disabled: bool = False) -> list[CronJob]:
        """List all jobs."""
        self._ensure_db()
        assert self._db
        query = "SELECT * FROM cron_jobs"
        if not include_disabled:
            query += " WHERE enabled = 1"
        query += " ORDER BY created_at_ms"
        rows = self._db.execute(query).fetchall()
        return [self._row_to_job(r) for r in rows]

    def add_job(
        self,
        name: str,
        schedule: CronSchedule,
        message: str,
        deliver: bool = False,
        channel: str | None = None,
        to: str | None = None,
        delete_after_run: bool = False,
    ) -> CronJob:
        """Add a new job. Skips creation if a job with the same name already exists."""
        self._ensure_db()

        # Dedup: return existing job if name matches (case-insensitive)
        existing = self.list_jobs(include_disabled=False)
        for j in existing:
            if j.name.lower() == name.lower():
                logger.warning(f"Cron: job '{name}' already exists ({j.id}), skipping")
                return j

        now = _now_ms()
        job = CronJob(
            id=str(uuid.uuid4())[:8],
            name=name,
            enabled=True,
            schedule=schedule,
            payload=CronPayload(
                kind="agent_turn",
                message=message,
                deliver=deliver,
                channel=channel,
                to=to,
            ),
            state=CronJobState(),
            created_at_ms=now,
            updated_at_ms=now,
            delete_after_run=delete_after_run,
        )

        self._save_job_metadata(job)

        # Register with APScheduler
        trigger = self._make_trigger(schedule)
        if trigger and self._scheduler:
            self._scheduler.add_job(
                self._execute_wrapper,
                trigger=trigger,
                id=job.id,
                args=[job.id],
                replace_existing=True,
            )

        # Populate next_run from scheduler
        job.state.next_run_at_ms = self._get_next_run_ms(job.id)

        logger.info(f"Cron: added job '{name}' ({job.id})")
        return job

    def remove_job(self, job_id: str) -> CronJob | None:
        """Remove a job by ID."""
        self._ensure_db()
        job = self._load_job_metadata(job_id)
        if not job:
            return None

        self._delete_job_metadata(job_id)
        if self._scheduler:
            try:
                self._scheduler.remove_job(job_id, jobstore="default")
            except Exception:
                pass

        logger.info(f"Cron: removed job {job_id}")
        return job

    def enable_job(self, job_id: str, enabled: bool = True) -> CronJob | None:
        """Enable or disable a job."""
        self._ensure_db()
        job = self._load_job_metadata(job_id)
        if not job:
            return None

        job.enabled = enabled
        job.updated_at_ms = _now_ms()
        self._save_job_metadata(job)

        if self._scheduler:
            if enabled:
                trigger = self._make_trigger(job.schedule)
                if trigger:
                    self._scheduler.add_job(
                        self._execute_wrapper,
                        trigger=trigger,
                        id=job.id,
                        args=[job.id],
                        replace_existing=True,
                    )
                job.state.next_run_at_ms = self._get_next_run_ms(job.id)
            else:
                try:
                    self._scheduler.remove_job(job.id, jobstore="default")
                except Exception:
                    pass
                job.state.next_run_at_ms = None

        return job

    async def run_job(self, job_id: str, force: bool = False) -> bool:
        """Manually run a job (blocking -- awaits completion)."""
        self._ensure_db()
        job = self._load_job_metadata(job_id)
        if not job:
            return False
        if not force and not job.enabled:
            return False
        await self._execute_job(job)
        return True

    def run_job_async(self, job_id: str) -> bool:
        """Fire-and-forget job execution. Returns True if job was found and started."""
        self._ensure_db()
        if job_id in self._executing:
            return True  # Already running
        job = self._load_job_metadata(job_id)
        if not job:
            return False
        asyncio.create_task(self._execute_job(job))
        return True

    def is_running(self, job_id: str) -> bool:
        """Check if a job is currently executing."""
        return job_id in self._executing

    def status(self) -> dict:
        """Get service status."""
        self._ensure_db()
        return {
            "enabled": self._running,
            "jobs": self._count_jobs(),
            "scheduler_running": bool(self._scheduler and self._scheduler.running),
        }
