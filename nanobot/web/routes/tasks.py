"""Tasks dashboard and notification API routes."""

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

router = APIRouter(tags=["tasks"])


def _format_ts(epoch_ms: int, tz_name: str = "UTC") -> str:
    """Format an epoch-ms timestamp in the configured timezone."""
    dt = datetime.fromtimestamp(epoch_ms / 1000, tz=timezone.utc)
    local_dt = dt.astimezone(ZoneInfo(tz_name))
    return local_dt.strftime("%Y-%m-%d %H:%M")


def _format_schedule(job, tz_name: str = "UTC") -> str:
    """Return a human-readable schedule description."""
    sched = job.schedule
    if sched.kind == "every" and sched.every_ms:
        total_s = sched.every_ms // 1000
        if total_s >= 3600:
            hours = total_s // 3600
            return f"Every {hours} hour{'s' if hours != 1 else ''}"
        minutes = total_s // 60
        return f"Every {minutes} minute{'s' if minutes != 1 else ''}"
    if sched.kind == "cron" and sched.expr:
        parts = sched.expr.strip().split()
        if len(parts) == 5:
            minute, hour, dom, month, dow = parts
            # Convert cron hour/minute from UTC to local timezone for display
            try:
                utc_hour = int(hour)
                utc_min = int(minute)
                utc_dt = datetime.now(tz=timezone.utc).replace(
                    hour=utc_hour,
                    minute=utc_min,
                    second=0,
                    microsecond=0,
                )
                local_dt = utc_dt.astimezone(ZoneInfo(tz_name))
                local_hour = f"{local_dt.hour:02d}"
                local_min = f"{local_dt.minute:02d}"
            except (ValueError, KeyError):
                local_hour = hour.zfill(2)
                local_min = minute.zfill(2)
            # Common patterns
            if dom == "*" and month == "*" and dow == "*":
                return f"Daily at {local_hour}:{local_min}"
            if dom == "*" and month == "*" and dow == "1":
                return f"Mondays at {local_hour}:{local_min}"
            if minute.startswith("*/") or hour.startswith("*/"):
                return f"Cron: {sched.expr}"
            if hour.startswith("*/"):
                return f"Every {hour[2:]} hours"
        return f"Cron: {sched.expr}"
    if sched.kind == "at" and sched.at_ms:
        return f"Once at {_format_ts(sched.at_ms, tz_name)}"
    return "Unknown"


def _job_to_dict(job, cron_service=None, tz_name: str = "UTC") -> dict:
    """Serialize a CronJob for JSON/template consumption."""
    last_run = ""
    if job.state.last_run_at_ms:
        last_run = _format_ts(job.state.last_run_at_ms, tz_name)
    next_run = ""
    if job.state.next_run_at_ms:
        next_run = _format_ts(job.state.next_run_at_ms, tz_name)

    running = False
    if cron_service:
        running = cron_service.is_running(job.id)

    return {
        "id": job.id,
        "name": job.name,
        "enabled": job.enabled,
        "schedule": _format_schedule(job, tz_name),
        "last_run": last_run,
        "last_status": job.state.last_status,
        "last_error": job.state.last_error,
        "next_run": next_run,
        "running": running,
        "message_preview": (job.payload.message or "")[:80],
    }


# ── Tasks page ─────────────────────────────────────────────────


@router.get("/tasks", response_class=HTMLResponse)
async def tasks_page(request: Request):
    auth = request.app.state.auth
    if not auth.require_auth(request):
        return RedirectResponse("/login")

    cron_service = getattr(request.app.state, "cron_service", None)
    tz = getattr(request.app.state, "timezone", "UTC")
    jobs = []
    if cron_service:
        jobs = [
            _job_to_dict(j, cron_service, tz) for j in cron_service.list_jobs(include_disabled=True)
        ]

    notification_store = getattr(request.app.state, "notification_store", None)
    recent_activity = []
    if notification_store:
        recent_activity = notification_store.list_recent(limit=15)

    templates = request.app.state.templates
    return templates.TemplateResponse(
        "tasks.html",
        {
            "request": request,
            "jobs": jobs,
            "recent_activity": recent_activity,
        },
    )


@router.get("/tasks/api/jobs")
async def api_jobs(request: Request):
    auth = request.app.state.auth
    if not auth.require_auth(request):
        return JSONResponse({"error": "unauthorized"}, status_code=401)

    cron_service = getattr(request.app.state, "cron_service", None)
    if not cron_service:
        return JSONResponse([])

    tz = getattr(request.app.state, "timezone", "UTC")
    jobs = [
        _job_to_dict(j, cron_service, tz) for j in cron_service.list_jobs(include_disabled=True)
    ]
    return JSONResponse(jobs)


@router.post("/tasks/api/jobs/{job_id}/toggle")
async def api_toggle_job(request: Request, job_id: str):
    auth = request.app.state.auth
    if not auth.require_auth(request):
        return JSONResponse({"error": "unauthorized"}, status_code=401)

    cron_service = getattr(request.app.state, "cron_service", None)
    if not cron_service:
        return JSONResponse({"error": "cron not available"}, status_code=503)

    # Find current state and flip
    jobs = cron_service.list_jobs(include_disabled=True)
    current_job = next((j for j in jobs if j.id == job_id), None)
    if not current_job:
        return JSONResponse({"error": "not found"}, status_code=404)

    result = cron_service.enable_job(job_id, enabled=not current_job.enabled)
    if result:
        return JSONResponse({"ok": True, "enabled": result.enabled})
    return JSONResponse({"error": "failed"}, status_code=500)


@router.post("/tasks/api/jobs/{job_id}/run")
async def api_run_job(request: Request, job_id: str):
    auth = request.app.state.auth
    if not auth.require_auth(request):
        return JSONResponse({"error": "unauthorized"}, status_code=401)

    cron_service = getattr(request.app.state, "cron_service", None)
    if not cron_service:
        return JSONResponse({"error": "cron not available"}, status_code=503)

    started = cron_service.run_job_async(job_id)
    if started:
        return JSONResponse({"ok": True})
    return JSONResponse({"error": "job not found"}, status_code=404)


@router.post("/tasks/api/jobs/{job_id}/delete")
async def api_delete_job(request: Request, job_id: str):
    auth = request.app.state.auth
    if not auth.require_auth(request):
        return JSONResponse({"error": "unauthorized"}, status_code=401)

    cron_service = getattr(request.app.state, "cron_service", None)
    if not cron_service:
        return JSONResponse({"error": "cron not available"}, status_code=503)

    removed = cron_service.remove_job(job_id)
    if removed:
        return JSONResponse({"ok": True})
    return JSONResponse({"error": "not found"}, status_code=404)


@router.get("/tasks/api/jobs/{job_id}/transcript")
async def api_job_transcript(request: Request, job_id: str):
    auth = request.app.state.auth
    if not auth.require_auth(request):
        return JSONResponse({"error": "unauthorized"}, status_code=401)

    agent = getattr(request.app.state, "agent", None)
    if not agent:
        return JSONResponse({"error": "agent not available"}, status_code=503)

    session_key = f"cron:{job_id}"
    session = agent.sessions.get_or_create(session_key)
    messages = session.get_history()

    return JSONResponse(
        {
            "session_key": session_key,
            "messages": [
                {"role": m.get("role", ""), "content": m.get("content", "")} for m in messages
            ],
        }
    )


# ── Notification API ────────────────────────────────────────────


@router.get("/notifications/api/recent")
async def api_notifications_recent(request: Request):
    auth = request.app.state.auth
    if not auth.require_auth(request):
        return JSONResponse({"error": "unauthorized"}, status_code=401)

    store = getattr(request.app.state, "notification_store", None)
    if not store:
        return JSONResponse([])

    return JSONResponse(store.list_recent(limit=20))


@router.get("/notifications/api/unread-count")
async def api_unread_count(request: Request):
    auth = request.app.state.auth
    if not auth.require_auth(request):
        return JSONResponse({"error": "unauthorized"}, status_code=401)

    store = getattr(request.app.state, "notification_store", None)
    count = store.count_unread() if store else 0
    return JSONResponse({"count": count})


@router.post("/notifications/api/mark-read")
async def api_mark_read(request: Request):
    auth = request.app.state.auth
    if not auth.require_auth(request):
        return JSONResponse({"error": "unauthorized"}, status_code=401)

    store = getattr(request.app.state, "notification_store", None)
    if not store:
        return JSONResponse({"ok": True})

    body = await request.json()
    notification_id = body.get("id")
    if notification_id:
        store.mark_read(int(notification_id))
    else:
        store.mark_all_read()

    return JSONResponse({"ok": True})
