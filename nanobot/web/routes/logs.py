"""Log viewer route."""

import time

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

router = APIRouter(prefix="/logs", tags=["logs"])


@router.get("", response_class=HTMLResponse)
async def logs_page(request: Request):
    auth = request.app.state.auth
    if not auth.require_auth(request):
        return RedirectResponse("/login")

    from nanobot.agent.errors import ErrorCategory, get_error_logger

    error_logger = get_error_logger()
    recent = error_logger.get_recent_errors(minutes=60) if error_logger else []

    categories = [c.value for c in ErrorCategory]

    templates = request.app.state.templates
    return templates.TemplateResponse(
        "logs.html",
        {
            "request": request,
            "recent_errors": recent[:50],
            "categories": categories,
        },
    )


@router.get("/api/stream")
async def api_stream(request: Request):
    auth = request.app.state.auth
    if not auth.require_auth(request):
        return JSONResponse({"error": "unauthorized"}, status_code=401)

    from nanobot.agent.errors import get_error_logger

    error_logger = get_error_logger()
    if not error_logger:
        return JSONResponse([])

    records = error_logger.get_recent_errors(minutes=10)
    for r in records:
        ts = r.get("timestamp", 0)
        if ts:
            r["time_str"] = time.strftime("%H:%M:%S", time.localtime(ts))
    return JSONResponse(records[:30])


@router.get("/api/filter")
async def api_filter(request: Request):
    auth = request.app.state.auth
    if not auth.require_auth(request):
        return JSONResponse({"error": "unauthorized"}, status_code=401)

    from nanobot.agent.errors import get_error_logger

    category = request.query_params.get("category", "")
    severity = request.query_params.get("severity", "")
    minutes = int(request.query_params.get("minutes", "60"))

    error_logger = get_error_logger()
    if not error_logger:
        return JSONResponse([])

    records = error_logger.get_recent_errors(minutes=minutes)

    if category:
        records = [r for r in records if r.get("category") == category]
    if severity:
        records = [r for r in records if r.get("severity") == severity]

    for r in records:
        ts = r.get("timestamp", 0)
        if ts:
            r["time_str"] = time.strftime("%H:%M:%S", time.localtime(ts))

    return JSONResponse(records[:50])
