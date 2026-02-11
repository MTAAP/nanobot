"""Health dashboard route."""

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

router = APIRouter(prefix="/health", tags=["health"])


@router.get("", response_class=HTMLResponse)
async def health_page(request: Request):
    auth = request.app.state.auth
    if not auth.require_auth(request):
        return RedirectResponse("/login")

    from nanobot.agent.errors import get_error_logger

    error_logger = get_error_logger()
    metrics = error_logger.get_metrics() if error_logger else {}
    top_errors = error_logger.get_top_errors(limit=10) if error_logger else []
    recent = error_logger.get_recent_errors(minutes=60) if error_logger else []

    # Determine status level
    errors_per_hour = metrics.get("errors_last_hour", 0)
    if errors_per_hour == 0:
        status = "HEALTHY"
    elif errors_per_hour <= 5:
        status = "GOOD"
    elif errors_per_hour <= 20:
        status = "WARNING"
    else:
        status = "CRITICAL"

    templates = request.app.state.templates
    return templates.TemplateResponse(
        "health.html",
        {
            "request": request,
            "status": status,
            "metrics": metrics,
            "top_errors": top_errors,
            "recent_errors": recent[:20],
        },
    )


@router.get("/api/metrics")
async def api_metrics(request: Request):
    auth = request.app.state.auth
    if not auth.require_auth(request):
        return JSONResponse({"error": "unauthorized"}, status_code=401)

    from nanobot.agent.errors import get_error_logger

    error_logger = get_error_logger()
    if not error_logger:
        return JSONResponse({"total_errors": 0, "errors_last_hour": 0})

    return JSONResponse(error_logger.get_metrics())


@router.get("/api/recent")
async def api_recent(request: Request):
    auth = request.app.state.auth
    if not auth.require_auth(request):
        return JSONResponse({"error": "unauthorized"}, status_code=401)

    from nanobot.agent.errors import get_error_logger

    minutes = int(request.query_params.get("minutes", "60"))
    error_logger = get_error_logger()
    if not error_logger:
        return JSONResponse([])

    return JSONResponse(error_logger.get_recent_errors(minutes=minutes))
