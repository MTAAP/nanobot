"""FastAPI web dashboard for K&P Marketing Assistant."""

from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from nanobot.web.auth import AuthManager


def create_app(
    intel_store: Any = None,
    pipedrive_client: Any = None,
    consent_store: Any = None,
    auth_manager: AuthManager | None = None,
    message_bus: Any = None,
    agent: Any = None,
    web_channel: Any = None,
    cron_service: Any = None,
    notification_store: Any = None,
    timezone: str = "UTC",
) -> FastAPI:
    """Create the FastAPI dashboard application."""
    app = FastAPI(
        title="K&P Marketing Assistant",
        docs_url=None,
        redoc_url=None,
    )

    # Store dependencies in app state
    app.state.intel_store = intel_store
    app.state.pipedrive = pipedrive_client
    app.state.consent = consent_store
    app.state.auth = auth_manager or AuthManager()
    app.state.message_bus = message_bus
    app.state.agent = agent
    app.state.web_channel = web_channel
    app.state.cron_service = cron_service
    app.state.notification_store = notification_store
    app.state.timezone = timezone

    # Static files and templates
    static_dir = Path(__file__).parent / "static"
    template_dir = Path(__file__).parent / "templates"
    static_dir.mkdir(exist_ok=True)

    app.mount(
        "/static",
        StaticFiles(directory=str(static_dir)),
        name="static",
    )
    templates = Jinja2Templates(directory=str(template_dir))
    app.state.templates = templates

    # Include routers
    from nanobot.web.routes.chat import router as chat_router
    from nanobot.web.routes.intelligence import router as intel_router
    from nanobot.web.routes.leads import router as leads_router
    from nanobot.web.routes.recommendations import router as recs_router
    from nanobot.web.routes.reports import router as reports_router
    from nanobot.web.routes.settings import router as settings_router
    from nanobot.web.routes.signals import router as signals_router
    from nanobot.web.routes.tasks import router as tasks_router

    app.include_router(signals_router)
    app.include_router(leads_router)
    app.include_router(recs_router)
    app.include_router(intel_router)
    app.include_router(reports_router)
    app.include_router(chat_router)
    app.include_router(tasks_router)
    app.include_router(settings_router)

    # Auth routes
    from nanobot.web.auth import create_auth_routes

    app.include_router(create_auth_routes(templates))

    @app.get("/")
    async def index(request: Request):
        user = app.state.auth.get_current_user(request)
        if not user:
            return RedirectResponse("/login")
        return RedirectResponse("/signals")

    return app
