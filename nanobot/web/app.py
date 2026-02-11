"""FastAPI web dashboard for nanobot."""

from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from nanobot.web.auth import AuthManager


def create_app(
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
        title="nanobot",
        docs_url=None,
        redoc_url=None,
    )

    # Store dependencies in app state
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
    from nanobot.web.routes.health import router as health_router
    from nanobot.web.routes.logs import router as logs_router
    from nanobot.web.routes.memory import router as memory_router
    from nanobot.web.routes.settings import router as settings_router
    from nanobot.web.routes.tasks import router as tasks_router

    app.include_router(chat_router)
    app.include_router(tasks_router)
    app.include_router(health_router)
    app.include_router(memory_router)
    app.include_router(logs_router)
    app.include_router(settings_router)

    # Auth routes
    from nanobot.web.auth import create_auth_routes

    app.include_router(create_auth_routes(templates))

    @app.get("/")
    async def index(request: Request):
        user = app.state.auth.get_current_user(request)
        if not user:
            return RedirectResponse("/login")
        return RedirectResponse("/chat")

    return app
