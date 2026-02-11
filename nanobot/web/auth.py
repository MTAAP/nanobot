"""Session-based authentication for the web dashboard."""

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

try:
    from itsdangerous import URLSafeTimedSerializer
except ImportError:
    URLSafeTimedSerializer = None

try:
    from passlib.hash import bcrypt
except ImportError:
    bcrypt = None


class AuthManager:
    """Simple session-based authentication manager."""

    COOKIE_NAME = "nanobot_session"
    MAX_AGE = 86400 * 7  # 7 days

    def __init__(
        self,
        username: str = "admin",
        password_hash: str = "",
        secret_key: str = "change-me-in-production",
    ):
        self._username = username
        self._password_hash = password_hash
        self._secret_key = secret_key
        if URLSafeTimedSerializer:
            self._serializer = URLSafeTimedSerializer(secret_key)
        else:
            self._serializer = None

    def verify_password(self, password: str) -> bool:
        """Verify a password against the stored hash."""
        if not self._password_hash:
            return bool(password)
        if bcrypt:
            try:
                return bcrypt.verify(password, self._password_hash)
            except Exception:
                return False
        return password == self._password_hash

    def verify_credentials(self, username: str, password: str) -> bool:
        """Verify username and password."""
        return username == self._username and self.verify_password(password)

    def create_session(self, username: str) -> str:
        """Create a signed session token."""
        if self._serializer:
            return self._serializer.dumps({"user": username})
        # Fallback without itsdangerous
        return f"session:{username}"

    def get_current_user(self, request: Request) -> str | None:
        """Get current user from session cookie."""
        token = request.cookies.get(self.COOKIE_NAME)
        if not token:
            return None
        if self._serializer:
            try:
                data = self._serializer.loads(token, max_age=self.MAX_AGE)
                return data.get("user")
            except Exception:
                return None
        # Fallback
        if token.startswith("session:"):
            return token.split(":", 1)[1]
        return None

    def require_auth(self, request: Request) -> str | None:
        """Check auth, return username or None (for redirect)."""
        return self.get_current_user(request)


def create_auth_routes(templates: Jinja2Templates) -> APIRouter:
    """Create login/logout routes."""
    router = APIRouter()

    @router.get("/login", response_class=HTMLResponse)
    async def login_page(request: Request):
        return templates.TemplateResponse("login.html", {"request": request})

    @router.post("/login")
    async def login(request: Request):
        form = await request.form()
        username = form.get("username", "")
        password = form.get("password", "")

        auth: AuthManager = request.app.state.auth
        if auth.verify_credentials(username, password):
            token = auth.create_session(username)
            response = RedirectResponse("/signals", status_code=303)
            response.set_cookie(
                AuthManager.COOKIE_NAME,
                token,
                max_age=AuthManager.MAX_AGE,
                httponly=True,
                samesite="lax",
            )
            return response

        return templates.TemplateResponse(
            "login.html",
            {"request": request, "error": "Invalid credentials"},
        )

    @router.get("/logout")
    async def logout(request: Request):
        response = RedirectResponse("/login")
        response.delete_cookie(AuthManager.COOKIE_NAME)
        return response

    return router
