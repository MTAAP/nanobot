"""Intelligence scan page route."""

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, RedirectResponse

router = APIRouter(prefix="/intelligence", tags=["intelligence"])


@router.get("", response_class=HTMLResponse)
async def intelligence_page(request: Request):
    auth = request.app.state.auth
    if not auth.require_auth(request):
        return RedirectResponse("/login")

    templates = request.app.state.templates
    return templates.TemplateResponse("intelligence.html", {"request": request})
