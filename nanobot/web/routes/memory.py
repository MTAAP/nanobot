"""Memory browser route."""

from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

router = APIRouter(prefix="/memory", tags=["memory"])


@router.get("", response_class=HTMLResponse)
async def memory_page(request: Request):
    auth = request.app.state.auth
    if not auth.require_auth(request):
        return RedirectResponse("/login")

    agent = getattr(request.app.state, "agent", None)

    vector_count = 0
    db_size = 0
    entity_count = 0
    relation_count = 0
    core_sections: list[dict] = []

    if agent:
        if agent.vector_store:
            vector_count = agent.vector_store.count()
            db_path = Path(agent.vector_store.db_path)
            if db_path.exists():
                db_size = db_path.stat().st_size

        if agent.core_memory:
            try:
                sections = agent.core_memory.read_all()
                core_sections = [
                    {"name": name, "content": content} for name, content in sections.items()
                ]
            except Exception:
                pass

        if agent.entity_store:
            try:
                with agent.entity_store._get_conn() as conn:
                    entity_count = conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
                    relation_count = conn.execute("SELECT COUNT(*) FROM relations").fetchone()[0]
            except Exception:
                pass

    templates = request.app.state.templates
    return templates.TemplateResponse(
        "memory.html",
        {
            "request": request,
            "vector_count": vector_count,
            "db_size": _fmt_size(db_size),
            "entity_count": entity_count,
            "relation_count": relation_count,
            "core_sections": core_sections,
        },
    )


@router.get("/api/stats")
async def api_stats(request: Request):
    auth = request.app.state.auth
    if not auth.require_auth(request):
        return JSONResponse({"error": "unauthorized"}, status_code=401)

    agent = getattr(request.app.state, "agent", None)
    if not agent:
        return JSONResponse({"vector_count": 0, "entity_count": 0})

    vector_count = agent.vector_store.count() if agent.vector_store else 0
    entity_count = 0
    if agent.entity_store:
        try:
            with agent.entity_store._get_conn() as conn:
                entity_count = conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
        except Exception:
            pass

    return JSONResponse(
        {
            "vector_count": vector_count,
            "entity_count": entity_count,
        }
    )


@router.get("/api/search")
async def api_search(request: Request):
    auth = request.app.state.auth
    if not auth.require_auth(request):
        return JSONResponse({"error": "unauthorized"}, status_code=401)

    query = request.query_params.get("q", "").strip()
    if not query:
        return JSONResponse([])

    agent = getattr(request.app.state, "agent", None)
    if not agent or not agent.vector_store:
        return JSONResponse([])

    results = await agent.vector_store.search(query=query, top_k=10)
    return JSONResponse(
        [
            {
                "text": r.get("text", "")[:300],
                "similarity": round(r.get("similarity", 0), 3),
                "type": r.get("metadata", {}).get("type", "unknown"),
                "created_at": r.get("created_at", ""),
            }
            for r in results
        ]
    )


@router.get("/api/entities")
async def api_entities(request: Request):
    auth = request.app.state.auth
    if not auth.require_auth(request):
        return JSONResponse({"error": "unauthorized"}, status_code=401)

    query = request.query_params.get("q", "").strip()
    agent = getattr(request.app.state, "agent", None)
    if not agent or not agent.entity_store:
        return JSONResponse([])

    if query:
        entities = agent.entity_store.search_entities(query, limit=20)
    else:
        entities = agent.entity_store.search_entities("%", limit=20)

    return JSONResponse(entities)


def _fmt_size(size: int) -> str:
    """Format byte size to human-readable string."""
    if size >= 1024 * 1024:
        return f"{size / (1024 * 1024):.1f} MB"
    if size >= 1024:
        return f"{size / 1024:.1f} KB"
    return f"{size} B"
