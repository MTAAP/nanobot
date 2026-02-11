"""Chat route with WebSocket endpoint for agent communication."""

import json
from typing import Any

from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

from nanobot.bus.events import InboundMessage

router = APIRouter(prefix="/chat", tags=["chat"])


@router.get("", response_class=HTMLResponse)
async def chat_page(request: Request):
    auth = request.app.state.auth
    if not auth.require_auth(request):
        return RedirectResponse("/login")

    templates = request.app.state.templates
    return templates.TemplateResponse(
        "chat.html",
        {
            "request": request,
        },
    )


@router.get("/sessions")
async def list_sessions(request: Request) -> JSONResponse:
    """List web chat sessions with preview text."""
    agent = request.app.state.agent
    if not agent:
        return JSONResponse([])

    all_sessions = agent.sessions.list_sessions()
    result = []
    for info in all_sessions:
        key = info.get("key", "")
        if not key.startswith("web:"):
            continue

        session_id = key[4:]  # strip "web:" prefix
        preview = ""
        message_count = 0

        # Read session file to get first user message and count
        path = info.get("path")
        if path:
            try:
                with open(path, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        data = json.loads(line)
                        if data.get("_type") == "metadata":
                            continue
                        message_count += 1
                        if not preview and data.get("role") == "user":
                            preview = (data.get("content") or "")[:120]
            except Exception:
                continue

        if message_count == 0:
            continue

        result.append(
            {
                "id": session_id,
                "preview": preview or "(no preview)",
                "updated_at": info.get("updated_at"),
                "message_count": message_count,
            }
        )

    return JSONResponse(result)


@router.get("/sessions/{session_id}/history")
async def session_history(request: Request, session_id: str) -> JSONResponse:
    """Load full message history for a session."""
    agent = request.app.state.agent
    if not agent:
        return JSONResponse({"messages": []})

    key = f"web:{session_id}"
    session = agent.sessions.get_or_create(key)

    messages = []
    for msg in session.messages:
        role = msg.get("role", "assistant")
        content = msg.get("content", "")
        if not content:
            continue
        messages.append(
            {
                "role": role,
                "content": content,
                "timestamp": msg.get("timestamp"),
            }
        )

    return JSONResponse({"messages": messages})


@router.delete("/sessions/{session_id}")
async def delete_session(request: Request, session_id: str) -> JSONResponse:
    """Delete a chat session."""
    agent = request.app.state.agent
    if not agent:
        return JSONResponse({"ok": False})

    key = f"web:{session_id}"
    deleted = agent.sessions.delete(key)
    return JSONResponse({"ok": deleted})


@router.websocket("/ws/{session_id}")
async def chat_ws(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time agent chat."""
    await websocket.accept()

    web_channel = websocket.app.state.web_channel
    bus = websocket.app.state.message_bus

    if not web_channel or not bus:
        await websocket.send_json(
            {
                "type": "error",
                "content": "Chat not available: agent not connected.",
            }
        )
        await websocket.close()
        return

    web_channel.register(session_id, websocket)

    try:
        # If context query params present, send auto-context message
        context = websocket.query_params.get("context")
        context_id = websocket.query_params.get("id")
        if context and context_id:
            store = websocket.app.state.intel_store
            prompt = _build_context_prompt(store, context, context_id)
            if prompt:
                await bus.publish_inbound(
                    InboundMessage(
                        channel="web",
                        sender_id="admin",
                        chat_id=session_id,
                        content=prompt,
                    )
                )

        # Listen for user messages
        while True:
            data = await websocket.receive_json()
            content = data.get("content", "").strip()
            if not content:
                continue
            await bus.publish_inbound(
                InboundMessage(
                    channel="web",
                    sender_id="admin",
                    chat_id=session_id,
                    content=content,
                )
            )
    except WebSocketDisconnect:
        pass
    finally:
        web_channel.unregister(session_id)


def _build_context_prompt(
    store: Any,
    context_type: str,
    context_id: str,
) -> str:
    """Build a context-aware prompt for the agent based on what the user clicked."""
    if not store:
        return ""

    try:
        cid = int(context_id)
    except (ValueError, TypeError):
        return ""

    if context_type == "signal":
        signal = store.get_signal(cid)
        if not signal:
            return ""
        return (
            f"The user wants to discuss signal #{signal['id']}: "
            f"{signal['company_name']} - {signal['signal_type']} - "
            f"{signal['title']}. "
            f"Description: {signal.get('description', 'N/A')}. "
            f"Relevance: {signal.get('relevance_score', 0):.0%}. "
            f"Analyze this consulting opportunity for K&P. "
            f"What's our best approach? Consider using scan_company "
            f"and lead_scoring tools."
        )

    if context_type == "lead":
        # Leads are grouped by company name, not by ID
        # The context_id here is actually company_name passed as string
        return (
            "The user wants to discuss lead opportunities for a company. "
            "Review all signals for this company, score the lead, "
            "suggest K&P services that fit, and develop an approach strategy. "
            "Use the lead_scoring and market_intelligence tools."
        )

    if context_type == "recommendation":
        rec = store.get_recommendation(cid)
        if not rec:
            return ""
        return (
            f"The user wants to refine recommendation #{rec['id']} for "
            f"{rec['company_name']}. Service area: {rec.get('service_area', 'N/A')}. "
            f"Consultant: {rec.get('consultant_name', 'unassigned')}. "
            f"Review the outreach plan, suggest talking points, and help "
            f"draft the email. Use market_report render_outreach if needed."
        )

    return ""
