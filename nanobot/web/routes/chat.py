"""Chat route with WebSocket endpoint for agent communication."""

import json

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
            except (OSError, json.JSONDecodeError):
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
