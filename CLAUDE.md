# nanobot

You are working on **nanobot**, a personal AI assistant framework for K&P Management Consulting (~24k LOC Python). It runs as a Docker container exposing a web dashboard (port 8080) and gateway API (port 18790).

## Critical: Question Ambiguities

Before implementing any feature or task, you MUST aggressively clarify ambiguities. Do not assume. Do not guess. Ask the user directly:

- **Scope**: "What exactly should this feature do? What should it NOT do?"
- **Edge cases**: "How should this handle [specific edge case]?"
- **Integration**: "How does this interact with existing [component]?"
- **Error states**: "What should happen when [failure scenario]?"
- **User expectations**: "What does success look like for this feature?"
- **Priorities**: "If we can't have everything, what's essential vs nice-to-have?"

Create detailed feature specifications BEFORE writing code. Push back on vague requirements.

## Current State

**Active branch**: `marketing-assistant` — marketing intelligence & outreach features

**Web dashboard** at `/` with pages: Signals, Leads, Recommendations, Intelligence, Reports, Compose, Chat, Settings. Chat sidebar uses WebSocket for real-time streaming with session persistence.

**Marketing subsystem** (`nanobot/marketing/`):
- `intel_store.py` — SQLite-backed signal & lead storage
- `scoring.py` — Lead scoring and consultant matching
- `pipedrive.py` — CRM integration
- `reports.py` — Report generation (daily brief, weekly, monthly)
- `consent.py` — GDPR consent tracking
- `templates/` — Outreach email templates (change, leadership, process, sales, strategy, turnaround)

**Channels**: Telegram, WhatsApp, Feishu/Lark, Discord, Email, Web

**Tools** (in `nanobot/agent/tools/`): CRM, intelligence scanning, lead scoring, email, follow-up, reports, cron, shell, spawn (subagents), MCP install, memory, filesystem, and more.

## Architecture

```
User -> Channel -> MessageBus -> AgentLoop -> LLM
                        |              |
                        v              v
                   SessionStore    ToolRegistry
                                       |
                                  MarketingTools -> IntelStore (SQLite)
                                                -> Pipedrive (CRM API)
```

- **Channels**: Extend `BaseChannel` — async start/stop lifecycle
- **Message Bus**: Decouples channels from agent via async queues
- **Agent Loop**: Processes messages, manages tool calls, streams responses
- **Tools**: Extend `Tool` base class with JSON Schema validation
- **Web UI**: Starlette app with Jinja2 templates + WebSocket chat
- **ProviderResolver**: Multi-provider LLM routing (subsystem -> compaction -> default -> priority scan)

## Key Directories

```
nanobot/
├── agent/          # Core agent loop, context assembly, tool execution
│   ├── tools/      # All tool implementations (base.py defines ABC)
│   └── memory/     # Agent memory subsystem
├── channels/       # Telegram, WhatsApp, Feishu, Discord, Email, Web
├── bus/            # Message bus (async queue-based)
├── config/         # Pydantic config schema and loading
├── llm/            # LiteLLM provider wrapper
├── marketing/      # Intel store, scoring, CRM, reports, templates
├── providers/      # ProviderResolver for multi-LLM routing
├── session/        # JSONL-based conversation storage
├── web/            # Starlette dashboard (routes/, static/, templates/)
├── cron/           # Scheduled job service (APScheduler + SQLite)
├── heartbeat/      # Periodic health checks and autonomous tasks
├── mcp/            # MCP server client integration
├── memory/         # Persistent memory (workspace-backed)
├── registry/       # Tool/skill registry
├── skills/         # Built-in skills (github, summarize, tmux, weather)
├── utils/          # Shared utilities
└── cli/            # Typer CLI commands
bridge/             # Node.js WhatsApp Web.js bridge
docker/             # Docker entrypoint scripts
tests/              # pytest test suite
```

## Tech Stack

- Python 3.12 (Docker) / 3.13 (local dev) with Hatchling build system
- Node.js 20+ for WhatsApp bridge only
- LiteLLM for LLM integration (100+ models)
- Starlette + Jinja2 for web dashboard
- Pydantic for configuration validation
- Typer for CLI
- Docker Compose for deployment
- Config location: `~/.nanobot/config.json`

## Commands

```bash
# === Deployment (primary) ===
docker compose up -d --build          # Rebuild and deploy
docker compose logs -f nanobot        # Follow logs
docker compose restart nanobot        # Restart without rebuild

# === Local Development ===
# No `python` on PATH — always use .venv/bin/ prefix
.venv/bin/pytest tests/               # Run tests
.venv/bin/ruff check nanobot/         # Lint
.venv/bin/ruff format --check nanobot/ # Format check

# Full verification — run after any code change
.venv/bin/pytest tests/ && .venv/bin/ruff check nanobot/ && .venv/bin/ruff format --check nanobot/

# === Docker testing ===
docker compose --profile test run --rm test  # Run tests + lint in container

# === WhatsApp Bridge ===
cd bridge && npm run build && npm start

# === Updating this file ===
# When you learn something new about the project (architecture, gotchas, patterns),
# update this CLAUDE.md immediately. Keep it current. This file IS the source of truth
# for how to work on this codebase. If a lesson isn't here, it will be forgotten.
```

## Key Files

| File | Purpose |
|------|---------|
| `nanobot/agent/loop.py` | Core agent processing loop |
| `nanobot/agent/context.py` | System prompt assembly |
| `nanobot/agent/tools/base.py` | Tool interface (`Tool` ABC) |
| `nanobot/channels/base.py` | Channel interface (`BaseChannel` ABC) |
| `nanobot/config/schema.py` | Pydantic config models |
| `nanobot/bus/queue.py` | Message bus implementation |
| `nanobot/llm/provider.py` | LiteLLM wrapper |
| `nanobot/providers/resolver.py` | Multi-provider LLM routing |
| `nanobot/web/app.py` | Starlette web app setup and routes |
| `nanobot/web/routes/chat.py` | WebSocket chat endpoint |
| `nanobot/web/static/chat.js` | Chat sidebar client (WebSocket + sessions) |
| `nanobot/marketing/intel_store.py` | SQLite signal/lead storage |
| `nanobot/marketing/scoring.py` | Lead scoring engine |
| `docker-compose.yml` | Docker deployment config |
| `Dockerfile` | Container build definition |

## Code Patterns You Must Follow

### Async-First
Everything is async. Use `async def` and `await` consistently:
```python
async def process_message(self, message: Message) -> AsyncIterator[str]:
    async for chunk in self.llm.stream(messages):
        yield chunk
```

### Abstract Base Classes
Extend `BaseChannel` for new channels, `Tool` for new tools:
```python
class MyTool(Tool):
    name = "my_tool"
    description = "Does something useful"

    def get_parameters(self) -> dict:
        return {"type": "object", "properties": {...}}

    async def execute(self, **kwargs) -> str:
        return "result"
```

### Error Handling in Tools
Return error strings. Do not raise exceptions:
```python
async def execute(self, **kwargs) -> str:
    try:
        result = await do_work()
        return result
    except SomeError as e:
        return f"Error: {e}"
```

### Type Hints
Use Python 3.11+ union syntax. Do not use `Optional`:
```python
def process(self, data: str | None = None) -> dict[str, Any]:
    ...
```

### Pydantic Config Models
Use `alias` for camelCase JSON fields. Test with alias names:
```python
# Definition
field_name: str = Field(alias="fieldName")

# Testing — must use alias in kwargs (no populate_by_name=True)
Model(**{"fieldName": "value"})  # correct
Model(field_name="value")        # WRONG — will not set the field
```

## Rules You Must Follow

- **Line length**: 100 characters max
- **Logging**: Use loguru. No emojis in debug messages.
- **File I/O**: Always use `encoding="utf-8"`
- **Single implementation**: When refactoring, remove legacy code completely. Do not leave alternatives.
- **Type hints**: Required on all function signatures
- **Imports**: Use absolute imports from `nanobot.` package
- **Verification**: Run tests + lint after any code change
- **Static files**: Changes to `nanobot/web/static/` require `docker compose up -d --build` to deploy

## Workflow

1. **Use git worktrees**: Always work in a git worktree to enable parallel work with other agents. Create a worktree before starting any implementation:
   ```bash
   git worktree add ../nanobot-feature-name feature-branch-name
   cd ../nanobot-feature-name
   ```
   This prevents conflicts when multiple agents work on the codebase simultaneously.
2. **Clarify first**: Question the user on all ambiguities before implementing
3. **Plan complex tasks**: Use Plan mode to explore the codebase before proposing changes
4. **Verify changes**: Run tests and linting after every modification
5. **Prove correctness**: Demonstrate that changes work with tests or examples
6. **Use subagents**: For multi-file exploration or complex refactoring
7. **Update this file**: After learning something new about the project, update CLAUDE.md immediately

## GitHub

This is a fork of `HKUDS/nanobot`. PRs must target **MTAAP/nanobot**, not the upstream.

After cloning, run once: `gh repo set-default MTAAP/nanobot`

## Lessons Learned

### WebSocket lifecycle in chat.js
The chat sidebar has two modes: **session list** (no WS) and **active chat** (WS connected). The `inActiveChat` flag distinguishes them. Without it, `chatWs.close()` triggers an async `onclose` that sets a reconnect timer after `disconnectWebSocket()` has already cleared it — causing zombie connections and stale status messages in the session list. Always guard `onopen`/`onclose` side effects with `inActiveChat`.

### Pydantic gotchas
- `Config` inner class uses deprecated class-based config (PydanticDeprecatedSince20 warning is known and accepted)
- Fields with `alias` can only be set via the alias name in kwargs unless `populate_by_name=True` is set on the model

### Docker deployment
- Static files (JS/CSS/templates) are baked into the image at build time — edits require `docker compose up -d --build`
- The entrypoint supports restart signals for MCP server installation
- Volumes mount `~/.nanobot` for config and workspace persistence
- No `python` binary on local PATH — always use `.venv/bin/python3` or `.venv/bin/pytest`

### Provider resolution
- `ProviderResolver` in `nanobot/providers/resolver.py` handles multi-provider routing
- Resolution chain: subsystem provider -> compaction provider -> default provider -> priority scan
- Providers cached in `AgentLoop._provider_cache` keyed by `name:key:base`

## Mistakes to Avoid

- Forgetting `encoding="utf-8"` on file operations
- Using `Optional[X]` instead of `X | None`
- Raising exceptions in tool `execute()` methods instead of returning error strings
- Leaving emoji in debug/log messages
- Implementing features without clarifying requirements first
- Assuming user intent instead of asking
- Creating PRs against the upstream `HKUDS/nanobot` instead of `MTAAP/nanobot`
- Forgetting to rebuild Docker after changing static files or Python source
- Testing Pydantic aliased fields with Python kwarg names instead of alias names
- Not guarding WebSocket event handlers with state flags (async close race conditions)
