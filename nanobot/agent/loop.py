"""Agent loop: the core processing engine."""

import asyncio
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from nanobot.agent.context import ContextBuilder
from nanobot.agent.guardrails import GuardrailEngine
from nanobot.agent.intent import IntentClassifier, QueryIntent
from nanobot.agent.subagent import SubagentManager
from nanobot.agent.tools.filesystem import EditFileTool, ListDirTool, ReadFileTool, WriteFileTool
from nanobot.agent.tools.message import MessageTool
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.tools.shell import ExecTool
from nanobot.agent.tools.spawn import SpawnTool
from nanobot.agent.tools.web import WebFetchTool, WebSearchTool
from nanobot.agent.tracing import Tracer
from nanobot.agent.usage import UsageRecord, UsageTracker
from nanobot.bus.events import InboundMessage, OutboundMessage
from nanobot.bus.progress import ProgressEvent, ProgressKind
from nanobot.bus.queue import MessageBus
from nanobot.channels.manager import ChannelManager
from nanobot.providers.base import LLMProvider, LLMResponse, ToolCallRequest
from nanobot.session.compaction import CompactionConfig as SessionCompactionConfig


_TOOL_CALL_BLOCK_RE = re.compile(
    r"<tool_call>\s*\w+.*?</tool_call>",
    re.DOTALL | re.IGNORECASE,
)


def _parse_tool_calls_from_content(content: str | None) -> list[ToolCallRequest]:
    """Parse XML-style tool calls from response content when the model emits them as text.

    Handles format: <tool_call>NAME<arg_key>k</arg_key><arg_value>v</arg_value>...</tool_call>
    Returns a list of ToolCallRequest; content is not modified.
    """
    if not content or "<tool_call>" not in content:
        return []
    out: list[ToolCallRequest] = []
    # Match <tool_call>toolname ... </tool_call>
    for i, block in enumerate(
        re.finditer(
            r"<tool_call>\s*(\w+)\s*(.*?)</tool_call>",
            content,
            re.DOTALL | re.IGNORECASE,
        )
    ):
        name = block.group(1).strip()
        inner = block.group(2)
        args: dict[str, Any] = {}
        for kv in re.finditer(
            r"<arg_key>\s*([^<]*?)\s*</arg_key>\s*<arg_value>\s*(.*?)\s*</arg_value>",
            inner,
            re.DOTALL | re.IGNORECASE,
        ):
            k = kv.group(1).strip()
            v = kv.group(2).strip()
            if k:
                args[k] = v
        if name:
            out.append(
                ToolCallRequest(
                    id=f"content_parse_{i}_{int(time.time() * 1000)}",
                    name=name,
                    arguments=args,
                )
            )
    return out


def _strip_tool_call_blocks_from_content(content: str | None) -> str:
    """Remove <tool_call>...</tool_call> blocks so they are not shown to the user."""
    if not content or "<tool_call>" not in content:
        return content or ""
    cleaned = _TOOL_CALL_BLOCK_RE.sub("", content)
    return cleaned.strip() or ""
from nanobot.session.compaction import SessionCompactor
from nanobot.session.manager import SessionManager

if TYPE_CHECKING:
    from nanobot.config.schema import (
        CompactionConfig,
        ContextConfig,
        DaemonConfig,
        ExecToolConfig,
        GuardrailConfig,
        IntentConfig,
        MCPConfig,
        MemoryConfig,
        MemoryExtractionConfig,
        StreamingConfig,
        TracingConfig,
    )
    from nanobot.cron.service import CronService
    from nanobot.mcp import MCPManager
    from nanobot.memory.vectors import VectorStore
    from nanobot.providers.resolver import ProviderResolver
    from nanobot.registry.store import AgentRegistry
    from nanobot.session.manager import Session


def _truncate_at_sentence(text: str, max_chars: int = 300) -> str:
    """Truncate text at the nearest sentence boundary within max_chars.

    Looks for sentence-ending punctuation (.!?) followed by a space or end
    of string. If no sentence boundary is found, falls back to a hard cut.
    """
    if len(text) <= max_chars:
        return text

    # Search for the last sentence boundary within max_chars
    truncated = text[:max_chars]
    for i in range(len(truncated) - 1, -1, -1):
        if truncated[i] in ".!?" and (i + 1 >= len(truncated) or truncated[i + 1] == " "):
            return truncated[: i + 1]

    # No sentence boundary found, hard truncate with ellipsis
    return truncated[:max_chars] + "..."


def _format_memory_age(created_at: str) -> str:
    """Format a memory's created_at timestamp as a human-readable age.

    Args:
        created_at: ISO format timestamp string.

    Returns:
        Human-readable age like "2 hours ago", "3 days ago", or "unknown age".
    """
    if not created_at:
        return "unknown age"
    try:
        ts = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        # Ensure both are offset-aware or both naive
        now = datetime.now(timezone.utc)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        delta = now - ts
        seconds = int(delta.total_seconds())
        if seconds < 60:
            return "just now"
        minutes = seconds // 60
        if minutes < 60:
            return f"{minutes} min ago"
        hours = minutes // 60
        if hours < 24:
            return f"{hours} hour{'s' if hours != 1 else ''} ago"
        days = hours // 24
        if days < 14:
            return f"{days} day{'s' if days != 1 else ''} ago"
        weeks = days // 7
        if weeks < 8:
            return f"{weeks} week{'s' if weeks != 1 else ''} ago"
        months = days // 30
        return f"{months} month{'s' if months != 1 else ''} ago"
    except Exception:
        return "unknown age"


class AgentLoop:
    """
    The agent loop is the core processing engine.

    It:
    1. Receives messages from the bus
    2. Builds context with history, memory, skills
    3. Calls the LLM
    4. Executes tool calls
    5. Sends responses back
    """

    def __init__(
        self,
        bus: MessageBus,
        provider: LLMProvider,
        workspace: Path,
        model: str | None = None,
        max_iterations: int = 20,
        brave_api_key: str | None = None,
        exec_config: "ExecToolConfig | None" = None,
        channel_manager: ChannelManager | None = None,
        cron_service: "CronService | None" = None,
        mcp_config: "MCPConfig | None" = None,
        context_config: "ContextConfig | None" = None,
        compaction_config: "CompactionConfig | None" = None,
        memory_config: "MemoryConfig | None" = None,
        provider_resolver: "ProviderResolver | None" = None,
        memory_extraction: "MemoryExtractionConfig | None" = None,
        restrict_to_workspace: bool = False,
        registry: "AgentRegistry | None" = None,
        daemon_config: "DaemonConfig | None" = None,
        intent_config: "IntentConfig | None" = None,
        streaming_config: "StreamingConfig | None" = None,
        tracing_config: "TracingConfig | None" = None,
        guardrail_config: "GuardrailConfig | None" = None,
        temperature: float = 0.7,
        tool_temperature: float = 0.0,
        timezone: str = "UTC",
    ):
        from nanobot.config.schema import (
            CompactionConfig,
            ContextConfig,
            ExecToolConfig,
            GuardrailConfig,
            IntentConfig,
            MCPConfig,
            MemoryConfig,
            StreamingConfig,
            TracingConfig,
        )

        self.bus = bus
        self.provider = provider
        self.workspace = workspace
        self.model = model or provider.get_default_model()
        self.max_iterations = max_iterations
        self.brave_api_key = brave_api_key
        self.temperature = temperature
        self.tool_temperature = tool_temperature
        self.exec_config = exec_config or ExecToolConfig()
        self.channel_manager = channel_manager
        self.cron_service = cron_service
        self.mcp_config = mcp_config or MCPConfig()
        self.context_config = context_config or ContextConfig()
        self.compaction_config = compaction_config or CompactionConfig()
        self.memory_config = memory_config or MemoryConfig()
        self.provider_resolver = provider_resolver

        # Tools-level workspace restriction (applies to file tools and exec)
        self.restrict_to_workspace = restrict_to_workspace or self.exec_config.restrict_to_workspace

        # Memory extraction config
        from nanobot.config.schema import MemoryExtractionConfig

        self._extraction_config = memory_extraction or MemoryExtractionConfig()

        # Intent classification config
        self._intent_config = intent_config or IntentConfig()
        self._intent_classifier = IntentClassifier(
            enabled=self._intent_config.enabled,
            llm_fallback=self._intent_config.llm_fallback,
        )

        # Vector store for semantic memory (initialized in run())
        self.vector_store: "VectorStore | None" = None
        # Core memory (initialized in _init_memory())
        self.core_memory = None
        # Entity store (initialized in _init_memory())
        self.entity_store = None
        # Proactive memory (initialized in _init_memory())
        self.proactive_memory = None

        self.timezone = timezone
        self.context = ContextBuilder(
            workspace,
            memory_enabled=self.memory_config.enabled,
            timezone=self.timezone,
        )
        self.sessions = SessionManager(workspace)
        self.tools = ToolRegistry()

        # Memory extraction and consolidation (lightweight vector store)
        self._extractor = None
        self._consolidator = None
        self._session_compactor = SessionCompactor(
            config=SessionCompactionConfig(),
        )
        self._extraction_interval = self._extraction_config.extraction_interval
        self._enable_pre_compaction_flush = self._extraction_config.enable_pre_compaction_flush
        self._enable_tool_lessons = self._extraction_config.enable_tool_lessons

        # Agent registry (ACP)
        self._registry = registry
        self._daemon_config = daemon_config

        # Self-evolution manager
        self._evolve_manager = None
        if daemon_config and daemon_config.self_evolve.enabled and self._registry:
            self._init_evolve_manager(daemon_config)

        self.subagents = SubagentManager(
            provider=provider,
            workspace=workspace,
            bus=bus,
            model=self.model,
            brave_api_key=brave_api_key,
            exec_config=self.exec_config,
            registry=self._registry,
            evolve_manager=self._evolve_manager,
            progress_callback=bus.publish_progress,
        )

        # MCP manager (initialized in run())
        self.mcp_manager: "MCPManager | None" = None

        # Cache for subsystem-specific LLM providers
        self._provider_cache: dict[str, LLMProvider] = {}

        # Serializes daemon execution with user message processing
        self._processing_lock = asyncio.Lock()

        # Tool failure reflexion: track consecutive failures per tool
        self._tool_failure_counts: dict[str, int] = {}
        self._tool_failure_threshold = 3

        # Config-driven observability
        self._streaming_config = streaming_config or StreamingConfig()
        self._tracing_config = tracing_config or TracingConfig()
        self._guardrail_config = guardrail_config or GuardrailConfig()

        # Observability: tracing, usage tracking, guardrails
        self._tracer = Tracer(enabled=self._tracing_config.enabled)
        self._usage_tracker = UsageTracker()
        self._guardrails = GuardrailEngine(
            enabled=self._guardrail_config.enabled,
            timeout_s=self._guardrail_config.timeout_s,
        )

        self._running = False
        # Names of tools that support set_context(channel, chat_id)
        self._context_tools = ("message", "spawn", "cron", "install_mcp_server", "follow_up")
        # Background tasks for fire-and-forget memory operations
        self._background_tasks: set[asyncio.Task] = set()
        self._register_default_tools()

    def _spawn_background(self, coro) -> None:
        """Schedule a coroutine as a fire-and-forget background task.

        The task is tracked so it doesn't get garbage-collected, and errors
        are logged rather than silently swallowed.
        """
        task = asyncio.create_task(coro)
        self._background_tasks.add(task)

        def _on_done(t: asyncio.Task) -> None:
            self._background_tasks.discard(t)
            if t.cancelled():
                return
            exc = t.exception()
            if exc:
                logger.warning(f"Background task failed: {exc}")

        task.add_done_callback(_on_done)

    def _update_tool_contexts(self, channel: str, chat_id: str) -> None:
        """Update channel/chat context on all tools that support it."""
        for name in self._context_tools:
            tool = self.tools.get(name)
            if tool and hasattr(tool, "set_context"):
                tool.set_context(channel, chat_id)

    def _register_default_tools(self) -> None:
        """Register the default set of tools."""
        # Compute allowed_dir for file tools when workspace restriction is active
        allowed_dir = self.workspace if self.restrict_to_workspace else None

        # File tools
        self.tools.register(ReadFileTool(allowed_dir=allowed_dir))
        self.tools.register(WriteFileTool(allowed_dir=allowed_dir))
        self.tools.register(EditFileTool(allowed_dir=allowed_dir))
        self.tools.register(ListDirTool(allowed_dir=allowed_dir))

        # Shell tool
        self.tools.register(
            ExecTool(
                working_dir=str(self.workspace),
                timeout=self.exec_config.timeout,
                restrict_to_workspace=self.restrict_to_workspace,
            )
        )

        # Web tools
        self.tools.register(WebSearchTool(api_key=self.brave_api_key))
        self.tools.register(WebFetchTool())

        # Message tool
        message_tool = MessageTool(send_callback=self.bus.publish_outbound)
        self.tools.register(message_tool)

        # Spawn tool (for subagents)
        spawn_tool = SpawnTool(manager=self.subagents, registry=self._registry)
        self.tools.register(spawn_tool)

        # Tmux tool (persistent shell sessions)
        from nanobot.agent.tools.tmux import TmuxTool

        self.tools.register(TmuxTool())

        # Cron tool (for self-scheduling)
        if self.cron_service:
            from nanobot.agent.tools.cron import CronTool

            self.tools.register(
                CronTool(cron_service=self.cron_service, default_timezone=self.timezone)
            )

        # Discord config tool (if Discord channel is enabled)
        if self.channel_manager and "discord" in self.channel_manager.channels:
            from nanobot.agent.tools.discord_config import DiscordSetNotificationChannelTool
            from nanobot.channels.discord import DiscordChannel

            discord_channel = self.channel_manager.channels["discord"]
            if isinstance(discord_channel, DiscordChannel):
                self.tools.register(DiscordSetNotificationChannelTool(discord_channel))

        # MCP install tool (always available for self-installation)
        from nanobot.agent.tools.mcp_install import InstallMCPServerTool

        self.tools.register(InstallMCPServerTool(workspace=self.workspace))

        # Follow-up tracking tool
        from nanobot.agent.tools.followup import FollowUpTool

        self.tools.register(
            FollowUpTool(db_path=Path.home() / ".nanobot" / "data" / "followups.db")
        )

        # Self-evolution tool (if enabled)
        if self._evolve_manager:
            from nanobot.agent.tools.evolve import SelfEvolveTool

            self.tools.register(SelfEvolveTool(self._evolve_manager))
            self.context.self_evolve_enabled = True

    def _init_evolve_manager(self, daemon_config: "DaemonConfig") -> None:
        """Initialize the self-evolution manager from config."""
        evolve_cfg = daemon_config.self_evolve

        # Extract GITHUB_TOKEN from MCP server config
        github_token = None
        if self.mcp_config and self.mcp_config.servers:
            gh_server = self.mcp_config.servers.get("github")
            if gh_server:
                github_token = gh_server.env.get("GITHUB_TOKEN")

        if not github_token:
            import os

            github_token = os.environ.get("GITHUB_TOKEN", "")

        if not github_token:
            logger.warning("Self-evolution enabled but no GITHUB_TOKEN found")
            return

        from nanobot.registry.evolve import SelfEvolveManager

        self._evolve_manager = SelfEvolveManager(
            workspace=self.workspace,
            repo_url=evolve_cfg.repo_url,
            github_token=github_token,
            protected_branches=evolve_cfg.protected_branches,
            test_command=evolve_cfg.test_command,
            lint_command=evolve_cfg.lint_command,
            auto_merge=evolve_cfg.auto_merge,
        )
        logger.info("Self-evolution manager initialized")

    def _resolve_subsystem_provider(self, provider_name: str | None) -> LLMProvider:
        """Resolve an LLM provider for a subsystem (compaction, extraction, etc.).

        If ``provider_name`` is set and differs from the main provider, creates
        and caches a separate LiteLLMProvider with the appropriate credentials.
        Otherwise returns the main provider.

        Args:
            provider_name: Named provider from config, or None for main.

        Returns:
            An LLMProvider instance.
        """
        if not provider_name or not self.provider_resolver:
            return self.provider

        api_key, api_base = self.provider_resolver.resolve(provider_name)

        # If resolution yielded the same credentials as main, reuse it
        if api_key == self.provider.api_key and api_base == self.provider.api_base:
            return self.provider

        # Check cache
        cache_key = f"{provider_name}:{api_key}:{api_base}"
        if cache_key in self._provider_cache:
            return self._provider_cache[cache_key]

        # Create new provider
        from nanobot.providers.litellm_provider import LiteLLMProvider

        new_provider = LiteLLMProvider(
            api_key=api_key,
            api_base=api_base,
            default_model=self.model,
        )
        self._provider_cache[cache_key] = new_provider
        logger.debug(f"Created subsystem provider for '{provider_name}'")
        return new_provider

    def _truncate_tool_result(self, result: str) -> str:
        """Truncate tool result to fit within budget."""
        from nanobot.utils.tokens import count_tokens, truncate_to_token_limit

        max_tokens = self.context_config.tool_result_budget
        current_tokens = count_tokens(result)

        if current_tokens <= max_tokens:
            return result

        logger.debug(f"Truncating tool result from {current_tokens} to {max_tokens} tokens")
        return truncate_to_token_limit(result, max_tokens)

    def _record_tool_lesson(
        self,
        tool_name: str,
        arguments: dict,
        error: str,
        count: int,
    ) -> None:
        """Record a tool failure lesson to TOOLS.md for daemon review."""
        import time as _time

        lesson = (
            f"\n\n## [LESSON] {tool_name} - "
            f"Repeated failure ({count} times)\n"
            f"Arguments pattern: {json.dumps(arguments)[:200]}\n"
            f"Error: {error[:300]}\n"
            f"Recorded: {_time.strftime('%Y-%m-%d %H:%M')}\n"
        )
        tools_md = self.workspace / "TOOLS.md"
        try:
            existing = ""
            if tools_md.exists():
                existing = tools_md.read_text(encoding="utf-8")
            tools_md.write_text(existing + lesson, encoding="utf-8")
            logger.info(f"Recorded tool lesson for '{tool_name}' after {count} failures")
        except Exception as e:
            logger.warning(f"Failed to record tool lesson: {e}")

    def _check_context_budget(self, messages: list[dict]) -> None:
        """Log warnings if approaching context limits.

        Uses character-based estimation (chars // 4) for fast approximate
        checks, only calling the full tokenizer when near the limit.
        """
        max_tokens = self.context_config.max_context_tokens

        # Fast estimation: ~4 chars per token
        total_chars = sum(len(m.get("content", "") or "") for m in messages)
        est_tokens = total_chars // 4
        threshold_80 = max_tokens * 0.8

        if est_tokens < threshold_80:
            return  # Safely under budget, skip expensive tokenization

        # Near budget — do precise count
        from nanobot.utils.tokens import count_messages_tokens

        total_tokens = count_messages_tokens(messages)
        threshold = max_tokens * 0.9

        if total_tokens > threshold:
            logger.warning(
                f"Context approaching limit: {total_tokens}/{max_tokens} tokens "
                f"({total_tokens / max_tokens * 100:.1f}%)"
            )
        elif total_tokens > threshold_80:
            logger.debug(f"Context at {total_tokens / max_tokens * 100:.1f}% capacity")

    async def _emit_progress(
        self,
        channel: str,
        chat_id: str,
        kind: ProgressKind,
        detail: str = "",
        tool_name: str | None = None,
        iteration: int = 0,
        total_iterations: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Emit a progress event to the message bus."""
        event = ProgressEvent(
            channel=channel,
            chat_id=chat_id,
            kind=kind,
            detail=detail,
            tool_name=tool_name,
            iteration=iteration,
            total_iterations=total_iterations,
            metadata=metadata or {},
        )
        await self.bus.publish_progress(event)

    def _record_usage(
        self,
        response_usage: dict[str, int],
        session_key: str,
        subsystem: str = "main",
    ) -> None:
        """Record token usage from an LLM response."""
        if not response_usage:
            return
        try:
            self._usage_tracker.record(
                UsageRecord(
                    model=self.model,
                    prompt_tokens=response_usage.get("prompt_tokens", 0),
                    completion_tokens=response_usage.get("completion_tokens", 0),
                    total_tokens=response_usage.get("total_tokens", 0),
                    session_key=session_key,
                    subsystem=subsystem,
                )
            )
        except Exception as e:
            logger.debug(f"Failed to record usage: {e}")

    async def _stream_final_response(
        self,
        messages: list[dict],
        channel: str,
        chat_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Stream the final LLM response with edit-in-place progress events.

        Instead of returning content from a chat() call, this re-calls the
        LLM with stream() and emits STREAMING progress events at intervals
        so channels can update their messages in real time.
        """
        content = ""
        last_emit = 0.0
        interval = self._streaming_config.edit_interval_ms / 1000.0
        min_chars = self._streaming_config.min_chunk_chars

        async for chunk in self.provider.stream(
            messages=messages,
            model=self.model,
            temperature=self.temperature,
        ):
            if chunk.content:
                content += chunk.content

                now = time.time()
                if len(content) >= min_chars and now - last_emit >= interval:
                    await self._emit_progress(
                        channel,
                        chat_id,
                        ProgressKind.STREAMING,
                        detail=content,
                        metadata=metadata,
                    )
                    last_emit = now

            if chunk.finish_reason:
                # Record streaming usage if available
                if chunk.usage:
                    self._record_usage(chunk.usage, f"{channel}:{chat_id}")
                break

        return content or ""

    # Patterns that indicate the LLM claims to have performed an action
    _ACTION_CLAIM_PATTERN = re.compile(
        r"\b(?:"
        r"I(?:'ve| have) (?:updated|written|created|set up|configured|initialized"
        r"|populated|added|installed|scheduled|removed|deleted|modified|saved"
        r"|recorded|re-initialized|activated|established)"
        r"|I (?:updated|wrote|created|set up|configured|initialized|populated"
        r"|added|installed|scheduled|removed|deleted|modified|saved|activated)"
        r"|(?:Changes|Updates|Modifications) (?:have been|were) (?:made|applied|saved)"
        r"|(?:File|Config|Settings|Database) (?:has been|was) (?:updated|written|created)"
        r")\b",
        re.IGNORECASE,
    )

    def _contains_unverified_actions(self, content: str) -> bool:
        """Detect if a response claims actions were performed.

        Used to catch cases where the LLM describes actions in text
        without actually calling tools.
        """
        if not content:
            return False
        return bool(self._ACTION_CLAIM_PATTERN.search(content))

    def _detect_clarification(self, content: str) -> bool:
        """Detect whether the response is a clarification request."""
        if not content:
            return False
        # Short response containing a question mark
        return "?" in content and len(content) < 500

    def _parse_clarification_options(
        self,
        content: str,
    ) -> list[dict]:
        """Parse numbered options from a clarification response.

        Returns a list of button component dicts for interactive replies.
        """
        options = re.findall(
            r"^\s*(\d+)[.)]\s+(.+)$",
            content,
            re.MULTILINE,
        )
        if len(options) < 2:
            return []
        return [
            {
                "type": "button",
                "label": f"{num}. {text[:40]}",
                "callback_data": text,
            }
            for num, text in options[:5]
        ]

    def _handle_guardrail_callback(self, content: str) -> None:
        """Route guardrail approve/deny callbacks to the engine."""
        parts = content.split(":", 2)
        if len(parts) != 3:
            return
        action, approval_id = parts[1], parts[2]
        if action == "approve":
            self._guardrails.approve(approval_id)
        elif action == "deny":
            self._guardrails.deny(approval_id)

    def _check_follow_up(
        self,
        tool_name: str,
        arguments: dict,
        result: str,
    ) -> str | None:
        """Check if a tool execution warrants a proactive follow-up hint.

        Returns a hint string to append to the tool result, or None.
        """
        if tool_name == "cron" and "add" in str(arguments.get("action", "")):
            return "[Hint: A cron job was added. Consider confirming the schedule with the user.]"
        if tool_name == "write_file":
            path = str(arguments.get("path", ""))
            if any(s in path for s in ("config", ".env", ".json", ".yaml")):
                return "[Hint: A configuration file was written. Verify the changes are correct.]"
        if tool_name == "exec":
            # Long-running command (result > 2000 chars suggests complexity)
            if len(result) > 2000:
                return (
                    "[Hint: This command produced extensive output. "
                    "Consider summarising key results for the user.]"
                )
        return None

    async def _maybe_compact(
        self,
        messages: list[dict],
        session: "Session",
    ) -> list[dict]:
        """Run compaction if context is near capacity and compaction is enabled."""
        from nanobot.agent.compaction import MessageCompactor, should_compact

        if not self.compaction_config.enabled:
            return messages

        max_tokens = self.context_config.max_context_tokens
        threshold = self.compaction_config.threshold

        if not await should_compact(messages, max_tokens, threshold):
            return messages

        # Resolve compaction provider (may differ from main)
        compaction_provider = self._resolve_subsystem_provider(self.compaction_config.provider)

        # Create compactor
        compactor = MessageCompactor(
            provider=compaction_provider,
            model=self.compaction_config.model or self.model,
            keep_recent=self.compaction_config.keep_recent,
        )

        # Get previous rolling summary
        previous_summary = session.get_rolling_summary()

        # Run compaction
        target_tokens = int(max_tokens * 0.6)  # Compact to 60% capacity
        compacted, new_summary = await compactor.compact_messages(
            messages, target_tokens, previous_summary
        )

        # Store new rolling summary
        if new_summary != previous_summary:
            session.set_rolling_summary(new_summary)

        return compacted

    async def _urgent_compact(
        self,
        messages: list[dict],
        session: "Session",
    ) -> list[dict]:
        """Force compaction when context is critically full (95%+)."""
        if not self.compaction_config.enabled:
            return messages

        from nanobot.utils.tokens import count_messages_tokens

        total = count_messages_tokens(messages)
        limit = self.context_config.max_context_tokens
        if total < limit * 0.95:
            return messages

        logger.warning(f"Urgent compaction: {total}/{limit} tokens ({total / limit * 100:.0f}%)")

        from nanobot.agent.compaction import MessageCompactor

        compaction_provider = self._resolve_subsystem_provider(
            self.compaction_config.provider,
        )
        compactor = MessageCompactor(
            provider=compaction_provider,
            model=self.compaction_config.model or self.model,
            keep_recent=self.compaction_config.keep_recent,
        )
        previous_summary = session.get_rolling_summary()
        target = int(limit * 0.5)
        compacted, new_summary = await compactor.compact_messages(
            messages,
            target,
            previous_summary,
        )
        if new_summary != previous_summary:
            session.set_rolling_summary(new_summary)
        return compacted

    async def run(self) -> None:
        """Run the agent loop, processing messages from the bus."""
        self._running = True

        # Initialize MCP if enabled
        await self._init_mcp()

        # Initialize memory if enabled
        self._init_memory()

        # Wire vector store into cron tool for cleanup on job removal
        if self.cron_service and self.vector_store:
            from nanobot.agent.tools.cron import CronTool

            cron_tool = self.tools.get("cron")
            if isinstance(cron_tool, CronTool):
                cron_tool._vector_store = self.vector_store

        # Initialize memory extraction pipeline
        self._init_extraction()

        # Wire intent classifier LLM provider if configured
        if self._intent_config.enabled and self._intent_config.llm_fallback:
            classifier_provider = self._resolve_subsystem_provider(
                self._intent_config.classifier_provider
            )
            self._intent_classifier.provider = classifier_provider
            self._intent_classifier.model = self._intent_config.classifier_model or self.model

        # Check for restart signal (e.g., after MCP server installation)
        await self._check_restart_signal()

        logger.info("Agent loop started")

        while self._running:
            try:
                # Wait for next message
                msg = await asyncio.wait_for(self.bus.consume_inbound(), timeout=1.0)

                # Process it
                try:
                    response = await self._process_message(msg)
                    if response:
                        await self.bus.publish_outbound(response)
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    # Send error response (include metadata for reaction/typing cleanup)
                    await self.bus.publish_outbound(
                        OutboundMessage(
                            channel=msg.channel,
                            chat_id=msg.chat_id,
                            content=f"Sorry, I encountered an error: {str(e)}",
                            metadata=msg.metadata,
                        )
                    )
            except asyncio.TimeoutError:
                continue

    async def stop(self) -> None:
        """Stop the agent loop."""
        self._running = False

        # Wait for in-flight background tasks (memory indexing, etc.)
        if self._background_tasks:
            logger.debug(f"Draining {len(self._background_tasks)} background tasks")
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
            self._background_tasks.clear()

        # Stop MCP servers
        if self.mcp_manager:
            await self.mcp_manager.stop()

        # Close extraction vector store
        if self._consolidator and hasattr(self._consolidator, "store"):
            try:
                self._consolidator.store.close()
            except Exception:
                pass

        logger.info("Agent loop stopping")

    async def _process_message(self, msg: InboundMessage) -> OutboundMessage | None:
        """
        Process a single inbound message.

        Acquires processing lock to serialize with daemon execution.
        Guardrail callbacks are routed outside the lock so they can
        unblock a tool execution waiting on approval.

        Args:
            msg: The inbound message to process.

        Returns:
            The response message, or None if no response needed.
        """
        # Handle guardrail callbacks outside the lock
        if msg.content.startswith("guardrail:"):
            self._handle_guardrail_callback(msg.content)
            return None

        async with self._processing_lock:
            trace_id = self._tracer.new_trace()
            async with self._tracer.span(
                "process_message",
                attributes={"channel": msg.channel, "trace_id": trace_id},
            ):
                return await self._process_message_unlocked(msg)

    async def _process_message_unlocked(self, msg: InboundMessage) -> OutboundMessage | None:
        """Inner message processing (called under lock)."""
        # Handle system messages (subagent announces)
        # The chat_id contains the original "channel:chat_id" to route back to
        if msg.channel == "system":
            return await self._process_system_message(msg)

        logger.info(f"Processing message from {msg.channel}:{msg.sender_id}")

        # Notify heartbeat of user activity for dynamic intervals
        if hasattr(self, "_heartbeat") and self._heartbeat:
            self._heartbeat.notify_user_activity()

        # Get or create session
        session = self.sessions.get_or_create(msg.session_key)

        # Update tool contexts
        self._update_tool_contexts(msg.channel, msg.chat_id)

        # Extract channel context from metadata (e.g., Discord channel history)
        channel_context = msg.metadata.get("channel_context", "") if msg.metadata else ""

        # Extract reply context (the message being replied to)
        reply_context = msg.metadata.get("reply_context", "") if msg.metadata else ""
        if reply_context:
            reply_header = f"[Replying to: {reply_context}]"
            if channel_context:
                channel_context = f"{channel_context}\n\n{reply_header}"
            else:
                channel_context = reply_header

        # Classify intent and auto-recall in parallel where possible.
        # Start optimistic memory recall concurrently with intent classification;
        # discard the result only if intent turns out to be FACTUAL.
        intent_task = asyncio.create_task(
            self._intent_classifier.classify(msg.content)
        )
        recall_task = asyncio.create_task(
            self._auto_recall_optimistic(msg.content)
        )

        intent = await intent_task
        logger.debug(f"Query intent: {intent.value}")

        # Get recall result (already computed in parallel)
        memory_context = await recall_task
        if intent == QueryIntent.FACTUAL:
            memory_context = None  # discard for factual queries
        if memory_context:
            if channel_context:
                channel_context = f"{memory_context}\n\n{channel_context}"
            else:
                channel_context = memory_context

        # For FACTUAL/ACTION queries, inject a tool-use directive into the message
        user_content = msg.content
        if intent == QueryIntent.FACTUAL:
            user_content = (
                "[SYSTEM: This query requires verifiable facts. "
                "You MUST use a tool to verify your answer.]\n\n" + user_content
            )
        elif intent == QueryIntent.ACTION:
            user_content = (
                "[SYSTEM: This is an action request. You MUST call the "
                "appropriate tools to execute it. Do NOT describe actions "
                "in text without calling tools. After calling a tool, check "
                "its result before reporting success.]\n\n" + user_content
            )

        # Subagent visibility: inject running subagent count into context
        running_count = self.subagents.get_running_count()
        if running_count > 0:
            subagent_note = f"[{running_count} background subagent(s) currently running]"
            channel_context = (
                f"{subagent_note}\n\n{channel_context}" if channel_context else subagent_note
            )

        # Build initial messages (use token-aware history)
        messages = self.context.build_messages(
            history=session.get_history(max_tokens=self.context_config.history_budget),
            current_message=user_content,
            media=msg.media if msg.media else None,
            channel_context=channel_context,
            system_prompt_budget=self.context_config.system_prompt_budget,
        )

        # Run compaction if needed
        messages = await self._maybe_compact(messages, session)

        # Determine tool_choice based on intent
        forced_tool_choice: str | None = None
        if intent in (QueryIntent.FACTUAL, QueryIntent.ACTION):
            forced_tool_choice = "required"

        # Agent loop
        iteration = 0
        tools_called = 0
        final_content = None

        while iteration < self.max_iterations:
            iteration += 1

            # Check context budget before LLM call
            self._check_context_budget(messages)
            messages = await self._urgent_compact(messages, session)

            # Call LLM -- use low temperature for tool-calling determinism
            tool_defs = self.tools.get_definitions()
            current_temp = self.tool_temperature if tool_defs else self.temperature
            # Emit thinking progress
            await self._emit_progress(
                msg.channel,
                msg.chat_id,
                ProgressKind.THINKING,
                detail=f"Iteration {iteration}/{self.max_iterations}",
                iteration=iteration,
                total_iterations=self.max_iterations,
                metadata=msg.metadata,
            )

            # Force tool use on first iteration for FACTUAL/ACTION queries
            tc = forced_tool_choice if iteration == 1 else None
            async with self._tracer.span(
                "llm_call",
                attributes={"model": self.model, "iteration": iteration},
            ):
                response = await self.provider.chat(
                    messages=messages,
                    tools=tool_defs,
                    model=self.model,
                    temperature=current_temp,
                    tool_choice=tc,
                )

            # Record token usage
            self._record_usage(response.usage, msg.session_key)

            # Fallback: some models emit tool calls as XML in content instead of structured tool_calls
            if not response.has_tool_calls and response.content and "<tool_call>" in response.content:
                parsed = _parse_tool_calls_from_content(response.content)
                if parsed:
                    logger.debug(f"Parsed {len(parsed)} tool call(s) from response content")
                    # Strip XML blocks so we do not store or send raw tool_call tags to the user
                    cleaned_content = _strip_tool_call_blocks_from_content(response.content)
                    response = LLMResponse(
                        content=cleaned_content,
                        tool_calls=parsed,
                        finish_reason=response.finish_reason,
                        usage=response.usage,
                    )

            # Handle tool calls
            if response.has_tool_calls:
                tools_called += len(response.tool_calls)
                # Add assistant message with tool calls
                tool_call_dicts = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments),  # Must be JSON string
                        },
                    }
                    for tc in response.tool_calls
                ]
                messages = self.context.add_assistant_message(
                    messages, response.content, tool_call_dicts
                )

                # Execute tools — parallelize independent tool calls
                tool_results: list[tuple[Any, str]] = []  # (tool_call, result)

                # Phase 1: Pre-validate and check guardrails (fast, synchronous-ish)
                executable: list[Any] = []
                for tool_call in response.tool_calls:
                    tool = self.tools.get(tool_call.name)
                    if not tool:
                        result = f"Error: unknown tool '{tool_call.name}'"
                        logger.warning(f"LLM called unknown tool: {tool_call.name}")
                        tool_results.append((tool_call, result))
                        continue

                    errors = tool.validate_params(tool_call.arguments)
                    if errors:
                        result = (
                            f"Error: invalid arguments for "
                            f"{tool_call.name} - {'; '.join(errors)}"
                        )
                        logger.warning(f"Tool validation failed: {tool_call.name} - {errors}")
                        tool_results.append((tool_call, result))
                        continue

                    # Guardrail check before execution
                    rule = self._guardrails.check(
                        tool_call.name,
                        tool_call.arguments,
                    )
                    if rule:
                        approval_id = self._guardrails.request_approval(
                            rule,
                            tool_call.name,
                            tool_call.arguments,
                        )
                        approved = await self._guardrails.wait_for_approval(
                            approval_id,
                        )
                        if not approved:
                            result = (
                                f"Error: {rule.description} blocked by guardrail "
                                f"(approval denied or timed out)"
                            )
                            tool_results.append((tool_call, result))
                            continue

                    executable.append(tool_call)

                # Phase 2: Execute validated tools in parallel
                if len(executable) > 1:
                    # Parallel execution for multiple independent tool calls
                    async def _exec_one(tc):
                        args_str = json.dumps(tc.arguments)
                        logger.debug(
                            f"Executing tool: {tc.name} with arguments: {args_str}"
                        )
                        await self._emit_progress(
                            msg.channel, msg.chat_id,
                            ProgressKind.TOOL_START,
                            tool_name=tc.name,
                            detail=f"Executing {tc.name}",
                            iteration=iteration,
                            total_iterations=self.max_iterations,
                            metadata=msg.metadata,
                        )
                        async with self._tracer.span(
                            "tool_exec", attributes={"tool": tc.name},
                        ):
                            res = await self.tools.execute(tc.name, tc.arguments)
                        tool_status = (
                            "error"
                            if isinstance(res, str) and res.startswith("Error")
                            else "ok"
                        )
                        await self._emit_progress(
                            msg.channel, msg.chat_id,
                            ProgressKind.TOOL_COMPLETE,
                            tool_name=tc.name,
                            detail=f"{tc.name}: {tool_status}",
                            iteration=iteration,
                            total_iterations=self.max_iterations,
                            metadata=msg.metadata,
                        )
                        return res

                    results = await asyncio.gather(
                        *[_exec_one(tc) for tc in executable],
                        return_exceptions=True,
                    )
                    for tc, res in zip(executable, results):
                        if isinstance(res, Exception):
                            tool_results.append((tc, f"Error executing {tc.name}: {res}"))
                        else:
                            tool_results.append((tc, res))
                elif len(executable) == 1:
                    # Single tool — no gather overhead
                    tc = executable[0]
                    args_str = json.dumps(tc.arguments)
                    logger.debug(
                        f"Executing tool: {tc.name} with arguments: {args_str}"
                    )
                    await self._emit_progress(
                        msg.channel, msg.chat_id,
                        ProgressKind.TOOL_START,
                        tool_name=tc.name,
                        detail=f"Executing {tc.name}",
                        iteration=iteration,
                        total_iterations=self.max_iterations,
                        metadata=msg.metadata,
                    )
                    async with self._tracer.span(
                        "tool_exec", attributes={"tool": tc.name},
                    ):
                        result = await self.tools.execute(tc.name, tc.arguments)
                    tool_status = (
                        "error"
                        if isinstance(result, str) and result.startswith("Error")
                        else "ok"
                    )
                    await self._emit_progress(
                        msg.channel, msg.chat_id,
                        ProgressKind.TOOL_COMPLETE,
                        tool_name=tc.name,
                        detail=f"{tc.name}: {tool_status}",
                        iteration=iteration,
                        total_iterations=self.max_iterations,
                        metadata=msg.metadata,
                    )
                    tool_results.append((tc, result))

                # Phase 3: Post-process results (reflexion, hints, truncation)
                for tool_call, result in tool_results:
                    if isinstance(result, str) and result.startswith("Error"):
                        count = self._tool_failure_counts.get(tool_call.name, 0) + 1
                        self._tool_failure_counts[tool_call.name] = count
                        if count >= self._tool_failure_threshold:
                            self._record_tool_lesson(
                                tool_call.name,
                                tool_call.arguments,
                                result,
                                count,
                            )
                            self._tool_failure_counts[tool_call.name] = 0
                    else:
                        self._tool_failure_counts.pop(tool_call.name, None)

                    # Proactive follow-up hints (F12)
                    hint = self._check_follow_up(
                        tool_call.name,
                        tool_call.arguments,
                        result,
                    )
                    if hint:
                        result = f"{result}\n{hint}"

                    result = self._truncate_tool_result(result)
                    messages = self.context.add_tool_result(
                        messages, tool_call.id, tool_call.name, result
                    )

                # Post-iteration compaction check (once per iteration, not per tool)
                messages = await self._maybe_compact(messages, session)
            else:
                # No tool calls -- final response
                if self._streaming_config.enabled:
                    final_content = await self._stream_final_response(
                        messages,
                        msg.channel,
                        msg.chat_id,
                        msg.metadata,
                    )
                else:
                    final_content = response.content
                break

        if final_content is None:
            final_content = "I've completed processing but have no response to give."

        # Post-loop action verification: detect hallucinated actions.
        # If the LLM claimed to perform actions but never called any tools,
        # inject a correction and re-enter the loop once.
        if (
            intent == QueryIntent.ACTION
            and tools_called == 0
            and self._contains_unverified_actions(final_content)
            and iteration < self.max_iterations
        ):
            logger.warning(
                "Action response claimed actions without tool calls -- "
                "re-entering loop with correction"
            )
            messages = self.context.add_assistant_message(messages, final_content)
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "[SYSTEM: Your previous response claimed to have performed "
                        "actions (writing files, updating configs, etc.) but you did "
                        "NOT call any tools. The actions did NOT happen. You MUST now "
                        "call the appropriate tools to actually execute the requested "
                        "actions. Do not describe -- execute.]"
                    ),
                }
            )
            # Single retry pass with forced tool use
            response = await self.provider.chat(
                messages=messages,
                tools=self.tools.get_definitions(),
                model=self.model,
                temperature=self.tool_temperature,
                tool_choice="required",
            )
            self._record_usage(response.usage, msg.session_key)

            if response.has_tool_calls:
                tool_call_dicts = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments),
                        },
                    }
                    for tc in response.tool_calls
                ]
                messages = self.context.add_assistant_message(
                    messages, response.content, tool_call_dicts
                )
                for tool_call in response.tool_calls:
                    result = await self.tools.execute(tool_call.name, tool_call.arguments)
                    result = self._truncate_tool_result(result)
                    messages = self.context.add_tool_result(
                        messages, tool_call.id, tool_call.name, result
                    )

                # Get final response after tool execution
                final_response = await self.provider.chat(
                    messages=messages,
                    tools=self.tools.get_definitions(),
                    model=self.model,
                    temperature=self.temperature,
                )
                self._record_usage(final_response.usage, msg.session_key)
                if final_response.content:
                    final_content = final_response.content

        # Save to session
        session.add_message("user", msg.content)
        session.add_message("assistant", final_content)
        self.sessions.save(session)

        # Log session usage summary
        self._usage_tracker.log_session_summary(msg.session_key)

        # Clarification detection (F2)
        components: list[dict] = []
        if self._detect_clarification(final_content):
            await self._emit_progress(
                msg.channel,
                msg.chat_id,
                ProgressKind.CLARIFICATION,
                detail="Asking for clarification",
                metadata=msg.metadata,
            )
            components = self._parse_clarification_options(final_content)

        # Index conversation to memory (fire-and-forget, don't block response)
        self._spawn_background(self._index_conversation(
            msg.session_key, msg.content, final_content,
        ))

        # Periodic extraction and consolidation of facts and lessons
        self._spawn_background(self._maybe_extract_and_consolidate(
            session, msg.session_key,
        ))

        return OutboundMessage(
            channel=msg.channel,
            chat_id=msg.chat_id,
            content=final_content,
            metadata=msg.metadata,
            components=components,
        )

    async def _process_system_message(self, msg: InboundMessage) -> OutboundMessage | None:
        """
        Process a system message (e.g., subagent announce).

        The chat_id field contains "original_channel:original_chat_id" to route
        the response back to the correct destination.
        """
        logger.info(f"Processing system message from {msg.sender_id}")

        # Parse origin from chat_id (format: "channel:chat_id")
        if ":" in msg.chat_id:
            parts = msg.chat_id.split(":", 1)
            origin_channel = parts[0]
            origin_chat_id = parts[1]
        else:
            # Fallback
            origin_channel = "cli"
            origin_chat_id = msg.chat_id

        # Use the origin session for context
        session_key = f"{origin_channel}:{origin_chat_id}"
        session = self.sessions.get_or_create(session_key)

        # Update tool contexts
        self._update_tool_contexts(origin_channel, origin_chat_id)

        # Build messages with the announce content (token-aware history)
        messages = self.context.build_messages(
            history=session.get_history(max_tokens=self.context_config.history_budget),
            current_message=msg.content,
            system_prompt_budget=self.context_config.system_prompt_budget,
        )

        # Run compaction if needed
        messages = await self._maybe_compact(messages, session)

        # Agent loop (limited for announce handling)
        iteration = 0
        final_content = None

        while iteration < self.max_iterations:
            iteration += 1

            # Check context budget before LLM call
            self._check_context_budget(messages)

            response = await self.provider.chat(
                messages=messages, tools=self.tools.get_definitions(), model=self.model
            )

            if response.has_tool_calls:
                tool_call_dicts = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                    }
                    for tc in response.tool_calls
                ]
                messages = self.context.add_assistant_message(
                    messages, response.content, tool_call_dicts
                )

                for tool_call in response.tool_calls:
                    args_str = json.dumps(tool_call.arguments)
                    logger.debug(f"Executing tool: {tool_call.name} with arguments: {args_str}")
                    result = await self.tools.execute(tool_call.name, tool_call.arguments)
                    # Truncate large tool results to fit within budget
                    result = self._truncate_tool_result(result)
                    messages = self.context.add_tool_result(
                        messages, tool_call.id, tool_call.name, result
                    )
            else:
                final_content = response.content
                break

        if final_content is None:
            final_content = "Background task completed."

        # Save to session (mark as system message in history)
        session.add_message("user", f"[System: {msg.sender_id}] {msg.content}")
        session.add_message("assistant", final_content)
        self.sessions.save(session)

        self._spawn_background(self._maybe_extract_and_consolidate(
            session, session_key,
        ))

        return OutboundMessage(
            channel=origin_channel, chat_id=origin_chat_id, content=final_content
        )

    async def process_direct(
        self,
        content: str,
        session_key: str = "cli:direct",
        channel: str | None = None,
        chat_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        Process a message directly (for CLI, cron, and daemon usage).

        Args:
            content: The message content.
            session_key: Session identifier (format: "channel:chat_id").
            channel: Explicit channel name. Parsed from session_key if omitted.
            chat_id: Explicit chat ID. Parsed from session_key if omitted.
            metadata: Optional metadata to attach (e.g., for channel context).

        Returns:
            The agent's response.
        """
        if channel and chat_id:
            ch, cid = channel, chat_id
        elif ":" in session_key:
            ch, cid = session_key.split(":", 1)
        else:
            ch, cid = "cli", session_key

        msg = InboundMessage(
            channel=ch,
            sender_id="system",
            chat_id=cid,
            content=content,
            metadata=metadata or {},
        )
        response = await self._process_message(msg)
        return response.content if response else ""

    async def _auto_recall_optimistic(
        self,
        user_message: str,
    ) -> str | None:
        """Optimistic auto-recall: performs memory search without intent filtering.

        Called concurrently with intent classification. The caller is
        responsible for discarding the result if intent turns out to be
        FACTUAL. Uses CONVERSATIONAL intent internally (no special
        annotations) — the caller can re-annotate if needed.
        """
        return await self._auto_recall(user_message, QueryIntent.CONVERSATIONAL)

    async def _auto_recall(
        self,
        user_message: str,
        intent: QueryIntent = QueryIntent.CONVERSATIONAL,
    ) -> str | None:
        """
        Automatically search memory for context relevant to the user's message.

        Memory injection is conditional on query intent:
        - FACTUAL: skip memory entirely (force tool use instead)
        - VERIFY_STATE: inject memory with [UNVERIFIED] annotations
        - MEMORY/ACTION/CONVERSATIONAL: inject memory normally with timestamps

        Args:
            user_message: The user's incoming message.
            intent: Classified query intent.

        Returns:
            Relevant memory context, or None if disabled or no results.
        """
        if not self.memory_config.enabled or not self.memory_config.auto_recall:
            return None

        if not self.vector_store:
            return None

        # Skip memory for factual queries -- tools should provide the answer
        if intent == QueryIntent.FACTUAL:
            logger.debug("Skipping memory injection for FACTUAL query")
            return None

        try:
            memory_context: list[str] = []

            # Deterministic recall: pure similarity, no time decay or type bias
            if self.memory_config.deterministic_recall:
                results = await self.vector_store.search(
                    query=user_message,
                    top_k=self.memory_config.search_top_k,
                    min_similarity=self.memory_config.min_similarity,
                    recency_weight=0.0,
                    type_weights={},
                )
            else:
                results = await self.vector_store.search(
                    query=user_message,
                    top_k=self.memory_config.search_top_k,
                    min_similarity=self.memory_config.min_similarity,
                    recency_weight=self.memory_config.recency_weight,
                    type_weights={"fact": 1.2, "conversation": 1.0},
                )

            if results:
                if intent == QueryIntent.VERIFY_STATE:
                    memory_context.append(
                        "[Recalled memories -- these describe MUTABLE STATE "
                        "that may have changed.\n"
                        "You MUST verify each claim with tools before "
                        "presenting it as current.]"
                    )
                else:
                    memory_context.append(
                        "[Relevant memories from past conversations "
                        "-- may be outdated. Verify factual claims "
                        "with tools.]"
                    )

                for r in results:
                    text = r.get("text", "")
                    text = _truncate_at_sentence(text, 300)
                    age = _format_memory_age(r.get("created_at", ""))
                    entry = f"- [{age}] {text}"

                    # Annotate mutable state entries
                    if intent == QueryIntent.VERIFY_STATE:
                        entry = self._annotate_mutable_state(entry)

                    memory_context.append(entry)

            # Proactive reminders
            if self.proactive_memory:
                try:
                    reminders = await self.proactive_memory.get_reminders()
                    if reminders:
                        memory_context.append("\n[Upcoming reminders]")
                        for r in reminders:
                            memory_context.append(f"- {r}")
                except Exception as e:
                    logger.warning(f"Proactive reminders failed: {e}")

            return "\n".join(memory_context) if memory_context else None

        except Exception as e:
            logger.warning(f"Auto-recall failed: {e}")
            return None

    def _annotate_mutable_state(self, entry: str) -> str:
        """Add verification hints to memory entries about mutable state."""
        import re as _re

        text_lower = entry.lower()
        if _re.search(r"\b(reminder|cron|schedule|job|recurring)\b", text_lower):
            return entry + "\n    -> VERIFY: use `cron` tool to check"
        if _re.search(r"\b(file|exists|created)\b", text_lower):
            return entry + "\n    -> VERIFY: use `read_file` or `exec`"
        if _re.search(r"\b(running|process|service|active)\b", text_lower):
            return entry + "\n    -> VERIFY: use `exec` to check"
        return entry

    async def _index_conversation(
        self,
        session_key: str,
        user_message: str,
        assistant_message: str,
    ) -> None:
        """
        Index a conversation turn to the vector store.

        Args:
            session_key: Session identifier.
            user_message: The user's message.
            assistant_message: The assistant's response.
        """
        if not self.memory_config.enabled or not self.memory_config.index_conversations:
            return

        if not self.vector_store:
            return

        try:
            # Create conversation turn text
            turn_text = f"User: {user_message}\nAssistant: {assistant_message}"

            # Sanitize before indexing
            from nanobot.memory.filters import sanitize_for_memory

            sanitized = sanitize_for_memory(turn_text)
            if sanitized is None:
                logger.debug("Conversation turn filtered by sanitization")
                return

            metadata = {
                "session_key": session_key,
                "type": "conversation",
            }

            await self.vector_store.add(sanitized, metadata)

            # Extract and index facts if enabled
            if self.memory_config.extract_facts:
                await self._extract_and_index_facts(
                    session_key,
                    user_message,
                    assistant_message,
                )

            # Extract entities if enabled
            if self.memory_config.enable_entities and self.entity_store:
                await self._extract_and_index_entities(
                    user_message,
                    assistant_message,
                )

            # Record interaction pattern for proactive learning
            if self.proactive_memory:
                try:
                    from datetime import datetime

                    self.proactive_memory.record_interaction_pattern(
                        session_key=session_key,
                        topic=user_message[:100],
                        timestamp=datetime.now().isoformat(),
                    )
                except Exception as e:
                    logger.debug(f"Pattern recording failed: {e}")

        except Exception as e:
            logger.warning(f"Failed to index conversation: {e}")

    async def _extract_and_index_facts(
        self,
        session_key: str,
        user_message: str,
        assistant_message: str,
    ) -> None:
        """Extract and index key facts from a conversation turn."""
        try:
            from nanobot.memory.extractor import FactExtractor

            # Use extraction model or fall back to compaction model or main model
            extraction_model = (
                self.memory_config.extraction_model or self.compaction_config.model or self.model
            )

            # Resolve extraction provider: extraction -> compaction -> main
            extraction_provider = self._resolve_subsystem_provider(
                self.memory_config.extraction_provider or self.compaction_config.provider
            )

            extractor = FactExtractor(
                provider=extraction_provider,
                model=extraction_model,
            )

            facts = await extractor.extract_from_turn(user_message, assistant_message)

            for fact in facts:
                metadata = {
                    "session_key": session_key,
                    "type": "fact",
                }
                await self.vector_store.add(fact, metadata)

            if facts:
                logger.debug(f"Indexed {len(facts)} facts from conversation")

        except Exception as e:
            logger.warning(f"Fact extraction failed: {e}")

    async def _extract_and_index_entities(
        self,
        user_message: str,
        assistant_message: str,
    ) -> None:
        """Extract entities and relations from a conversation turn."""
        if not self.entity_store:
            return

        try:
            extraction_model = (
                self.memory_config.extraction_model or self.compaction_config.model or self.model
            )

            # Resolve extraction provider: extraction -> compaction -> main
            extraction_provider = self._resolve_subsystem_provider(
                self.memory_config.extraction_provider or self.compaction_config.provider
            )

            prompt = (
                "Extract named entities and their relationships "
                "from this conversation.\n\n"
                f"User: {user_message}\nAssistant: {assistant_message}\n\n"
                "Output as JSON array of objects, each with:\n"
                '- "name": entity name\n'
                '- "type": one of person/project/organization/'
                "technology/location/other\n"
                '- "relations": array of '
                '{"relation": "verb", "target": "other entity name"}'
                " (optional)\n\n"
                "Only include clearly identified entities. "
                "If none, output [].\nJSON:"
            )

            response = await extraction_provider.chat(
                messages=[{"role": "user", "content": prompt}],
                model=extraction_model,
                max_tokens=512,
                temperature=0.2,
            )

            content = (response.content or "").strip()
            if not content:
                logger.debug("Entity extraction returned empty response")
                return

            # Handle markdown code blocks
            if content.startswith("```"):
                content = content.split("\n", 1)[1].rsplit("```", 1)[0].strip()

            if not content:
                return

            entities = json.loads(content)
            if not isinstance(entities, list):
                return

            for entity in entities:
                name = entity.get("name", "").strip()
                etype = entity.get("type", "other").strip()
                if not name:
                    continue

                self.entity_store.upsert_entity(name, etype)

                for rel in entity.get("relations", []):
                    relation = rel.get("relation", "").strip()
                    target = rel.get("target", "").strip()
                    if relation and target:
                        self.entity_store.add_relation(name, relation, target)

            if entities:
                logger.debug(f"Extracted {len(entities)} entities from conversation")

        except json.JSONDecodeError as e:
            logger.debug(f"Entity extraction JSON parse failed: {e}")
        except Exception as e:
            logger.warning(f"Entity extraction failed: {e}")

    def _init_memory(self) -> None:
        """Initialize semantic memory system."""
        if not self.memory_config.enabled:
            return

        try:
            from nanobot.agent.tools.memory_search import MemorySearchTool
            from nanobot.llm.embeddings import EmbeddingService
            from nanobot.memory.vectors import VectorStore

            # Resolve embedding provider credentials
            embed_key, embed_base = None, None
            if self.provider_resolver:
                embed_key, embed_base = self.provider_resolver.resolve(
                    self.memory_config.embedding_provider
                )
            else:
                embed_key = self.provider.api_key

            # Create embedding service
            embedding_service = EmbeddingService(
                model=self.memory_config.embedding_model,
                api_key=embed_key,
                api_base=embed_base,
            )

            # Create vector store
            self.vector_store = VectorStore(
                db_path=self.memory_config.db_path,
                embedding_service=embedding_service,
            )

            # Register memory search tool
            self.tools.register(MemorySearchTool(self.vector_store))

            # Initialize core memory if enabled
            if self.memory_config.enable_core_memory:
                try:
                    from nanobot.agent.tools.core_memory import (
                        CoreMemoryReadTool,
                        CoreMemoryUpdateTool,
                    )
                    from nanobot.memory.core import CoreMemory

                    self.core_memory = CoreMemory(self.workspace)
                    self.tools.register(CoreMemoryReadTool(self.core_memory))
                    self.tools.register(CoreMemoryUpdateTool(self.core_memory))
                    # Pass core memory to context builder
                    self.context.core_memory = self.core_memory
                    logger.debug("Core memory initialized")
                except Exception as e:
                    logger.warning(f"Failed to initialize core memory: {e}")

            # Initialize entity store if enabled
            if self.memory_config.enable_entities:
                try:
                    from nanobot.memory.entities import EntityStore

                    self.entity_store = EntityStore(
                        self.memory_config.entities_db_path,
                    )
                    logger.debug("Entity store initialized")
                except Exception as e:
                    logger.warning(f"Failed to initialize entity store: {e}")

            # Initialize proactive memory if enabled
            if self.memory_config.enable_proactive:
                try:
                    from nanobot.memory.proactive import ProactiveMemory

                    self.proactive_memory = ProactiveMemory(
                        vector_store=self.vector_store,
                        entity_store=self.entity_store,
                        data_dir=self.workspace / "memory",
                    )
                    logger.debug("Proactive memory initialized")
                except Exception as e:
                    logger.warning(f"Failed to initialize proactive memory: {e}")

            # Register memory forget tool
            try:
                from nanobot.agent.tools.memory_forget import (
                    MemoryForgetTool,
                )

                self.tools.register(MemoryForgetTool(self.vector_store))
            except Exception as e:
                logger.warning(f"Failed to register memory forget tool: {e}")

            logger.info(
                f"Memory initialized: {self.vector_store.count()} entries, "
                f"model={self.memory_config.embedding_model}"
            )

        except Exception as e:
            logger.error(f"Failed to initialize memory: {e}")
            self.vector_store = None

    def _init_extraction(self) -> None:
        """Initialize the memory extraction/consolidation pipeline."""
        if not self._extraction_config.enabled:
            return

        try:
            from nanobot.agent.memory.consolidator import MemoryConsolidator
            from nanobot.agent.memory.extractor import MemoryExtractor
            from nanobot.agent.memory.store import (
                EmbeddingService,
                VectorMemoryStore,
            )

            cfg = self._extraction_config
            self._extractor = MemoryExtractor(
                model=cfg.extraction_model,
                max_facts=cfg.max_facts_per_extraction,
            )

            # Create a dedicated vector store for extracted facts
            from pathlib import Path

            vector_db_path = Path("memory") / "extraction_vectors.db"

            # Resolve embedding credentials (mirrors _init_memory pattern)
            embed_key, embed_base = None, None
            if self.provider_resolver:
                embed_key, embed_base = self.provider_resolver.resolve(cfg.embedding_provider)
            else:
                embed_key = self.provider.api_key

            embedding_service = EmbeddingService(
                model=cfg.embedding_model,
                api_key=embed_key,
                api_base=embed_base,
            )
            extraction_store = VectorMemoryStore(
                db_path=vector_db_path,
                base_dir=self.workspace,
                embedding_service=embedding_service,
                max_memories=cfg.max_memories,
            )

            # Resolve extraction provider if available
            extraction_provider = self._resolve_subsystem_provider(
                self.memory_config.extraction_provider
                if hasattr(self.memory_config, "extraction_provider")
                else None
            )

            self._consolidator = MemoryConsolidator(
                store=extraction_store,
                model=cfg.extraction_model,
                candidate_threshold=cfg.candidate_threshold,
                provider=(extraction_provider if extraction_provider != self.provider else None),
            )

            # Pass extractor to session compactor
            self._session_compactor = SessionCompactor(
                config=SessionCompactionConfig(),
                extractor=self._extractor,
            )

            logger.info(
                f"Memory extraction initialized: model={cfg.extraction_model}, "
                f"interval={cfg.extraction_interval}",
            )
        except Exception as e:
            logger.warning("Failed to initialize memory extraction: %s", e)
            self._extractor = None
            self._consolidator = None

    async def _pre_compaction_flush(
        self,
        history: list[dict[str, Any]],
        namespace: str,
    ) -> None:
        """Run silent memory extraction before compaction."""
        if not self._consolidator or not self._extractor or len(history) < 10:
            return
        try:
            extracted = await self._extractor.extract_for_pre_compaction(history)
            if extracted:
                await self._consolidator.consolidate(extracted, namespace)
                logger.debug(
                    "Pre-compaction flush: consolidated %d facts",
                    len(extracted),
                )
        except Exception as e:
            logger.warning("Pre-compaction memory flush failed: %s", e)

    async def _maybe_extract_and_consolidate(
        self,
        session: "Session",
        namespace: str,
    ) -> None:
        """Trigger extraction/consolidation when conditions are met."""
        if not self._consolidator or not self._extractor:
            return

        user_count = sum(1 for m in session.messages if m.get("role") == "user")
        if user_count <= 0 or user_count % self._extraction_interval != 0:
            return

        history = session.get_history()[-20:]
        await self._extract_and_consolidate(history, namespace)

    async def _extract_and_consolidate(
        self,
        messages: list[dict[str, Any]],
        namespace: str,
    ) -> None:
        """Extract facts and lessons from conversation and consolidate."""
        if not self._consolidator or not self._extractor:
            return

        try:
            # Facts
            extracted = await self._extractor.extract(messages)
            if extracted:
                await self._consolidator.consolidate(extracted, namespace)
                logger.debug("Extracted and consolidated %d facts", len(extracted))

            # Lessons (stored in dedicated namespace)
            lessons = await self._extractor.extract_lessons(messages)
            if lessons:
                await self._consolidator.consolidate(lessons, namespace)
                logger.debug("Extracted and consolidated %d lessons", len(lessons))

            # Tool-specific lessons
            if self._enable_tool_lessons:
                tool_lessons = self._extractor.extract_tool_lessons(messages)
                if tool_lessons:
                    await self._consolidator.consolidate(tool_lessons, namespace)
                    logger.debug(
                        "Extracted and consolidated %d tool lessons",
                        len(tool_lessons),
                    )
        except Exception as e:
            logger.warning("Memory extraction/consolidation failed: %s", e)

    async def _init_mcp(self) -> None:
        """Initialize MCP manager and register MCP tools."""
        if not self.mcp_config.enabled:
            return

        from nanobot.mcp import MCPManager, create_mcp_tool_proxies

        self.mcp_manager = MCPManager(self.mcp_config)
        await self.mcp_manager.start()

        # Register MCP tool proxies
        mcp_tools = create_mcp_tool_proxies(self.mcp_manager)
        for tool in mcp_tools:
            self.tools.register(tool)
            logger.debug(f"Registered MCP tool: {tool.name}")

        # Log any server errors for diagnosis
        for server_name, error in self.mcp_manager.server_errors.items():
            logger.warning(f"MCP server {server_name} failed: {error}")

        logger.info(f"MCP initialized: {len(mcp_tools)} tools available")

    async def _check_restart_signal(self) -> None:
        """Check for restart signal and schedule verification job if needed."""
        from nanobot.restart import check_and_clear_restart_signal

        signal = check_and_clear_restart_signal(self.workspace)
        if not signal:
            return

        logger.info(f"Restart signal detected: {signal.get('reason', 'unknown')}")

        # Schedule verification job if present
        verify_job = signal.get("verify_job")
        if verify_job and self.cron_service:
            from datetime import datetime

            from nanobot.cron.types import CronSchedule

            at_time = verify_job.get("at_time")
            if at_time:
                try:
                    dt = datetime.fromisoformat(at_time.replace("Z", "+00:00"))
                    at_ms = int(dt.timestamp() * 1000)

                    self.cron_service.add_job(
                        name=verify_job.get("name", "verify_mcp"),
                        schedule=CronSchedule(kind="at", at_ms=at_ms),
                        message=verify_job.get("message", "Verify MCP installation"),
                        deliver=verify_job.get("deliver", True),
                        channel=verify_job.get("channel"),
                        to=verify_job.get("to"),
                        delete_after_run=True,
                    )
                    logger.info(f"Scheduled verification job: {verify_job.get('name')}")
                except Exception as e:
                    logger.error(f"Failed to schedule verification job: {e}")
