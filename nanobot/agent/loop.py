"""Agent loop: the core processing engine."""

import asyncio
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from nanobot.agent.context import ContextBuilder
from nanobot.agent.memory import MemoryStore
from nanobot.agent.memory.consolidator import MemoryConsolidator
from nanobot.agent.memory.extractor import MemoryExtractor
from nanobot.agent.subagent import SubagentManager
from nanobot.agent.tools.filesystem import EditFileTool, ListDirTool, ReadFileTool, WriteFileTool
from nanobot.agent.tools.message import MessageTool
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.tools.shell import ExecTool
from nanobot.agent.tools.spawn import SpawnTool
from nanobot.agent.tools.web import WebFetchTool, WebSearchTool
from nanobot.bus.events import InboundMessage, OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.channels.manager import ChannelManager
from nanobot.providers.base import LLMProvider
from nanobot.session.manager import SessionManager
from nanobot.session.compaction import CompactionConfig, SessionCompactor

if TYPE_CHECKING:
    from nanobot.config.schema import (
        ExecToolConfig,
        MemoryExtractionConfig,
        MCPConfig,
    )
    from nanobot.cron.service import CronService
    from nanobot.mcp import MCPManager


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
        extraction_model: str = "gpt-4o-mini",
        embedding_model: str = "text-embedding-3-small",
        max_memories: int = 1000,
        extraction_interval: int = 10,
        memory_extraction: "MemoryExtractionConfig | None" = None,
    ):
        from nanobot.config.schema import (
            ExecToolConfig,
            MemoryExtractionConfig,
            MCPConfig,
        )

        self.bus = bus
        self.provider = provider
        self.workspace = workspace
        self.model = model or provider.get_default_model()
        self.max_iterations = max_iterations
        self.brave_api_key = brave_api_key
        self.exec_config = exec_config or ExecToolConfig()
        self.channel_manager = channel_manager
        self.cron_service = cron_service
        self.mcp_config = mcp_config or MCPConfig()

        mem_cfg = memory_extraction or MemoryExtractionConfig()
        _embed = mem_cfg.embedding_model if memory_extraction else embedding_model
        _max_mem = mem_cfg.max_memories if memory_extraction else max_memories
        _ext_model = mem_cfg.extraction_model if memory_extraction else extraction_model
        _interval = mem_cfg.extraction_interval if memory_extraction else extraction_interval

        # Context and memory
        self.context = ContextBuilder(
            workspace,
            use_vector_memory=mem_cfg.enabled,
            embedding_model=_embed,
            max_memories=_max_mem,
        )
        self.memory = MemoryStore(workspace)
        self.sessions = SessionManager(workspace)
        self.tools = ToolRegistry()

        # Memory extraction and consolidation
        self._extractor = MemoryExtractor(
            model=_ext_model,
            max_facts=mem_cfg.max_facts_per_extraction,
        )
        self._consolidator = (
            MemoryConsolidator(
                store=self.context.vector_memory,
                model=_ext_model,
                candidate_threshold=mem_cfg.candidate_threshold,
            )
            if getattr(self.context, "vector_memory", None) and mem_cfg.enabled
            else None
        )
        self._compactor = SessionCompactor(config=CompactionConfig())
        self._extraction_interval = _interval
        self._enable_pre_compaction_flush = mem_cfg.enable_pre_compaction_flush
        self._enable_tool_lessons = mem_cfg.enable_tool_lessons
        self.subagents = SubagentManager(
            provider=provider,
            workspace=workspace,
            bus=bus,
            model=self.model,
            brave_api_key=brave_api_key,
            exec_config=self.exec_config,
        )

        # MCP manager (initialized in run())
        self.mcp_manager: "MCPManager | None" = None

        self._running = False
        self._register_default_tools()

    def _register_default_tools(self) -> None:
        """Register the default set of tools."""
        # File tools
        self.tools.register(ReadFileTool())
        self.tools.register(WriteFileTool())
        self.tools.register(EditFileTool())
        self.tools.register(ListDirTool())

        # Shell tool
        self.tools.register(
            ExecTool(
                working_dir=str(self.workspace),
                timeout=self.exec_config.timeout,
                restrict_to_workspace=self.exec_config.restrict_to_workspace,
            )
        )

        # Web tools
        self.tools.register(WebSearchTool(api_key=self.brave_api_key))
        self.tools.register(WebFetchTool())

        # Message tool
        message_tool = MessageTool(send_callback=self.bus.publish_outbound)
        self.tools.register(message_tool)

        # Spawn tool (for subagents)
        spawn_tool = SpawnTool(manager=self.subagents)
        self.tools.register(spawn_tool)

        # Cron tool (for self-scheduling)
        if self.cron_service:
            from nanobot.agent.tools.cron import CronTool

            self.tools.register(CronTool(cron_service=self.cron_service))

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

    async def run(self) -> None:
        """Run the agent loop, processing messages from the bus."""
        self._running = True

        # Initialize MCP if enabled
        await self._init_mcp()

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

        # Stop MCP servers
        if self.mcp_manager:
            await self.mcp_manager.stop()

        # Close context resources (e.g. vector memory)
        try:
            self.context.close()
        except Exception:
            pass

        logger.info("Agent loop stopping")

    async def _process_message(self, msg: InboundMessage) -> OutboundMessage | None:
        """
        Process a single inbound message.

        Args:
            msg: The inbound message to process.

        Returns:
            The response message, or None if no response needed.
        """
        # Handle system messages (subagent announces)
        # The chat_id contains the original "channel:chat_id" to route back to
        if msg.channel == "system":
            return await self._process_system_message(msg)

        logger.info(f"Processing message from {msg.channel}:{msg.sender_id}")

        # Get or create session
        session = self.sessions.get_or_create(msg.session_key)

        # Update tool contexts
        message_tool = self.tools.get("message")
        if isinstance(message_tool, MessageTool):
            message_tool.set_context(msg.channel, msg.chat_id)

        spawn_tool = self.tools.get("spawn")
        if isinstance(spawn_tool, SpawnTool):
            spawn_tool.set_context(msg.channel, msg.chat_id)

        # Update cron tool context
        from nanobot.agent.tools.cron import CronTool

        cron_tool = self.tools.get("cron")
        if isinstance(cron_tool, CronTool):
            cron_tool.set_context(msg.channel, msg.chat_id)

        # Update MCP install tool context
        from nanobot.agent.tools.mcp_install import InstallMCPServerTool

        mcp_install_tool = self.tools.get("install_mcp_server")
        if isinstance(mcp_install_tool, InstallMCPServerTool):
            mcp_install_tool.set_context(msg.channel, msg.chat_id)

        # Extract channel context from metadata (e.g., Discord channel history)
        channel_context = msg.metadata.get("channel_context", "") if msg.metadata else ""

        # Build initial messages (use get_history for LLM-formatted messages)
        history = session.get_history()
        if (
            getattr(self, "_enable_pre_compaction_flush", True)
            and len(history) >= self._compactor.config.threshold
            and self._consolidator
        ):
            await self._pre_compaction_flush(history, msg.session_key)
        if len(history) > self._compactor.config.threshold:
            history = self._compactor.compact(history)

        messages = self.context.build_messages(
            history=history,
            current_message=msg.content,
            media=msg.media if msg.media else None,
            channel_context=channel_context,
            namespace=msg.session_key,
        )

        # Agent loop
        iteration = 0
        final_content = None

        while iteration < self.max_iterations:
            iteration += 1

            # Call LLM
            response = await self.provider.chat(
                messages=messages, tools=self.tools.get_definitions(), model=self.model
            )

            # Handle tool calls
            if response.has_tool_calls:
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

                # Execute tools
                for tool_call in response.tool_calls:
                    args_str = json.dumps(tool_call.arguments)
                    logger.debug(f"Executing tool: {tool_call.name} with arguments: {args_str}")
                    result = await self.tools.execute(tool_call.name, tool_call.arguments)
                    messages = self.context.add_tool_result(
                        messages, tool_call.id, tool_call.name, result
                    )
            else:
                # No tool calls, we're done
                final_content = response.content
                break

        if final_content is None:
            final_content = "I've completed processing but have no response to give."

        # Save to session
        session.add_message("user", msg.content)
        session.add_message("assistant", final_content)
        self.sessions.save(session)

        # Periodic extraction and consolidation of facts and lessons
        await self._maybe_extract_and_consolidate(session, msg.session_key)

        return OutboundMessage(
            channel=msg.channel, chat_id=msg.chat_id, content=final_content, metadata=msg.metadata
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
        history_sys = session.get_history()
        if (
            getattr(self, "_enable_pre_compaction_flush", True)
            and len(history_sys) >= self._compactor.config.threshold
            and self._consolidator
        ):
            await self._pre_compaction_flush(history_sys, session_key)

        # Update tool contexts
        message_tool = self.tools.get("message")
        if isinstance(message_tool, MessageTool):
            message_tool.set_context(origin_channel, origin_chat_id)

        spawn_tool = self.tools.get("spawn")
        if isinstance(spawn_tool, SpawnTool):
            spawn_tool.set_context(origin_channel, origin_chat_id)

        # Update cron tool context
        from nanobot.agent.tools.cron import CronTool

        cron_tool = self.tools.get("cron")
        if isinstance(cron_tool, CronTool):
            cron_tool.set_context(origin_channel, origin_chat_id)

        # Update MCP install tool context
        from nanobot.agent.tools.mcp_install import InstallMCPServerTool

        mcp_install_tool = self.tools.get("install_mcp_server")
        if isinstance(mcp_install_tool, InstallMCPServerTool):
            mcp_install_tool.set_context(origin_channel, origin_chat_id)

        # Build messages with the announce content
        history = session.get_history()
        if len(history) > self._compactor.config.threshold:
            history = self._compactor.compact(history)

        messages = self.context.build_messages(
            history=history,
            current_message=msg.content,
            namespace=session_key,
        )

        # Agent loop (limited for announce handling)
        iteration = 0
        final_content = None

        while iteration < self.max_iterations:
            iteration += 1

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
        await self._maybe_extract_and_consolidate(session, session_key)

        return OutboundMessage(
            channel=origin_channel,
            chat_id=origin_chat_id,
            content=final_content,
        )

    async def _pre_compaction_flush(
        self,
        history: list[dict[str, Any]],
        namespace: str,
    ) -> None:
        """Run silent memory extraction before compaction to persist important facts."""
        if not self._consolidator or len(history) < 10:
            return
        try:
            loop = asyncio.get_running_loop()
            extracted = await loop.run_in_executor(
                None,
                self._extractor.extract_for_pre_compaction,
                history,
            )
            if extracted:
                await loop.run_in_executor(
                    None,
                    self._consolidator.consolidate,
                    extracted,
                    namespace,
                )
                logger.debug(
                    "Pre-compaction flush: consolidated %s facts", len(extracted)
                )
        except Exception as e:
            logger.warning(f"Pre-compaction memory flush failed: {e}")

    async def _maybe_extract_and_consolidate(
        self,
        session: "Session",
        namespace: str,
    ) -> None:
        """Trigger memory extraction/consolidation when conditions are met."""
        if not self._consolidator:
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
        """Extract facts and lessons from conversation and consolidate into vector memory."""
        if not self._consolidator:
            return

        try:
            loop = asyncio.get_running_loop()

            # Facts
            extracted = await loop.run_in_executor(
                None,
                self._extractor.extract,
                messages,
            )
            if extracted:
                await loop.run_in_executor(
                    None,
                    self._consolidator.consolidate,
                    extracted,
                    namespace,
                )
                logger.debug(f"Extracted and consolidated {len(extracted)} facts")

            # Lessons (stored in dedicated namespace)
            lessons = await loop.run_in_executor(
                None,
                self._extractor.extract_lessons,
                messages,
            )
            if lessons:
                await loop.run_in_executor(
                    None,
                    self._consolidator.consolidate,
                    lessons,
                    namespace,
                )
                logger.debug(f"Extracted and consolidated {len(lessons)} lessons")

            # Tool-specific lessons (routed to tools namespace by consolidator)
            if getattr(self, "_enable_tool_lessons", True):
                tool_lessons = await loop.run_in_executor(
                    None,
                    self._extractor.extract_tool_lessons,
                    messages,
                )
                if tool_lessons:
                    await loop.run_in_executor(
                        None,
                        self._consolidator.consolidate,
                        tool_lessons,
                        namespace,
                    )
                    logger.debug(
                        f"Extracted and consolidated {len(tool_lessons)} tool lessons"
                    )
        except Exception as e:
            logger.warning(f"Memory extraction/consolidation failed: {e}")

    async def process_direct(self, content: str, session_key: str = "cli:direct") -> str:
        """
        Process a message directly (for CLI usage).

        Args:
            content: The message content.
            session_key: Session identifier.

        Returns:
            The agent's response.
        """
        msg = InboundMessage(channel="cli", sender_id="user", chat_id="direct", content=content)

        response = await self._process_message(msg)
        return response.content if response else ""

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
