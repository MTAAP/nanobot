"""Subagent manager for background task execution."""

import asyncio
import json
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from nanobot.agent.tools.filesystem import ListDirTool, ReadFileTool, WriteFileTool
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.tools.shell import ExecTool
from nanobot.agent.tools.web import WebFetchTool, WebSearchTool
from nanobot.bus.events import InboundMessage
from nanobot.bus.progress import ProgressCallback, ProgressEvent, ProgressKind
from nanobot.bus.queue import MessageBus
from nanobot.providers.base import LLMProvider

if TYPE_CHECKING:
    from nanobot.config.schema import ExecToolConfig
    from nanobot.registry.evolve import SelfEvolveManager
    from nanobot.registry.store import AgentRegistry


class SubagentManager:
    """
    Manages background subagent execution.

    Subagents are lightweight agent instances that run in the background
    to handle specific tasks. They share the same LLM provider but have
    isolated context and a focused system prompt.
    """

    def __init__(
        self,
        provider: LLMProvider,
        workspace: Path,
        bus: MessageBus,
        model: str | None = None,
        brave_api_key: str | None = None,
        exec_config: "ExecToolConfig | None" = None,
        registry: "AgentRegistry | None" = None,
        evolve_manager: "SelfEvolveManager | None" = None,
        progress_callback: ProgressCallback | None = None,
    ):
        from nanobot.config.schema import ExecToolConfig

        self.provider = provider
        self.workspace = workspace
        self.bus = bus
        self.model = model or provider.get_default_model()
        self.brave_api_key = brave_api_key
        self.exec_config = exec_config or ExecToolConfig()
        self._registry = registry
        self._evolve_manager = evolve_manager
        self._progress_callback = progress_callback
        self._running_tasks: dict[str, asyncio.Task[None]] = {}

    async def spawn(
        self,
        task: str,
        label: str | None = None,
        origin_channel: str = "cli",
        origin_chat_id: str = "direct",
        registry_task_id: str | None = None,
    ) -> str:
        """
        Spawn a subagent to execute a task in the background.

        Args:
            task: The task description for the subagent.
            label: Optional human-readable label for the task.
            origin_channel: The channel to announce results to.
            origin_chat_id: The chat ID to announce results to.
            registry_task_id: Optional task ID from the agent registry.

        Returns:
            Status message indicating the subagent was started.
        """
        task_id = str(uuid.uuid4())[:8]
        display_label = label or task[:30] + ("..." if len(task) > 30 else "")

        origin = {
            "channel": origin_channel,
            "chat_id": origin_chat_id,
        }

        # Create background task
        bg_task = asyncio.create_task(
            self._run_subagent(task_id, task, display_label, origin, registry_task_id)
        )
        self._running_tasks[task_id] = bg_task

        # Cleanup when done
        bg_task.add_done_callback(lambda _: self._running_tasks.pop(task_id, None))

        logger.info(f"Spawned subagent [{task_id}]: {display_label}")
        return f"Subagent [{display_label}] started (id: {task_id}). I'll notify you when it completes."

    async def _run_subagent(
        self,
        task_id: str,
        task: str,
        label: str,
        origin: dict[str, str],
        registry_task_id: str | None = None,
    ) -> None:
        """Execute the subagent task and announce the result."""
        logger.info(f"Subagent [{task_id}] starting task: {label}")

        agent_id = f"subagent-{task_id}"
        pulse_task: asyncio.Task | None = None

        try:
            # Build subagent tools (no message tool, no spawn tool)
            tools = ToolRegistry()
            tools.register(ReadFileTool())
            tools.register(WriteFileTool())
            tools.register(ListDirTool())
            tools.register(
                ExecTool(
                    working_dir=str(self.workspace),
                    timeout=self.exec_config.timeout,
                    restrict_to_workspace=self.exec_config.restrict_to_workspace,
                )
            )
            tools.register(WebSearchTool(api_key=self.brave_api_key))
            tools.register(WebFetchTool())

            # Registry integration: handshake + proof tool + evolve tool
            if self._registry and registry_task_id:
                from nanobot.registry.handshake import AgentHandshake, HandshakeError
                from nanobot.registry.store import AgentState, TaskState

                # Perform handshake
                handshake = AgentHandshake(self._registry, self.workspace)
                try:
                    await handshake.perform(
                        agent_id=agent_id,
                        task_id=registry_task_id,
                        capabilities=["read_file", "write_file", "exec"],
                        available_tool_names=list(tools.tool_names),
                    )
                except HandshakeError as e:
                    logger.error(f"Subagent [{task_id}] handshake failed: {e}")
                    await self._announce_result(
                        task_id, label, task, f"Handshake failed: {e}", origin, "error"
                    )
                    return

                # Transition task to IN_PROGRESS
                await self._registry.update_task_state(
                    registry_task_id, TaskState.IN_PROGRESS, reason="subagent started"
                )

                # Register proof tool
                from nanobot.agent.tools.proof import SubmitProofTool

                tools.register(SubmitProofTool(registry=self._registry, task_id=registry_task_id))

                # Register evolve tool if available
                if self._evolve_manager:
                    from nanobot.agent.tools.evolve import SelfEvolveTool

                    tools.register(SelfEvolveTool(self._evolve_manager))

                # Start pulse loop
                pulse_task = asyncio.create_task(self._pulse_loop(agent_id, interval=60))

            # Build messages with subagent-specific prompt
            system_prompt = self._build_subagent_prompt(
                task, has_registry=bool(self._registry and registry_task_id)
            )
            messages: list[dict[str, Any]] = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": task},
            ]

            # Run agent loop (limited iterations)
            max_iterations = 15
            iteration = 0
            final_result: str | None = None

            while iteration < max_iterations:
                iteration += 1

                response = await self.provider.chat(
                    messages=messages,
                    tools=tools.get_definitions(),
                    model=self.model,
                )

                if response.has_tool_calls:
                    # Add assistant message with tool calls
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
                    messages.append(
                        {
                            "role": "assistant",
                            "content": response.content or "",
                            "tool_calls": tool_call_dicts,
                        }
                    )

                    # Execute tools
                    for tool_call in response.tool_calls:
                        logger.debug(f"Subagent [{task_id}] executing: {tool_call.name}")
                        if self._progress_callback:
                            await self._progress_callback(
                                ProgressEvent(
                                    channel=origin["channel"],
                                    chat_id=origin["chat_id"],
                                    kind=ProgressKind.TOOL_START,
                                    tool_name=tool_call.name,
                                    detail=f"[subagent] Executing {tool_call.name}",
                                    iteration=iteration,
                                )
                            )
                        result = await tools.execute(tool_call.name, tool_call.arguments)
                        if self._progress_callback:
                            status = (
                                "error"
                                if isinstance(result, str) and result.startswith("Error")
                                else "ok"
                            )
                            await self._progress_callback(
                                ProgressEvent(
                                    channel=origin["channel"],
                                    chat_id=origin["chat_id"],
                                    kind=ProgressKind.TOOL_COMPLETE,
                                    tool_name=tool_call.name,
                                    detail=f"[subagent] {tool_call.name}: {status}",
                                    iteration=iteration,
                                )
                            )
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "name": tool_call.name,
                                "content": result,
                            }
                        )
                else:
                    final_result = response.content
                    break

            if final_result is None:
                final_result = "Task completed but no final response was generated."

            # Update registry state on completion
            if self._registry and registry_task_id:
                from nanobot.registry.store import AgentState

                try:
                    await self._registry.update_agent_state(
                        agent_id, AgentState.COMPLETED, reason="task finished"
                    )
                except Exception:
                    pass  # May already be in a terminal state or DB error

            logger.info(f"Subagent [{task_id}] completed successfully")
            await self._announce_result(task_id, label, task, final_result, origin, "ok")

        except Exception as e:
            error_msg = f"Error: {str(e)}"
            logger.error(f"Subagent [{task_id}] failed: {e}")

            # Update registry state on failure
            if self._registry and registry_task_id:
                from nanobot.registry.store import AgentState, TaskState

                try:
                    await self._registry.update_agent_state(
                        agent_id, AgentState.FAILED, reason=str(e)
                    )
                except Exception:
                    pass
                try:
                    await self._registry.update_task_state(
                        registry_task_id, TaskState.FAILED, reason=str(e)
                    )
                except Exception:
                    pass

            await self._announce_result(task_id, label, task, error_msg, origin, "error")

        finally:
            if pulse_task:
                pulse_task.cancel()
                try:
                    await pulse_task
                except asyncio.CancelledError:
                    pass

            # Transition agent back to IDLE for reuse
            if self._registry:
                try:
                    agent = await self._registry.get_agent(agent_id)
                    if agent and agent["state"] in ("completed", "failed"):
                        from nanobot.registry.store import AgentState

                        await self._registry.update_agent_state(
                            agent_id, AgentState.IDLE, reason="cleanup"
                        )
                except Exception:
                    pass

    async def _pulse_loop(self, agent_id: str, interval: int = 60) -> None:
        """Periodically record heartbeat pulses for the agent."""
        while True:
            await asyncio.sleep(interval)
            if self._registry:
                try:
                    await self._registry.record_pulse(agent_id)
                except Exception as e:
                    logger.debug(f"Pulse failed for {agent_id}: {e}")

    async def _announce_result(
        self,
        task_id: str,
        label: str,
        task: str,
        result: str,
        origin: dict[str, str],
        status: str,
    ) -> None:
        """Announce the subagent result to the main agent via the message bus."""
        status_text = "completed successfully" if status == "ok" else "failed"

        announce_content = f"""[Subagent '{label}' {status_text}]

Task: {task}

Result:
{result}

Summarize this naturally for the user. Keep it brief (1-2 sentences). Do not mention technical details like "subagent" or task IDs."""

        # Inject as system message to trigger main agent
        msg = InboundMessage(
            channel="system",
            sender_id="subagent",
            chat_id=f"{origin['channel']}:{origin['chat_id']}",
            content=announce_content,
        )

        await self.bus.publish_inbound(msg)
        logger.debug(
            f"Subagent [{task_id}] announced result to {origin['channel']}:{origin['chat_id']}"
        )

    def _build_subagent_prompt(self, task: str, has_registry: bool = False) -> str:
        """Build a focused system prompt for the subagent."""
        base = f"""# Subagent

You are a subagent spawned by the main agent to complete a specific task.

## Your Task
{task}

## Rules
1. Stay focused - complete only the assigned task, nothing else
2. Your final response will be reported back to the main agent
3. Do not initiate conversations or take on side tasks
4. Be concise but informative in your findings

## What You Can Do
- Read and write files in the workspace
- Execute shell commands
- Search the web and fetch web pages
- Complete the task thoroughly

## What You Cannot Do
- Send messages directly to users (no message tool available)
- Spawn other subagents
- Access the main agent's conversation history

## Workspace
Your workspace is at: {self.workspace}

When you have completed the task, provide a clear summary of your findings or actions."""

        if has_registry:
            base += """

## Proof of Work
After completing your task, you MUST submit proof using the submit_proof tool.
Choose the appropriate proof type:
- git: For code changes (branch, commit hash)
- file: For file creation/modification (path, sha256 hash)
- command: For shell commands (command, exit code)
- test: For test results (passed/failed counts)
- pr: For pull requests (PR URL, number, branch)

## Self-Evolution (if available)
If you have access to self_evolve, follow this workflow:
1. setup_repo - Clone/pull the nanobot repo
2. create_branch - Create a feature branch
3. (Make changes using read_file/write_file on the repo)
4. run_tests - Verify changes
5. run_lint - Check code style
6. commit_push - Commit and push changes
7. create_pr - Create a pull request
8. submit_proof with type=pr"""

        return base

    def get_running_count(self) -> int:
        """Return the number of currently running subagents."""
        return len(self._running_tasks)
