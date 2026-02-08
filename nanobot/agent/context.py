"""Context builder for assembling agent prompts."""

import base64
import mimetypes
import platform
from pathlib import Path
from typing import Any

from loguru import logger

from nanobot.agent.memory import MemoryStore
from nanobot.agent.skills import SkillsLoader

# Template for TOOLS.md when created for the first time
TOOLS_MD_TEMPLATE = """# Tool Usage Patterns

This file tracks your knowledge about tools - when to use them, common mistakes, and best practices.
Update this file when you learn something new about using a tool effectively.

## MCP Tools

MCP (Model Context Protocol) tools are provided by external servers. After installing a new MCP server,
document the tools here with usage patterns.

<!-- Example:
### github_create_issue
- Use when: User explicitly requests creating a GitHub issue
- Avoid when: User just mentions a bug without asking to file it
- Best practices: Always confirm repo and issue details first

### filesystem_read_file
- Use when: Need to read file contents from allowed directories
- Avoid when: File path is outside allowed directories
- Learned: Works better with absolute paths
-->

## Native Tools

### read_file
- Use when: Need to read file contents
- Best practices: Prefer this over exec with cat

### write_file
- Use when: Creating or overwriting files
- Best practices: Include full content, not patches

### edit_file
- Use when: Making targeted changes to existing files
- Best practices: Use for surgical edits, write_file for full rewrites

### exec
- Use when: Need to run shell commands
- Avoid when: A more specific tool exists (use read_file over cat)
- Best practices: Set appropriate timeout for long-running commands

### web_search
- Use when: Need current information not in knowledge base
- Best practices: Use specific search queries

### web_fetch
- Use when: Need to retrieve content from a specific URL
- Best practices: Check if URL is accessible first

### message
- Use when: Need to send proactive messages to chat channels
- Avoid when: Just responding in conversation (reply directly instead)

### spawn
- Use when: Need to run long-running or parallel background tasks
- Best practices: Use for tasks that don't need immediate response

### cron
- Use when: Need to schedule tasks for specific times or intervals
- Best practices: Use 'at' for one-time, 'cron' for recurring

### install_mcp_server
- Use when: User wants to add new MCP server capabilities
- Best practices: Verify package name and command before installing
"""


class ContextBuilder:
    """
    Builds the context (system prompt + messages) for the agent.

    Assembles bootstrap files, memory, skills, and conversation history
    into a coherent prompt for the LLM.
    """

    BOOTSTRAP_FILES = [
        "AGENTS.md",
        "SOUL.md",
        "USER.md",
        "TOOLS.md",
        "IDENTITY.md",
    ]

    def __init__(
        self,
        workspace: Path,
        memory_enabled: bool = False,
        core_memory: Any = None,
    ):
        self.workspace = workspace
        self.memory = MemoryStore(workspace)
        self.skills = SkillsLoader(workspace)
        self.memory_enabled = memory_enabled
        self.core_memory = core_memory
        self.self_evolve_enabled = False
        self._ensure_tools_md()

        # Bootstrap file caching
        self._bootstrap_cache: str | None = None
        self._bootstrap_mtimes: dict[str, float] = {}

    def _ensure_tools_md(self) -> None:
        """Create TOOLS.md with template if it doesn't exist."""
        tools_md = self.workspace / "TOOLS.md"
        if not tools_md.exists():
            try:
                tools_md.write_text(TOOLS_MD_TEMPLATE, encoding="utf-8")
                logger.info("Created TOOLS.md template")
            except Exception as e:
                logger.warning(f"Failed to create TOOLS.md: {e}")

    def build_system_prompt(
        self,
        skill_names: list[str] | None = None,
        budget: int | None = None,
    ) -> str:
        """
        Build the system prompt from bootstrap files, memory, and skills.

        Args:
            skill_names: Optional list of skills to include.
            budget: Max estimated token count (chars // 4). Truncates low-priority
                sections (skills summary, then bootstrap extras) when exceeded.

        Returns:
            Complete system prompt.
        """
        parts = []

        # Core identity (never truncated)
        parts.append(self._get_identity())

        # Bootstrap files
        bootstrap = self._load_bootstrap_files()
        if bootstrap:
            parts.append(bootstrap)

        # Memory context
        memory = self.memory.get_memory_context()
        if memory:
            parts.append(f"# Memory\n\n{memory}")

        # Core memory (always-in-context scratchpad)
        if self.core_memory:
            core_ctx = self.core_memory.get_context()
            if core_ctx:
                parts.append(core_ctx)

        # Long-term memory instructions
        if self.memory_enabled:
            parts.append(self._get_memory_instructions())

        # Clarification guidance
        parts.append(self._get_clarification_instructions())

        # Skills - progressive loading
        # 1. Always-loaded skills: include full content
        always_skills = self.skills.get_always_skills()
        if always_skills:
            always_content = self.skills.load_skills_for_context(always_skills)
            if always_content:
                parts.append(f"# Active Skills\n\n{always_content}")

        # 2. Available skills: only show summary (agent uses read_file to load)
        skills_summary = self.skills.build_skills_summary()
        if skills_summary:
            parts.append(f"""# Skills

The following skills extend your capabilities. To use a skill, read its SKILL.md file using the read_file tool.
Skills with available="false" need dependencies installed first - you can try installing them with apt/brew.

{skills_summary}""")

        result = "\n\n---\n\n".join(parts)

        # Enforce budget by dropping lowest-priority sections from the end
        if budget and len(result) // 4 > budget:
            est = len(result) // 4
            logger.warning(f"System prompt ~{est} tokens exceeds budget {budget}, truncating")
            # Drop skills summary first, then bootstrap extras
            while len(parts) > 1 and len("\n\n---\n\n".join(parts)) // 4 > budget:
                removed = parts.pop()
                logger.debug(f"Dropped section ({len(removed)} chars) to meet budget")
            result = "\n\n---\n\n".join(parts)

        return result

    @staticmethod
    def _get_runtime_info() -> str:
        """Get runtime environment info (OS, architecture, Python version)."""
        system = platform.system()
        os_name = "macOS" if system == "Darwin" else system
        return f"{os_name} {platform.machine()}, Python {platform.python_version()}"

    def _get_identity(self) -> str:
        """Get the core identity section, loading from IDENTITY.md if available."""
        from datetime import datetime

        now = datetime.now().strftime("%Y-%m-%d %H:%M (%A)")
        workspace_path = str(self.workspace.expanduser().resolve())

        # Try to load identity from IDENTITY.md
        identity_file = self.workspace / "IDENTITY.md"
        if identity_file.exists():
            try:
                identity_content = identity_file.read_text(encoding="utf-8")
                return self._build_identity_with_context(identity_content, now, workspace_path)
            except Exception as e:
                logger.warning(f"Failed to load IDENTITY.md, using defaults: {e}")

        # Fallback to hardcoded defaults if IDENTITY.md doesn't exist
        return self._get_default_identity(now, workspace_path)

    def _build_identity_with_context(
        self, identity_content: str, now: str, workspace_path: str
    ) -> str:
        """Build identity from IDENTITY.md content with dynamic context appended."""
        return f"""{identity_content}

## Current Context

**Time (approximate)**: {now} -- always verify with `exec("date")` for precision
**Runtime**: {self._get_runtime_info()}
**Workspace**: {workspace_path}
- Memory files: {workspace_path}/memory/MEMORY.md
- Daily notes: {workspace_path}/memory/YYYY-MM-DD.md
- Custom skills: {workspace_path}/skills/{{skill-name}}/SKILL.md

**IMPORTANT**: Only create files inside your workspace. Never create nested workspace
directories (e.g. `workspace/temp/workspace/`). For code modifications, use the
`self_evolve` tool which manages its own repo clone.

{self._get_capabilities_section(workspace_path)}

{self._get_tool_usage_section(workspace_path)}"""

    def _get_default_identity(self, now: str, workspace_path: str) -> str:
        """Get the hardcoded default identity (fallback when IDENTITY.md doesn't exist)."""
        return f"""# nanobot

You are nanobot, a helpful AI assistant. You have access to tools that allow you to:
- Read, write, and edit files
- Execute shell commands
- Search the web and fetch web pages
- Send messages to users on chat channels
- Spawn subagents for complex background tasks
- Schedule tasks and reminders

## Current Time (approximate)
{now} -- always verify with `exec("date")` for precision

## Runtime
{self._get_runtime_info()}

## Workspace
Your workspace is at: {workspace_path}
- Memory files: {workspace_path}/memory/MEMORY.md
- Daily notes: {workspace_path}/memory/YYYY-MM-DD.md
- Custom skills: {workspace_path}/skills/{{skill-name}}/SKILL.md

**IMPORTANT**: Only create files inside your workspace. Never create nested workspace
directories (e.g. `workspace/temp/workspace/`). For code modifications, use the
`self_evolve` tool which manages its own repo clone.

{self._get_capabilities_section(workspace_path)}

{self._get_tool_usage_section(workspace_path)}

Always be helpful, accurate, and concise. When using tools, explain what you're doing.
When remembering something, write to {workspace_path}/memory/MEMORY.md"""

    def _get_capabilities_section(self, workspace_path: str) -> str:
        """Get the proactive capabilities section."""
        return f"""## Proactive Capabilities

### Scheduled Jobs (cron tool)
You can schedule tasks using the `cron` tool:
- **One-time reminders**: `schedule_type="at"` with ISO datetime (e.g., "2025-02-06T09:00:00")
- **Recurring tasks**: `schedule_type="every"` with seconds (min 60), or `schedule_type="cron"` with expression
- **Deliver to user**: Set `deliver=true` to message the user when the job runs

**IMPORTANT**: When creating cron jobs for reminders, notifications, or any task where the user
should see the result, you MUST set `deliver=true` and specify the `channel` and `to` fields.
Without `deliver=true`, the job runs silently and the user never sees the output.
The `channel` and `to` fields default to the current conversation if not specified.

Example cron expressions:
- `0 9 * * *` = 9 AM daily
- `0 9 * * 1-5` = 9 AM weekdays
- `*/30 * * * *` = every 30 minutes
- `0 */2 * * *` = every 2 hours

### Heartbeat Tasks (HEARTBEAT.md)
The file `{workspace_path}/HEARTBEAT.md` is checked every ~30 minutes.
- Add tasks as markdown checkboxes: `- [ ] Task description`
- You'll be asked to read and act on any tasks listed

**When to use cron vs heartbeat:**
- Use **cron** for precise timing (9 AM daily, every 2 hours)
- Use **HEARTBEAT.md** for approximate periodic checks (~30 min)

IMPORTANT: When responding to direct questions or conversations, reply directly with your text response.
Only use the 'message' tool when you need to send a message to a specific chat channel (like WhatsApp).
For normal conversation, just respond with text - do not call the message tool.
{self._get_self_evolve_section()}
{self._get_mutable_state_section()}"""

    def _get_self_evolve_section(self) -> str:
        """Get the self-evolution tool guidance (empty if disabled)."""
        if not self.self_evolve_enabled:
            return ""
        return """

### Self-Evolution (self_evolve tool)
You can modify your own source code using the `self_evolve` tool. ALWAYS use this tool for:
- Setting up the repo: action="setup_repo"
- Creating feature branches: action="create_branch"
- Running tests/lint: action="run_tests", action="run_lint"
- Committing and pushing: action="commit_push"
- Creating pull requests: action="create_pr"

CRITICAL: NEVER claim to have run git commands or pushed code without actually calling the self_evolve tool.
If you haven't called the tool, the operation did NOT happen. Do not fabricate or hallucinate git results.
Always verify operations by checking tool output before reporting success."""

    def _get_mutable_state_section(self) -> str:
        """Get mutable state verification instructions."""
        return """
## Mutable State Verification

When recalled memories describe reminders, schedules, cron jobs, or any
state that can change over time, you MUST verify with the appropriate tool
before telling the user. Memories are snapshots -- the actual state may differ.

**Verification mapping:**
- Reminders/cron jobs/schedules: use `cron` tool (action="list") to check current state
- File existence/contents: use `read_file` or `exec("ls -la ...")`
- Running processes/services: use `exec("ps aux | grep ...")`
- System configuration: use appropriate `exec` command"""

    def _get_tool_usage_section(self, workspace_path: str) -> str:
        """Get the tool usage knowledge section."""
        return f"""## Action Integrity

CRITICAL: When asked to perform an action (write a file, set up a cron job, update config,
install something, etc.), you MUST call the appropriate tool. NEVER claim to have performed
an action unless you actually called a tool and received a successful result.

**Rules:**
1. To write/update a file -> call `write_file` or `edit_file`. Saying "I updated X" without
   calling a tool means the file was NOT changed.
2. To create a cron job -> call `cron` tool. Describing the schedule is not the same as creating it.
3. After calling a tool, CHECK THE RESULT before reporting success. If the tool returned an error,
   report the error -- do not claim success.
4. NEVER use phrases like "I have updated", "I've written", "I've configured" in your response
   unless the corresponding tool call succeeded in this conversation turn.

**Example -- WRONG behavior:**
User: "Write my preferences to USER.md"
You: "I have updated USER.md with your preferences." (NO tool was called -- file unchanged!)

**Example -- CORRECT behavior:**
User: "Write my preferences to USER.md"
You: [call write_file tool] -> verify result -> "Done, I've written your preferences to USER.md."

## Ground Truth First

NEVER answer factual questions from memory or training data alone. For verifiable facts,
you MUST use a tool to get the current, accurate answer.

**Queries that ALWAYS require tool verification:**
- Current time/date: use `exec` with `date` command
- Weather: use `web_search`
- Calculations: use `exec` with appropriate command
- System status (disk, processes, network): use `exec`
- File existence/contents: use `read_file` or `exec`

**Example -- correct behavior:**
User: "What time is it?"
You: [call exec tool with command="date"] then report the result.

Do NOT guess, approximate, or rely on the time shown in your system prompt for user-facing answers.

## Tool Usage Knowledge

You maintain a knowledge file at {workspace_path}/TOOLS.md that tracks:
- When to use each tool
- Common mistakes to avoid
- Best practices learned from experience

When you learn something new about using a tool effectively, update TOOLS.md.
Read TOOLS.md at the start of complex tasks to refresh your knowledge.

## MCP (Model Context Protocol) Tools

MCP tools extend your capabilities by connecting to external servers. You can:
- Install new MCP servers using the `install_mcp_server` tool
- After installation, nanobot restarts to load the new server
- MCP tool names are prefixed with the server name (e.g., `github_create_issue`)

When you install a new MCP server, document its tools in TOOLS.md."""

    def _get_memory_instructions(self) -> str:
        """Get instructions for using semantic memory."""
        return """# Long-term Memory

You have access to semantic memory from all past conversations.

## Memory Tools

- `memory_search` - Search past conversations and facts. Supports
  time-filtered search (today, this_week, this_month, last_N_days) and
  type-filtered search (fact, conversation).
- `core_memory_read` - Read your persistent core memory scratchpad.
- `core_memory_update` - Update a section of core memory with key user
  info, preferences, or project context.
- `memory_forget` - Remove a specific memory entry by ID.

## When to Search

BEFORE answering questions about:
- Prior work or decisions made together
- User preferences or habits
- Dates, names, or specific facts discussed before
- Commitments or tasks from earlier conversations

Always run `memory_search` first to recall relevant context. This helps you:
- Maintain continuity across conversations
- Remember user preferences without being told again
- Recall important decisions and their reasoning
- Reference past work accurately

## Auto-Recall

Relevant memories are automatically recalled with time-weighted relevance
and injected into conversation context. If you see
[Relevant memories from past conversations], review them before responding.

## Memory Reliability

Recalled memories are snapshots from past conversations. They may be:
- **Outdated**: Facts that were true when recorded but have since changed
- **Stale**: Reminders, schedules, or cron jobs that may have been modified or removed
- **Incomplete**: Partial context from a longer conversation

When memories describe mutable state (reminders, schedules, cron jobs, timers),
you MUST verify with the appropriate tool before presenting them as current truth.

## Core Memory

Core memory is a small persistent scratchpad always visible in your
context. Use `core_memory_update` to store important user info,
preferences, and active project context. This avoids repeated lookups."""

    def _get_clarification_instructions(self) -> str:
        """Get instructions for when the agent should ask clarifying questions."""
        return """# Clarification Protocol

When to ask for clarification:
- The request is ambiguous with multiple valid interpretations
- A destructive or irreversible action is requested (deleting files, modifying configs)
- Required parameters are missing and cannot be reasonably inferred
- The task scope is unclear (could be simple fix or major refactor)

How to clarify:
- Ask ONE focused question at a time
- Offer 2-3 concrete options when possible
- Include your best guess: "I'll assume X unless you say otherwise"
- Format options as a numbered list for easy selection

When NOT to clarify:
- Simple greetings or casual conversation
- Clear, direct requests with obvious intent
- Follow-up messages in an ongoing task
- When context from conversation history makes intent obvious
- When a reasonable default exists and the action is reversible"""

    def _is_bootstrap_stale(self) -> bool:
        """Check if any bootstrap files have been modified since last cache."""
        for filename in self.BOOTSTRAP_FILES:
            file_path = self.workspace / filename
            if file_path.exists():
                mtime = file_path.stat().st_mtime
                cached_mtime = self._bootstrap_mtimes.get(filename)
                if cached_mtime is None or mtime > cached_mtime:
                    return True
            elif filename in self._bootstrap_mtimes:
                # File was deleted
                return True
        return False

    def _load_bootstrap_files(self) -> str:
        """Load all bootstrap files from workspace (with caching)."""
        # Return cached content if not stale
        if self._bootstrap_cache is not None and not self._is_bootstrap_stale():
            return self._bootstrap_cache

        parts = []
        new_mtimes: dict[str, float] = {}

        for filename in self.BOOTSTRAP_FILES:
            file_path = self.workspace / filename
            if file_path.exists():
                content = file_path.read_text(encoding="utf-8")
                parts.append(f"## {filename}\n\n{content}")
                new_mtimes[filename] = file_path.stat().st_mtime

        # Update cache
        self._bootstrap_cache = "\n\n".join(parts) if parts else ""
        self._bootstrap_mtimes = new_mtimes

        return self._bootstrap_cache

    def build_messages(
        self,
        history: list[dict[str, Any]],
        current_message: str,
        skill_names: list[str] | None = None,
        media: list[str] | None = None,
        channel_context: str = "",
        system_prompt_budget: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Build the complete message list for an LLM call.

        Args:
            history: Previous conversation messages.
            current_message: The new user message.
            skill_names: Optional skills to include.
            media: Optional list of local file paths for images/media.
            channel_context: Optional recent channel messages for context.
            system_prompt_budget: Max estimated token count for system prompt.

        Returns:
            List of messages including system prompt.
        """
        messages = []

        # System prompt
        system_prompt = self.build_system_prompt(skill_names, budget=system_prompt_budget)
        messages.append({"role": "system", "content": system_prompt})

        # History
        messages.extend(history)

        # If channel context provided, prepend to user message
        if channel_context:
            current_message = (
                f"[Recent channel messages for context]\n"
                f"{channel_context}\n\n"
                f"[Current message]\n{current_message}"
            )

        # Current message (with optional image attachments)
        user_content = self._build_user_content(current_message, media)
        messages.append({"role": "user", "content": user_content})

        return messages

    def _build_user_content(self, text: str, media: list[str] | None) -> str | list[dict[str, Any]]:
        """Build user message content with optional base64-encoded images."""
        if not media:
            return text

        images = []
        for path in media:
            p = Path(path)
            mime, _ = mimetypes.guess_type(path)
            if not p.is_file() or not mime or not mime.startswith("image/"):
                continue
            b64 = base64.b64encode(p.read_bytes()).decode()
            images.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}})

        if not images:
            return text
        return images + [{"type": "text", "text": text}]

    def add_tool_result(
        self, messages: list[dict[str, Any]], tool_call_id: str, tool_name: str, result: str
    ) -> list[dict[str, Any]]:
        """
        Add a tool result to the message list.

        Args:
            messages: Current message list.
            tool_call_id: ID of the tool call.
            tool_name: Name of the tool.
            result: Tool execution result.

        Returns:
            Updated message list.
        """
        messages.append(
            {"role": "tool", "tool_call_id": tool_call_id, "name": tool_name, "content": result}
        )
        return messages

    def add_assistant_message(
        self,
        messages: list[dict[str, Any]],
        content: str | None,
        tool_calls: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Add an assistant message to the message list.

        Args:
            messages: Current message list.
            content: Message content.
            tool_calls: Optional tool calls.

        Returns:
            Updated message list.
        """
        msg: dict[str, Any] = {"role": "assistant", "content": content or ""}

        if tool_calls:
            msg["tool_calls"] = tool_calls

        messages.append(msg)
        return messages
