"""MCP server installation tool."""

import asyncio
import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from loguru import logger

from nanobot.agent.tools.base import Tool
from nanobot.config.loader import get_config_path, load_config, save_config
from nanobot.restart import write_restart_signal


class InstallMCPServerTool(Tool):
    """
    Tool to install MCP servers and configure them for nanobot.

    After installation, nanobot will restart to load the new server.
    """

    def __init__(self, workspace: Path):
        """
        Initialize the install tool.

        Args:
            workspace: Path to the workspace directory for restart signal.
        """
        self._workspace = workspace
        self._context_channel: str | None = None
        self._context_chat_id: str | None = None

    def set_context(self, channel: str, chat_id: str) -> None:
        """Set the current conversation context for delivery after restart."""
        self._context_channel = channel
        self._context_chat_id = chat_id

    @property
    def name(self) -> str:
        return "install_mcp_server"

    @property
    def description(self) -> str:
        return (
            "Install an MCP server package and add it to nanobot's configuration. "
            "After installation, nanobot will restart to load the new server. "
            "Supports npm/npx (Node.js) and pip/uvx (Python) packages."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "server_name": {
                    "type": "string",
                    "description": "Unique name for this server (e.g., 'github', 'filesystem')",
                },
                "package": {
                    "type": "string",
                    "description": (
                        "Package to install (npm or pip package name, "
                        "e.g., '@anthropic/mcp-server-filesystem' or 'mcp-server-github')"
                    ),
                },
                "command": {
                    "type": "string",
                    "description": "Command to run the server (e.g., 'npx', 'uvx', 'python')",
                },
                "args": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Arguments for the command (e.g., ['-y', 'mcp-server-github'])",
                },
                "env": {
                    "type": "object",
                    "additionalProperties": {"type": "string"},
                    "description": "Environment variables (e.g., {'GITHUB_TOKEN': 'xxx'})",
                },
                "install_command": {
                    "type": "string",
                    "description": (
                        "Optional: custom install command. If not provided, "
                        "auto-detects npm or pip based on package name"
                    ),
                },
            },
            "required": ["server_name", "package", "command", "args"],
        }

    async def execute(
        self,
        server_name: str,
        package: str,
        command: str,
        args: list[str],
        env: dict[str, str] | None = None,
        install_command: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Execute the MCP server installation."""
        try:
            # Validate server name
            if not server_name or not server_name.replace("_", "").replace("-", "").isalnum():
                return "Error: server_name must be alphanumeric (with optional underscores/dashes)"

            # Check if server already exists
            config = load_config()
            if server_name in config.tools.mcp.servers:
                return f"Error: MCP server '{server_name}' already exists in config"

            # Run installation
            install_result = await self._run_install(package, install_command)
            if install_result.startswith("Error:"):
                return install_result

            # Update config
            from nanobot.config.schema import MCPServerConfig

            server_config = MCPServerConfig(
                command=command,
                args=args,
                env=env or {},
                enabled=True,
            )

            # Enable MCP if not already enabled
            config.tools.mcp.enabled = True
            config.tools.mcp.servers[server_name] = server_config

            # Save config
            save_config(config)
            logger.info(f"Added MCP server '{server_name}' to config")

            # Write restart signal with verification job
            verify_time = (
                (datetime.now(UTC) + timedelta(minutes=2)).isoformat().replace("+00:00", "Z")
            )

            verify_job = {
                "name": f"verify_mcp_{server_name}",
                "at_time": verify_time,
                "message": (
                    f"Verify MCP server installation: {server_name}\n"
                    f"- Check if server started correctly\n"
                    f"- List available tools from {server_name}\n"
                    f"- Update TOOLS.md with new tool descriptions\n"
                    f"- Report any errors to the user"
                ),
                "deliver": True,
                "channel": self._context_channel,
                "to": self._context_chat_id,
            }

            write_restart_signal(
                workspace=self._workspace,
                reason=f"MCP server installed: {server_name}",
                verify_job=verify_job,
            )

            return (
                f"Installed {package} as MCP server '{server_name}'.\n"
                f"Config saved to: {get_config_path()}\n"
                f"Nanobot will restart to load the new server.\n"
                f"A verification job will run ~2 minutes after restart to confirm the installation."
            )

        except Exception as e:
            logger.error(f"Failed to install MCP server: {e}")
            return f"Error: {e}"

    async def _run_install(self, package: str, install_command: str | None) -> str:
        """Run the package installation command."""
        if install_command:
            cmd = install_command
        elif self._is_npm_package(package):
            # Check if npm is available
            if not shutil.which("npm"):
                return "Error: npm not found. Please install Node.js first."
            cmd = f"npm install -g {package}"
        else:
            # Assume pip package
            if not shutil.which("pip") and not shutil.which("pip3"):
                return "Error: pip not found. Please install Python pip first."
            pip_cmd = "pip3" if shutil.which("pip3") else "pip"
            cmd = f"{pip_cmd} install {package}"

        logger.info(f"Running install command: {cmd}")

        try:
            process = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=300)

            if process.returncode != 0:
                error_msg = stderr.decode("utf-8").strip() or stdout.decode("utf-8").strip()
                return f"Error installing {package}: {error_msg}"

            logger.info(f"Successfully installed {package}")
            return "OK"

        except asyncio.TimeoutError:
            return "Error: Installation timed out after 5 minutes"
        except Exception as e:
            return f"Error: {e}"

    def _is_npm_package(self, package: str) -> bool:
        """Check if the package looks like an npm package."""
        # npm packages often start with @ (scoped) or contain /
        # or common npm package patterns
        return package.startswith("@") or "/" in package or package.startswith("mcp-server-")
