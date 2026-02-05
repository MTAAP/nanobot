"""MCP client for communicating with a single MCP server via stdio."""

import asyncio
import json
from typing import Any

from loguru import logger

from nanobot.mcp.protocol import (
    MCP_PROTOCOL_VERSION,
    JsonRpcRequest,
    JsonRpcResponse,
    MCPServerInfo,
    MCPToolCallResult,
    MCPToolInfo,
)


class MCPClient:
    """
    Client for communicating with a single MCP server via stdio.

    Manages the subprocess lifecycle and JSON-RPC message exchange.
    """

    def __init__(
        self,
        name: str,
        command: str,
        args: list[str],
        env: dict[str, str] | None = None,
        timeout: int = 30,
    ):
        """
        Initialize an MCP client.

        Args:
            name: Unique name for this server connection.
            command: Command to run (e.g., "npx", "uvx", "python").
            args: Arguments for the command.
            env: Environment variables for the subprocess.
            timeout: Timeout in seconds for operations.
        """
        self.name = name
        self.command = command
        self.args = args
        self.env = env
        self.timeout = timeout

        self._process: asyncio.subprocess.Process | None = None
        self._request_id = 0
        self._pending: dict[int | str, asyncio.Future[JsonRpcResponse]] = {}
        self._read_task: asyncio.Task[None] | None = None
        self._stderr_task: asyncio.Task[None] | None = None
        self._server_info: MCPServerInfo | None = None
        self._tools: list[MCPToolInfo] = []
        self._started = False
        self._start_error: str | None = None

    @property
    def is_running(self) -> bool:
        """Check if the server process is running."""
        return self._process is not None and self._process.returncode is None

    @property
    def server_info(self) -> MCPServerInfo | None:
        """Get server info from initialization."""
        return self._server_info

    @property
    def tools(self) -> list[MCPToolInfo]:
        """Get discovered tools from this server."""
        return self._tools

    @property
    def start_error(self) -> str | None:
        """Get any error that occurred during startup."""
        return self._start_error

    async def start(self) -> bool:
        """
        Start the MCP server process and initialize the connection.

        Returns:
            True if started successfully, False otherwise.
        """
        if self._started:
            return self.is_running

        self._started = True
        self._start_error = None

        try:
            # Build environment
            import os

            process_env = os.environ.copy()
            if self.env:
                process_env.update(self.env)

            # Start the process
            logger.debug(f"Starting MCP server {self.name}: {self.command} {' '.join(self.args)}")

            self._process = await asyncio.create_subprocess_exec(
                self.command,
                *self.args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=process_env,
                limit=10_000_000,  # 10MB buffer limit for large MCP responses
            )

            # Start reading responses and stderr
            self._read_task = asyncio.create_task(self._read_loop())
            self._stderr_task = asyncio.create_task(self._read_stderr())

            # Initialize the connection
            await self._initialize()

            # Discover tools
            await self._discover_tools()

            logger.info(
                f"MCP server {self.name} started: {self._server_info.name if self._server_info else 'unknown'} "
                f"with {len(self._tools)} tools"
            )
            return True

        except asyncio.TimeoutError:
            self._start_error = f"Timeout waiting for MCP server {self.name} to initialize"
            logger.error(self._start_error)
            await self.stop()
            return False
        except Exception as e:
            self._start_error = f"Failed to start MCP server {self.name}: {e}"
            logger.error(self._start_error)
            await self.stop()
            return False

    async def stop(self) -> None:
        """Stop the MCP server process."""
        if self._read_task:
            self._read_task.cancel()
            try:
                await self._read_task
            except asyncio.CancelledError:
                pass
            self._read_task = None

        if self._stderr_task:
            self._stderr_task.cancel()
            try:
                await self._stderr_task
            except asyncio.CancelledError:
                pass
            self._stderr_task = None

        if self._process:
            try:
                self._process.terminate()
                await asyncio.wait_for(self._process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                self._process.kill()
                await self._process.wait()
            except Exception as e:
                logger.warning(f"Error stopping MCP server {self.name}: {e}")
            finally:
                self._process = None

        # Clear pending requests
        for future in self._pending.values():
            if not future.done():
                future.cancel()
        self._pending.clear()

        logger.debug(f"MCP server {self.name} stopped")

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """
        Call a tool on the MCP server.

        Args:
            tool_name: Name of the tool to call.
            arguments: Arguments for the tool.

        Returns:
            Tool result as a string.
        """
        if not self.is_running:
            return f"Error: MCP server {self.name} is not running"

        try:
            response = await self._send_request(
                "tools/call",
                {"name": tool_name, "arguments": arguments},
            )

            if response.error:
                error_msg = response.error.get("message", str(response.error))
                return f"Error: {error_msg}"

            result = MCPToolCallResult.from_dict(response.result or {})
            if result.is_error:
                return f"Error: {result.to_string()}"

            return result.to_string()

        except asyncio.TimeoutError:
            return f"Error: Timeout calling tool {tool_name} on {self.name}"
        except Exception as e:
            return f"Error calling tool {tool_name}: {e}"

    async def _initialize(self) -> None:
        """Initialize the MCP connection."""
        response = await self._send_request(
            "initialize",
            {
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {
                    "name": "nanobot",
                    "version": "0.1.0",
                },
            },
        )

        if response.error:
            raise RuntimeError(f"Initialize failed: {response.error}")

        self._server_info = MCPServerInfo.from_dict(response.result or {})

        # Send initialized notification
        await self._send_notification("notifications/initialized", {})

    async def _discover_tools(self) -> None:
        """Discover available tools from the server."""
        if not self._server_info or not self._server_info.capabilities.tools:
            logger.debug(f"MCP server {self.name} does not support tools")
            return

        response = await self._send_request("tools/list", {})

        if response.error:
            logger.warning(f"Failed to list tools from {self.name}: {response.error}")
            return

        tools_data = response.result or {}
        self._tools = [MCPToolInfo.from_dict(t) for t in tools_data.get("tools", [])]
        logger.debug(f"Discovered {len(self._tools)} tools from {self.name}")

    async def _send_request(
        self, method: str, params: dict[str, Any] | None = None
    ) -> JsonRpcResponse:
        """Send a JSON-RPC request and wait for response."""
        if not self._process or not self._process.stdin:
            raise RuntimeError("Server not running")

        self._request_id += 1
        request_id = self._request_id

        request = JsonRpcRequest(method=method, id=request_id, params=params)
        message = json.dumps(request.to_dict()) + "\n"

        # Create future for response
        future: asyncio.Future[JsonRpcResponse] = asyncio.get_event_loop().create_future()
        self._pending[request_id] = future

        try:
            # Send request
            self._process.stdin.write(message.encode("utf-8"))
            await self._process.stdin.drain()

            # Wait for response
            response = await asyncio.wait_for(future, timeout=self.timeout)
            return response

        finally:
            self._pending.pop(request_id, None)

    async def _send_notification(self, method: str, params: dict[str, Any] | None = None) -> None:
        """Send a JSON-RPC notification (no response expected)."""
        if not self._process or not self._process.stdin:
            raise RuntimeError("Server not running")

        from nanobot.mcp.protocol import JsonRpcNotification

        notification = JsonRpcNotification(method=method, params=params)
        message = json.dumps(notification.to_dict()) + "\n"

        self._process.stdin.write(message.encode("utf-8"))
        await self._process.stdin.drain()

    async def _read_loop(self) -> None:
        """Read responses from the server."""
        if not self._process or not self._process.stdout:
            logger.debug(f"MCP {self.name}: No stdout available")
            return

        logger.debug(f"MCP {self.name}: Read loop started")
        try:
            while True:
                line = await self._process.stdout.readline()
                if not line:
                    logger.debug(f"MCP {self.name}: EOF on stdout")
                    break

                logger.debug(f"MCP {self.name}: Received line ({len(line)} bytes)")
                try:
                    data = json.loads(line.decode("utf-8"))
                    response = JsonRpcResponse.from_dict(data)

                    # Match response to pending request
                    if response.id is not None and response.id in self._pending:
                        future = self._pending[response.id]
                        if not future.done():
                            future.set_result(response)
                            logger.debug(f"MCP {self.name}: Resolved request {response.id}")

                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON from MCP server {self.name}: {line}")
                except Exception as e:
                    logger.warning(f"Error processing message from {self.name}: {e}")

        except asyncio.CancelledError:
            logger.debug(f"MCP {self.name}: Read loop cancelled")
        except Exception as e:
            logger.error(f"Read loop error for {self.name}: {e}")

    async def _read_stderr(self) -> None:
        """Read stderr from the server for logging."""
        if not self._process or not self._process.stderr:
            return

        try:
            while True:
                line = await self._process.stderr.readline()
                if not line:
                    break
                logger.debug(f"MCP {self.name} stderr: {line.decode('utf-8').rstrip()}")
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.warning(f"Stderr read error for {self.name}: {e}")
