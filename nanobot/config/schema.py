"""Configuration schema using Pydantic."""

from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class WhatsAppConfig(BaseModel):
    """WhatsApp channel configuration."""

    enabled: bool = False
    bridge_url: str = "ws://localhost:3001"
    allow_from: list[str] = Field(default_factory=list)  # Allowed phone numbers


class TelegramConfig(BaseModel):
    """Telegram channel configuration."""

    enabled: bool = False
    token: str = ""  # Bot token from @BotFather
    allow_from: list[str] = Field(default_factory=list)  # Allowed user IDs or usernames
    proxy: str | None = (
        None  # HTTP/SOCKS5 proxy URL, e.g. "http://127.0.0.1:7890" or "socks5://127.0.0.1:1080"
    )


class FeishuConfig(BaseModel):
    """Feishu/Lark channel configuration using WebSocket long connection."""

    enabled: bool = False
    app_id: str = ""  # App ID from Feishu Open Platform
    app_secret: str = ""  # App Secret from Feishu Open Platform
    encrypt_key: str = ""  # Encrypt Key for event subscription (optional)
    verification_token: str = ""  # Verification Token for event subscription (optional)
    allow_from: list[str] = Field(default_factory=list)  # Allowed user open_ids


class DiscordConfig(BaseModel):
    """Discord channel configuration."""

    enabled: bool = False
    token: str = ""  # Bot token from Discord Developer Portal
    guild_id: str = ""  # The server ID where nanobot lives
    default_channel_id: str = ""  # Channel for proactive messages (cron, notifications)
    trigger_word: str = "nano"  # Respond when this word appears in messages
    allow_from: list[str] = Field(default_factory=list)  # Allowed user IDs
    context_messages: int = Field(default=0, alias="contextMessages")  # Recent channel messages
    # Reaction emojis (customizable)
    emoji_processing: str = "\u23f3"  # Hourglass
    emoji_complete: str = "\u2705"  # Checkmark
    emoji_error: str = "\u274c"  # X mark


class ChannelsConfig(BaseModel):
    """Configuration for chat channels."""

    whatsapp: WhatsAppConfig = Field(default_factory=WhatsAppConfig)
    telegram: TelegramConfig = Field(default_factory=TelegramConfig)
    feishu: FeishuConfig = Field(default_factory=FeishuConfig)
    discord: DiscordConfig = Field(default_factory=DiscordConfig)


class MemoryExtractionConfig(BaseModel):
    """Memory extraction and consolidation configuration."""

    enabled: bool = True
    extraction_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    max_memories: int = 1000
    extraction_interval: int = 10
    max_facts_per_extraction: int = 5
    max_lessons_per_extraction: int = 3
    enable_pre_compaction_flush: bool = True
    enable_tool_lessons: bool = True
    lesson_generalization_model: str = "gpt-4o-mini"
    candidate_threshold: float = 0.5


class AgentDefaults(BaseModel):
    """Default agent configuration."""

    workspace: str = "~/.nanobot/workspace"
    model: str = "anthropic/claude-opus-4-5"
    max_tokens: int = 8192
    temperature: float = 0.7
    max_tool_iterations: int = 20
    memory_extraction: MemoryExtractionConfig = Field(
        default_factory=MemoryExtractionConfig
    )


class AgentsConfig(BaseModel):
    """Agent configuration."""

    defaults: AgentDefaults = Field(default_factory=AgentDefaults)


class ProviderConfig(BaseModel):
    """LLM provider configuration."""

    api_key: str = ""
    api_base: str | None = None


class ProvidersConfig(BaseModel):
    """Configuration for LLM providers."""

    anthropic: ProviderConfig = Field(default_factory=ProviderConfig)
    openai: ProviderConfig = Field(default_factory=ProviderConfig)
    openrouter: ProviderConfig = Field(default_factory=ProviderConfig)
    groq: ProviderConfig = Field(default_factory=ProviderConfig)
    zhipu: ProviderConfig = Field(default_factory=ProviderConfig)
    vllm: ProviderConfig = Field(default_factory=ProviderConfig)
    gemini: ProviderConfig = Field(default_factory=ProviderConfig)


class GatewayConfig(BaseModel):
    """Gateway/server configuration."""

    host: str = "0.0.0.0"
    port: int = 18790


class WebSearchConfig(BaseModel):
    """Web search tool configuration."""

    api_key: str = ""  # Brave Search API key
    max_results: int = 5


class WebToolsConfig(BaseModel):
    """Web tools configuration."""

    search: WebSearchConfig = Field(default_factory=WebSearchConfig)


class ExecToolConfig(BaseModel):
    """Shell exec tool configuration."""

    timeout: int = 60
    restrict_to_workspace: bool = False  # If true, block commands accessing paths outside workspace


class MCPServerConfig(BaseModel):
    """Configuration for a single MCP server."""

    command: str  # e.g., "npx", "uvx", "python"
    args: list[str] = Field(default_factory=list)  # Command arguments
    env: dict[str, str] = Field(default_factory=dict)  # Environment variables
    enabled: bool = True
    timeout: int = 30  # Startup/request timeout in seconds


class MCPToolConfig(BaseModel):
    """Configuration for an individual MCP tool."""

    enabled: bool = True  # Can disable specific tools


class MCPConfig(BaseModel):
    """MCP (Model Context Protocol) configuration."""

    enabled: bool = False
    servers: dict[str, MCPServerConfig] = Field(default_factory=dict)
    tools: dict[str, MCPToolConfig] = Field(default_factory=dict)  # Tool-specific overrides


class ToolsConfig(BaseModel):
    """Tools configuration."""

    web: WebToolsConfig = Field(default_factory=WebToolsConfig)
    exec: ExecToolConfig = Field(default_factory=ExecToolConfig)
    mcp: MCPConfig = Field(default_factory=MCPConfig)


class Config(BaseSettings):
    """Root configuration for nanobot."""

    agents: AgentsConfig = Field(default_factory=AgentsConfig)
    channels: ChannelsConfig = Field(default_factory=ChannelsConfig)
    providers: ProvidersConfig = Field(default_factory=ProvidersConfig)
    gateway: GatewayConfig = Field(default_factory=GatewayConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)

    @property
    def workspace_path(self) -> Path:
        """Get expanded workspace path."""
        return Path(self.agents.defaults.workspace).expanduser()

    def get_api_key(self) -> str | None:
        """Get API key in priority order: OpenRouter > Anthropic > OpenAI > Gemini > Zhipu > Groq > vLLM."""
        return (
            self.providers.openrouter.api_key
            or self.providers.anthropic.api_key
            or self.providers.openai.api_key
            or self.providers.gemini.api_key
            or self.providers.zhipu.api_key
            or self.providers.groq.api_key
            or self.providers.vllm.api_key
            or None
        )

    def get_api_base(self) -> str | None:
        """Get API base URL if using OpenRouter, Zhipu or vLLM."""
        if self.providers.openrouter.api_key:
            return self.providers.openrouter.api_base or "https://openrouter.ai/api/v1"
        if self.providers.zhipu.api_key:
            return self.providers.zhipu.api_base
        if self.providers.vllm.api_base:
            return self.providers.vllm.api_base
        return None

    class Config:
        env_prefix = "NANOBOT_"
        env_nested_delimiter = "__"
