"""LiteLLM provider implementation for multi-provider support."""

import os
from typing import Any

import litellm
from litellm import acompletion

from nanobot.providers.base import LLMProvider, LLMResponse, ToolCallRequest

# Data-driven prefix rules for model routing.
# Each entry: (model_keywords, litellm_prefix, skip_if_already_prefixed)
_prefix_rules: list[tuple[tuple[str, ...], str, tuple[str, ...]]] = [
    (("glm", "zhipu"), "zai", ("zhipu/", "zai/", "openrouter/", "hosted_vllm/")),
    (("qwen", "dashscope"), "dashscope", ("dashscope/", "openrouter/")),
    (("moonshot", "kimi"), "moonshot", ("moonshot/", "openrouter/")),
    (("deepseek",), "deepseek", ("deepseek/", "openrouter/")),
    (("gemini",), "gemini", ("gemini/", "openrouter/", "openai/", "zai/", "hosted_vllm/")),
]


class LiteLLMProvider(LLMProvider):
    """
    LLM provider using LiteLLM for multi-provider support.

    Supports OpenRouter, Anthropic, OpenAI, Gemini, DeepSeek, DashScope,
    Moonshot, AiHubMix, and many other providers through a unified interface.
    """

    def __init__(
        self,
        api_key: str | None = None,
        api_base: str | None = None,
        default_model: str = "anthropic/claude-opus-4-5",
        extra_headers: dict[str, str] | None = None,
    ):
        super().__init__(api_key, api_base)
        self.default_model = default_model
        self.extra_headers = extra_headers

        # Detect known providers by api_key prefix, api_base, or model name
        self.is_openrouter = (api_key and api_key.startswith("sk-or-")) or (
            api_base and "openrouter" in api_base
        )

        model_lower = default_model.lower()
        self.is_zhipu = (
            "zhipu" in model_lower
            or "glm" in model_lower
            or "zai" in model_lower
            or (bool(api_base) and "z.ai" in (api_base or ""))
        )

        self.is_aihubmix = bool(api_base) and "aihubmix" in (api_base or "")

        # Track if using custom endpoint (vLLM, etc.) - exclude known providers
        self.is_vllm = (
            bool(api_base) and not self.is_openrouter and not self.is_zhipu and not self.is_aihubmix
        )

        # Configure LiteLLM based on provider
        if api_key:
            if self.is_openrouter:
                os.environ["OPENROUTER_API_KEY"] = api_key
            elif self.is_zhipu:
                os.environ["ZHIPUAI_API_KEY"] = api_key
            elif self.is_aihubmix:
                os.environ["OPENAI_API_KEY"] = api_key
            elif self.is_vllm:
                os.environ["OPENAI_API_KEY"] = api_key
            elif "deepseek" in model_lower:
                os.environ.setdefault("DEEPSEEK_API_KEY", api_key)
            elif "dashscope" in model_lower or "qwen" in model_lower:
                os.environ.setdefault("DASHSCOPE_API_KEY", api_key)
            elif "moonshot" in model_lower or "kimi" in model_lower:
                os.environ.setdefault("MOONSHOT_API_KEY", api_key)
            elif "anthropic" in default_model:
                os.environ.setdefault("ANTHROPIC_API_KEY", api_key)
            elif "openai" in default_model or "gpt" in default_model:
                os.environ.setdefault("OPENAI_API_KEY", api_key)
            elif "gemini" in model_lower:
                os.environ.setdefault("GEMINI_API_KEY", api_key)
            elif "groq" in default_model:
                os.environ.setdefault("GROQ_API_KEY", api_key)

        # Disable LiteLLM logging noise
        litellm.suppress_debug_info = True

    def _apply_model_prefix(self, model: str) -> str:
        """Apply the correct LiteLLM prefix to a model name.

        Uses the data-driven ``_prefix_rules`` table. Special cases for
        OpenRouter, Zhipu with custom api_base, vLLM, and AiHubMix are
        handled before the table lookup.
        """
        # OpenRouter: always prefix
        if self.is_openrouter and not model.startswith("openrouter/"):
            return f"openrouter/{model}"

        # Zhipu with custom api_base: use openai/ prefix for direct auth
        if self.is_zhipu and not model.startswith("openrouter/"):
            if self.api_base and not model.startswith("openai/"):
                return f"openai/{model}"
            if not self.api_base and not model.startswith("zai/"):
                return f"zai/{model}"

        # AiHubMix: openai-compatible endpoint
        if self.is_aihubmix and not model.startswith("openai/"):
            return f"openai/{model}"

        # vLLM: hosted_vllm/ prefix per LiteLLM docs
        if self.is_vllm:
            return f"hosted_vllm/{model}"

        # Data-driven prefix rules
        model_lower = model.lower()
        for keywords, prefix, skip_prefixes in _prefix_rules:
            if any(kw in model_lower for kw in keywords):
                if not model.startswith(skip_prefixes):
                    return f"{prefix}/{model}"

        return model

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> LLMResponse:
        """
        Send a chat completion request via LiteLLM.

        Args:
            messages: List of message dicts with 'role' and 'content'.
            tools: Optional list of tool definitions in OpenAI format.
            model: Model identifier (e.g., 'anthropic/claude-sonnet-4-5').
            max_tokens: Maximum tokens in response.
            temperature: Sampling temperature.

        Returns:
            LLMResponse with content and/or tool calls.
        """
        model = self._apply_model_prefix(model or self.default_model)

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        # Pass credentials directly so LiteLLM can authenticate per-call
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.api_base:
            kwargs["api_base"] = self.api_base
        if self.extra_headers:
            kwargs["extra_headers"] = self.extra_headers

        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        try:
            response = await acompletion(**kwargs)
            return self._parse_response(response)
        except Exception as e:
            # Return error as content for graceful handling
            return LLMResponse(
                content=f"Error calling LLM: {str(e)}",
                finish_reason="error",
            )

    def _parse_response(self, response: Any) -> LLMResponse:
        """Parse LiteLLM response into our standard format."""
        choice = response.choices[0]
        message = choice.message

        tool_calls = []
        if hasattr(message, "tool_calls") and message.tool_calls:
            for tc in message.tool_calls:
                # Parse arguments from JSON string if needed
                args = tc.function.arguments
                if isinstance(args, str):
                    import json

                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        from loguru import logger

                        logger.warning(
                            f"Malformed tool call arguments for {tc.function.name}: skipping"
                        )
                        continue

                tool_calls.append(
                    ToolCallRequest(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=args,
                    )
                )

        usage = {}
        if hasattr(response, "usage") and response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

        return LLMResponse(
            content=message.content,
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason or "stop",
            usage=usage,
        )

    def get_default_model(self) -> str:
        """Get the default model."""
        return self.default_model
