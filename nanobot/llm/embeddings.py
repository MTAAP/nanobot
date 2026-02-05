"""Embedding service using LiteLLM."""

import asyncio
import os
from typing import Any

import litellm
from loguru import logger


class EmbeddingService:
    """
    Generates embeddings using LiteLLM.

    Supports various embedding models through OpenRouter, OpenAI, etc.
    """

    def __init__(
        self,
        model: str = "openai/text-embedding-3-small",
        api_key: str | None = None,
        api_base: str | None = None,
    ):
        """
        Initialize the embedding service.

        Args:
            model: Embedding model to use.
            api_key: API key for the provider.
            api_base: Optional API base URL.
        """
        self.model = model
        self.api_key = api_key
        self.api_base = api_base

        # Detect OpenRouter
        self.is_openrouter = (api_key and api_key.startswith("sk-or-")) or (
            api_base and "openrouter" in (api_base or "")
        )

        # Set up environment for LiteLLM
        if api_key:
            if self.is_openrouter:
                os.environ["OPENROUTER_API_KEY"] = api_key
            elif "openai" in model.lower():
                os.environ.setdefault("OPENAI_API_KEY", api_key)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors.
        """
        if not texts:
            return []

        # Format model name for LiteLLM
        model = self.model
        if self.is_openrouter and not model.startswith("openrouter/"):
            model = f"openrouter/{model}"

        kwargs: dict[str, Any] = {
            "model": model,
            "input": texts,
        }

        if self.api_base:
            kwargs["api_base"] = self.api_base

        max_retries = 2
        backoff_seconds = [1, 2]
        last_error: Exception | None = None

        for attempt in range(max_retries + 1):
            try:
                response = await litellm.aembedding(**kwargs)

                # Extract embeddings from response
                embeddings = []
                for item in response.data:
                    embeddings.append(item["embedding"])

                return embeddings

            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    wait = backoff_seconds[attempt]
                    logger.warning(
                        f"Embedding attempt {attempt + 1} failed "
                        f"(model={model}, input_count={len(texts)}), "
                        f"retrying in {wait}s"
                    )
                    await asyncio.sleep(wait)

        total_chars = sum(len(t) for t in texts)
        logger.error(
            f"Embedding failed after {max_retries + 1} attempts: "
            f"model={model}, input_count={len(texts)}, "
            f"total_chars={total_chars}, error={last_error}"
        )
        raise last_error  # type: ignore[misc]

    async def embed_single(self, text: str) -> list[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed.

        Returns:
            Embedding vector.
        """
        embeddings = await self.embed([text])
        return embeddings[0] if embeddings else []
