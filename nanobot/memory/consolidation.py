"""Memory consolidation engine for merging similar memories."""

import json
import sqlite3
from typing import Any

from loguru import logger

from nanobot.providers.base import LLMProvider

MERGE_PROMPT = """You are merging similar memories into a single consolidated memory.

Here are the individual memories to merge:
{memories}

Rules:
- Create ONE concise memory that captures all unique information
- Preserve the most specific and recent details
- Do not lose any important facts
- Keep the same tone and format as the originals
- Output ONLY the consolidated memory text, nothing else

Consolidated memory:"""


class MemoryConsolidator:
    """Consolidates similar memories to reduce bloat and improve retrieval."""

    def __init__(
        self,
        vector_store: Any,
        provider: LLMProvider,
        model: str | None = None,
    ):
        """
        Initialize the consolidator.

        Args:
            vector_store: Vector store instance (duck typed).
            provider: LLM provider for generating merged summaries.
            model: Model to use (None = provider default).
        """
        self.vector_store = vector_store
        self.provider = provider
        self.model = model

    async def consolidate(
        self,
        similarity_threshold: float = 0.85,
        min_cluster_size: int = 2,
        dry_run: bool = False,
    ) -> dict[str, int]:
        """
        Find clusters of similar memories and merge them.

        Strategy:
        1. Load all entries from the vector store
        2. Greedy clustering: for each entry, find entries above threshold
        3. For clusters with >= min_cluster_size entries, use LLM to merge
        4. Delete originals and insert the consolidated summary

        Args:
            similarity_threshold: Minimum similarity to consider entries related.
            min_cluster_size: Minimum entries to form a cluster.
            dry_run: If True, only report what would be done.

        Returns:
            Stats dict with clusters_found, entries_merged, entries_created.
        """
        entries = self._load_all_entries()
        if len(entries) < min_cluster_size:
            logger.debug(f"Only {len(entries)} entries, need at least {min_cluster_size}")
            return {
                "clusters_found": 0,
                "entries_merged": 0,
                "entries_created": 0,
            }

        clusters = self._find_clusters(entries, similarity_threshold, min_cluster_size)

        if not clusters:
            logger.debug("No clusters found above similarity threshold")
            return {
                "clusters_found": 0,
                "entries_merged": 0,
                "entries_created": 0,
            }

        logger.debug(f"Found {len(clusters)} clusters to consolidate")

        if dry_run:
            total_entries = sum(len(c) for c in clusters)
            return {
                "clusters_found": len(clusters),
                "entries_merged": total_entries,
                "entries_created": len(clusters),
            }

        entries_merged = 0
        entries_created = 0

        for cluster in clusters:
            try:
                merged_text = await self._merge_cluster(cluster)
                if not merged_text:
                    continue

                # Collect metadata from the most recent entry
                newest = max(
                    cluster,
                    key=lambda e: e.get("created_at", ""),
                )
                metadata = newest.get("metadata", {})
                metadata["consolidated"] = True
                metadata["source_count"] = len(cluster)

                # Delete original entries
                for entry in cluster:
                    entry_id = entry.get("id")
                    if entry_id is not None:
                        await self.vector_store.delete(entry_id)

                # Insert consolidated entry
                await self.vector_store.add(merged_text, metadata=metadata)

                entries_merged += len(cluster)
                entries_created += 1

            except Exception as e:
                logger.error(f"Failed to consolidate cluster: {e}")
                continue

        logger.debug(
            f"Consolidation complete: merged {entries_merged} entries into {entries_created}"
        )
        return {
            "clusters_found": len(clusters),
            "entries_merged": entries_merged,
            "entries_created": entries_created,
        }

    async def _merge_cluster(self, entries: list[dict]) -> str:
        """
        Use LLM to merge a cluster of similar memories into one.

        Args:
            entries: List of memory entry dicts with 'text' key.

        Returns:
            Consolidated memory text, or empty string on failure.
        """
        texts = [e["text"] for e in entries]
        numbered = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(texts))
        prompt = MERGE_PROMPT.format(memories=numbered)

        try:
            response = await self.provider.chat(
                messages=[{"role": "user", "content": prompt}],
                model=self.model,
                max_tokens=512,
                temperature=0.3,
            )
            return (response.content or "").strip()
        except Exception as e:
            logger.error(f"LLM merge failed: {e}")
            return ""

    async def prune_old(
        self,
        max_age_days: int = 365,
        dry_run: bool = False,
    ) -> dict[str, int]:
        """
        Remove memories older than max_age_days that were never accessed.

        Only prunes entries with access_count == 0 (never recalled).

        Args:
            max_age_days: Maximum age in days for unaccessed memories.
            dry_run: If True, only report what would be done.

        Returns:
            Stats dict with pruned_count.
        """
        from datetime import datetime, timedelta

        cutoff = (datetime.now() - timedelta(days=max_age_days)).isoformat()

        entries = self._load_all_entries()
        to_prune = []
        for entry in entries:
            created = entry.get("created_at", "")
            if not created or created >= cutoff:
                continue
            if entry.get("access_count", 0) == 0:
                to_prune.append(entry)

        if dry_run:
            return {"pruned_count": len(to_prune)}

        pruned = 0
        for entry in to_prune:
            entry_id = entry.get("id")
            if entry_id is not None:
                try:
                    await self.vector_store.delete(entry_id)
                    pruned += 1
                except Exception as e:
                    logger.error(f"Failed to prune entry: {e}")

        logger.debug(f"Pruned {pruned} old unaccessed memories")
        return {"pruned_count": pruned}

    def _load_all_entries(self) -> list[dict[str, Any]]:
        """Load all entries from the vector store's SQLite database."""
        entries = []
        try:
            db_path = self.vector_store.db_path
            with sqlite3.connect(db_path) as conn:
                cursor = conn.execute(
                    "SELECT id, text, embedding, metadata, created_at, access_count FROM vectors"
                )
                for row in cursor:
                    (
                        entry_id,
                        text,
                        embedding_blob,
                        metadata_json,
                        created_at,
                        access_count,
                    ) = row
                    embedding = json.loads(embedding_blob.decode("utf-8"))
                    metadata = json.loads(metadata_json) if metadata_json else {}
                    entries.append(
                        {
                            "id": entry_id,
                            "text": text,
                            "embedding": embedding,
                            "metadata": metadata,
                            "created_at": created_at,
                            "access_count": access_count or 0,
                        }
                    )
        except Exception as e:
            logger.error(f"Failed to load entries: {e}")
        return entries

    def _find_clusters(
        self,
        entries: list[dict],
        threshold: float,
        min_size: int,
    ) -> list[list[dict]]:
        """
        Greedy clustering of entries by embedding similarity.

        Args:
            entries: All vector store entries with embeddings.
            threshold: Minimum cosine similarity for clustering.
            min_size: Minimum cluster size.

        Returns:
            List of clusters (each a list of entry dicts).
        """
        used = set()
        clusters: list[list[dict]] = []

        for i, entry_a in enumerate(entries):
            if i in used:
                continue
            cluster = [entry_a]
            used.add(i)

            for j, entry_b in enumerate(entries):
                if j in used or j <= i:
                    continue
                sim = self._cosine_similarity(entry_a["embedding"], entry_b["embedding"])
                if sim >= threshold:
                    cluster.append(entry_b)
                    used.add(j)

            if len(cluster) >= min_size:
                clusters.append(cluster)

        return clusters

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        from nanobot.memory.vectors import _cosine_similarity_fast

        return _cosine_similarity_fast(a, b)
