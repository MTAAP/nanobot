"""SQLite-based vector store for semantic memory search."""

import asyncio
import hashlib
import json
import math
import os
import sqlite3
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

from nanobot.llm.embeddings import EmbeddingService
from nanobot.utils.helpers import ensure_dir

CURRENT_SCHEMA_VERSION = 1


class VectorStore:
    """
    SQLite-based vector store for semantic memory.

    Stores embeddings with metadata and supports cosine similarity search.
    Handles 100k+ entries efficiently.
    """

    def __init__(
        self,
        db_path: str | Path,
        embedding_service: EmbeddingService,
    ):
        """
        Initialize the vector store.

        Args:
            db_path: Path to SQLite database file.
            embedding_service: Service for generating embeddings.
        """
        self.db_path = Path(db_path).expanduser()
        self.embedding_service = embedding_service
        self._write_lock = asyncio.Lock()
        self._query_embedding_cache: OrderedDict[str, list[float]] = OrderedDict()
        self._cache_max_size = 128

        # Ensure directory exists
        ensure_dir(self.db_path.parent)

        # Initialize database
        self._init_db()

    def _init_db(self) -> None:
        """Initialize the SQLite database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=10000")

            # Create schema version table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS schema_version (
                    version INTEGER NOT NULL
                )
            """)

            cursor = conn.execute("SELECT version FROM schema_version")
            row = cursor.fetchone()

            if row is None:
                # Check if vectors table already exists (pre-versioning DB)
                existing = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='vectors'"
                ).fetchone()

                if existing:
                    # Existing DB without version tracking: migrate from v0
                    conn.execute(
                        "INSERT INTO schema_version (version) VALUES (0)",
                    )
                    conn.commit()
                    self._run_migrations(conn, 0)
                else:
                    # Truly fresh database: create at current version
                    conn.execute("""
                        CREATE TABLE vectors (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            content_hash TEXT UNIQUE NOT NULL,
                            text TEXT NOT NULL,
                            embedding BLOB NOT NULL,
                            metadata TEXT,
                            created_at TEXT NOT NULL,
                            access_count INTEGER DEFAULT 0,
                            last_accessed_at TEXT
                        )
                    """)
                    conn.execute("""
                        CREATE INDEX IF NOT EXISTS idx_content_hash
                        ON vectors(content_hash)
                    """)
                    conn.execute("""
                        CREATE INDEX IF NOT EXISTS idx_created_at
                        ON vectors(created_at)
                    """)
                    conn.execute(
                        "INSERT INTO schema_version (version) VALUES (?)",
                        (CURRENT_SCHEMA_VERSION,),
                    )
                    conn.commit()
            else:
                db_version = row[0]
                if db_version < CURRENT_SCHEMA_VERSION:
                    self._run_migrations(conn, db_version)

            # Defensive: ensure access tracking columns exist regardless of version.
            # Handles edge cases where schema_version was set but ALTER failed.
            cols = {r[1] for r in conn.execute("PRAGMA table_info(vectors)")}
            if "access_count" not in cols:
                conn.execute("ALTER TABLE vectors ADD COLUMN access_count INTEGER DEFAULT 0")
                conn.commit()
                logger.info("Added missing access_count column to vectors table")
            if "last_accessed_at" not in cols:
                conn.execute("ALTER TABLE vectors ADD COLUMN last_accessed_at TEXT")
                conn.commit()
                logger.info("Added missing last_accessed_at column to vectors table")

        # Set file permissions
        if self.db_path.exists():
            os.chmod(self.db_path, 0o600)

    def _run_migrations(self, conn: sqlite3.Connection, from_version: int) -> None:
        """Run schema migrations from from_version to current."""
        if from_version < 1:
            # Migration to v1: add access tracking columns
            cols = {r[1] for r in conn.execute("PRAGMA table_info(vectors)")}
            if "access_count" not in cols:
                conn.execute("ALTER TABLE vectors ADD COLUMN access_count INTEGER DEFAULT 0")
            if "last_accessed_at" not in cols:
                conn.execute("ALTER TABLE vectors ADD COLUMN last_accessed_at TEXT")

        conn.execute(
            "UPDATE schema_version SET version = ?",
            (CURRENT_SCHEMA_VERSION,),
        )
        conn.commit()
        logger.info(
            f"Migrated vector store schema from v{from_version} to v{CURRENT_SCHEMA_VERSION}"
        )

    def _compute_hash(self, text: str) -> str:
        """Compute content hash for deduplication."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]

    def _serialize_embedding(self, embedding: list[float]) -> bytes:
        """Serialize embedding to bytes for storage."""
        return json.dumps(embedding).encode("utf-8")

    def _deserialize_embedding(self, data: bytes) -> list[float]:
        """Deserialize embedding from bytes."""
        return json.loads(data.decode("utf-8"))

    def _cache_query_embedding(self, query: str, embedding: list[float]) -> None:
        """Cache a query embedding with FIFO eviction."""
        key = hashlib.md5(query.encode("utf-8")).hexdigest()
        if key in self._query_embedding_cache:
            self._query_embedding_cache.move_to_end(key)
        else:
            if len(self._query_embedding_cache) >= self._cache_max_size:
                self._query_embedding_cache.popitem(last=False)
            self._query_embedding_cache[key] = embedding

    def _get_cached_embedding(self, query: str) -> list[float] | None:
        """Get a cached query embedding if available."""
        key = hashlib.md5(query.encode("utf-8")).hexdigest()
        return self._query_embedding_cache.get(key)

    async def add(
        self,
        text: str,
        metadata: dict[str, Any] | None = None,
        skip_dedup: bool = False,
    ) -> bool:
        """
        Add a text entry to the vector store.

        Args:
            text: Text to index.
            metadata: Optional metadata (session_key, timestamp, etc.).
            skip_dedup: If True, skip semantic deduplication check.

        Returns:
            True if added, False if already exists or is a duplicate.
        """
        content_hash = self._compute_hash(text)

        async with self._write_lock:
            # Check for exact duplicate by hash
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "SELECT id FROM vectors WHERE content_hash = ?",
                    (content_hash,),
                )
                if cursor.fetchone():
                    logger.debug(f"Skipping duplicate entry: {content_hash[:8]}...")
                    return False

            # Generate embedding
            try:
                embedding = await self.embedding_service.embed_single(text)
            except Exception as e:
                logger.error(f"Failed to generate embedding: {e}")
                return False

            # Semantic dedup: check for similar existing entries
            if not skip_dedup:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.execute("SELECT id, embedding FROM vectors")
                    for row in cursor:
                        existing_emb = self._deserialize_embedding(row[1])
                        sim = self._cosine_similarity(embedding, existing_emb)
                        if sim > 0.85:
                            logger.debug(f"Skipping semantically similar entry (sim={sim:.3f})")
                            return False

            # Store in database
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO vectors
                    (content_hash, text, embedding, metadata,
                     created_at, access_count, last_accessed_at)
                    VALUES (?, ?, ?, ?, ?, 0, NULL)
                    """,
                    (
                        content_hash,
                        text,
                        self._serialize_embedding(embedding),
                        json.dumps(metadata) if metadata else None,
                        datetime.now().isoformat(),
                    ),
                )
                conn.commit()

        logger.debug(f"Added vector entry: {content_hash[:8]}...")
        return True

    async def add_batch(
        self,
        entries: list[tuple[str, dict[str, Any] | None]],
    ) -> int:
        """
        Add multiple entries efficiently.

        Args:
            entries: List of (text, metadata) tuples.

        Returns:
            Number of entries added.
        """
        if not entries:
            return 0

        async with self._write_lock:
            # Filter out duplicates
            to_add = []
            with sqlite3.connect(self.db_path) as conn:
                for text, metadata in entries:
                    content_hash = self._compute_hash(text)
                    cursor = conn.execute(
                        "SELECT id FROM vectors WHERE content_hash = ?",
                        (content_hash,),
                    )
                    if not cursor.fetchone():
                        to_add.append((text, metadata, content_hash))

            if not to_add:
                return 0

            # Generate embeddings in batch
            texts = [t[0] for t in to_add]
            try:
                embeddings = await self.embedding_service.embed(texts)
            except Exception as e:
                logger.error(f"Failed to generate embeddings: {e}")
                return 0

            # Store in database
            added = 0
            with sqlite3.connect(self.db_path) as conn:
                for i, (text, metadata, content_hash) in enumerate(to_add):
                    try:
                        conn.execute(
                            """
                            INSERT OR IGNORE INTO vectors
                            (content_hash, text, embedding,
                             metadata, created_at,
                             access_count, last_accessed_at)
                            VALUES (?, ?, ?, ?, ?, 0, NULL)
                            """,
                            (
                                content_hash,
                                text,
                                self._serialize_embedding(embeddings[i]),
                                json.dumps(metadata) if metadata else None,
                                datetime.now().isoformat(),
                            ),
                        )
                        added += 1
                    except Exception as e:
                        logger.warning(f"Failed to add entry: {e}")
                conn.commit()

        logger.debug(f"Added {added} vector entries in batch")
        return added

    async def search(
        self,
        query: str,
        top_k: int = 5,
        min_similarity: float = 0.3,
        recency_weight: float = 0.005,
        after: str | None = None,
        before: str | None = None,
        type_weights: dict[str, float] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Search for similar entries with time-decayed relevance.

        Args:
            query: Search query text.
            top_k: Maximum results to return.
            min_similarity: Minimum cosine similarity threshold.
            recency_weight: Decay rate for time-based scoring.
            after: ISO date string, only include entries after this.
            before: ISO date string, only include entries before this.
            type_weights: Score multipliers by entry type.

        Returns:
            List of matching entries with similarity scores.
        """
        _default_type_weights = {"fact": 1.2, "conversation": 1.0}
        if type_weights is None:
            type_weights = _default_type_weights

        # Check embedding cache
        cached = self._get_cached_embedding(query)
        if cached is not None:
            query_embedding = cached
        else:
            try:
                query_embedding = await self.embedding_service.embed_single(query)
                self._cache_query_embedding(query, query_embedding)
            except Exception as e:
                logger.error(f"Failed to generate query embedding: {e}")
                return []

        now = datetime.now()

        # Build SQL query with optional time filters
        sql = "SELECT id, text, embedding, metadata, created_at FROM vectors"
        conditions = []
        params: list[str] = []

        if after is not None:
            conditions.append("created_at >= ?")
            params.append(after)
        if before is not None:
            conditions.append("created_at <= ?")
            params.append(before)

        if conditions:
            sql += " WHERE " + " AND ".join(conditions)

        # Load vectors and compute similarities
        results = []
        matched_ids: list[int] = []

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(sql, params)
            for row in cursor:
                entry_id, text, embedding_blob, metadata_json, created_at = row
                embedding = self._deserialize_embedding(embedding_blob)
                similarity = self._cosine_similarity(query_embedding, embedding)

                if similarity < min_similarity:
                    continue

                metadata = json.loads(metadata_json) if metadata_json else {}

                # Time decay
                try:
                    created = datetime.fromisoformat(created_at)
                    days_old = (now - created).total_seconds() / 86400
                except (ValueError, TypeError):
                    days_old = 0.0

                final_score = similarity * math.exp(-recency_weight * days_old)

                # Type-weighted scoring
                entry_type = metadata.get("type", "conversation")
                weight = type_weights.get(entry_type, 1.0)
                final_score *= weight

                matched_ids.append(entry_id)
                results.append(
                    {
                        "id": entry_id,
                        "text": text,
                        "similarity": similarity,
                        "score": final_score,
                        "metadata": metadata,
                        "created_at": created_at,
                    }
                )

        # Sort by final score and take top_k
        results.sort(key=lambda x: x["score"], reverse=True)
        results = results[:top_k]

        # Update access tracking for returned results
        if results:
            returned_ids = [r["id"] for r in results]
            now_iso = now.isoformat()
            with sqlite3.connect(self.db_path) as conn:
                placeholders = ",".join("?" * len(returned_ids))
                conn.execute(
                    f"UPDATE vectors SET access_count = access_count + 1, "
                    f"last_accessed_at = ? "
                    f"WHERE id IN ({placeholders})",
                    [now_iso, *returned_ids],
                )
                conn.commit()

        return results

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        if len(a) != len(b):
            return 0.0

        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)

    def count(self) -> int:
        """Get the number of entries in the store."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM vectors")
            return cursor.fetchone()[0]

    def clear(self) -> None:
        """Clear all entries from the store."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM vectors")
            conn.commit()
        logger.info("Cleared vector store")

    async def delete(self, entry_id: int) -> bool:
        """
        Delete an entry by ID.

        Args:
            entry_id: ID of the entry to delete.

        Returns:
            True if deleted, False if not found.
        """
        async with self._write_lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("DELETE FROM vectors WHERE id = ?", (entry_id,))
                conn.commit()
                return cursor.rowcount > 0

    async def delete_by_query(
        self,
        query: str,
        min_similarity: float = 0.85,
    ) -> int:
        """
        Delete entries matching a semantic query.

        Args:
            query: Query text to match against.
            min_similarity: Minimum cosine similarity for deletion.

        Returns:
            Number of entries deleted.
        """
        try:
            query_embedding = await self.embedding_service.embed_single(query)
        except Exception as e:
            logger.error(f"Failed to generate query embedding for delete: {e}")
            return 0

        ids_to_delete: list[int] = []

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT id, embedding FROM vectors")
            for row in cursor:
                entry_id, embedding_blob = row
                embedding = self._deserialize_embedding(embedding_blob)
                similarity = self._cosine_similarity(query_embedding, embedding)
                if similarity >= min_similarity:
                    ids_to_delete.append(entry_id)

        if not ids_to_delete:
            return 0

        async with self._write_lock:
            with sqlite3.connect(self.db_path) as conn:
                placeholders = ",".join("?" * len(ids_to_delete))
                conn.execute(
                    f"DELETE FROM vectors WHERE id IN ({placeholders})",
                    ids_to_delete,
                )
                conn.commit()

        logger.debug(f"Deleted {len(ids_to_delete)} entries by query")
        return len(ids_to_delete)

    def get_stats(self) -> dict[str, Any]:
        """
        Get statistics about the vector store.

        Returns:
            Dict with count, total_size, oldest, and newest dates.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM vectors")
            count = cursor.fetchone()[0]

            cursor = conn.execute("SELECT MIN(created_at), MAX(created_at) FROM vectors")
            row = cursor.fetchone()
            oldest = row[0] if row else None
            newest = row[1] if row else None

            # Total size of the DB file
            total_size = self.db_path.stat().st_size if self.db_path.exists() else 0

        return {
            "count": count,
            "total_size": total_size,
            "oldest": oldest,
            "newest": newest,
        }
