"""Lightweight entity-based knowledge graph backed by SQLite."""

import json
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger


class EntityStore:
    """Lightweight knowledge graph for entity relationships."""

    def __init__(self, db_path: str | Path):
        """
        Initialize the entity store.

        Args:
            db_path: Path to the SQLite database file.
        """
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        """Create entities and relations tables."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS entities (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    type TEXT NOT NULL,
                    attributes TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS relations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_id INTEGER NOT NULL,
                    relation TEXT NOT NULL,
                    target_id INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (source_id) REFERENCES entities(id)
                        ON DELETE CASCADE,
                    FOREIGN KEY (target_id) REFERENCES entities(id)
                        ON DELETE CASCADE,
                    UNIQUE(source_id, relation, target_id)
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_entity_name
                ON entities(name)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_relation_source
                ON relations(source_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_relation_target
                ON relations(target_id)
            """)
            conn.execute("PRAGMA foreign_keys = ON")
            conn.commit()

        # Set file permissions to owner-only
        try:
            os.chmod(self.db_path, 0o600)
        except OSError:
            pass

    def _get_conn(self) -> sqlite3.Connection:
        """Get a database connection with foreign keys enabled."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def upsert_entity(
        self,
        name: str,
        entity_type: str,
        attributes: dict[str, Any] | None = None,
    ) -> int:
        """
        Insert or update an entity. Merges attributes on update.

        Args:
            name: Entity name (unique identifier).
            entity_type: Type of entity (e.g. person, project).
            attributes: Optional key-value attributes.

        Returns:
            Entity ID.
        """
        now = datetime.now().isoformat()
        with self._get_conn() as conn:
            cursor = conn.execute(
                "SELECT id, attributes FROM entities WHERE name = ?",
                (name,),
            )
            row = cursor.fetchone()

            if row:
                entity_id, existing_json = row
                existing = json.loads(existing_json) if existing_json else {}
                if attributes:
                    existing.update(attributes)
                conn.execute(
                    "UPDATE entities SET type = ?, attributes = ?, updated_at = ? WHERE id = ?",
                    (
                        entity_type,
                        json.dumps(existing, ensure_ascii=False),
                        now,
                        entity_id,
                    ),
                )
                conn.commit()
                logger.debug(f"Updated entity: {name}")
                return entity_id
            else:
                cursor = conn.execute(
                    "INSERT INTO entities "
                    "(name, type, attributes, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (
                        name,
                        entity_type,
                        json.dumps(
                            attributes or {},
                            ensure_ascii=False,
                        ),
                        now,
                        now,
                    ),
                )
                conn.commit()
                entity_id = cursor.lastrowid or 0
                logger.debug(f"Created entity: {name} (id={entity_id})")
                return entity_id

    def _ensure_entity(self, name: str) -> int:
        """Get or create an entity by name. Returns entity ID."""
        with self._get_conn() as conn:
            cursor = conn.execute(
                "SELECT id FROM entities WHERE name = ?",
                (name,),
            )
            row = cursor.fetchone()
            if row:
                return row[0]
        return self.upsert_entity(name, "unknown")

    def add_relation(self, source: str, relation: str, target: str) -> bool:
        """
        Add a relation between two entities.

        Creates entities if they do not exist.

        Args:
            source: Source entity name.
            relation: Relation type (e.g. "works_at", "knows").
            target: Target entity name.

        Returns:
            True if relation was added, False if it already exists.
        """
        source_id = self._ensure_entity(source)
        target_id = self._ensure_entity(target)

        with self._get_conn() as conn:
            try:
                conn.execute(
                    "INSERT INTO relations "
                    "(source_id, relation, target_id, created_at) "
                    "VALUES (?, ?, ?, ?)",
                    (
                        source_id,
                        relation,
                        target_id,
                        datetime.now().isoformat(),
                    ),
                )
                conn.commit()
                logger.debug(f"Added relation: {source} --{relation}--> {target}")
                return True
            except sqlite3.IntegrityError:
                logger.debug(f"Relation already exists: {source} --{relation}--> {target}")
                return False

    def query_entity(self, name: str) -> dict[str, Any] | None:
        """
        Get an entity with all its relations.

        Args:
            name: Entity name to look up.

        Returns:
            Dict with name, type, attributes, relations; or None.
        """
        with self._get_conn() as conn:
            cursor = conn.execute(
                "SELECT id, name, type, attributes, "
                "created_at, updated_at "
                "FROM entities WHERE name = ?",
                (name,),
            )
            row = cursor.fetchone()
            if not row:
                return None

            entity_id, ename, etype, attr_json, created, updated = row
            attributes = json.loads(attr_json) if attr_json else {}

            relations = self._get_relations_for_id(conn, entity_id, ename)

            return {
                "name": ename,
                "type": etype,
                "attributes": attributes,
                "created_at": created,
                "updated_at": updated,
                "relations": relations,
            }

    def _get_relations_for_id(
        self,
        conn: sqlite3.Connection,
        entity_id: int,
        entity_name: str,
    ) -> list[dict[str, str]]:
        """Get all relations for an entity ID."""
        relations: list[dict[str, str]] = []

        # Outgoing relations
        cursor = conn.execute(
            "SELECT r.relation, e.name "
            "FROM relations r "
            "JOIN entities e ON e.id = r.target_id "
            "WHERE r.source_id = ?",
            (entity_id,),
        )
        for rel_type, target_name in cursor:
            relations.append(
                {
                    "relation": rel_type,
                    "target": target_name,
                    "direction": "outgoing",
                }
            )

        # Incoming relations
        cursor = conn.execute(
            "SELECT r.relation, e.name "
            "FROM relations r "
            "JOIN entities e ON e.id = r.source_id "
            "WHERE r.target_id = ?",
            (entity_id,),
        )
        for rel_type, source_name in cursor:
            relations.append(
                {
                    "relation": rel_type,
                    "target": source_name,
                    "direction": "incoming",
                }
            )

        return relations

    def search_entities(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        """
        Search entities by name using LIKE matching.

        Args:
            query: Search string (supports SQL LIKE wildcards).
            limit: Maximum results to return.

        Returns:
            List of matching entity dicts.
        """
        pattern = f"%{query}%"
        results: list[dict[str, Any]] = []

        with self._get_conn() as conn:
            cursor = conn.execute(
                "SELECT name, type, attributes, "
                "created_at, updated_at "
                "FROM entities WHERE name LIKE ? "
                "ORDER BY updated_at DESC LIMIT ?",
                (pattern, limit),
            )
            for row in cursor:
                name, etype, attr_json, created, updated = row
                attributes = json.loads(attr_json) if attr_json else {}
                results.append(
                    {
                        "name": name,
                        "type": etype,
                        "attributes": attributes,
                        "created_at": created,
                        "updated_at": updated,
                    }
                )

        return results

    def get_relations(self, entity_name: str) -> list[dict[str, str]]:
        """
        Get all relations for an entity by name.

        Args:
            entity_name: Name of the entity.

        Returns:
            List of relation dicts with relation, target, direction.
        """
        with self._get_conn() as conn:
            cursor = conn.execute(
                "SELECT id FROM entities WHERE name = ?",
                (entity_name,),
            )
            row = cursor.fetchone()
            if not row:
                return []
            return self._get_relations_for_id(conn, row[0], entity_name)

    def remove_entity(self, name: str) -> bool:
        """
        Remove an entity and all its relations.

        Args:
            name: Entity name to remove.

        Returns:
            True if entity was removed, False if not found.
        """
        with self._get_conn() as conn:
            cursor = conn.execute(
                "SELECT id FROM entities WHERE name = ?",
                (name,),
            )
            row = cursor.fetchone()
            if not row:
                return False

            entity_id = row[0]
            conn.execute(
                "DELETE FROM relations WHERE source_id = ? OR target_id = ?",
                (entity_id, entity_id),
            )
            conn.execute(
                "DELETE FROM entities WHERE id = ?",
                (entity_id,),
            )
            conn.commit()
            logger.debug(f"Removed entity: {name}")
            return True

    def get_stats(self) -> dict[str, int]:
        """
        Return counts of entities and relations.

        Returns:
            Dict with entity_count and relation_count.
        """
        with self._get_conn() as conn:
            entities = conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
            relations = conn.execute("SELECT COUNT(*) FROM relations").fetchone()[0]

        return {
            "entity_count": entities,
            "relation_count": relations,
        }
