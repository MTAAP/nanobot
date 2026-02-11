"""Local intelligence store for market signals and recommendations."""

import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any


class IntelStore:
    """SQLite-backed store for market intelligence data."""

    def __init__(
        self,
        db_path: str | Path = "~/.nanobot/data/intel.db",
    ):
        self._db_path = Path(db_path).expanduser()
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._create_tables()

    def _create_tables(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                company_name TEXT NOT NULL,
                signal_type TEXT NOT NULL,
                title TEXT NOT NULL,
                description TEXT DEFAULT '',
                source_url TEXT NOT NULL,
                source_name TEXT NOT NULL,
                relevance_score REAL DEFAULT 0.5,
                kp_service_match TEXT DEFAULT '',
                detected_at TEXT NOT NULL,
                status TEXT DEFAULT 'new',
                reviewed_by TEXT DEFAULT '',
                reviewed_at TEXT DEFAULT ''
            );

            CREATE TABLE IF NOT EXISTS recommendations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                signal_id INTEGER REFERENCES signals(id),
                company_name TEXT NOT NULL,
                consultant_name TEXT DEFAULT '',
                service_area TEXT DEFAULT '',
                outreach_channel TEXT DEFAULT 'email',
                pitch_summary TEXT DEFAULT '',
                status TEXT DEFAULT 'pending',
                created_at TEXT NOT NULL,
                approved_by TEXT DEFAULT '',
                approved_at TEXT DEFAULT '',
                sent_at TEXT DEFAULT ''
            );

            CREATE TABLE IF NOT EXISTS consultants (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT DEFAULT '',
                specializations TEXT DEFAULT '[]',
                industries TEXT DEFAULT '[]',
                regions TEXT DEFAULT '[]',
                active INTEGER DEFAULT 1
            );

            CREATE TABLE IF NOT EXISTS intelligence_reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                report_type TEXT NOT NULL,
                title TEXT NOT NULL,
                content TEXT DEFAULT '',
                generated_at TEXT NOT NULL,
                delivered_at TEXT DEFAULT '',
                delivered_to TEXT DEFAULT ''
            );

            CREATE TABLE IF NOT EXISTS outreach_tracking (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                recommendation_id INTEGER REFERENCES recommendations(id),
                company_name TEXT NOT NULL,
                contact_email TEXT DEFAULT '',
                sent_at TEXT NOT NULL,
                follow_up_at TEXT DEFAULT '',
                responded_at TEXT DEFAULT '',
                response_status TEXT DEFAULT 'awaiting',
                heat_status TEXT DEFAULT 'warm',
                notes TEXT DEFAULT '',
                updated_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_signals_status
                ON signals(status);
            CREATE INDEX IF NOT EXISTS idx_signals_type
                ON signals(signal_type);
            CREATE INDEX IF NOT EXISTS idx_recommendations_status
                ON recommendations(status);
        """)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    # --- Signals ---

    def add_signal(
        self,
        company_name: str,
        signal_type: str,
        title: str,
        source_url: str,
        source_name: str,
        description: str = "",
        relevance_score: float = 0.5,
        kp_service_match: str = "",
    ) -> dict[str, Any]:
        now = datetime.utcnow().isoformat()
        cursor = self._conn.execute(
            """INSERT INTO signals
            (company_name, signal_type, title, description,
             source_url, source_name, relevance_score,
             kp_service_match, detected_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                company_name,
                signal_type,
                title,
                description,
                source_url,
                source_name,
                relevance_score,
                kp_service_match,
                now,
            ),
        )
        self._conn.commit()
        return self._row_to_dict(
            self._conn.execute(
                "SELECT * FROM signals WHERE id = ?",
                (cursor.lastrowid,),
            ).fetchone()
        )

    def get_signals(
        self,
        status: str | None = None,
        signal_type: str | None = None,
        min_relevance: float | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        query = "SELECT * FROM signals WHERE 1=1"
        params: list[Any] = []
        if status:
            query += " AND status = ?"
            params.append(status)
        if signal_type:
            query += " AND signal_type = ?"
            params.append(signal_type)
        if min_relevance is not None:
            query += " AND relevance_score >= ?"
            params.append(min_relevance)
        query += " ORDER BY detected_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        rows = self._conn.execute(query, params).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def update_signal_status(
        self,
        signal_id: int,
        status: str,
        reviewed_by: str = "",
    ) -> bool:
        now = datetime.utcnow().isoformat()
        cursor = self._conn.execute(
            "UPDATE signals SET status = ?, reviewed_by = ?, reviewed_at = ? WHERE id = ?",
            (status, reviewed_by, now, signal_id),
        )
        self._conn.commit()
        return cursor.rowcount > 0

    def get_signal(self, signal_id: int) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT * FROM signals WHERE id = ?",
            (signal_id,),
        ).fetchone()
        return self._row_to_dict(row) if row else None

    # --- Recommendations ---

    def add_recommendation(
        self,
        company_name: str,
        signal_id: int | None = None,
        consultant_name: str = "",
        service_area: str = "",
        outreach_channel: str = "email",
        pitch_summary: str = "",
    ) -> dict[str, Any]:
        now = datetime.utcnow().isoformat()
        cursor = self._conn.execute(
            """INSERT INTO recommendations
            (signal_id, company_name, consultant_name,
             service_area, outreach_channel, pitch_summary,
             created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                signal_id,
                company_name,
                consultant_name,
                service_area,
                outreach_channel,
                pitch_summary,
                now,
            ),
        )
        self._conn.commit()
        return self._row_to_dict(
            self._conn.execute(
                "SELECT * FROM recommendations WHERE id = ?",
                (cursor.lastrowid,),
            ).fetchone()
        )

    def get_recommendations(
        self,
        status: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        query = "SELECT * FROM recommendations WHERE 1=1"
        params: list[Any] = []
        if status:
            query += " AND status = ?"
            params.append(status)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        rows = self._conn.execute(query, params).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def update_recommendation_status(
        self,
        rec_id: int,
        status: str,
        approved_by: str = "",
    ) -> bool:
        now = datetime.utcnow().isoformat()
        updates = "status = ?"
        params: list[Any] = [status]
        if status == "approved":
            updates += ", approved_by = ?, approved_at = ?"
            params.extend([approved_by, now])
        elif status == "sent":
            updates += ", sent_at = ?"
            params.append(now)
        params.append(rec_id)
        cursor = self._conn.execute(
            f"UPDATE recommendations SET {updates} WHERE id = ?",
            params,
        )
        self._conn.commit()
        return cursor.rowcount > 0

    # --- Single record fetches ---

    def get_recommendation(self, rec_id: int) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT * FROM recommendations WHERE id = ?", (rec_id,)
        ).fetchone()
        return self._row_to_dict(row) if row else None

    # --- Outreach Tracking ---

    def create_outreach_tracking(
        self,
        recommendation_id: int,
        company_name: str,
        contact_email: str = "",
    ) -> dict[str, Any]:
        now = datetime.utcnow().isoformat()
        cursor = self._conn.execute(
            """INSERT INTO outreach_tracking
            (recommendation_id, company_name, contact_email, sent_at, updated_at)
            VALUES (?, ?, ?, ?, ?)""",
            (recommendation_id, company_name, contact_email, now, now),
        )
        self._conn.commit()
        return self._row_to_dict(
            self._conn.execute(
                "SELECT * FROM outreach_tracking WHERE id = ?",
                (cursor.lastrowid,),
            ).fetchone()
        )

    def update_outreach_response(
        self,
        recommendation_id: int,
        responded: bool = True,
    ) -> bool:
        now = datetime.utcnow().isoformat()
        row = self._conn.execute(
            "SELECT * FROM outreach_tracking WHERE recommendation_id = ?",
            (recommendation_id,),
        ).fetchone()
        if not row:
            return False
        tracking = self._row_to_dict(row)
        if responded:
            # Calculate heat based on response time
            sent_at = tracking.get("sent_at", "")
            heat = "warm"
            if sent_at:
                try:
                    sent_dt = datetime.fromisoformat(sent_at)
                    days_elapsed = (datetime.utcnow() - sent_dt).days
                    if days_elapsed <= 3:
                        heat = "hot"
                except (ValueError, TypeError):
                    pass
            self._conn.execute(
                """UPDATE outreach_tracking
                SET responded_at = ?, response_status = 'responded',
                    heat_status = ?, updated_at = ?
                WHERE recommendation_id = ?""",
                (now, heat, now, recommendation_id),
            )
        self._conn.commit()
        return True

    def get_outreach_tracking(self, recommendation_id: int) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT * FROM outreach_tracking WHERE recommendation_id = ?",
            (recommendation_id,),
        ).fetchone()
        return self._row_to_dict(row) if row else None

    def get_all_outreach_tracking(
        self,
        heat_status: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        query = "SELECT * FROM outreach_tracking WHERE 1=1"
        params: list[Any] = []
        if heat_status:
            query += " AND heat_status = ?"
            params.append(heat_status)
        query += " ORDER BY sent_at DESC LIMIT ?"
        params.append(limit)
        rows = self._conn.execute(query, params).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def update_stale_outreach(self) -> int:
        """Mark outreach with no response after 7 days as cold."""
        cutoff = (datetime.utcnow() - timedelta(days=7)).isoformat()
        now = datetime.utcnow().isoformat()
        cursor = self._conn.execute(
            """UPDATE outreach_tracking
            SET heat_status = 'cold', response_status = 'no_response',
                updated_at = ?
            WHERE response_status = 'awaiting' AND sent_at < ?""",
            (now, cutoff),
        )
        self._conn.commit()
        return cursor.rowcount

    def mark_no_response(self, recommendation_id: int) -> bool:
        """Mark an outreach as no-response and set heat to cold."""
        now = datetime.utcnow().isoformat()
        cursor = self._conn.execute(
            """UPDATE outreach_tracking
            SET response_status = 'no_response', heat_status = 'cold',
                updated_at = ?
            WHERE recommendation_id = ?""",
            (now, recommendation_id),
        )
        self._conn.commit()
        return cursor.rowcount > 0

    # --- Consultants ---

    def add_consultant(
        self,
        name: str,
        email: str = "",
        specializations: list[str] | None = None,
        industries: list[str] | None = None,
        regions: list[str] | None = None,
    ) -> dict[str, Any]:
        cursor = self._conn.execute(
            """INSERT INTO consultants
            (name, email, specializations, industries, regions)
            VALUES (?, ?, ?, ?, ?)""",
            (
                name,
                email,
                json.dumps(specializations or []),
                json.dumps(industries or []),
                json.dumps(regions or []),
            ),
        )
        self._conn.commit()
        return self._row_to_dict(
            self._conn.execute(
                "SELECT * FROM consultants WHERE id = ?",
                (cursor.lastrowid,),
            ).fetchone()
        )

    def get_consultants(
        self,
        specialization: str | None = None,
        industry: str | None = None,
        active_only: bool = True,
    ) -> list[dict[str, Any]]:
        query = "SELECT * FROM consultants WHERE 1=1"
        params: list[Any] = []
        if active_only:
            query += " AND active = 1"
        rows = self._conn.execute(query, params).fetchall()
        results = []
        for r in rows:
            d = self._row_to_dict(r)
            d["specializations"] = json.loads(d.get("specializations", "[]"))
            d["industries"] = json.loads(d.get("industries", "[]"))
            d["regions"] = json.loads(d.get("regions", "[]"))
            if specialization and (specialization not in d["specializations"]):
                continue
            if industry and industry not in d["industries"]:
                continue
            results.append(d)
        return results

    # --- Reports ---

    def add_report(
        self,
        report_type: str,
        title: str,
        content: str,
        delivered_to: str = "",
    ) -> dict[str, Any]:
        now = datetime.utcnow().isoformat()
        cursor = self._conn.execute(
            """INSERT INTO intelligence_reports
            (report_type, title, content, generated_at,
             delivered_to)
            VALUES (?, ?, ?, ?, ?)""",
            (report_type, title, content, now, delivered_to),
        )
        self._conn.commit()
        return self._row_to_dict(
            self._conn.execute(
                "SELECT * FROM intelligence_reports WHERE id = ?",
                (cursor.lastrowid,),
            ).fetchone()
        )

    def get_reports(
        self,
        report_type: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        query = "SELECT * FROM intelligence_reports WHERE 1=1"
        params: list[Any] = []
        if report_type:
            query += " AND report_type = ?"
            params.append(report_type)
        query += " ORDER BY generated_at DESC LIMIT ?"
        params.append(limit)
        rows = self._conn.execute(query, params).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def get_report(self, report_id: int) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT * FROM intelligence_reports WHERE id = ?",
            (report_id,),
        ).fetchone()
        return self._row_to_dict(row) if row else None

    # --- Stats ---

    def get_signal_stats(self) -> dict[str, Any]:
        stats: dict[str, Any] = {}
        rows = self._conn.execute(
            "SELECT status, COUNT(*) as cnt FROM signals GROUP BY status"
        ).fetchall()
        stats["by_status"] = {r["status"]: r["cnt"] for r in rows}
        rows = self._conn.execute(
            "SELECT signal_type, COUNT(*) as cnt FROM signals GROUP BY signal_type"
        ).fetchall()
        stats["by_type"] = {r["signal_type"]: r["cnt"] for r in rows}
        total = self._conn.execute("SELECT COUNT(*) FROM signals").fetchone()
        stats["total"] = total[0] if total else 0
        return stats

    @staticmethod
    def _row_to_dict(
        row: sqlite3.Row | None,
    ) -> dict[str, Any]:
        if row is None:
            return {}
        return dict(row)
