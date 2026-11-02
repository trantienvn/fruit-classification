"""Lưu metadata dự đoán trong SQLite, không lưu ảnh người dùng."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator

from app.web.labels import localized_class_name


class HistoryService:
    """Kho lịch sử dự đoán cục bộ."""

    def __init__(self, database_path: Path | str) -> None:
        self.database_path = Path(database_path)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=10)
        connection.row_factory = sqlite3.Row
        return connection

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def initialize(self) -> None:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connection() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT NOT NULL,
                    class_id INTEGER NOT NULL,
                    class_name TEXT NOT NULL,
                    fruit TEXT NOT NULL,
                    ripeness TEXT NOT NULL,
                    confidence REAL,
                    processing_time_ms REAL NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_predictions_created_at "
                "ON predictions(created_at DESC)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_predictions_class_name "
                "ON predictions(class_name)"
            )

    @staticmethod
    def _serialize(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "filename": row["filename"],
            "predicted_class": row["class_id"],
            "class_name": row["class_name"],
            "class_name_vi": localized_class_name(row["class_name"]),
            "fruit_type": row["fruit"],
            "ripeness": row["ripeness"],
            "confidence": row["confidence"],
            "processing_time_ms": row["processing_time_ms"],
            "created_at": row["created_at"],
        }

    def add(self, prediction: dict[str, Any]) -> dict[str, Any]:
        created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with self._connection() as connection:
            cursor = connection.execute(
                """
                INSERT INTO predictions (
                    filename, class_id, class_name, fruit, ripeness,
                    confidence, processing_time_ms, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    prediction["filename"],
                    prediction["predicted_class"],
                    prediction["class_name"],
                    prediction["fruit_type"],
                    prediction["ripeness"],
                    prediction.get("confidence"),
                    prediction["processing_time_ms"],
                    created_at,
                ),
            )
            row = connection.execute(
                "SELECT * FROM predictions WHERE id = ?", (cursor.lastrowid,)
            ).fetchone()
        return self._serialize(row)

    def list(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
        query: str | None = None,
        class_name: str | None = None,
    ) -> tuple[list[dict[str, Any]], int]:
        clauses: list[str] = []
        parameters: list[Any] = []
        if query:
            clauses.append("lower(filename) LIKE ?")
            parameters.append(f"%{query.strip().lower()}%")
        if class_name:
            clauses.append("class_name = ?")
            parameters.append(class_name)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""

        with self._connection() as connection:
            total = connection.execute(
                f"SELECT COUNT(*) FROM predictions {where}", parameters
            ).fetchone()[0]
            rows = connection.execute(
                f"SELECT * FROM predictions {where} "
                "ORDER BY id DESC LIMIT ? OFFSET ?",
                (*parameters, limit, offset),
            ).fetchall()
        return [self._serialize(row) for row in rows], total

    def clear(self) -> int:
        with self._connection() as connection:
            cursor = connection.execute("DELETE FROM predictions")
        return cursor.rowcount

    def ping(self) -> bool:
        with self._connection() as connection:
            connection.execute("SELECT 1").fetchone()
        return True

    def statistics(self, class_names: list[str]) -> dict[str, Any]:
        fruits = ("Apple", "Banana", "Mango")
        ripeness_names = ("Unripe", "Ripe", "Overripe")
        by_class = {name: 0 for name in class_names}
        by_fruit = {name: 0 for name in fruits}
        by_ripeness = {name: 0 for name in ripeness_names}

        with self._connection() as connection:
            total, latest_at = connection.execute(
                "SELECT COUNT(*), MAX(created_at) FROM predictions"
            ).fetchone()
            rows = connection.execute(
                """
                SELECT class_name, fruit, ripeness, COUNT(*) AS count
                FROM predictions
                GROUP BY class_name, fruit, ripeness
                """
            ).fetchall()
            for row in rows:
                by_class[row["class_name"]] = row["count"]
                by_fruit[row["fruit"]] = row["count"]
                by_ripeness[row["ripeness"]] = row["count"]

            today = datetime.now(timezone.utc).date()
            start = today - timedelta(days=6)
            daily_rows = connection.execute(
                """
                SELECT substr(created_at, 1, 10) AS day, COUNT(*) AS count
                FROM predictions
                WHERE created_at >= ?
                GROUP BY day
                """,
                (datetime.combine(start, datetime.min.time(), timezone.utc).isoformat(),),
            ).fetchall()
        daily_counts = {row["day"]: row["count"] for row in daily_rows}
        by_day = [
            {"date": (start + timedelta(days=offset)).isoformat(),
             "count": daily_counts.get((start + timedelta(days=offset)).isoformat(), 0)}
            for offset in range(7)
        ]

        return {
            "total_predictions": total,
            "unique_classes": sum(1 for count in by_class.values() if count),
            "latest_at": latest_at,
            "by_class": [
                {"class_name": name, "count": count}
                for name, count in by_class.items()
            ],
            "by_fruit": [
                {"fruit": name, "count": count} for name, count in by_fruit.items()
            ],
            "by_ripeness": [
                {"ripeness": name, "count": count}
                for name, count in by_ripeness.items()
            ],
            "by_day": by_day,
        }
