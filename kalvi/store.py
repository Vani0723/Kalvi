"""Local lecture store (SQLite). Nothing ever leaves the laptop."""

from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS lectures (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    title       TEXT NOT NULL,
    started_at  REAL NOT NULL,
    ended_at    REAL,
    language    TEXT,
    device      TEXT,
    audio_path  TEXT,
    source      TEXT DEFAULT 'live'
);
CREATE TABLE IF NOT EXISTS segments (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    lecture_id  INTEGER NOT NULL REFERENCES lectures(id) ON DELETE CASCADE,
    start_s     REAL NOT NULL,
    end_s       REAL NOT NULL,
    text        TEXT NOT NULL,
    language    TEXT,
    latency_ms  REAL
);
CREATE INDEX IF NOT EXISTS idx_segments_lecture ON segments(lecture_id, start_s);
"""


class Store:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.executescript(SCHEMA)
        self._lock = threading.Lock()

    def _exec(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        with self._lock:
            cur = self._conn.execute(sql, params)
            self._conn.commit()
            return cur

    def _query(self, sql: str, params: tuple = ()) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(r) for r in self._conn.execute(sql, params).fetchall()]

    # Lectures -----------------------------------------------------------
    def create_lecture(self, title: str, language: str, device: str, source: str = "live") -> int:
        cur = self._exec(
            "INSERT INTO lectures (title, started_at, language, device, source) VALUES (?, ?, ?, ?, ?)",
            (title, time.time(), language, device, source),
        )
        return int(cur.lastrowid)

    def set_audio_path(self, lecture_id: int, path: str) -> None:
        self._exec("UPDATE lectures SET audio_path = ? WHERE id = ?", (path, lecture_id))

    def end_lecture(self, lecture_id: int) -> None:
        self._exec("UPDATE lectures SET ended_at = ? WHERE id = ?", (time.time(), lecture_id))

    def list_lectures(self) -> list[dict[str, Any]]:
        return self._query(
            """SELECT l.*, COUNT(s.id) AS segment_count, COALESCE(MAX(s.end_s), 0) AS duration_s
               FROM lectures l LEFT JOIN segments s ON s.lecture_id = l.id
               GROUP BY l.id ORDER BY l.started_at DESC"""
        )

    def get_lecture(self, lecture_id: int) -> dict[str, Any] | None:
        rows = self._query("SELECT * FROM lectures WHERE id = ?", (lecture_id,))
        if not rows:
            return None
        lecture = rows[0]
        lecture["segments"] = self.segments(lecture_id)
        return lecture

    def delete_lecture(self, lecture_id: int) -> dict[str, Any] | None:
        lecture = self.get_lecture(lecture_id)
        if lecture:
            self._exec("DELETE FROM lectures WHERE id = ?", (lecture_id,))
        return lecture

    # Segments -----------------------------------------------------------
    def add_segment(self, lecture_id: int, start_s: float, end_s: float, text: str, language: str | None, latency_ms: float) -> dict[str, Any]:
        cur = self._exec(
            "INSERT INTO segments (lecture_id, start_s, end_s, text, language, latency_ms) VALUES (?, ?, ?, ?, ?, ?)",
            (lecture_id, start_s, end_s, text, language, latency_ms),
        )
        return {
            "id": int(cur.lastrowid), "lecture_id": lecture_id, "start_s": start_s, "end_s": end_s,
            "text": text, "language": language, "latency_ms": latency_ms,
        }

    def segments(self, lecture_id: int) -> list[dict[str, Any]]:
        return self._query("SELECT * FROM segments WHERE lecture_id = ? ORDER BY start_s", (lecture_id,))

    def close(self) -> None:
        with self._lock:
            self._conn.close()
