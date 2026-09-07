"""Persistent store for typed adaptive lessons."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from .models import TypedLesson

SCHEMA = """
CREATE TABLE IF NOT EXISTS typed_lessons (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    scope TEXT NOT NULL,
    enabled INTEGER NOT NULL,
    confidence REAL NOT NULL,
    source_session_id TEXT,
    captured_at TEXT NOT NULL,
    expires_at TEXT,
    decay_half_life_days INTEGER,
    last_applied_at TEXT,
    match_json TEXT NOT NULL,
    prefer_json TEXT NOT NULL,
    limit_usd_per_session REAL,
    on_breach TEXT,
    metadata_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_typed_lessons_kind ON typed_lessons(kind);
CREATE INDEX IF NOT EXISTS idx_typed_lessons_scope ON typed_lessons(scope);
"""


class TypedLessonStore:
    def __init__(self, root: Path | str, *, create: bool = True) -> None:
        self.root = Path(root).resolve()
        self.db_path = self.root / "lessons.sqlite"
        self._create = create
        if create or self.db_path.exists():
            self.root.mkdir(parents=True, exist_ok=True)
            with self._connect() as conn:
                conn.executescript(SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        return conn

    def upsert_lesson(self, lesson: TypedLesson) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO typed_lessons (
                    id, kind, scope, enabled, confidence, source_session_id,
                    captured_at, expires_at, decay_half_life_days, last_applied_at,
                    match_json, prefer_json, limit_usd_per_session, on_breach, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    kind = excluded.kind,
                    scope = excluded.scope,
                    enabled = excluded.enabled,
                    confidence = excluded.confidence,
                    source_session_id = excluded.source_session_id,
                    captured_at = excluded.captured_at,
                    expires_at = excluded.expires_at,
                    decay_half_life_days = excluded.decay_half_life_days,
                    last_applied_at = excluded.last_applied_at,
                    match_json = excluded.match_json,
                    prefer_json = excluded.prefer_json,
                    limit_usd_per_session = excluded.limit_usd_per_session,
                    on_breach = excluded.on_breach,
                    metadata_json = excluded.metadata_json
                """,
                (
                    lesson.id,
                    lesson.kind,
                    lesson.scope,
                    1 if lesson.enabled else 0,
                    lesson.confidence,
                    lesson.source_session_id,
                    lesson.captured_at.isoformat(),
                    lesson.expires_at.isoformat() if lesson.expires_at else None,
                    lesson.decay_half_life_days,
                    lesson.last_applied_at.isoformat() if lesson.last_applied_at else None,
                    json.dumps(lesson.match, ensure_ascii=False, sort_keys=True),
                    json.dumps(lesson.prefer, ensure_ascii=False, sort_keys=True),
                    lesson.limit_usd_per_session,
                    lesson.on_breach,
                    json.dumps(lesson.metadata, ensure_ascii=False, sort_keys=True),
                ),
            )

    def get_lesson(self, lesson_id: str) -> TypedLesson | None:
        if not self.db_path.exists():
            return None
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM typed_lessons WHERE id = ?", (lesson_id,)).fetchone()
        return None if row is None else self._row_to_lesson(row)

    def list_lessons(self) -> list[TypedLesson]:
        if not self.db_path.exists():
            return []
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM typed_lessons ORDER BY captured_at DESC").fetchall()
        return [self._row_to_lesson(row) for row in rows]

    def list_active_lessons(self, *, scope: str = "user", at: datetime | None = None) -> list[TypedLesson]:
        return [lesson for lesson in self.list_lessons() if lesson.is_active_at(at, scope=scope)]  # type: ignore[arg-type]

    def set_enabled(self, lesson_id: str, enabled: bool) -> TypedLesson:
        lesson = self.get_lesson(lesson_id)
        if lesson is None:
            raise ValueError(f"typed lesson not found: {lesson_id}")
        updated = lesson.model_copy(update={"enabled": enabled})
        self.upsert_lesson(updated)
        return updated

    def mark_applied(self, lesson_id: str, *, at: datetime | None = None) -> TypedLesson:
        lesson = self.get_lesson(lesson_id)
        if lesson is None:
            raise ValueError(f"typed lesson not found: {lesson_id}")
        updated = lesson.model_copy(update={"last_applied_at": at or datetime.now(UTC)})
        self.upsert_lesson(updated)
        return updated

    def _row_to_lesson(self, row: sqlite3.Row) -> TypedLesson:
        return TypedLesson(
            id=row["id"],
            kind=row["kind"],
            scope=row["scope"],
            enabled=bool(row["enabled"]),
            confidence=float(row["confidence"]),
            source_session_id=row["source_session_id"],
            captured_at=datetime.fromisoformat(row["captured_at"]),
            expires_at=datetime.fromisoformat(row["expires_at"]) if row["expires_at"] else None,
            decay_half_life_days=int(row["decay_half_life_days"]) if row["decay_half_life_days"] is not None else None,
            last_applied_at=datetime.fromisoformat(row["last_applied_at"]) if row["last_applied_at"] else None,
            match=json.loads(row["match_json"] or "{}"),
            prefer=json.loads(row["prefer_json"] or "{}"),
            limit_usd_per_session=(
                float(row["limit_usd_per_session"]) if row["limit_usd_per_session"] is not None else None
            ),
            on_breach=row["on_breach"],
            metadata=json.loads(row["metadata_json"] or "{}"),
        )


__all__ = ["TypedLessonStore"]
