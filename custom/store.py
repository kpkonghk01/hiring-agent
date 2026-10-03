"""One SQLite file holding Extracted Resumes and Evaluations (docs/adr/0002).

Payloads are stored as JSON text and rehydrated by the caller, so schema changes
live in the Pydantic models, not in migrations.
"""

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from models import JSONResume

_SCHEMA = """
CREATE TABLE IF NOT EXISTS extracted_resumes (
    resume_hash  TEXT PRIMARY KEY,
    resume_json  TEXT NOT NULL,
    extracted_by TEXT NOT NULL,   -- provider:model, for reference only
    extracted_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS evaluations (
    resume_hash     TEXT NOT NULL,
    role_hash       TEXT NOT NULL,
    judge           TEXT NOT NULL,  -- provider:model
    role_name       TEXT NOT NULL,
    evaluation_json TEXT NOT NULL,
    github_json     TEXT,           -- GitHub Enrichment snapshot the judge saw; NULL if none
    evaluated_at    TEXT NOT NULL,
    PRIMARY KEY (resume_hash, role_hash, judge)
);
"""


@dataclass(frozen=True)
class EvaluationKey:
    resume_hash: str
    role_hash: str
    judge: str


@dataclass(frozen=True)
class StoredEvaluation:
    evaluation: dict
    github: Optional[dict]


class Store:
    def __init__(self, path: str):
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path)
        self._conn.executescript(_SCHEMA)

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "Store":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def get_extracted(self, resume_hash: str) -> Optional[JSONResume]:
        row = self._conn.execute(
            "SELECT resume_json FROM extracted_resumes WHERE resume_hash = ?", (resume_hash,)
        ).fetchone()
        return JSONResume.model_validate_json(row[0]) if row else None

    def put_extracted(self, resume_hash: str, resume: JSONResume, extracted_by: str) -> None:
        """Store (or replace) a Resume's Extracted Resume.

        Replacing it invalidates every Evaluation of that Resume, since they were
        judged on the old extraction — both happen in one transaction.
        """
        with self._conn:
            self._conn.execute("DELETE FROM evaluations WHERE resume_hash = ?", (resume_hash,))
            self._conn.execute(
                "INSERT OR REPLACE INTO extracted_resumes VALUES (?, ?, ?, ?)",
                (resume_hash, resume.model_dump_json(), extracted_by, _now()),
            )

    def get_evaluation(self, key: EvaluationKey) -> Optional[StoredEvaluation]:
        row = self._conn.execute(
            "SELECT evaluation_json, github_json FROM evaluations "
            "WHERE resume_hash = ? AND role_hash = ? AND judge = ?",
            (key.resume_hash, key.role_hash, key.judge),
        ).fetchone()
        if not row:
            return None
        return StoredEvaluation(
            evaluation=json.loads(row[0]), github=json.loads(row[1]) if row[1] else None
        )

    def put_evaluation(
        self, key: EvaluationKey, role_name: str, evaluation: dict, github: Optional[dict]
    ) -> None:
        with self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO evaluations VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    key.resume_hash,
                    key.role_hash,
                    key.judge,
                    role_name,
                    json.dumps(evaluation, ensure_ascii=False),
                    json.dumps(github, ensure_ascii=False) if github is not None else None,
                    _now(),
                ),
            )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
