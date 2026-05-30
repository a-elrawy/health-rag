"""Research logging to JSONL (one object per line) and SQLite (queryable table).

Each record captures everything needed to audit a decision: timestamp,
question, retrieved doc/chunk ids + scores, sufficiency decision, guardrail
decision, final answer, embedding backend, and the LLM model/prompt when used.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import JSONL_LOG_PATH, LOG_DIR, SQLITE_LOG_PATH


@dataclass
class ResearchLogRecord:
    timestamp: str
    question: str
    retrieved: List[Dict[str, Any]]  # [{document_id, chunk_id, similarity_score}]
    similarity_scores: List[float]
    evidence_sufficient: bool
    sufficiency_reason: str
    guardrail_triggered: bool
    guardrail_categories: List[str]
    final_answer: str
    embedding_backend: str
    model_name: Optional[str] = None
    prompt_summary: Optional[str] = None
    prompt_template: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)


_SCHEMA = """
CREATE TABLE IF NOT EXISTS research_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    question TEXT NOT NULL,
    retrieved_json TEXT NOT NULL,
    similarity_scores_json TEXT NOT NULL,
    evidence_sufficient INTEGER NOT NULL,
    sufficiency_reason TEXT,
    guardrail_triggered INTEGER NOT NULL,
    guardrail_categories_json TEXT,
    final_answer TEXT,
    embedding_backend TEXT,
    model_name TEXT,
    prompt_summary TEXT,
    prompt_template TEXT,
    extra_json TEXT
);
"""


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class ResearchLogger:
    def __init__(
        self,
        jsonl_path: Path = JSONL_LOG_PATH,
        sqlite_path: Path = SQLITE_LOG_PATH,
    ) -> None:
        self.jsonl_path = jsonl_path
        self.sqlite_path = sqlite_path
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.sqlite_path) as conn:
            conn.execute(_SCHEMA)
            conn.commit()

    def log(self, record: ResearchLogRecord) -> None:
        self._write_jsonl(record)
        self._write_sqlite(record)

    def _write_jsonl(self, record: ResearchLogRecord) -> None:
        with self.jsonl_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")

    def _write_sqlite(self, record: ResearchLogRecord) -> None:
        with sqlite3.connect(self.sqlite_path) as conn:
            conn.execute(
                """
                INSERT INTO research_log (
                    timestamp, question, retrieved_json, similarity_scores_json,
                    evidence_sufficient, sufficiency_reason, guardrail_triggered,
                    guardrail_categories_json, final_answer, embedding_backend,
                    model_name, prompt_summary, prompt_template, extra_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.timestamp,
                    record.question,
                    json.dumps(record.retrieved, ensure_ascii=False),
                    json.dumps(record.similarity_scores),
                    int(record.evidence_sufficient),
                    record.sufficiency_reason,
                    int(record.guardrail_triggered),
                    json.dumps(record.guardrail_categories),
                    record.final_answer,
                    record.embedding_backend,
                    record.model_name,
                    record.prompt_summary,
                    record.prompt_template,
                    json.dumps(record.extra, ensure_ascii=False),
                ),
            )
            conn.commit()
