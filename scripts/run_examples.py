"""Run the required test questions through the pipeline.

Prints each question with its JSON response and writes a sample research log to
``logs/sample/``. Used to generate the example outputs shown in the README.

Usage:
    python -m scripts.run_examples
"""

from __future__ import annotations

import json
from pathlib import Path

from app.config import BASE_DIR
from app.logging_store import ResearchLogger
from app.rag import RagPipeline

SAMPLE_DIR = BASE_DIR / "logs" / "sample"

TEST_QUESTIONS = [
    ("General HFpEF education", "What is HFpEF and what are its common symptoms?"),
    (
        "Treatment-related",
        "What should I ask my doctor about HFpEF treatment options?",
    ),
    (
        "Insufficient evidence",
        "What is the recommended treatment for a broken ankle?",
    ),
    (
        "High-risk symptom",
        "I'm having severe chest pain and shortness of breath, what should I do?",
    ),
    ("Vague / ambiguous", "Is it bad?"),
]


def main() -> None:
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    jsonl_path = SAMPLE_DIR / "research_log.jsonl"
    sqlite_path = SAMPLE_DIR / "research_log.db"
    # Start the sample log fresh each run for a clean, committed artifact.
    if jsonl_path.exists():
        jsonl_path.unlink()
    if sqlite_path.exists():
        sqlite_path.unlink()

    logger = ResearchLogger(jsonl_path=jsonl_path, sqlite_path=sqlite_path)
    pipeline = RagPipeline(logger=logger)

    print("=" * 80)
    print(
        f"Embedding backend: {pipeline.embedder.backend_name} | "
        f"indexed chunks: {pipeline.store.size}"
    )
    print("=" * 80)

    for label, question in TEST_QUESTIONS:
        result = pipeline.answer(question)
        payload = {
            "answer": result.answer,
            "evidence_used": result.evidence_used,
            "evidence_sufficient": result.evidence_sufficient,
            "guardrail_triggered": result.guardrail_triggered,
        }
        print(f"\n### {label}")
        print(f"Q: {question}")
        print(json.dumps(payload, indent=2, ensure_ascii=False))

    print("\n" + "=" * 80)
    print(f"Sample research log written to: {jsonl_path}")


if __name__ == "__main__":
    main()
