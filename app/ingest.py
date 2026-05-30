"""Document loading and chunking.

Each file -> a document with a stable ``document_id`` (from its filename).
Documents are split on Markdown headings, then windowed into overlapping
word-based chunks, each with a ``chunk_id`` unique within its document.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List

from .config import settings


@dataclass
class Chunk:
    document_id: str
    chunk_id: str
    text: str
    source_path: str


def _document_id_from_path(path: Path) -> str:
    """Map a filename to a readable, stable document id (e.g. ``doc_hfpef_overview``)."""
    return f"doc_{path.stem}"


def _split_into_sections(text: str) -> List[str]:
    """Split markdown text on headings so chunks stay topically coherent."""
    lines = text.splitlines()
    sections: List[str] = []
    current: List[str] = []
    for line in lines:
        if re.match(r"^#{1,6}\s", line) and current:
            sections.append("\n".join(current).strip())
            current = [line]
        else:
            current.append(line)
    if current:
        sections.append("\n".join(current).strip())
    return [s for s in sections if s]


def _chunk_section(
    section: str, size_words: int, overlap_words: int
) -> List[str]:
    """Window a section into overlapping word chunks."""
    words = section.split()
    if len(words) <= size_words:
        return [section.strip()] if section.strip() else []

    chunks: List[str] = []
    step = max(1, size_words - overlap_words)
    for start in range(0, len(words), step):
        window = words[start : start + size_words]
        if not window:
            break
        chunks.append(" ".join(window))
        if start + size_words >= len(words):
            break
    return chunks


def load_documents(data_dir: Path | None = None) -> List[Chunk]:
    """Load every ``.txt``/``.md`` file under ``data_dir`` into a flat chunk list."""
    from .config import DATA_DIR

    directory = data_dir or DATA_DIR
    paths = sorted(
        p for p in directory.iterdir() if p.suffix.lower() in {".txt", ".md"}
    )
    if not paths:
        raise FileNotFoundError(f"No .txt or .md documents found in {directory}")

    chunks: List[Chunk] = []
    for path in paths:
        document_id = _document_id_from_path(path)
        raw = path.read_text(encoding="utf-8")
        sections = _split_into_sections(raw)

        counter = 0
        for section in sections:
            for piece in _chunk_section(
                section,
                settings.chunk_size_words,
                settings.chunk_overlap_words,
            ):
                text = piece.strip()
                if not text:
                    continue
                chunks.append(
                    Chunk(
                        document_id=document_id,
                        chunk_id=f"chunk_{counter}",
                        text=text,
                        source_path=str(path),
                    )
                )
                counter += 1
    return chunks
