"""Evidence sufficiency gate: refuse rather than guess when support is weak.

Evidence is sufficient only if BOTH hold:
1. Relevance floor: best chunk >= ``min_top_similarity``.
2. Corroboration: >= ``min_supporting_chunks`` chunks >= ``min_support_similarity``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .config import settings
from .vectorstore import RetrievedChunk


@dataclass
class SufficiencyDecision:
    sufficient: bool
    top_similarity: float
    supporting_count: int
    reason: str


def evaluate_sufficiency(results: List[RetrievedChunk]) -> SufficiencyDecision:
    if not results:
        return SufficiencyDecision(
            sufficient=False,
            top_similarity=0.0,
            supporting_count=0,
            reason="No chunks were retrieved.",
        )

    top_similarity = results[0].similarity_score
    supporting_count = sum(
        1 for r in results if r.similarity_score >= settings.min_support_similarity
    )

    passes_floor = top_similarity >= settings.min_top_similarity
    passes_support = supporting_count >= settings.min_supporting_chunks
    sufficient = passes_floor and passes_support

    if sufficient:
        reason = (
            f"Top similarity {top_similarity:.3f} >= {settings.min_top_similarity} "
            f"and {supporting_count} supporting chunk(s) "
            f">= {settings.min_support_similarity}."
        )
    elif not passes_floor:
        reason = (
            f"Best match {top_similarity:.3f} is below the relevance floor "
            f"{settings.min_top_similarity}; question likely outside the knowledge base."
        )
    else:
        reason = (
            f"Only {supporting_count} chunk(s) above {settings.min_support_similarity}; "
            f"need at least {settings.min_supporting_chunks}."
        )

    return SufficiencyDecision(
        sufficient=sufficient,
        top_similarity=top_similarity,
        supporting_count=supporting_count,
        reason=reason,
    )
