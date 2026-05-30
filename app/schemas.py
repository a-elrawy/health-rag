"""Pydantic models defining the public API contract for POST /ask."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        description="The patient's question in natural language.",
        examples=["What should I ask my doctor about HFpEF treatment options?"],
    )


class EvidenceItem(BaseModel):
    document_id: str
    chunk_id: str
    similarity_score: float = Field(
        ..., description="Cosine similarity between the query and the chunk, in [0, 1]."
    )


class Groundedness(BaseModel):
    """Post-hoc faithfulness check of the answer against its cited evidence."""

    score: float = Field(
        ..., description="Mean per-sentence support similarity in [0, 1]."
    )
    supported_fraction: float = Field(
        ..., description="Fraction of answer sentences supported by the evidence."
    )
    scored_sentences: int


class AskResponse(BaseModel):
    answer: str
    evidence_used: List[EvidenceItem]
    evidence_sufficient: bool
    guardrail_triggered: bool
    # Additive extension beyond the required schema: a transparency signal of how
    # well the generated answer is grounded in the cited evidence. Null when no
    # answer was generated (guardrail/refusal).
    groundedness: Optional[Groundedness] = None
