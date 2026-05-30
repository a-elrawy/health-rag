"""Faithfulness / groundedness check (hallucination guard).

A lightweight, offline proxy for entailment: each answer sentence's max cosine
similarity to the cited chunks must clear ``support_threshold``. Reports the
mean support (``groundedness_score``), the supported fraction, and any
unsupported sentences. A production system would swap in a trained entailment
model or LLM judge behind this same interface.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List

import numpy as np

# Boilerplate that is not a factual claim and should not be scored.
_BOILERPLATE_PATTERNS = [
    r"this is general information",
    r"general educational information",
    r"not (a )?(substitute|personal medical advice)",
    r"^\(sources?:",
    r"please ask your doctor",
]


def _is_boilerplate(sentence: str) -> bool:
    low = sentence.lower()
    return any(re.search(p, low) for p in _BOILERPLATE_PATTERNS)


def _split_sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text)
    return [p.strip() for p in parts if p.strip()]


@dataclass
class FaithfulnessReport:
    groundedness_score: float
    supported_fraction: float
    scored_sentences: int
    unsupported_sentences: List[str] = field(default_factory=list)
    note: str = ""


def assess_faithfulness(
    answer: str,
    evidence_texts: List[str],
    embedder,
    support_threshold: float = 0.5,
) -> FaithfulnessReport:
    """Score how well ``answer`` is supported by ``evidence_texts``."""
    if not evidence_texts:
        return FaithfulnessReport(
            groundedness_score=0.0,
            supported_fraction=0.0,
            scored_sentences=0,
            note="No evidence to check against.",
        )

    sentences = [s for s in _split_sentences(answer) if not _is_boilerplate(s)]
    if not sentences:
        return FaithfulnessReport(
            groundedness_score=1.0,
            supported_fraction=1.0,
            scored_sentences=0,
            note="No factual sentences to score (boilerplate only).",
        )

    sent_vecs = embedder.embed(sentences)
    evid_vecs = embedder.embed(evidence_texts)
    # Vectors are L2-normalized by the backends, so dot product == cosine.
    sims = sent_vecs @ evid_vecs.T  # (n_sentences, n_evidence)
    best_per_sentence = sims.max(axis=1)

    supported = best_per_sentence >= support_threshold
    unsupported = [
        sentences[i] for i in range(len(sentences)) if not supported[i]
    ]

    return FaithfulnessReport(
        groundedness_score=round(float(np.mean(best_per_sentence)), 4),
        supported_fraction=round(float(np.mean(supported)), 4),
        scored_sentences=len(sentences),
        unsupported_sentences=unsupported,
    )
