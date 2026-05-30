"""RAG pipeline orchestration.

Built once at startup (load + chunk + embed + FAISS index). Each `answer()`
call runs: guardrail -> retrieve -> sufficiency gate -> generate (or refuse) ->
faithfulness check -> log -> respond.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from . import guardrails
from .config import settings
from .embeddings import EmbeddingModel
from .faithfulness import assess_faithfulness
from .generator import REFUSAL_MESSAGE, generate_answer
from .ingest import load_documents
from .logging_store import ResearchLogger, ResearchLogRecord, now_iso
from .sufficiency import evaluate_sufficiency
from .vectorstore import FaissVectorStore, RetrievedChunk


@dataclass
class PipelineResponse:
    answer: str
    evidence_used: List[dict]
    evidence_sufficient: bool
    guardrail_triggered: bool
    groundedness: Optional[dict] = None


class RagPipeline:
    def __init__(
        self,
        data_dir: Optional[Path] = None,
        logger: Optional[ResearchLogger] = None,
    ) -> None:
        self.chunks = load_documents(data_dir)
        self.embedder = EmbeddingModel(settings.embedding_model_name)
        embeddings = self.embedder.embed([c.text for c in self.chunks])
        self.store = FaissVectorStore(self.chunks, embeddings)
        self.logger = logger or ResearchLogger()

    # -- helpers -----------------------------------------------------------
    @staticmethod
    def _evidence_payload(results: List[RetrievedChunk]) -> List[dict]:
        return [
            {
                "document_id": r.chunk.document_id,
                "chunk_id": r.chunk.chunk_id,
                "similarity_score": round(r.similarity_score, 4),
            }
            for r in results
        ]

    def _retrieve(self, question: str) -> List[RetrievedChunk]:
        query_vec = self.embedder.embed([question])[0]
        return self.store.search(query_vec, settings.top_k)

    # -- main entrypoint ---------------------------------------------------
    def answer(self, question: str) -> PipelineResponse:
        timestamp = now_iso()

        # 1) Safety guardrail first: never generate advice for an apparent emergency.
        guardrail = guardrails.check_high_risk(question)
        if guardrail.triggered:
            response = PipelineResponse(
                answer=guardrail.message or guardrails.ESCALATION_MESSAGE,
                evidence_used=[],
                evidence_sufficient=False,
                guardrail_triggered=True,
            )
            self.logger.log(
                ResearchLogRecord(
                    timestamp=timestamp,
                    question=question,
                    retrieved=[],
                    similarity_scores=[],
                    evidence_sufficient=False,
                    sufficiency_reason="Skipped: safety guardrail triggered.",
                    guardrail_triggered=True,
                    guardrail_categories=guardrail.matched_categories,
                    final_answer=response.answer,
                    embedding_backend=self.embedder.backend_name,
                    model_name=None,
                    prompt_summary="No generation: emergency escalation message returned.",
                )
            )
            return response

        # 2) Retrieve relevant chunks.
        results = self._retrieve(question)
        evidence_payload = self._evidence_payload(results)
        scores = [r.similarity_score for r in results]

        # 3) Evidence sufficiency gate.
        decision = evaluate_sufficiency(results)

        if not decision.sufficient:
            response = PipelineResponse(
                answer=REFUSAL_MESSAGE,
                # Surface what was retrieved for transparency.
                evidence_used=evidence_payload,
                evidence_sufficient=False,
                guardrail_triggered=False,
            )
            self.logger.log(
                ResearchLogRecord(
                    timestamp=timestamp,
                    question=question,
                    retrieved=evidence_payload,
                    similarity_scores=scores,
                    evidence_sufficient=False,
                    sufficiency_reason=decision.reason,
                    guardrail_triggered=False,
                    guardrail_categories=[],
                    final_answer=response.answer,
                    embedding_backend=self.embedder.backend_name,
                    model_name=None,
                    prompt_summary="No generation: evidence insufficient, refused.",
                )
            )
            return response

        # 4) Generate a grounded, patient-friendly answer.
        generation = generate_answer(question, results)

        # 5) Faithfulness check: is the answer supported by the cited evidence?
        evidence_texts = [r.chunk.text for r in results]
        faith = assess_faithfulness(generation.answer, evidence_texts, self.embedder)
        groundedness = {
            "score": faith.groundedness_score,
            "supported_fraction": faith.supported_fraction,
            "scored_sentences": faith.scored_sentences,
        }

        response = PipelineResponse(
            answer=generation.answer,
            evidence_used=evidence_payload,
            evidence_sufficient=True,
            guardrail_triggered=False,
            groundedness=groundedness,
        )

        extra = dict(generation.metadata)
        extra["groundedness"] = groundedness
        if faith.unsupported_sentences:
            extra["unsupported_sentences"] = faith.unsupported_sentences

        self.logger.log(
            ResearchLogRecord(
                timestamp=timestamp,
                question=question,
                retrieved=evidence_payload,
                similarity_scores=scores,
                evidence_sufficient=True,
                sufficiency_reason=decision.reason,
                guardrail_triggered=False,
                guardrail_categories=[],
                final_answer=response.answer,
                embedding_backend=self.embedder.backend_name,
                model_name=generation.model_name,
                prompt_summary=generation.prompt_summary,
                prompt_template=generation.prompt_template or None,
                extra=extra,
            )
        )
        return response
