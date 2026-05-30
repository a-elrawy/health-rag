"""FastAPI application exposing the document-grounded health assistant.

Run with:
    uvicorn app.main:app --reload

The pipeline (documents, embeddings, FAISS index) is built once at startup and
reused across requests.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from .config import settings
from .rag import RagPipeline
from .schemas import AskRequest, AskResponse

# Module-level handle populated on startup.
pipeline: RagPipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    pipeline = RagPipeline()
    print(
        f"[startup] Indexed {pipeline.store.size} chunks "
        f"using embedding backend: {pipeline.embedder.backend_name}; "
        f"LLM generation: {'on' if settings.use_llm else 'OFF (no API key set)'}."
    )
    yield


app = FastAPI(
    title="Health AI RAG Backend",
    description=(
        "Document-grounded RAG assistant for HFpEF and cardio-kidney-metabolic "
        "conditions, with an evidence-sufficiency gate, safety guardrails, and "
        "research logging."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "indexed_chunks": pipeline.store.size if pipeline else 0,
        "embedding_backend": pipeline.embedder.backend_name if pipeline else None,
        "llm_generation": settings.use_llm,
    }


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    assert pipeline is not None, "Pipeline not initialized"
    result = pipeline.answer(request.question)
    return AskResponse(
        answer=result.answer,
        evidence_used=result.evidence_used,
        evidence_sufficient=result.evidence_sufficient,
        guardrail_triggered=result.guardrail_triggered,
        groundedness=result.groundedness,
    )
