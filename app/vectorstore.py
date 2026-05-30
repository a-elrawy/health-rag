"""FAISS vector store.

Vectors are L2-normalized, so an inner-product index (``IndexFlatIP``) returns
cosine similarity directly. An exact flat index is ideal for this tiny corpus.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import faiss
import numpy as np

from .ingest import Chunk


@dataclass
class RetrievedChunk:
    chunk: Chunk
    similarity_score: float


class FaissVectorStore:
    def __init__(self, chunks: List[Chunk], embeddings: np.ndarray) -> None:
        if len(chunks) != embeddings.shape[0]:
            raise ValueError("Number of chunks and embeddings must match.")
        self._chunks = chunks
        self._dim = int(embeddings.shape[1])
        self._index = faiss.IndexFlatIP(self._dim)
        self._index.add(np.ascontiguousarray(embeddings, dtype="float32"))

    @property
    def size(self) -> int:
        return len(self._chunks)

    def search(self, query_embedding: np.ndarray, top_k: int) -> List[RetrievedChunk]:
        query = np.ascontiguousarray(query_embedding.reshape(1, -1), dtype="float32")
        k = min(top_k, len(self._chunks))
        scores, indices = self._index.search(query, k)

        results: List[RetrievedChunk] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            # Clamp tiny floating point overshoots so scores stay within [0, 1].
            clamped = float(max(0.0, min(1.0, score)))
            results.append(
                RetrievedChunk(chunk=self._chunks[idx], similarity_score=clamped)
            )
        return results
