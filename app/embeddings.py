"""Local semantic embeddings via sentence-transformers.

Vectors are L2-normalized so inner-product search in FAISS equals cosine
similarity. ``backend_name`` is recorded in the research log.
"""

from __future__ import annotations

from typing import List

import numpy as np


class EmbeddingModel:
    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        self.backend_name = f"sentence-transformers:{model_name}"

    def embed(self, texts: List[str]) -> np.ndarray:
        return self._model.encode(
            texts, convert_to_numpy=True, normalize_embeddings=True
        ).astype("float32")
