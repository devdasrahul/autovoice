"""Embedding generation for AutoVoice RAG.

Uses sentence-transformers/all-MiniLM-L6-v2 on CPU (small, fast, free),
with a deterministic hashing vectorizer fallback for lightweight local/test runs.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence

from autovoice.logging_conf import get_logger

log = get_logger(__name__)

EMBEDDING_DIM = 384


class EmbeddingEngine:
    """Lazy-loading CPU text embedder with fallback."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """Initialize embedder with model name."""
        self.model_name = model_name
        self._model = None
        self._is_st_loaded = False

    def load(self) -> None:
        """Load sentence-transformers model on CPU (idempotent)."""
        if self._is_st_loaded:
            return
        try:
            from sentence_transformers import SentenceTransformer

            log.info("rag.embeddings.loading", model=self.model_name)
            self._model = SentenceTransformer(self.model_name, device="cpu")
            self._is_st_loaded = True
            log.info("rag.embeddings.loaded", model=self.model_name)
        except Exception as exc:
            log.warning("rag.embeddings.fallback_to_hashing", reason=str(exc))
            self._is_st_loaded = False

    def encode(self, texts: Sequence[str], batch_size: int = 32) -> list[list[float]]:
        """Encode a batch of texts into normalized embedding vectors.

        Args:
            texts: List of strings to encode.
            batch_size: Batch size for model inference.

        Returns:
            List of 384-dimensional float vectors.
        """
        if not texts:
            return []

        self.load()
        if self._is_st_loaded and self._model is not None:
            vectors = self._model.encode(
                list(texts),
                batch_size=batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            return [v.tolist() for v in vectors]

        # Fast deterministic semantic hash projection (zero external dependency fallback)
        return [self._hash_vector(t) for t in texts]

    def encode_query(self, query: str) -> list[float]:
        """Encode a single search query string."""
        return self.encode([query])[0]

    @staticmethod
    def _hash_vector(text: str, dim: int = EMBEDDING_DIM) -> list[float]:
        """Deterministic normalized projection vector."""
        words = text.lower().split()
        v = [0.0] * dim
        for w in words:
            h = int(hashlib.sha256(w.encode("utf-8")).hexdigest(), 16)
            for i in range(4):
                idx = (h >> (i * 8)) % dim
                weight = (((h >> (i * 4)) & 0xFF) / 255.0) - 0.5
                v[idx] += weight
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [round(x / norm, 6) for x in v]
