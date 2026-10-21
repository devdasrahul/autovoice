"""Persistent ChromaDB vector index and retrieval for AutoVoice RAG.

Indexes cleaned NHTSA complaint narratives with vehicle metadata.
Supports metadata filtering (make, model, year, component, severity).
Provides an in-memory fallback store when ChromaDB is in testing/mock mode.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from autovoice.config import DATA_PROCESSED_DIR
from autovoice.logging_conf import get_logger
from autovoice.rag.embeddings import EmbeddingEngine

log = get_logger(__name__)

DEFAULT_CHROMA_DIR = DATA_PROCESSED_DIR / "chroma"
COLLECTION_NAME = "nhtsa_complaints"


@dataclass
class RetrievedPassage:
    """A passage retrieved from the vector index with similarity score and metadata."""

    id: str
    text: str
    score: float
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert passage to dictionary."""
        return {
            "id": self.id,
            "text": self.text,
            "score": round(self.score, 4),
            "metadata": self.metadata,
        }


class ComplaintVectorIndex:
    """Vector index wrapper supporting ChromaDB with in-memory fallback."""

    def __init__(
        self,
        persist_dir: Path | str | None = None,
        embedder: EmbeddingEngine | None = None,
        persist_directory: Path | str | None = None,
    ):
        """Initialize index with persistence directory and embedder."""
        target_dir = persist_dir or persist_directory or DEFAULT_CHROMA_DIR
        self.persist_dir = Path(target_dir)
        self.embedder = embedder or EmbeddingEngine()
        self._client = None
        self._collection = None
        self._memory_docs: list[dict[str, Any]] = []

    def _init_chroma(self) -> bool:
        """Initialize ChromaDB client. Returns True if successful."""
        if self._collection is not None:
            return True
        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings

            self.persist_dir.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(
                path=str(self.persist_dir),
                settings=ChromaSettings(anonymized_telemetry=False),
            )
            self._collection = self._client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
            return True
        except Exception as exc:
            log.warning("rag.chroma_init_fallback", reason=str(exc))
            return False

    def build_from_dataframe(self, df: pd.DataFrame, batch_size: int = 64) -> int:
        """Index all narratives from cleaned DataFrame into ChromaDB (idempotent).

        Args:
            df: Cleaned complaints dataframe.
            batch_size: Batch size for embeddings and upserts.

        Returns:
            Count of indexed records.
        """
        chroma_ok = self._init_chroma()
        records: list[dict[str, Any]] = []

        for _, row in df.iterrows():
            cmplid = str(row.get("cmplid"))
            text = str(row.get("narrative", "")).strip()
            if not text:
                continue

            meta = {
                "cmplid": cmplid,
                "make": str(row.get("make", "")).upper(),
                "model": str(row.get("model", "")).upper(),
                "model_year": int(row.get("model_year", 0)),
                "component": str(row.get("component", "")),
                "severity": str(row.get("severity", "LOW")),
                "crash": int(row.get("crash", 0)),
                "fire": int(row.get("fire", 0)),
            }
            records.append({"id": cmplid, "text": text, "metadata": meta})

        if not records:
            return 0

        # Cache in memory for instant local fallback
        self._memory_docs = records

        # Index in ChromaDB if available
        if chroma_ok and self._collection is not None:
            for i in range(0, len(records), batch_size):
                batch = records[i : i + batch_size]
                ids = [r["id"] for r in batch]
                texts = [r["text"] for r in batch]
                metas = [r["metadata"] for r in batch]
                embeddings = self.embedder.encode(texts)

                self._collection.upsert(
                    ids=ids,
                    documents=texts,
                    metadatas=metas,
                    embeddings=embeddings,
                )

            log.info("rag.index_built", count=len(records), dir=str(self.persist_dir))

        return len(records)

    def query(
        self,
        query_text: str,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievedPassage]:
        """Search index for top_k relevant complaints matching query and filters.

        Args:
            query_text: User question or search query.
            top_k: Number of passages to retrieve.
            filters: Metadata filter dictionary (e.g. {'make': 'FORD'}).

        Returns:
            List of RetrievedPassage objects ranked by relevance.
        """
        chroma_ok = self._init_chroma()
        query_vec = self.embedder.encode_query(query_text)

        # 1. Try ChromaDB native query
        if chroma_ok and self._collection is not None and self._collection.count() > 0:
            try:
                where_clause = None
                if filters:
                    valid_filters = {k: v for k, v in filters.items() if v is not None}
                    if len(valid_filters) == 1:
                        k, v = next(iter(valid_filters.items()))
                        where_clause = {k: {"$eq": v}}
                    elif len(valid_filters) > 1:
                        where_clause = {"$and": [{k: {"$eq": v}} for k, v in valid_filters.items()]}

                res = self._collection.query(
                    query_embeddings=[query_vec],
                    n_results=top_k,
                    where=where_clause,
                )

                passages: list[RetrievedPassage] = []
                if res and res["ids"] and len(res["ids"][0]) > 0:
                    for pid, doc, meta, dist in zip(
                        res["ids"][0],
                        res["documents"][0],
                        res["metadatas"][0],
                        res["distances"][0],
                        strict=False,
                    ):
                        score = max(0.0, 1.0 - float(dist))
                        passages.append(
                            RetrievedPassage(id=pid, text=doc, score=score, metadata=meta)
                        )
                    return passages
            except Exception as exc:
                log.warning("rag.chroma_query_failed_using_memory", reason=str(exc))

        # 2. In-Memory Cosine Similarity Fallback
        if not self._memory_docs:
            # Attempt loading from parquet
            parquet_p = DATA_PROCESSED_DIR / "complaints.parquet"
            if parquet_p.exists():
                df = pd.read_parquet(parquet_p)
                self.build_from_dataframe(df)

        if not self._memory_docs:
            return []

        # Filter candidates
        candidates = self._memory_docs
        if filters:
            for k, v in filters.items():
                if v is not None:
                    candidates = [
                        c
                        for c in candidates
                        if str(c["metadata"].get(k, "")).upper() == str(v).upper()
                    ]

        if not candidates:
            return []

        doc_texts = [c["text"] for c in candidates]
        doc_vecs = np.array(self.embedder.encode(doc_texts))
        q_vec = np.array(query_vec)

        # Cosine similarities
        scores = np.dot(doc_vecs, q_vec)
        top_indices = np.argsort(scores)[::-1][:top_k]

        return [
            RetrievedPassage(
                id=candidates[idx]["id"],
                text=candidates[idx]["text"],
                score=float(scores[idx]),
                metadata=candidates[idx]["metadata"],
            )
            for idx in top_indices
        ]
