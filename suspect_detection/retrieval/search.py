import logging
import os
from typing import Optional
from dataclasses import dataclass

from config import EMBEDDING_MODEL
from core.models import Chunk
from .vector import VectorStore
from .fts import FTSStore

logger = logging.getLogger(__name__)


@dataclass
class HybridResult:
    chunk: Chunk
    score: float
    vector_score: Optional[float] = None
    fts_score: Optional[float] = None
    snippet: Optional[str] = None


class HybridSearch:
    def __init__(
        self,
        vector_store: VectorStore,
        fts_store: FTSStore,
        vector_weight: float = 0.5,
        fts_weight: float = 0.5
    ):
        self.vector_store = vector_store
        self.fts_store = fts_store
        self.vector_weight = vector_weight
        self.fts_weight = fts_weight

    def search(
        self,
        query: str,
        top_k: int = 5,
        patient_id: Optional[str] = None,
        mode: str = "hybrid"
    ) -> list[HybridResult]:
        if mode == "vector":
            return self._vector_search(query, top_k, patient_id)
        elif mode == "fts":
            return self._fts_search(query, top_k, patient_id)
        else:
            return self._hybrid_search(query, top_k, patient_id)

    def _vector_search(
        self,
        query: str,
        top_k: int,
        patient_id: Optional[str]
    ) -> list[HybridResult]:
        results = self.vector_store.search(query, top_k=top_k, patient_id=patient_id)
        return [
            HybridResult(
                chunk=r.chunk,
                score=r.score,
                vector_score=r.score,
                fts_score=None
            )
            for r in results
        ]

    def _fts_search(
        self,
        query: str,
        top_k: int,
        patient_id: Optional[str]
    ) -> list[HybridResult]:
        results = self.fts_store.search(query, top_k=top_k, patient_id=patient_id)
        return [
            HybridResult(
                chunk=r.chunk,
                score=r.score,
                vector_score=None,
                fts_score=r.score,
                snippet=r.snippet
            )
            for r in results
        ]

    def _hybrid_search(
        self,
        query: str,
        top_k: int,
        patient_id: Optional[str]
    ) -> list[HybridResult]:
        # Get results
        fetch_k = top_k * 2  # Fetch more for better fusion

        vector_results = self.vector_store.search(query, top_k=fetch_k, patient_id=patient_id)
        fts_results = self.fts_store.search(query, top_k=fetch_k, patient_id=patient_id)

        # Score maps
        vector_scores: dict[str, float] = {}
        fts_scores: dict[str, float] = {}
        chunks: dict[str, Chunk] = {}
        snippets: dict[str, str] = {}

        # Vector scores
        for r in vector_results:
            vector_scores[r.chunk.id] = r.score
            chunks[r.chunk.id] = r.chunk

        # FTS scores (normalized)
        if fts_results:
            max_fts = max(r.score for r in fts_results) if fts_results else 1.0
            for r in fts_results:
                normalized = r.score / max_fts if max_fts > 0 else 0
                fts_scores[r.chunk.id] = normalized
                chunks[r.chunk.id] = r.chunk
                snippets[r.chunk.id] = r.snippet

        # Combine scores
        all_chunk_ids = set(vector_scores.keys()) | set(fts_scores.keys())
        combined: list[HybridResult] = []

        for chunk_id in all_chunk_ids:
            vs = vector_scores.get(chunk_id, 0.0)
            fs = fts_scores.get(chunk_id, 0.0)

            # Weighted
            combined_score = (self.vector_weight * vs) + (self.fts_weight * fs)

            combined.append(HybridResult(
                chunk=chunks[chunk_id],
                score=combined_score,
                vector_score=vs if chunk_id in vector_scores else None,
                fts_score=fs if chunk_id in fts_scores else None,
                snippet=snippets.get(chunk_id)
            ))

        # Sort
        combined.sort(key=lambda x: x.score, reverse=True)

        return combined[:top_k]


class SearchIndex:
    def __init__(
        self,
        index_dir: str,
        embedding_model: str = None
    ):
        self.index_dir = index_dir
        self.embedding_model = embedding_model or EMBEDDING_MODEL

        # Stores
        self.vector_store = VectorStore(embedding_model=self.embedding_model)
        self.fts_store = FTSStore(db_path=os.path.join(index_dir, "fts.db"))

        # Hybrid search
        self.hybrid = HybridSearch(self.vector_store, self.fts_store)

        # Load if available
        self._try_load()

    def _try_load(self):
        vector_path = os.path.join(self.index_dir, "vector")
        if os.path.exists(os.path.join(vector_path, "config.json")):
            try:
                self.vector_store.load(vector_path)
                logger.debug(f"Loaded {len(self.vector_store)} chunks from vector index")
            except Exception as e:
                logger.warning(f"Failed to load vector index: {e}")

    def add_chunks(self, chunks: list[Chunk]):
        self.vector_store.add_chunks(chunks)
        self.fts_store.add_chunks(chunks)

    def search(
        self,
        query: str,
        top_k: int = 5,
        patient_id: Optional[str] = None,
        mode: str = "hybrid"
    ) -> list[HybridResult]:
        return self.hybrid.search(query, top_k, patient_id, mode)

    def patient_exists(self, patient_id: str) -> bool:
        return self.fts_store.patient_exists(patient_id)

    def get_patient_documents(self, patient_id: str) -> list[Chunk]:
        return self.fts_store.get_patient_chunks(patient_id)

    def save(self):
        os.makedirs(self.index_dir, exist_ok=True)
        vector_path = os.path.join(self.index_dir, "vector")
        self.vector_store.save(vector_path)
        # FTS persisted via SQLite

    def list_patients(self) -> list[str]:
        return self.fts_store.list_patients()

    def __len__(self):
        return len(self.vector_store)


# Singleton
_search_index: Optional[SearchIndex] = None


def get_search_index(index_dir: str = None, embedding_model: str = None) -> SearchIndex:
    global _search_index

    if _search_index is None:
        if index_dir is None:
            from config import INDEX_DIR
            index_dir = INDEX_DIR
        if embedding_model is None:
            embedding_model = EMBEDDING_MODEL
        _search_index = SearchIndex(index_dir=index_dir, embedding_model=embedding_model)
        logger.info(f"Created SearchIndex singleton with {len(_search_index)} chunks")

    return _search_index
