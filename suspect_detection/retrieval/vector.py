import os
import json
import pickle
from typing import Optional
from dataclasses import dataclass

import numpy as np

from core.models import Chunk


@dataclass
class SearchResult:
    chunk: Chunk
    score: float


class VectorStore:
    def __init__(
        self,
        embedding_model: str = "NeuML/pubmedbert-base-embeddings",
        index_path: Optional[str] = None
    ):
        self.embedding_model_name = embedding_model
        self.index_path = index_path
        self.embedder = None
        self.index = None
        self.chunks: list[Chunk] = []  # Ordered list matching FAISS index
        self.chunk_to_idx: dict[str, int] = {}  # chunk.id -> index position
        self._dimension = None

    def _load_embedder(self):
        if self.embedder is None:
            from sentence_transformers import SentenceTransformer
            self.embedder = SentenceTransformer(self.embedding_model_name)
            self._dimension = self.embedder.get_sentence_embedding_dimension()

    def _init_index(self):
        import faiss
        self._load_embedder()
        self.index = faiss.IndexFlatIP(self._dimension)  # Inner product (cosine after norm)

    def embed(self, texts: list[str]) -> np.ndarray:
        self._load_embedder()
        embeddings = self.embedder.encode(texts, normalize_embeddings=True)
        return embeddings.astype(np.float32)

    def add_chunks(self, chunks: list[Chunk]):
        if not chunks:
            return

        if self.index is None:
            self._init_index()

        # Generate embeddings
        texts = [chunk.content for chunk in chunks]
        embeddings = self.embed(texts)

        # Add to FAISS
        start_idx = len(self.chunks)
        self.index.add(embeddings)

        # Store chunk references
        for i, chunk in enumerate(chunks):
            self.chunks.append(chunk)
            self.chunk_to_idx[chunk.id] = start_idx + i

    def search(
        self,
        query: str,
        top_k: int = 5,
        patient_id: Optional[str] = None,
        min_score: float = 0.0
    ) -> list[SearchResult]:
        if self.index is None or self.index.ntotal == 0:
            return []

        # Embed query
        query_embedding = self.embed([query])

        # Search more than needed if filtering by patient
        search_k = top_k * 10 if patient_id else top_k

        # FAISS search
        scores, indices = self.index.search(query_embedding, min(search_k, self.index.ntotal))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or score < min_score:
                continue

            chunk = self.chunks[idx]

            # Filter by patient if specified
            if patient_id and chunk.metadata.get("patient_id") != patient_id:
                continue

            results.append(SearchResult(chunk=chunk, score=float(score)))

            if len(results) >= top_k:
                break

        return results

    def save(self, path: str):
        import faiss

        os.makedirs(path, exist_ok=True)

        # Save FAISS index
        if self.index is not None:
            faiss.write_index(self.index, os.path.join(path, "index.faiss"))

        # Save chunks as JSON (for readability)
        chunks_data = []
        for chunk in self.chunks:
            chunks_data.append({
                "id": chunk.id,
                "content": chunk.content,
                "metadata": chunk.metadata
            })

        with open(os.path.join(path, "chunks.json"), "w") as f:
            json.dump(chunks_data, f, indent=2)

        # Save config
        config = {
            "embedding_model": self.embedding_model_name,
            "dimension": self._dimension,
            "num_chunks": len(self.chunks)
        }
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

    def load(self, path: str):
        import faiss

        # Load config
        with open(os.path.join(path, "config.json")) as f:
            config = json.load(f)

        self.embedding_model_name = config["embedding_model"]
        self._dimension = config["dimension"]

        # Load FAISS index
        index_path = os.path.join(path, "index.faiss")
        if os.path.exists(index_path):
            self.index = faiss.read_index(index_path)

        # Load chunks
        with open(os.path.join(path, "chunks.json")) as f:
            chunks_data = json.load(f)

        self.chunks = []
        self.chunk_to_idx = {}
        for i, data in enumerate(chunks_data):
            chunk = Chunk(
                id=data["id"],
                content=data["content"],
                metadata=data["metadata"]
            )
            self.chunks.append(chunk)
            self.chunk_to_idx[chunk.id] = i

    def clear(self):
        self.index = None
        self.chunks = []
        self.chunk_to_idx = {}

    def __len__(self):
        return len(self.chunks)
