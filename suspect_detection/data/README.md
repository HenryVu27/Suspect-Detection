# Data Storage Structure

This directory contains the search indices for the suspect detection system.

## Directory Layout

```
data/
├── README.md              # This file
└── index/
    ├── fts.db             # SQLite database with FTS5 index
    └── vector/
        ├── index.faiss    # FAISS vector index (binary)
        ├── chunks.json    # Chunk content + metadata
        └── config.json    # Index configuration
```

---

## SQLite FTS Store (`fts.db`)

### Tables

**`chunks`** - Main storage table
```sql
CREATE TABLE chunks (
    chunk_id    TEXT PRIMARY KEY,  -- e.g., "CVD-2025-001_progress_note_2024-10-15_0"
    content     TEXT,              -- Full chunk text
    patient_id  TEXT,              -- e.g., "CVD-2025-001"
    doc_type    TEXT,              -- e.g., "progress_note", "lab", "hra"
    date        TEXT,              -- e.g., "2024-10-15"
    source_file TEXT               -- Original file path
);

CREATE INDEX idx_patient ON chunks(patient_id);
```

**`chunks_fts`** - FTS5 virtual table (inverted index)
```sql
CREATE VIRTUAL TABLE chunks_fts USING fts5(
    chunk_id,
    content,
    patient_id,
    doc_type,
    date,
    source_file,
    tokenize='porter unicode61'  -- Porter stemming + unicode support
);
```

### Key Queries

```sql
-- Full-text search with BM25 ranking
SELECT *, bm25(chunks_fts) as score
FROM chunks_fts
WHERE chunks_fts MATCH 'diabetes medication'
ORDER BY score
LIMIT 10;

-- Search within a specific patient
SELECT *, bm25(chunks_fts) as score
FROM chunks_fts
WHERE chunks_fts MATCH 'eliquis' AND patient_id = 'CVD-2025-002'
ORDER BY score;

-- Get snippet with highlighted matches
SELECT snippet(chunks_fts, 1, '<b>', '</b>', '...', 32) as snippet
FROM chunks_fts
WHERE chunks_fts MATCH 'diabetes';
```

---

## FAISS Vector Store (`vector/`)

### Files

| File | Contents |
|------|----------|
| `index.faiss` | Binary FAISS index using `IndexFlatIP` (Inner Product for cosine similarity) |
| `chunks.json` | Array of chunk objects with content and metadata |
| `config.json` | Index configuration |

### config.json Structure
```json
{
    "embedding_model": "all-MiniLM-L6-v2",
    "dimension": 384,
    "num_chunks": 42
}
```

### chunks.json Structure
```json
[
    {
        "id": "CVD-2025-001_progress_note_2024-10-15_0",
        "content": "Patient presents with...",
        "metadata": {
            "patient_id": "CVD-2025-001",
            "doc_type": "progress_note",
            "date": "2024-10-15",
            "source_file": "/path/to/file.txt"
        }
    }
]
```

### How FAISS Search Works

1. Query text is embedded using `sentence-transformers`
2. FAISS finds nearest vectors by inner product (cosine similarity after normalization)
3. Returns indices into the chunks array
4. Look up actual content from `chunks.json`

```python
# Pseudocode
query_vector = embedder.encode("diabetes symptoms")
scores, indices = faiss_index.search(query_vector, top_k=5)
results = [chunks[i] for i in indices]
```

---

## Key Functions

### Indexing (`retrieval/search.py`)

```python
class SearchIndex:
    def add_chunks(self, chunks: list[Chunk]):
        """Add chunks to both FAISS and SQLite FTS."""
        self.vector_store.add_chunks(chunks)  # FAISS
        self.fts_store.add_chunks(chunks)     # SQLite

    def save(self):
        """Persist FAISS to disk. SQLite is auto-persisted."""
        self.vector_store.save(vector_path)
```

### Vector Search (`retrieval/vector.py`)

```python
class VectorStore:
    def embed(self, texts: list[str]) -> np.ndarray:
        """Generate embeddings using sentence-transformers."""
        return self.embedder.encode(texts, normalize_embeddings=True)

    def search(self, query: str, top_k: int, patient_id: str = None):
        """Semantic search with optional patient filter."""
        query_embedding = self.embed([query])
        scores, indices = self.index.search(query_embedding, top_k)
        # Filter by patient_id if specified
        return [SearchResult(chunk, score) for ...]
```

### FTS Search (`retrieval/fts.py`)

```python
class FTSStore:
    def search(self, query: str, top_k: int, patient_id: str = None):
        """BM25-ranked full-text search."""
        # Uses SQLite FTS5 MATCH syntax
        # Returns results with snippets

    def get_patient_chunks(self, patient_id: str) -> list[Chunk]:
        """Get all chunks for a patient (no search, just retrieval)."""

    def get_chunks_by_type(self, patient_id: str, doc_type: str):
        """Get chunks filtered by document type."""
```

### Hybrid Search (`retrieval/search.py`)

```python
class HybridSearch:
    def search(self, query: str, mode: str = "hybrid"):
        """
        Modes:
        - "hybrid": Combine FAISS + FTS with weighted score fusion
        - "vector": FAISS only (semantic similarity)
        - "fts": SQLite FTS only (keyword matching)
        """
        if mode == "hybrid":
            vector_results = self.vector_store.search(query)
            fts_results = self.fts_store.search(query)
            # Normalize and combine scores
            combined_score = (vector_weight * vs) + (fts_weight * fs)
```

---

## Building the Index

```bash
# Build index for all patients
python scripts/index_documents.py --rebuild

# Build for specific patient
python scripts/index_documents.py --patient CVD-2025-001

# Test search
python scripts/test_search.py --query "diabetes" --mode hybrid
```

---

## Embedding Model

Default: `all-MiniLM-L6-v2` from sentence-transformers

| Property | Value |
|----------|-------|
| Dimension | 384 |
| Max Sequence | 256 tokens |
| Speed | Fast (good for demo) |
| Quality | Decent for general text |

To change model, update `EMBEDDING_MODEL` in `config.py` and rebuild index.

---

## Notes

- SQLite FTS uses **Porter stemming** - "running" matches "run"
- FAISS uses **normalized vectors** - cosine similarity via inner product
- Hybrid search **normalizes FTS scores** since BM25 values vary widely
- Patient filtering happens **post-search** in FAISS (no native support)
