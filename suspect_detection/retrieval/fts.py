import os
import sqlite3
from typing import Optional
from dataclasses import dataclass

from core.models import Chunk


@dataclass
class FTSResult:
    chunk: Chunk
    score: float
    snippet: str


class FTSStore:
    def __init__(self, db_path: str = ":memory:"):
        self.db_path = db_path
        self.conn = None
        self._init_db()

    def _init_db(self):
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row

        # FTS5 table
        self.conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                chunk_id,
                content,
                patient_id,
                doc_type,
                date,
                source_file,
                section,
                tokenize='porter unicode61'
            )
        """)

        # Metadata table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY,
                content TEXT,
                patient_id TEXT,
                doc_type TEXT,
                date TEXT,
                source_file TEXT,
                section TEXT
            )
        """)

        # Patient index
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_patient ON chunks(patient_id)
        """)

        self.conn.commit()

    def add_chunks(self, chunks: list[Chunk]):
        if not chunks:
            return

        cursor = self.conn.cursor()

        for chunk in chunks:
            patient_id = chunk.metadata.get("patient_id", "")
            doc_type = chunk.metadata.get("doc_type", "")
            date = chunk.metadata.get("date", "")
            source_file = chunk.metadata.get("source_file", "")
            section = chunk.metadata.get("section", "")

            # Regular table
            cursor.execute("""
                INSERT OR REPLACE INTO chunks
                (chunk_id, content, patient_id, doc_type, date, source_file, section)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (chunk.id, chunk.content, patient_id, doc_type, date, source_file, section))

            # FTS table
            cursor.execute("""
                INSERT OR REPLACE INTO chunks_fts
                (chunk_id, content, patient_id, doc_type, date, source_file, section)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (chunk.id, chunk.content, patient_id, doc_type, date, source_file, section))

        self.conn.commit()

    def search(
        self,
        query: str,
        top_k: int = 5,
        patient_id: Optional[str] = None
    ) -> list[FTSResult]:
        # Build query
        fts_query = self._escape_query(query)

        if not fts_query.strip():
            return []

        # SQL
        if patient_id:
            sql = """
                SELECT
                    chunk_id,
                    content,
                    patient_id,
                    doc_type,
                    date,
                    source_file,
                    section,
                    bm25(chunks_fts) as score,
                    snippet(chunks_fts, 1, '<b>', '</b>', '...', 32) as snippet
                FROM chunks_fts
                WHERE chunks_fts MATCH ? AND patient_id = ?
                ORDER BY score
                LIMIT ?
            """
            params = (fts_query, patient_id, top_k)
        else:
            sql = """
                SELECT
                    chunk_id,
                    content,
                    patient_id,
                    doc_type,
                    date,
                    source_file,
                    section,
                    bm25(chunks_fts) as score,
                    snippet(chunks_fts, 1, '<b>', '</b>', '...', 32) as snippet
                FROM chunks_fts
                WHERE chunks_fts MATCH ?
                ORDER BY score
                LIMIT ?
            """
            params = (fts_query, top_k)

        cursor = self.conn.execute(sql, params)
        rows = cursor.fetchall()

        results = []
        for row in rows:
            metadata = {
                "patient_id": row["patient_id"],
                "doc_type": row["doc_type"],
                "date": row["date"],
                "source_file": row["source_file"]
            }
            if row["section"]:
                metadata["section"] = row["section"]

            chunk = Chunk(
                id=row["chunk_id"],
                content=row["content"],
                metadata=metadata
            )
            # BM25 scores (negated)
            results.append(FTSResult(
                chunk=chunk,
                score=-row["score"],
                snippet=row["snippet"]
            ))

        return results

    def patient_exists(self, patient_id: str) -> bool:
        cursor = self.conn.execute(
            "SELECT 1 FROM chunks WHERE patient_id = ? LIMIT 1",
            (patient_id,)
        )
        return cursor.fetchone() is not None

    def get_patient_chunks(self, patient_id: str) -> list[Chunk]:
        cursor = self.conn.execute("""
            SELECT chunk_id, content, patient_id, doc_type, date, source_file, section
            FROM chunks
            WHERE patient_id = ?
            ORDER BY date
        """, (patient_id,))

        chunks = []
        for row in cursor.fetchall():
            metadata = {
                "patient_id": row["patient_id"],
                "doc_type": row["doc_type"],
                "date": row["date"],
                "source_file": row["source_file"]
            }
            if row["section"]:
                metadata["section"] = row["section"]

            chunk = Chunk(
                id=row["chunk_id"],
                content=row["content"],
                metadata=metadata
            )
            chunks.append(chunk)

        return chunks

    def _escape_query(self, query: str) -> str:
        # Remove special chars
        special_chars = ['"', "'", "(", ")", "*", ":", "^", "-"]
        escaped = query
        for char in special_chars:
            escaped = escaped.replace(char, " ")

        # Split into terms
        terms = [t.strip() for t in escaped.split() if t.strip()]
        if not terms:
            return ""

        # Match terms
        if len(terms) == 1:
            return terms[0]
        else:
            # OR match
            return " OR ".join(terms)

    def count(self) -> int:
        cursor = self.conn.execute("SELECT COUNT(*) FROM chunks")
        return cursor.fetchone()[0]

    def list_patients(self) -> list[str]:
        cursor = self.conn.execute("""
            SELECT DISTINCT patient_id FROM chunks ORDER BY patient_id
        """)
        return [row[0] for row in cursor.fetchall()]

    def clear(self):
        self.conn.execute("DELETE FROM chunks")
        self.conn.execute("DELETE FROM chunks_fts")
        self.conn.commit()

    def close(self):
        if self.conn:
            self.conn.close()
            self.conn = None

    def __len__(self):
        return self.count()
