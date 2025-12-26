#!/usr/bin/env python3
import os
import shutil
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from config import PATIENT_DATA_PATH, INDEX_DIR, EMBEDDING_MODEL
from retrieval.loader import DocumentLoader
from retrieval.chunker import Chunker
from retrieval.search import SearchIndex


def index_patient(patient_id: str, loader: DocumentLoader, chunker: Chunker, index: SearchIndex) -> int:
    print(f"  Loading documents for {patient_id}...")
    documents = loader.load_patient_documents(patient_id)

    if not documents:
        print(f"  No documents found for {patient_id}")
        return 0

    print(f"  Found {len(documents)} documents")

    all_chunks = []
    for doc in documents:
        chunks = chunker.chunk_document(doc)
        all_chunks.extend(chunks)

    print(f"  Created {len(all_chunks)} chunks")
    index.add_chunks(all_chunks)
    return len(all_chunks)


def main():
    if not os.path.exists(PATIENT_DATA_PATH):
        print(f"Error: Data directory not found: {PATIENT_DATA_PATH}")
        sys.exit(1)

    if os.path.exists(INDEX_DIR):
        print(f"Removing existing index at {INDEX_DIR}")
        shutil.rmtree(INDEX_DIR)

    os.makedirs(INDEX_DIR, exist_ok=True)

    print(f"Index directory: {INDEX_DIR}")
    print(f"Data directory: {PATIENT_DATA_PATH}")
    print(f"Embedding model: {EMBEDDING_MODEL}\n")

    loader = DocumentLoader(PATIENT_DATA_PATH)
    chunker = Chunker()
    index = SearchIndex(index_dir=INDEX_DIR, embedding_model=EMBEDDING_MODEL)

    start_time = time.time()
    patients = loader.list_patients()

    if not patients:
        print("No patients found to index")
        sys.exit(1)

    print(f"Indexing {len(patients)} patient(s)...")
    total_chunks = 0
    for patient_id in patients:
        print(f"\nPatient: {patient_id}")
        total_chunks += index_patient(patient_id, loader, chunker, index)

    print("\nSaving index...")
    index.save()

    elapsed = time.time() - start_time
    print(f"Done! {len(patients)} patients, {total_chunks} chunks in {elapsed:.2f}s")


if __name__ == "__main__":
    main()
