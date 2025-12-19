#!/usr/bin/env python3
import os
import sys
import argparse
import time

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from config import PATIENT_DATA_PATH, INDEX_DIR, EMBEDDING_MODEL
from retrieval.loader import DocumentLoader
from retrieval.chunker import Chunker
from retrieval.search import SearchIndex


def index_patient(
    patient_id: str,
    loader: DocumentLoader,
    chunker: Chunker,
    index: SearchIndex
) -> int:
    """Index a single patient's documents."""
    print(f"  Loading documents for {patient_id}...")
    documents = loader.load_patient_documents(patient_id)

    if not documents:
        print(f"  No documents found for {patient_id}")
        return 0

    print(f"  Found {len(documents)} documents")

    # Chunk documents
    all_chunks = []
    for doc in documents:
        chunks = chunker.chunk_document(doc)
        all_chunks.extend(chunks)

    print(f"  Created {len(all_chunks)} chunks")

    # Add to index
    index.add_chunks(all_chunks)

    return len(all_chunks)


def main():
    parser = argparse.ArgumentParser(description="Index patient documents")
    parser.add_argument("--patient", type=str, help="Index specific patient only")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild index from scratch")
    parser.add_argument("--index-dir", type=str, default=INDEX_DIR, help="Index directory")
    parser.add_argument("--data-dir", type=str, default=PATIENT_DATA_PATH, help="Patient data directory")
    args = parser.parse_args()

    # Check data directory
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory not found: {args.data_dir}")
        sys.exit(1)

    # Rebuild: remove existing index
    if args.rebuild and os.path.exists(args.index_dir):
        import shutil
        print(f"Removing existing index at {args.index_dir}")
        shutil.rmtree(args.index_dir)

    # Create index directory
    os.makedirs(args.index_dir, exist_ok=True)

    print(f"Index directory: {args.index_dir}")
    print(f"Data directory: {args.data_dir}")
    print(f"Embedding model: {EMBEDDING_MODEL}")
    print()

    # Initialize components
    loader = DocumentLoader(args.data_dir)
    chunker = Chunker()
    index = SearchIndex(index_dir=args.index_dir, embedding_model=EMBEDDING_MODEL)

    start_time = time.time()

    # Get patients to index
    if args.patient:
        patients = [args.patient]
    else:
        patients = loader.list_patients()

    if not patients:
        print("No patients found to index")
        sys.exit(1)

    print(f"Indexing {len(patients)} patient(s)...")
    print("=" * 50)

    total_chunks = 0
    for patient_id in patients:
        print(f"\nPatient: {patient_id}")
        chunks = index_patient(patient_id, loader, chunker, index)
        total_chunks += chunks

    # Save index
    print("\n" + "=" * 50)
    print("Saving index...")
    index.save()

    elapsed = time.time() - start_time
    print(f"\nDone!")
    print(f"  Total patients: {len(patients)}")
    print(f"  Total chunks: {total_chunks}")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Index saved to: {args.index_dir}")


if __name__ == "__main__":
    main()
