import logging
import os

from agents.state import AgentState
from config import INDEX_DIR, PATIENT_DATA_PATH, CHUNKS_PER_QUERY, MAX_TOTAL_CHUNKS

logger = logging.getLogger(__name__)

# Clinical entity search queries
CLINICAL_SEARCH_QUERIES = [
    # Medications
    "medications prescriptions drugs dosage pharmacy",
    "medication list current medications prescribed",
    # Labs
    "laboratory results lab values blood test",
    "HbA1c eGFR creatinine lipid panel CBC",
    # Conditions/Diagnoses
    "diagnosis assessment problem list conditions",
    "ICD-10 diagnoses active problems medical history",
    # Prior year
    "prior year previous history chronic conditions",
    # Symptoms
    "symptoms complaints chief complaint presenting",
    "patient reports fatigue pain shortness of breath",
]


def load_documents_node(state: AgentState) -> dict:
    from retrieval.search import get_search_index

    patient_id = state.get("patient_id")
    if not patient_id:
        return {
            "error": "No patient_id provided",
            "response": "Error: No patient ID provided for analysis.",
            "next_step": "direct_reply",
        }

    logger.info(f"Retrieving relevant documents for patient: {patient_id}")

    try:
        index = get_search_index(index_dir=INDEX_DIR)

        # Check if patient exists
        if not index.patient_exists(patient_id):
            # Fallback to file load
            return _load_documents_fallback(patient_id)

        # Hybrid search
        retrieved_chunks = _retrieve_relevant_chunks(index, patient_id)

        if not retrieved_chunks:
            logger.warning("No relevant chunks found via search, using all documents")
            retrieved_chunks = index.get_patient_documents(patient_id)[:MAX_TOTAL_CHUNKS]

        # Convert to documents
        documents = _chunks_to_documents(retrieved_chunks)

        total_chars = sum(len(doc.get("content", "")) for doc in documents)
        logger.info(
            f"Retrieved {len(documents)} relevant document sections "
            f"({total_chars:,} chars) from {len(retrieved_chunks)} chunks"
        )

        return {"documents": documents, "next_step": "extraction"}

    except Exception as e:
        logger.error(f"Failed to retrieve documents: {e}")
        return {
            "error": str(e),
            "response": f"Error loading documents: {e}",
            "next_step": "direct_reply",
        }


def _retrieve_relevant_chunks(index, patient_id: str) -> list:
    seen_ids = set()
    all_results = []

    for query in CLINICAL_SEARCH_QUERIES:
        try:
            results = index.search(
                query=query,
                top_k=CHUNKS_PER_QUERY,
                patient_id=patient_id,
                mode="hybrid"
            )

            for result in results:
                chunk_id = result.chunk.id
                if chunk_id not in seen_ids:
                    seen_ids.add(chunk_id)
                    all_results.append((result.score, result.chunk))

        except Exception as e:
            logger.warning(f"Search failed for query '{query[:30]}...': {e}")
            continue

    # Sort by score, take top
    all_results.sort(key=lambda x: x[0], reverse=True)
    top_chunks = [chunk for _, chunk in all_results[:MAX_TOTAL_CHUNKS]]

    logger.info(f"Retrieved {len(top_chunks)} unique chunks from {len(CLINICAL_SEARCH_QUERIES)} queries")
    return top_chunks


def _chunks_to_documents(chunks: list) -> list[dict]:
    docs_by_source = {}

    for chunk in chunks:
        source = chunk.metadata.get("source_file", "unknown")
        if source not in docs_by_source:
            docs_by_source[source] = {
                "id": source,
                "type": chunk.metadata.get("doc_type", "other"),
                "date": chunk.metadata.get("date"),
                "source": os.path.basename(source),
                "content": chunk.content,
            }
        else:
            # Append chunk (avoid dupes)
            if chunk.content not in docs_by_source[source]["content"]:
                docs_by_source[source]["content"] += "\n\n" + chunk.content

    return list(docs_by_source.values())


def _load_documents_fallback(patient_id: str) -> dict:
    from retrieval.loader import DocumentLoader

    logger.info(f"Using fallback document loader for {patient_id}")

    try:
        loader = DocumentLoader(PATIENT_DATA_PATH)
        raw_docs = loader.load_patient_documents(patient_id)

        if not raw_docs:
            return {
                "error": f"No documents found for patient {patient_id}",
                "response": f"No documents found for patient {patient_id}. Please verify the patient ID.",
                "next_step": "direct_reply",
            }

        documents = [
            {
                "id": doc.source_file,
                "type": doc.doc_type,
                "date": doc.date,
                "source": os.path.basename(doc.source_file),
                "content": doc.content,
            }
            for doc in raw_docs
        ]

        logger.info(f"Loaded {len(documents)} documents via fallback")
        return {"documents": documents, "next_step": "extraction"}

    except Exception as e:
        logger.error(f"Fallback document load failed: {e}")
        return {
            "error": str(e),
            "response": f"Error loading documents: {e}",
            "next_step": "direct_reply",
        }
