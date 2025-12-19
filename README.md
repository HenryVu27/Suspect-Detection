# Suspect Detection System

A clinical suspect detection system that analyzes patient medical records to identify potential gaps, inconsistencies, and overlooked conditions using a multi-agent LangGraph workflow powered by Google Gemini.

## What It Does

The system detects:
- Medications without documented diagnoses
- Abnormal lab values suggesting undiagnosed conditions
- Chronic conditions missing from current documentation
- Symptom patterns indicating undiagnosed conditions
- Contradictions across clinical documents

## Agent Framework

The system uses LangGraph with a supervisor pattern:

```
orchestrator -> classifies user intent
     |
load_documents -> retrieves relevant documents via hybrid search
     |
extraction -> extracts clinical entities (meds, labs, conditions, symptoms)
     |
supervisor -> routes to detection agents based on available data
     |
     +-- cross_reference: medication/lab gaps
     +-- dropoff: missing chronic conditions
     +-- symptom_cluster: symptom pattern analysis
     +-- contradiction: conflicting information
     |
aggregate -> combines findings
     |
self_reflect -> validates findings
     |
report -> generates response
```

Key components:
- agents/graph.py - workflow definition
- agents/state.py - state management
- agents/nodes/ - node implementations
- retrieval/ - document loading, chunking, and search

## Setup

Install dependencies:

```
pip install -r requirements.txt
```

Create a .env file with your Gemini API key:

```
GEMINI_API_KEY=your_key_here
```

## Building the Index

Build chunks and index all patient documents:

```
python scripts/index_documents.py --rebuild
```

Index a specific patient:

```
python scripts/index_documents.py --patient CVD-2025-001
```

This creates:
- data/index/vector/ - FAISS vector index for semantic search
- data/index/fts.db - SQLite FTS5 index for keyword search

## Starting the Server

Option 1 - Interactive CLI:

```
python main.py
```

Option 2 - API Server:

```
cd api
python server.py
```

Server runs at http://localhost:8000 with API docs at http://localhost:8000/docs

## Querying

Via CLI:

```
python main.py
You: Analyze patient CVD-2025-001
You: List patients
You: quit
```

Via API:

```
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Analyze patient CVD-2025-001"}'
```

List patients:

```
curl http://localhost:8000/api/patients
```

Supported query types:
- Full analysis: "Analyze patient CVD-2025-001"
- Information requests: "What medications is this patient on?"
- List patients: "List available patients"
- Follow-up questions: "What should I focus on?"
- Medical questions: "What is HbA1c?"

## Project Structure

```
suspect_detection/
  agents/
    graph.py          - LangGraph workflow
    state.py          - state definitions
    orchestrator.py   - high-level API
    gemini_client.py  - Gemini integration
    nodes/            - node implementations
  retrieval/
    loader.py         - document loading
    chunker.py        - document chunking
    vector.py         - FAISS vector store
    fts.py            - SQLite FTS store
    search.py         - hybrid search
  api/
    server.py         - FastAPI server
  scripts/
    index_documents.py - indexing CLI
  config.py           - configuration
  main.py             - CLI entry point
```
