import logging
import os
import sys
import uuid
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.orchestrator import Orchestrator, get_orchestrator
from agents.graph import create_graph
from retrieval.loader import DocumentLoader
from config import PATIENT_DATA_PATH, LOG_LEVEL, LOG_DIR, LOG_FILE

# Configure logging with both console and file output
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),  # Console output
        logging.FileHandler(LOG_FILE, mode="a"),  # File output
    ],
)
logger = logging.getLogger(__name__)
logger.info(f"Logging to file: {LOG_FILE}")


# Request/response models
class ChatRequest(BaseModel):
    message: str
    reset: bool = False
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    patient_id: Optional[str] = None
    findings_count: int = 0
    session_id: str


class PatientInfo(BaseModel):
    patient_id: str
    document_count: int


# Pre-compile graph on startup for faster first requests
_shared_graph = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Application lifespan manager."""
    global _shared_graph

    logger.info("Starting Suspect Detection API...")

    # Initialize shared graph
    _shared_graph = create_graph()
    logger.info("LangGraph initialized")

    yield

    logger.info("Shutting down Suspect Detection API...")


app = FastAPI(
    title="Suspect Detection Agent",
    description="LangGraph-powered clinical suspect detection system",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_or_create_session(session_id: Optional[str] = None) -> tuple[Orchestrator, str]:
    """Get or create an orchestrator for a session.

    Args:
        session_id: Optional existing session ID

    Returns:
        Tuple of (orchestrator, session_id)
    """
    if not session_id:
        session_id = str(uuid.uuid4())

    orchestrator = get_orchestrator(session_id)
    return orchestrator, session_id


# API endpoints
@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "graph_ready": _shared_graph is not None,
        "version": "2.0.0",
    }


@app.post("/api/chat")
async def chat(request: ChatRequest) -> ChatResponse:
    """Process a chat message.

    Supports session-based isolation. Each session maintains its own state.
    """
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    orchestrator, session_id = get_or_create_session(request.session_id)

    if request.reset:
        orchestrator.reset()
        session_id = orchestrator.session_id  # Get new session ID after reset

    try:
        response = await orchestrator.run_async(request.message)

        return ChatResponse(
            response=response,
            patient_id=orchestrator.patient_id,
            findings_count=len(orchestrator.findings),
            session_id=session_id,
        )

    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/patients")
async def list_patients() -> list[PatientInfo]:
    """List all available patients with document counts."""
    try:
        loader = DocumentLoader(PATIENT_DATA_PATH)
        patient_ids = loader.list_patients()

        return [
            PatientInfo(
                patient_id=pid,
                document_count=len(loader.load_patient_documents(pid)),
            )
            for pid in patient_ids
        ]

    except Exception as e:
        logger.error(f"List patients error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/patient/{patient_id}/documents")
async def get_patient_documents(patient_id: str):
    """Get documents for a specific patient."""
    try:
        loader = DocumentLoader(PATIENT_DATA_PATH)
        docs = loader.load_patient_documents(patient_id)

        if not docs:
            raise HTTPException(status_code=404, detail=f"No documents found for patient {patient_id}")

        result = []
        for doc in docs:
            result.append({
                "doc_type": doc.doc_type,
                "date": doc.date,
                "source_file": os.path.basename(doc.source_file),
                "content": doc.content,
            })

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get documents error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Static files
WEB_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "web")
if os.path.exists(WEB_DIR):
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


@app.get("/")
async def serve_index():
    """Serve the web UI."""
    index_path = os.path.join(WEB_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {
        "message": "Suspect Detection API v2.0",
        "docs": "/docs",
        "endpoints": {
            "chat": "POST /api/chat",
            "patients": "GET /api/patients",
            "health": "GET /api/health",
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
