import os
from dotenv import load_dotenv

load_dotenv()

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
PATIENT_DATA_PATH = os.path.join(PROJECT_ROOT, "patient_data")
INDEX_DIR = os.path.join(BASE_DIR, "data", "index")

# Gemini API settings
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Model settings
GEMINI_FLASH_MODEL = "gemini-2.0-flash"      # Fast tasks: routing, simple queries
GEMINI_PRO_MODEL = "gemini-2.5-pro"        # Complex reasoning (using flash to avoid rate limits)

# Embedding model (local sentence-transformers)
EMBEDDING_MODEL = "NeuML/pubmedbert-base-embeddings"

# LLM generation settings
DEFAULT_TEMPERATURE = 0.3
DEFAULT_MAX_TOKENS = 4096

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_DIR = os.path.join(BASE_DIR, "logs")
LOG_FILE = os.path.join(LOG_DIR, "app.log")

# Validation
MAX_REFINEMENT_ATTEMPTS = 2
CONFIDENCE_THRESHOLD = 0.6
REFINEMENT_THRESHOLD = 0.3
VALIDATION_MAX_DOCS = 5
VALIDATION_MAX_CHARS = 2000

# Chunking
MIN_CHUNK_SIZE = 100
MAX_CHUNK_SIZE = 1500
CHUNK_OVERLAP = 100
MAX_CHUNK_TOKENS = 480
MIN_AVAILABLE_TOKENS = 50
SMALL_SECTION_THRESHOLD = 200
MIN_SPLIT_SIZE = 50
HEADER_MAX_LINES = 10

# Document retrieval
CHUNKS_PER_QUERY = 5
MAX_TOTAL_CHUNKS = 20

# Lab thresholds
LAB_HBA1C = 6.5
LAB_EGFR = 60
LAB_BNP = 100
LAB_NT_PROBNP = 300
LAB_CREATININE = 1.5
LAB_TSH = 4.5
LAB_LDL = 190

# Symptom cluster min matches
SLEEP_APNEA_MIN_MATCHES = 3
HEART_FAILURE_MIN_MATCHES = 3
DEPRESSION_MIN_MATCHES = 4
HYPOTHYROIDISM_MIN_MATCHES = 3
