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
