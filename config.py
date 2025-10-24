import os
from dotenv import load_dotenv
load_dotenv()
   

# Paths
CODE_DIR = os.path.join(os.path.dirname(__file__), os.getenv("CODE_DIR_REL", "code"))
CHROMA_DIR = os.path.join(os.getcwd(), os.getenv("CHROMA_DIR_REL", "chroma_db"))
DEFAULT_PDF = os.path.join(os.getenv("CODE_DIR_REL", "code"), os.getenv("DEFAULT_PDF_NAME", "Resume.pdf"))

# Server settings
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))

# Models
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", None)
LLM_MODEL = os.getenv("LLM_MODEL", None)
#Model Key
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY", None)

# Chunking
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
OVERLAP = int(os.getenv("OVERLAP", "100"))

# Retrieval
K_RETRIEVE = int(os.getenv("K_RETRIEVE", "4"))
TOP_K = int(os.getenv("TOP_K", "3"))
PROMPT_TEMPLATE = os.getenv("PROMPT_TEMPLATE", None)
REFINED_PROMPT_TEMPLATE=os.getenv("REFINED_PROMPT_TEMPLATE", None)
COLLECTION_NAME = os.getenv("COLLECTION_NAME", None)
QDRANT_URL = os.getenv("QDRANT_URL", None)