import os
from dotenv import load_dotenv
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
CHROMA_DB_DIR = os.getenv("CHROMA_DB_DIR", "./chroma_db")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL","models/embedding-001"
)  # example
GENERATION_MODEL = os.getenv("GENERATION_MODEL", "gemini-2.5-flash")      # example
TOP_K = int(os.getenv("TOP_K", "3"))
