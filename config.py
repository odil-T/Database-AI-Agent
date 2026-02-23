import os

from dotenv import load_dotenv
from pathlib import Path


load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

EMBEDDING_MODEL_DIR = BASE_DIR / "models" / "bge-large-en-v1.5"
TABLE_SCHEMAS_DIR = BASE_DIR / "table_schemas"
CHROMA_VECTORSTORE_DIR = BASE_DIR / "chroma_langchain_db"

COHERE_API_KEY = os.getenv("COHERE_API_KEY")