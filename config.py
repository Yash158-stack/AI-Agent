# config.py
import os
from functools import lru_cache
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

DEFAULT_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def get_embeddings_model():
    """Thread-safe singleton for HuggingFace embeddings to prevent duplicate memory allocation."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
