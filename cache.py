import pickle
import numpy as np
from db import SessionLocal, QueryCache
from langchain_huggingface import HuggingFaceEmbeddings


def normalize_query(q: str) -> str:
    return q.lower().strip()
embeddings_model = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

SIMILARITY_THRESHOLD = 0.75
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
def get_cached_response(query: str):
    db = SessionLocal()
    try:
        query_vec = embeddings_model.embed_query(normalize_query(query))
        entries = db.query(QueryCache).all()

        for entry in entries:
            stored_vec = pickle.loads(entry.embedding)
            score = cosine_similarity(query_vec, stored_vec)

            if score >= SIMILARITY_THRESHOLD:
                print(f"✅ SEMANTIC DB HIT (score={score:.2f})")
                return entry.response

        print("❌ SEMANTIC DB MISS - calling LLM")
        return None
    finally:
        db.close()

def should_cache_response(response: str) -> bool:
    negative_phrases = [
        "couldn't find",
        "no relevant",
        "not found",
        "not present",
        "does not contain"
    ]
    r = response.lower()
    return not any(p in r for p in negative_phrases)

def save_response(query: str, response: str):
    if not should_cache_response(response):
        print("⚠️ Negative response not cached")
        return

    db = SessionLocal()
    try:
        vec = embeddings_model.embed_query(normalize_query(query))
        entry = QueryCache(
            query=query,
            response=response,
            embedding=pickle.dumps(vec)
        )
        db.add(entry)
        db.commit()
        print("💾 Saved response + embedding to DB")
    finally:
        db.close()
