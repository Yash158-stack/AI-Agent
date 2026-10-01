# cache.py
import numpy as np

try:
    from config import get_embeddings_model
except ImportError:
    from functools import lru_cache
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError:
        from langchain_community.embeddings import HuggingFaceEmbeddings

    @lru_cache(maxsize=1)
    def get_embeddings_model():
        return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

from db import QueryCache, SessionLocal

SIMILARITY_THRESHOLD = 0.88


def normalize_query(query: str) -> str:
    return (query or "").lower().strip()


def _deserialize_embedding(blob: bytes) -> np.ndarray:
    """Safe deserialization of embedding vector from raw bytes with legacy pickle fallback."""
    if not blob:
        return None

    # Legacy pickle streams start with protocol byte \x80
    if blob.startswith(b"\x80"):
        try:
            import pickle
            unpickled = pickle.loads(blob)
            return np.asarray(unpickled, dtype=np.float32)
        except Exception:
            pass

    try:
        # Fast path: raw float32 buffer of expected 384 dimensions (1536 bytes)
        arr = np.frombuffer(blob, dtype=np.float32)
        if arr.size == 384:
            return arr
    except Exception:
        pass

    try:
        # Fallback for any other legacy pickled formats
        import pickle
        unpickled = pickle.loads(blob)
        return np.asarray(unpickled, dtype=np.float32)
    except Exception:
        return None


def _cache_scope(document_set_id, prompt_version, model_id):
    if not document_set_id:
        return None
    return {
        "document_set_id": document_set_id,
        "prompt_version": prompt_version or "",
        "model_id": model_id or "",
    }


def get_cached_response(
    query: str,
    document_set_id: str = None,
    prompt_version: str = "",
    model_id: str = "",
):
    scope = _cache_scope(document_set_id, prompt_version, model_id)
    if not scope:
        return None

    db = SessionLocal()
    try:
        embeddings_model = get_embeddings_model()
        query_vec = np.asarray(
            embeddings_model.embed_query(normalize_query(query)),
            dtype=np.float32,
        )
        query_norm = np.linalg.norm(query_vec)
        if query_norm == 0:
            return None

        entries = (
            db.query(QueryCache)
            .filter(
                QueryCache.document_set_id == scope["document_set_id"],
                QueryCache.prompt_version == scope["prompt_version"],
                QueryCache.model_id == scope["model_id"],
            )
            .all()
        )

        if not entries:
            return None

        vectors = []
        valid_entries = []
        for entry in entries:
            vec = _deserialize_embedding(entry.embedding)
            if vec is not None and vec.shape == query_vec.shape:
                vectors.append(vec)
                valid_entries.append(entry)

        if not vectors:
            return None

        # Vectorized batch cosine similarity
        matrix = np.vstack(vectors)
        norms = np.linalg.norm(matrix, axis=1) * query_norm
        norms[norms == 0] = 1e-9
        similarities = np.dot(matrix, query_vec) / norms

        best_idx = int(np.argmax(similarities))
        best_score = float(similarities[best_idx])

        if best_score >= SIMILARITY_THRESHOLD:
            print(f"✅ Semantic DB hit (score={best_score:.2f})")
            return valid_entries[best_idx].response

        print(f"❌ Semantic DB miss (best_score={best_score:.2f}) - calling LLM")
        return None
    finally:
        db.close()


def should_cache_response(response: str) -> bool:
    if not response:
        return False

    stripped = response.strip()
    # Reject error strings or exceptions so temporary failures aren't cached
    if (
        stripped.startswith("⚠️")
        or "error:" in stripped.lower()
        or "exception" in stripped.lower()
        or "traceback" in stripped.lower()
    ):
        return False

    negative_phrases = [
        "couldn't find",
        "no relevant",
        "not found",
        "not present",
        "does not contain",
    ]
    lowered = stripped.lower()
    return bool(lowered) and not any(
        phrase in lowered for phrase in negative_phrases
    )


def save_response(
    query: str,
    response: str,
    document_set_id: str = None,
    prompt_version: str = "",
    model_id: str = "",
):
    scope = _cache_scope(document_set_id, prompt_version, model_id)
    if not scope:
        print("Skipping cache write without document scope")
        return

    if not should_cache_response(response):
        print("Negative, empty, or error response not cached")
        return

    db = SessionLocal()
    try:
        embeddings_model = get_embeddings_model()
        vec = np.asarray(
            embeddings_model.embed_query(normalize_query(query)),
            dtype=np.float32,
        )
        entry = QueryCache(
            query=query,
            response=response,
            embedding=vec.tobytes(),
            document_set_id=scope["document_set_id"],
            prompt_version=scope["prompt_version"],
            model_id=scope["model_id"],
        )
        db.add(entry)
        db.commit()
        print("💾 Saved scoped response + binary embedding to DB")
    finally:
        db.close()
