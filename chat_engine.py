# chat_engine.py
import os
from agents.orchestrator import orchestrator
from agents.prompts import GLOBAL_SYSTEM_PROMPT
from cache import get_cached_response, save_response

try:
    from config import DEFAULT_GEMINI_MODEL
except ImportError:
    DEFAULT_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

PROMPT_VERSION = "rag-v3-history-grounded"
MODEL_ID = DEFAULT_GEMINI_MODEL


def _call_retriever(retriever, query):
    try:
        if hasattr(retriever, "invoke"):
            return retriever.invoke(query)
        if hasattr(retriever, "get_relevant_documents"):
            return retriever.get_relevant_documents(query)
        if hasattr(retriever, "similarity_search"):
            return retriever.similarity_search(query, k=10)
    except Exception as e:
        print(f"⚠️ Retriever error: {e}")
        return []
    return []


def _citation_label(metadata):
    source = metadata.get("source_name") or metadata.get("source") or "uploaded document"
    chunk = metadata.get("chunk_index")
    if chunk:
        return f"{source}, chunk {chunk}"
    return source


def extract_context_and_images(docs):
    text_parts = []
    images = []
    citations = []
    seen_citations = set()
    seen_images = set()

    for doc in docs:
        metadata = getattr(doc, "metadata", {}) or {}
        label = _citation_label(metadata)
        text = getattr(doc, "page_content", "")

        if text:
            text_parts.append(f"[Source: {label}]\n{text}")
            if label not in seen_citations:
                citations.append(label)
                seen_citations.add(label)

        image_paths = metadata.get("image_paths") or metadata.get("image_path")
        if image_paths:
            if isinstance(image_paths, list):
                for p in image_paths:
                    if p and p not in seen_images and os.path.exists(p):
                        images.append(p)
                        seen_images.add(p)
            elif image_paths not in seen_images and os.path.exists(image_paths):
                images.append(image_paths)
                seen_images.add(image_paths)

    return "\n\n".join(text_parts), images, citations


def _append_sources(output, citations):
    if not citations or "sources used:" in output.lower():
        return output

    rendered = "\n".join(f"- {citation}" for citation in citations[:8])
    return f"{output.rstrip()}\n\nSources used:\n{rendered}"


def _format_chat_history(chat_history, max_turns=6):
    if not chat_history:
        return ""
    recent = chat_history[-max_turns:]
    lines = []
    for role, content in recent:
        if isinstance(content, str):
            clean_role = "User" if "You" in role else "Assistant"
            lines.append(f"{clean_role}: {content.strip()}")
    if not lines:
        return ""
    return "\n".join(lines)


def handle_conversation(
    user_query,
    retriever,
    chat_history,
    button_state=None,
    document_set_id=None,
):
    cached = get_cached_response(
        user_query,
        document_set_id=document_set_id,
        prompt_version=PROMPT_VERSION,
        model_id=MODEL_ID,
    )
    if cached:
        chat_history.append(("You", user_query))
        chat_history.append(("AI (Cached)", cached))
        return cached, chat_history

    docs = _call_retriever(retriever, user_query)
    context_text, images, citations = extract_context_and_images(docs)

    history_text = _format_chat_history(chat_history, max_turns=6)
    context_blocks = []
    if history_text:
        context_blocks.append(f"=== CONVERSATION HISTORY ===\n{history_text}")
    if context_text:
        context_blocks.append(f"=== DOCUMENT CONTEXT ===\n{context_text}")

    enhanced = f"{GLOBAL_SYSTEM_PROMPT}\n\n" + "\n\n".join(context_blocks)

    try:
        result = orchestrator(user_query, enhanced, button_state=button_state)
    except Exception as e:
        result = {"agent": "System", "output": f"⚠️ Orchestrator error: {e}"}

    output = _append_sources(result.get("output", ""), citations)

    save_response(
        user_query,
        output,
        document_set_id=document_set_id,
        prompt_version=PROMPT_VERSION,
        model_id=MODEL_ID,
    )

    chat_history.append(("You", user_query))
    chat_history.append((f"AI ({result.get('agent', 'Agent')})", output))

    if images:
        chat_history.append(("AI (Images)", {"images": images}))

    return output, chat_history


def stream_text_chunks(text: str, chunk_size: int = 4):
    """Yield word/token chunks for smooth streaming in Streamlit UI."""
    words = text.split(" ")
    for i in range(0, len(words), chunk_size):
        yield " ".join(words[i:i + chunk_size]) + (" " if i + chunk_size < len(words) else "")

