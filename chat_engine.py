# chat_engine.py
import os
from agents.orchestrator import orchestrator
from agents.prompts import GLOBAL_SYSTEM_PROMPT

def _call_retriever(retriever, query):
    try:
        if hasattr(retriever, "invoke"):
            return retriever.invoke(query)
        if hasattr(retriever, "get_relevant_documents"):
            return retriever.get_relevant_documents(query)
        if hasattr(retriever, "similarity_search"):
            return retriever.similarity_search(query, k=10)
    except:
        return []
    return []

def extract_context_and_images(docs):
    text_parts = []
    images = []

    for d in docs:
        t = getattr(d, "page_content", "")
        if t:
            text_parts.append(t)

        meta = getattr(d, "metadata", {}) or {}
        img = meta.get("image_paths") or meta.get("image_path")
        if img:
            if isinstance(img, list):
                images.extend(img)
            else:
                images.append(img)

    return "\n\n".join(text_parts), images

def handle_conversation(user_query, retriever, chat_history, button_state=None):

    docs = _call_retriever(retriever, user_query)
    context_text, images = extract_context_and_images(docs)

    enhanced = (
        f"{GLOBAL_SYSTEM_PROMPT}\n\n=== DOCUMENT CONTEXT ===\n{context_text}"
    )

    result = orchestrator(user_query, enhanced, button_state=button_state)
    output = result.get("output", "")

    chat_history.append(("You", user_query))
    chat_history.append((f"AI ({result.get('agent','Agent')})", output))

    if images:
        chat_history.append(("AI (Images)", {"images": images}))

    return output, chat_history
