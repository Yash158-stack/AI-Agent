# chat_engine.py
import os
import google.generativeai as genai
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
    except Exception:
        pass
    return []

def extract_context_and_images(docs):
    """
    Return (combined_text, list_of_image_paths).
    Each doc is expected to have `page_content` and optional `metadata` with 'image_paths'.
    """
    combined_parts = []
    image_files = []

    for d in docs:
        # collect text
        text = getattr(d, "page_content", None)
        if text:
            combined_parts.append(text)

        # collect image paths if present in metadata
        meta = getattr(d, "metadata", None) or {}
        if isinstance(meta, dict):
            ip = meta.get("image_paths") or meta.get("image_path") or meta.get("image")
            if ip:
                # ip may be a single path or a list
                if isinstance(ip, (list, tuple)):
                    image_files.extend(ip)
                else:
                    image_files.append(ip)

    combined_text = "\n\n".join(combined_parts)
    # dedupe and keep existing files only
    seen = []
    valid_images = []
    for p in image_files:
        if p and p not in seen and os.path.exists(p):
            seen.append(p)
            valid_images.append(p)
    return combined_text, valid_images

def _user_wants_images(user_query: str):
    """
    Heuristic: detect if the user is explicitly asking to see or explain images.
    """
    if not user_query:
        return False
    q = user_query.lower()
    keywords = [
        "image", "images", "picture", "photo", "screenshot", "diagram", "figure",
        "show me", "display", "what does the image", "what does the screenshot",
        "show the image", "show the screenshot", "show screenshots", "show pictures",
        "visual", "visualize"
    ]
    return any(k in q for k in keywords)

def handle_conversation(user_query, retriever, chat_history, button_state=None):
    """
    - retrieve docs (robust)
    - extract combined text + image paths
    - call orchestrator (which runs the appropriate agent)
    - append assistant reply to history
    - if the user requested images (or we detect image-related question) append image entries to history
    """
    # ---------- 1. RETRIEVE CONTEXT ----------
    docs = _call_retriever(retriever, user_query)
    if not docs:
        docs = _call_retriever(retriever, user_query.strip())

    context_text, image_files = extract_context_and_images(docs)

    # Attach global reasoning prompt as prefix and include a short image list hint
    image_hint = ""
    if image_files:
        image_hint = "\n\n=== EXTRACTED IMAGE PATHS ===\n" + "\n".join(image_files)

    enhanced_context = f"{GLOBAL_SYSTEM_PROMPT}\n\n=== DOCUMENT CONTEXT ===\n{context_text}{image_hint}"

    # ---------- 2. CALL ORCHESTRATOR ----------
    result = orchestrator(
        user_query,
        enhanced_context,
        button_state=button_state
    )
    # result expected: {"agent": "...", "output": "..."}

    # ---------- 3. FALLBACK ----------
    if (not context_text.strip()) and result.get("agent") in ["QAAgent", "Summarizer", "NotesAgent"]:
        fallback = (
            "I couldn't find relevant info in the uploaded documents.\n\n"
            "Try asking something directly related to the document content."
        )
        result["output"] = fallback

    # ---------- 4. UPDATE CHAT HISTORY ----------
    # keep older tuple format (role, text) — text can be string or dict
    chat_history.append(("You", user_query if user_query else "(button)"))

    assistant_text = result.get("output", "")
    chat_history.append((f"AI ({result.get('agent','Agent')})", assistant_text))

    # ---------- 5. APPEND IMAGES WHEN APPROPRIATE ----------
    # Append images as a separate history entry with a dict payload so app.py can render them.
    # Conditions to append:
    #  - image files exist AND
    #  - user explicitly asked to see/explain images OR the agent's output contains an explicit image instruction
    append_images = False

    if image_files:
        if _user_wants_images(user_query):
            append_images = True
        else:
            # also check if the agent response mentions images in a way that implies user intent
            out_lower = (assistant_text or "").lower()
            if any(tok in out_lower for tok in ["see the image", "see below image", "image(s) below", "refer to the image", "the image shows"]):
                append_images = True

    if append_images and image_files:
        # append as a dict so app.py can detect & render
        chat_history.append((f"AI (Images)", {"images": image_files}))

    return result.get("output",""), chat_history
