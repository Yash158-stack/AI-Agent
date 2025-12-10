# chat_engine.py
import os
from agents.orchestrator import orchestrator
from agents.prompts import GLOBAL_SYSTEM_PROMPT
from groq_llm import generate as groq_generate

def _call_retriever(retriever, query):
    try:
        if hasattr(retriever, "invoke"):
            return retriever.invoke(query)
        if hasattr(retriever, "get_relevant_documents"):
            return retriever.get_relevant_documents(query)
        if hasattr(retriever, "similarity_search"):
            return retriever.similarity_search(query, k=10)
    except Exception:
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

    # Try orchestrator first (existing agent flow). If it fails, fallback to direct LLM call.
    output = ""
    agent_name = "Orchestrator"
    try:
        result = orchestrator(user_query, enhanced, button_state=button_state)
        output = result.get("output", "")
        agent_name = result.get("agent", "Orchestrator")
    except Exception as e:
        # fallback to Groq LLM
        prompt = f"{enhanced}\n\nUser: {user_query}\nAssistant:"
        llm_resp = groq_generate(prompt, max_tokens=512, temperature=0.0)
        output = llm_resp.get("text", f"[groq error: {str(e)}]")
        agent_name = "Groq-LLM"

    chat_history.append(("You", user_query))
    chat_history.append((f"AI ({agent_name})", output))

    if images:
        chat_history.append(("AI (Images)", {"images": images}))

    return output, chat_history
