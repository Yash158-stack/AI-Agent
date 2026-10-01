# app.py
import os
import uuid
import shutil
import atexit
import streamlit as st
from dotenv import load_dotenv

# ---------------- CONFIG & ENVIRONMENT ----------------
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    import google.generativeai as genai
    genai.configure(api_key=GEMINI_API_KEY)

try:
    from config import get_embeddings_model
except ImportError:
    from functools import lru_cache
    from langchain_huggingface import HuggingFaceEmbeddings

    @lru_cache(maxsize=1)
    def get_embeddings_model():
        return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
from ingest import compute_document_set_id, index_files
from chat_engine import handle_conversation
from langchain_community.vectorstores import FAISS

st.set_page_config(page_title="AI Academic Assistant", layout="wide")

# ---------------- SESSION SETUP ----------------
BASE = "user_data"
os.makedirs(BASE, exist_ok=True)

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

SESSION = os.path.join(BASE, st.session_state.session_id)
UPLOAD_DIR = os.path.join(SESSION, "uploads")
FAISS_DIR = os.path.join(SESSION, "faiss_db")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(FAISS_DIR, exist_ok=True)

st.session_state.setdefault("saved_files", [])
st.session_state.setdefault("indexed", False)
st.session_state.setdefault("chat_history", [])
st.session_state.setdefault("pending_button_query", None)
st.session_state.setdefault("pending_button_state", None)
st.session_state.setdefault("indexed_files", [])
st.session_state.setdefault("document_set_id", None)
st.session_state.setdefault("last_index_result", None)


def _safe_upload_name(name):
    return os.path.basename(name).replace("\\", "_").replace("/", "_")


def _cleanup():
    try:
        shutil.rmtree(SESSION)
    except Exception:
        pass


atexit.register(_cleanup)

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.title("Upload Documents")
    files = st.file_uploader(
        "Upload PDF / DOCX / PPTX / Images",
        type=["pdf", "docx", "pptx", "jpg", "png", "jpeg", "webp"],
        accept_multiple_files=True
    )

    if files is not None:
        changed = False
        active_paths = []

        for f in files:
            safe_name = _safe_upload_name(f.name)
            p = os.path.join(UPLOAD_DIR, safe_name)
            active_paths.append(p)
            content = f.getvalue()

            existing = None
            if os.path.exists(p):
                with open(p, "rb") as current:
                    existing = current.read()

            if existing != content:
                with open(p, "wb") as out:
                    out.write(content)
                changed = True

        if set(active_paths) != set(st.session_state.saved_files):
            st.session_state.saved_files = active_paths
            changed = True

        # Indexing / Reset logic
        if not st.session_state.saved_files:
            if st.session_state.indexed or st.session_state.document_set_id:
                shutil.rmtree(FAISS_DIR, ignore_errors=True)
                os.makedirs(FAISS_DIR, exist_ok=True)
                st.session_state.indexed = False
                st.session_state.document_set_id = None
                st.session_state.indexed_files = []
                st.session_state.last_index_result = None
                st.rerun()

        else:
            current_document_set_id = compute_document_set_id(
                st.session_state.saved_files
            )

            if changed or current_document_set_id != st.session_state.document_set_id:
                progress = st.progress(0)
                msg = st.empty()

                def cb(p, t):
                    progress.progress(p)
                    msg.write(t)

                result = index_files(st.session_state.saved_files, FAISS_DIR, cb)
                st.session_state.last_index_result = result
                st.session_state.document_set_id = result.get("document_set_id")
                st.session_state.indexed_files = result.get("files", [])
                st.session_state.indexed = bool(result.get("total_chunks"))
                st.rerun()

# ---------------- RETRIEVER ----------------
def load_retriever():
    idx = os.path.join(FAISS_DIR, "index.faiss")
    if not os.path.exists(idx):
        return None
    if (
        os.path.commonpath([os.path.abspath(idx), os.path.abspath(SESSION)])
        != os.path.abspath(SESSION)
    ):
        return None

    emb = get_embeddings_model()
    db = FAISS.load_local(FAISS_DIR, emb, allow_dangerous_deserialization=True)
    return db.as_retriever(search_kwargs={"k": 10})


retriever = load_retriever() if st.session_state.indexed else None

# ---------------- MAIN UI ----------------
st.title("Ask AI About Your Documents")

if not GEMINI_API_KEY:
    st.error("⚠️ Set GEMINI_API_KEY in your environment or .env before asking questions.")
    st.stop()

if not retriever:
    if st.session_state.saved_files and not st.session_state.indexed:
        st.warning(
            "⚠️ Could not extract readable text from uploaded documents. "
            "If you uploaded scanned images or PDFs, please ensure Tesseract OCR is installed."
        )
    else:
        st.info("Upload documents in the sidebar to get started.")
    st.stop()

# Action Buttons
c1, c2, c3 = st.columns(3)
with c1:
    b1 = st.button("Summarize")
with c2:
    b2 = st.button("Important Questions")
with c3:
    b3 = st.button("Create Notes")

if b1:
    st.session_state.pending_button_query = "summarize the document"
    st.session_state.pending_button_state = {"summary": True}
    st.rerun()

if b2:
    st.session_state.pending_button_query = "give me important questions"
    st.session_state.pending_button_state = {"questions": True}
    st.rerun()

if b3:
    st.session_state.pending_button_query = "create notes"
    st.session_state.pending_button_state = {"notes": True}
    st.rerun()

# Chat Input Handling
query = None
typed = st.chat_input("Ask anything...")

button_state = None
if typed:
    query = typed
elif st.session_state.pending_button_query:
    query = st.session_state.pending_button_query
    button_state = st.session_state.pending_button_state
    st.session_state.pending_button_query = None
    st.session_state.pending_button_state = None

if query:
    with st.spinner("Thinking..."):
        reply, st.session_state.chat_history = handle_conversation(
            query,
            retriever,
            st.session_state.chat_history,
            button_state=button_state,
            document_set_id=st.session_state.document_set_id,
        )
    st.rerun()

# Display Chat History & Associated Images
for role, content in st.session_state.chat_history:
    if isinstance(content, str):
        with st.chat_message("assistant" if "AI" in role else "user"):
            st.write(content)

    elif isinstance(content, dict) and "images" in content:
        raw_paths = content.get("images", [])
        paths = [p for p in raw_paths if p and os.path.exists(p)]
        if paths:
            cols = st.columns(min(3, len(paths)))
            for i, p in enumerate(paths):
                with cols[i % 3]:
                    st.image(p, width=220)
