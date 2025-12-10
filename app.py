# app.py
import os
import uuid
import shutil
import atexit
import time
import streamlit as st
from dotenv import load_dotenv

from ingest import index_files
from chat_engine import handle_conversation
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ---------------- CONFIG ----------------
st.set_page_config(page_title="AI Academic Assistant", layout="wide")
load_dotenv()

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
st.session_state.setdefault("indexed_files", [])

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

    if files:
        new = []
        for f in files:
            p = os.path.join(UPLOAD_DIR, f.name)
            if not os.path.exists(p):
                with open(p, "wb") as out:
                    out.write(f.read())

            if p not in st.session_state.saved_files:
                st.session_state.saved_files.append(p)

            if f.name not in st.session_state.indexed_files:
                new.append(p)

        # indexing
        if new:
            progress = st.progress(0)
            msg = st.empty()

            def cb(p, t):
                progress.progress(p)
                msg.write(t)

            index_files(new, FAISS_DIR, cb)
            st.session_state.indexed_files.extend(
                [os.path.basename(x) for x in new]
            )
            st.session_state.indexed = True
            st.rerun()

# ---------------- RETRIEVER ----------------
def load_retriever():
    idx = os.path.join(FAISS_DIR, "index.faiss")
    if not os.path.exists(idx):
        return None
    emb = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.load_local(FAISS_DIR, emb, allow_dangerous_deserialization=True)
    return db.as_retriever(search_kwargs={"k": 10})

retriever = load_retriever() if st.session_state.indexed else None

# ---------------- BUTTON COOLDOWN ----------------
# Minimal throttle to prevent duplicate clicks
COOLDOWN_SECONDS = int(os.getenv("BUTTON_COOLDOWN", "6"))

if "last_click_time" not in st.session_state:
    st.session_state["last_click_time"] = 0.0

def can_click():
    return (time.time() - st.session_state["last_click_time"]) > COOLDOWN_SECONDS

def register_click():
    st.session_state["last_click_time"] = time.time()

# ---------------- MAIN UI ----------------
st.title("Ask AI About Your Documents")

if not retriever:
    st.info("Upload documents in the sidebar to get started.")
    st.stop()

# Buttons (with cooldown to avoid accidental bursts)
c1, c2, c3 = st.columns(3)
with c1:
    b1 = st.button("Summarize", disabled=not can_click())
with c2:
    b2 = st.button("Important Questions", disabled=not can_click())
with c3:
    b3 = st.button("Create Notes", disabled=not can_click())

if b1 and can_click():
    register_click()
    st.session_state.pending_button_query = "summarize the document"
    st.rerun()

if b2 and can_click():
    register_click()
    st.session_state.pending_button_query = "give me important questions"
    st.rerun()

if b3 and can_click():
    register_click()
    st.session_state.pending_button_query = "create notes"
    st.rerun()

# Chat Input
query = None
typed = st.chat_input("Ask anything...")

if typed:
    query = typed
elif st.session_state.pending_button_query:
    query = st.session_state.pending_button_query
    st.session_state.pending_button_query = None

if query:
    with st.spinner("Thinking..."):
        reply, st.session_state.chat_history = handle_conversation(
            query, retriever, st.session_state.chat_history
        )
    st.rerun()

# Display chat
for role, content in st.session_state.chat_history:
    if isinstance(content, str):
        with st.chat_message("assistant" if "AI" in role else "user"):
            st.write(content)

    elif isinstance(content, dict) and "images" in content:
        paths = content["images"]
        cols = st.columns(min(3, len(paths)))
        for i, p in enumerate(paths):
            with cols[i % 3]:
                st.image(p, width=220)
