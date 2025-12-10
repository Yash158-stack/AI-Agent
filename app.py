# app.py
# --------------------------
# CLEAN, WORKING, STABLE APP WITH BUTTON LOGIC + IMAGE RENDERING
# --------------------------

import os
import uuid
import shutil
import atexit
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai

from ingest import index_files
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from chat_engine import handle_conversation


# ----------------- Config -----------------
st.set_page_config(page_title="AI Academic Assistant", layout="wide")
load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# ----------------- Session -----------------
BASE_USER_DIR = "user_data"
os.makedirs(BASE_USER_DIR, exist_ok=True)

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

SESSION_DIR = os.path.join(BASE_USER_DIR, st.session_state.session_id)
UPLOAD_DIR = os.path.join(SESSION_DIR, "uploads")
FAISS_DIR = os.path.join(SESSION_DIR, "faiss_db")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(FAISS_DIR, exist_ok=True)

# session vars
st.session_state.setdefault("saved_files", [])
st.session_state.setdefault("indexed", False)
st.session_state.setdefault("chat_history", [])
st.session_state.setdefault("pending_button_query", None)
st.session_state.setdefault("indexed_files", [])


# ----------------- Cleanup -----------------
def _atexit_cleanup():
    try:
        shutil.rmtree(SESSION_DIR)
    except:
        pass

atexit.register(_atexit_cleanup)


with st.sidebar:
    st.title("AI Academic Assistant")

    uploaded_files = st.file_uploader(
        "Upload PDF/DOCX/PPTX/IMAGE",
        type=["pdf", "docx", "pptx", "jpg", "png", "jpeg", "webp"],
        accept_multiple_files=True
    )

    if uploaded_files:
        new_files = []

        for f in uploaded_files:
            path = os.path.join(UPLOAD_DIR, f.name)

            # Save file
            if not os.path.exists(path):
                with open(path, "wb") as out:
                    out.write(f.read())

            # Add to saved_files (needed for state)
            if path not in st.session_state.saved_files:
                st.session_state.saved_files.append(path)

            # Mark new for indexing
            if f.name not in st.session_state.indexed_files:
                new_files.append(path)

        # If any new files → index them
        if new_files:
            progress = st.progress(0)
            status = st.empty()

            def progress_callback(p, msg):
                progress.progress(p)
                status.write(msg)

            index_files(new_files, FAISS_DIR, progress_callback)

            # Mark indexed
            st.session_state.indexed_files.extend([os.path.basename(p) for p in new_files])
            st.session_state.indexed = True   # 🔥 VERY IMPORTANT

            st.rerun()

# ----------------- Load Retriever -----------------
def load_retriever():
    idx = os.path.join(FAISS_DIR, "index.faiss")
    if not os.path.exists(idx):
        return None

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.load_local(FAISS_DIR, embeddings, allow_dangerous_deserialization=True)
    return db.as_retriever(search_kwargs={"k": 10})


retriever = load_retriever() if st.session_state.indexed else None


# ----------------- Main UI -----------------
st.title("Ask AI About Your Documents")

col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    btn_summary = st.button("Summarize Document")
with col2:
    btn_questions = st.button("Important Questions")
with col3:
    btn_notes = st.button("Create Notes")


# --- Button → pending query mapping ---
if btn_summary:
    st.session_state.pending_button_query = "summarize the document"
    st.rerun()

if btn_questions:
    st.session_state.pending_button_query = "give me important questions"
    st.rerun()

if btn_notes:
    st.session_state.pending_button_query = "create notes"
    st.rerun()


# ----------------- Chat Input -----------------
query = None
button_query = st.session_state.pending_button_query

if retriever:

    user_text = st.chat_input("Type your question...")

    # User typed input
    if user_text:
        query = user_text

    # Button clicked previously
    elif button_query:
        query = button_query
        st.session_state.pending_button_query = None


    # If we now have a query → process it
    if query:

        if query.lower() in ["exit", "quit", "bye"]:
            st.session_state.chat_history.clear()
            st.rerun()

        with st.spinner("Thinking..."):
            reply, st.session_state.chat_history = handle_conversation(
                query, retriever, st.session_state.chat_history
            )

        st.rerun()


# ----------------- DISPLAY CHAT (AFTER processing) -----------------
    for role, content in st.session_state.chat_history:
    
        if isinstance(content, str):
            with st.chat_message("assistant" if role != "You" else "user"):
                st.write(content)
    
        elif isinstance(content, dict) and "images" in content:
        
            # Show images as clean columns (NO expanders, NO popups)
            img_paths = content["images"]
    
            n = len(img_paths)
            cols = st.columns(min(n, 3))  # max 3 per row
    
            row_i = 0
            for idx, img_path in enumerate(img_paths):
                with cols[idx % 3]:
                    try:
                        st.image(img_path, width=250)
                    except:
                        st.write(f"⚠️ Could not display: {img_path}")
    
                # New row every 3 images
                if (idx % 3) == 2 and idx < n - 1:
                    cols = st.columns(min(n - idx - 1, 3))
    
        else:
            with st.chat_message("assistant" if role != "You" else "user"):
                st.write(repr(content))
    