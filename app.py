# app.py
"""
LearnAssist Academic Studio: Startup-Grade SaaS UI/UX
Features:
- Academic Slate/Indigo Glassmorphic Design System
- Onboarding Hero with 1-Click Pre-packaged ML Study Deck
- Persistent Sidebar Knowledge Base HUD & Telemetry
- 4-Tab Multi-Mode Studio:
    Tab 1: Academic Copilot (grounded conversational chat & citation chips)
    Tab 2: Smart Notes Studio (isolated reader, TOC, markdown download)
    Tab 3: Active Recall Lab (interactive MCQs, flip flashcards, essay rubrics)
    Tab 4: Document & Evidence Inspector (chunk keyword search & diagram gallery)
"""
import os
import re
import uuid
import shutil
import atexit
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

# ---------------- CONFIG & ENVIRONMENT ----------------
load_dotenv()

st.set_page_config(
    page_title="LearnAssist • Academic Studio",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject SaaS Design System CSS
from assets.styles import inject_custom_css, agent_badge_html, citation_chips_html
inject_custom_css()

try:
    from config import get_embeddings_model, DEFAULT_GEMINI_MODEL
except ImportError:
    from functools import lru_cache
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError:
        from langchain_community.embeddings import HuggingFaceEmbeddings

    DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"

    @lru_cache(maxsize=1)
    def get_embeddings_model():
        return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

from ingest import compute_document_set_id, index_files, text_splitting_recursive, _base_metadata
from chat_engine import handle_conversation, stream_text_chunks, extract_context_and_images
from langchain_community.vectorstores import FAISS
from sample_data import load_sample_deck, SAMPLE_DECK_FILENAME
from quiz_parser import parse_quiz_output
from agents.summary_agent import SummaryAgent
from agents.notes_agent import NotesAgent
from agents.question_agent import QuestionAgent

try:
    import pytesseract
except ImportError:
    pytesseract = None

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

# State Defaults
st.session_state.setdefault("saved_files", [])
st.session_state.setdefault("indexed", False)
st.session_state.setdefault("chat_history", [])
st.session_state.setdefault("indexed_files", [])
st.session_state.setdefault("document_set_id", None)
st.session_state.setdefault("last_index_result", None)
st.session_state.setdefault("notes_content", None)
st.session_state.setdefault("notes_summary", None)
st.session_state.setdefault("notes_type", "Comprehensive Notes")
st.session_state.setdefault("quiz_data", None)
st.session_state.setdefault("quiz_raw", None)
st.session_state.setdefault("quiz_answers", {})
st.session_state.setdefault("quiz_submitted", False)
st.session_state.setdefault("flashcard_index", 0)
st.session_state.setdefault("flashcard_flipped", False)
st.session_state.setdefault("mastered_cards", set())
st.session_state.setdefault("active_chunks", [])
st.session_state.setdefault("copilot_seed_prompt", None)
st.session_state.setdefault("cache_hits", 0)
st.session_state.setdefault("gemini_api_key", os.getenv("GEMINI_API_KEY", ""))


def _safe_upload_name(name):
    return os.path.basename(name).replace("\\", "_").replace("/", "_")


def _clear_mcq_widget_states():
    """Clear Streamlit radio widget keys so re-taking or loading new quizzes doesn't retain old selections."""
    for k in list(st.session_state.keys()):
        if k.startswith("mcq_radio_"):
            del st.session_state[k]


def _activate_deck(res):
    """Activate a newly loaded or indexed study deck with clean state."""
    _clear_mcq_widget_states()
    st.session_state.saved_files = [res["file_path"]]
    st.session_state.indexed_files = [res["filename"]]
    st.session_state.document_set_id = res["document_set_id"]
    st.session_state.indexed = True
    st.session_state.notes_content = res.get("notes")
    st.session_state.notes_summary = res.get("summary")
    st.session_state.quiz_data = res.get("quiz_data")
    st.session_state.quiz_raw = res.get("quiz_raw")
    st.session_state.quiz_answers = {}
    st.session_state.quiz_submitted = False
    st.session_state.flashcard_index = 0
    st.session_state.flashcard_flipped = False
    st.session_state.mastered_cards = set()
    st.session_state.copilot_seed_prompt = None
    st.session_state.active_chunks = res.get("chunks", [])
    st.session_state.last_index_result = {
        "path": FAISS_DIR,
        "total_chunks": res.get("total_chunks", 0),
        "files_indexed": 1,
        "document_set_id": res["document_set_id"],
        "files": [res["filename"]],
    }


def _clear_deck():
    """Reset the entire active deck and session states cleanly."""
    _clear_mcq_widget_states()
    shutil.rmtree(FAISS_DIR, ignore_errors=True)
    shutil.rmtree(UPLOAD_DIR, ignore_errors=True)
    os.makedirs(FAISS_DIR, exist_ok=True)
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    st.session_state.saved_files = []
    st.session_state.indexed = False
    st.session_state.document_set_id = None
    st.session_state.indexed_files = []
    st.session_state.last_index_result = None
    st.session_state.notes_content = None
    st.session_state.notes_summary = None
    st.session_state.quiz_data = None
    st.session_state.quiz_raw = None
    st.session_state.quiz_answers = {}
    st.session_state.quiz_submitted = False
    st.session_state.flashcard_index = 0
    st.session_state.flashcard_flipped = False
    st.session_state.mastered_cards = set()
    st.session_state.copilot_seed_prompt = None
    st.session_state.chat_history = []
    st.session_state.active_chunks = []


def _cleanup():
    try:
        shutil.rmtree(SESSION)
    except Exception:
        pass


atexit.register(_cleanup)

# Configure Gemini
api_key = st.session_state.get("gemini_api_key") or os.getenv("GEMINI_API_KEY")
if api_key:
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
    except Exception:
        pass


def load_retriever():
    idx = os.path.join(FAISS_DIR, "index.faiss")
    if not os.path.exists(idx):
        return None
    try:
        emb = get_embeddings_model()
        db = FAISS.load_local(FAISS_DIR, emb, allow_dangerous_deserialization=True)
        return db.as_retriever(search_kwargs={"k": 8})
    except Exception as e:
        print(f"⚠️ Error loading retriever: {e}")
        return None


def get_all_indexed_documents():
    """Retrieve all chunk Documents from the local FAISS index for inspector."""
    idx = os.path.join(FAISS_DIR, "index.faiss")
    if not os.path.exists(idx):
        return []
    try:
        emb = get_embeddings_model()
        db = FAISS.load_local(FAISS_DIR, emb, allow_dangerous_deserialization=True)
        return list(db.docstore._dict.values())
    except Exception:
        return []


def get_deck_context(retriever, query="main concepts, principles, definitions and core methodology"):
    """Retrieve grounded context for generating study notes or quizzes."""
    if not retriever:
        return ""
    try:
        docs = retriever.invoke(query) if hasattr(retriever, "invoke") else retriever.get_relevant_documents(query)
        context_text, _, _ = extract_context_and_images(docs)
        return context_text
    except Exception:
        return ""


# ---------------- SIDEBAR HUD & KNOWLEDGE BASE ----------------
with st.sidebar:
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 14px;">
        <span style="font-size: 1.8rem;">🎓</span>
        <div>
            <div style="font-weight: 800; font-size: 1.15rem; color: #FFFFFF; letter-spacing: -0.01em;">LearnAssist</div>
            <div style="font-size: 0.72rem; color: #818CF8; font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em;">Academic Studio v2.0</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 1. Knowledge Base HUD Card
    st.markdown('<div class="saas-card" style="padding: 14px 16px; margin-bottom: 12px;">', unsafe_allow_html=True)
    status_color = "#10B981" if st.session_state.indexed else "#94A3B8"
    status_label = "🟢 Deck Active" if st.session_state.indexed else "⚪ No Deck Loaded"
    st.markdown(f'<div style="font-size: 0.8rem; font-weight: 700; color: {status_color}; margin-bottom: 6px;">{status_label}</div>', unsafe_allow_html=True)

    total_files = len(st.session_state.saved_files)
    file_label = f"{total_files} file{'s' if total_files != 1 else ''}"
    total_chunks = st.session_state.last_index_result.get("total_chunks", 0) if st.session_state.last_index_result else len(st.session_state.active_chunks)

    col_h1, col_h2 = st.columns(2)
    with col_h1:
        st.markdown(f'<div class="hud-metric-label">Sources</div><div class="hud-metric-val">{total_files}</div>', unsafe_allow_html=True)
    with col_h2:
        st.markdown(f'<div class="hud-metric-label">Chunks</div><div class="hud-metric-val">{total_chunks}</div>', unsafe_allow_html=True)

    # File Badges
    if st.session_state.saved_files:
        st.markdown('<div style="margin-top: 8px;">', unsafe_allow_html=True)
        for p in st.session_state.saved_files[:4]:
            fn = os.path.basename(p)
            ext = os.path.splitext(fn)[1].lower().replace(".", "").upper() or "DOC"
            st.markdown(f'<span class="citation-chip" style="font-size: 0.7rem; margin-bottom: 4px;">🏷️ {ext} | {fn[:18]}..</span>', unsafe_allow_html=True)
        if len(st.session_state.saved_files) > 4:
            st.markdown(f'<span style="font-size: 0.72rem; color: #94A3B8;">+{len(st.session_state.saved_files)-4} more</span>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # 2. Performance & Telemetry HUD Card
    st.markdown('<div class="saas-card" style="padding: 14px 16px; margin-bottom: 12px;">', unsafe_allow_html=True)
    st.markdown('<div class="hud-metric-label" style="margin-bottom: 6px;">⚡ Telemetry & Acceleration</div>', unsafe_allow_html=True)
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.markdown(f'<div style="font-size: 0.7rem; color: #94A3B8;">Cache Hits</div><div style="font-size: 1.15rem; font-weight: 700; color: #67E8F9;">{st.session_state.cache_hits}</div>', unsafe_allow_html=True)
    with col_t2:
        ocr_label = "Tesseract" if pytesseract else "Native"
        st.markdown(f'<div style="font-size: 0.7rem; color: #94A3B8;">OCR Engine</div><div style="font-size: 0.85rem; font-weight: 700; color: #A5B4FC; margin-top: 4px;">{ocr_label}</div>', unsafe_allow_html=True)
    st.markdown(f'<div style="font-size: 0.72rem; color: #64748B; margin-top: 6px;">Model: {DEFAULT_GEMINI_MODEL} (384-dim)</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # 3. File Ingestion
    st.markdown('<div style="font-weight: 700; font-size: 0.85rem; color: #E2E8F0; margin-bottom: 6px;">📂 Upload Study Materials</div>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "PDF, DOCX, PPTX, TXT, or Images",
        type=["pdf", "docx", "pptx", "txt", "md", "jpg", "png", "jpeg", "webp"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        active_paths = []
        changed = False
        for f in uploaded_files:
            safe_name = _safe_upload_name(f.name)
            p = os.path.join(UPLOAD_DIR, safe_name)
            active_paths.append(p)
            content = f.getvalue()
            existing = None
            if os.path.exists(p):
                with open(p, "rb") as cf:
                    existing = cf.read()
            if existing != content:
                with open(p, "wb") as out:
                    out.write(content)
                changed = True

        if set(active_paths) != set(st.session_state.saved_files):
            st.session_state.saved_files = active_paths
            changed = True

        if st.session_state.saved_files:
            current_doc_set_id = compute_document_set_id(st.session_state.saved_files)
            if changed or current_doc_set_id != st.session_state.document_set_id:
                prog = st.progress(0)
                status_txt = st.empty()

                def _cb(val, msg):
                    prog.progress(val)
                    status_txt.markdown(f'<span style="font-size: 0.8rem; color: #A5B4FC;">{msg}</span>', unsafe_allow_html=True)

                res = index_files(st.session_state.saved_files, FAISS_DIR, _cb)
                st.session_state.last_index_result = res
                st.session_state.document_set_id = res.get("document_set_id")
                st.session_state.indexed_files = res.get("files", [])
                st.session_state.indexed = bool(res.get("total_chunks"))
                # Pre-populate active chunks
                all_docs = get_all_indexed_documents()
                st.session_state.active_chunks = [d.page_content for d in all_docs]
                st.rerun()

    # Quick Action Buttons
    st.markdown('<div style="margin-top: 14px;">', unsafe_allow_html=True)
    if st.button("🚀 Load Sample ML Deck", use_container_width=True):
        with st.spinner("Loading Machine Learning Fundamentals study deck..."):
            res = load_sample_deck(UPLOAD_DIR, FAISS_DIR, get_embeddings_model)
            _activate_deck(res)
            st.rerun()

    if st.button("🗑️ Reset / Clear Deck", use_container_width=True):
        _clear_deck()
        st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

# Check active retriever
retriever = load_retriever() if st.session_state.indexed else None


# ---------------- ONBOARDING HERO (WHEN NOT INDEXED) ----------------
if not st.session_state.indexed or not retriever:
    st.markdown("""
    <div class="hero-container">
        <div class="hero-badge">🚀 Academic Studio 2.0 • Autonomous Multi-Agent</div>
        <div class="hero-title">LearnAssist Academic Studio</div>
        <div class="hero-subtitle">
            Transform dense lecture slides, textbooks, and research papers into interactive study workspaces: 
            grounded conversational Q&A, structured notes, and gamified active recall exams.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # In-App API Key Configuration Drawer
    if not api_key:
        with st.expander("🔑 Configure Google Gemini API Key (Required for AI Agents)", expanded=True):
            st.markdown(
                "A Gemini API key is required to power multi-agent reasoning and grounded conversational chat. "
                "You can generate a free API key instantly in [Google AI Studio](https://aistudio.google.com/app/apikey)."
            )
            col_k1, col_k2 = st.columns([4, 1])
            with col_k1:
                entered_key = st.text_input("Enter GEMINI_API_KEY", type="password", key="api_key_drawer_input", label_visibility="collapsed")
            with col_k2:
                if st.button("Save Key", use_container_width=True):
                    if entered_key.strip():
                        st.session_state.gemini_api_key = entered_key.strip()
                        try:
                            import google.generativeai as genai
                            genai.configure(api_key=entered_key.strip())
                            st.success("API Key configured for this session!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error configuring Gemini: {e}")

    # Feature Grid
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">📚</div>
            <div class="feature-title">Multi-Format RAG Studio</div>
            <div class="feature-desc">
                Ingest PDF textbooks, PPTX slides with pinned figures, DOCX notes, and scanned diagrams with local OCR.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_f2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🧠</div>
            <div class="feature-title">Autonomous Agents</div>
            <div class="feature-desc">
                Specialized agents synthesize executive summaries, comprehensive study notes, and active recall question banks.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_f3:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">⚡</div>
            <div class="feature-title">Semantic Vector Cache</div>
            <div class="feature-desc">
                Instant sub-second response retrieval via local SQLite embedding similarity, saving API quota and latency.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div style="text-align: center; margin-top: 32px;">', unsafe_allow_html=True)
    col_c1, col_c2, col_c3 = st.columns([1, 2, 1])
    with col_c2:
        if st.button("🚀 Explore with 1-Click Sample Deck (Machine Learning)", use_container_width=True, type="primary"):
            with st.spinner("Synthesizing Machine Learning Fundamentals study deck..."):
                res = load_sample_deck(UPLOAD_DIR, FAISS_DIR, get_embeddings_model)
                _activate_deck(res)
                st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    st.info("💡 Or drop your lecture slides, notes, or textbook PDFs into the sidebar upload area to build your custom deck.")
    st.stop()


# ---------------- STUDIO WORKSPACE (INDEXED STATE) ----------------

# Top Navigation Action Bar
col_top_title, col_btn1, col_btn2, col_btn3 = st.columns([5, 2, 2, 2])
with col_top_title:
    st.markdown("""
    <div style="display: flex; align-items: baseline; gap: 8px;">
        <span style="font-size: 1.5rem; font-weight: 800; color: #FFFFFF;">Academic Studio</span>
        <span style="font-size: 0.85rem; color: #94A3B8;">• Grounded Study Workspace</span>
    </div>
    """, unsafe_allow_html=True)

with col_btn1:
    if st.button("📊 Summarize Deck", use_container_width=True):
        if not api_key:
            st.error("⚠️ Please configure GEMINI_API_KEY to synthesize summaries.")
        else:
            with st.spinner("Synthesizing executive summary..."):
                ctx = get_deck_context(retriever, "comprehensive summary of main ideas and conclusions")
                res = SummaryAgent.run("provide an executive summary of the document", ctx)
                st.session_state.notes_summary = res.get("output", "")
                st.session_state.notes_content = st.session_state.notes_summary
                st.session_state.notes_type = "Executive Summary"
                st.toast("✨ Executive Summary generated in Tab 2: Smart Notes Studio!", icon="📊")

with col_btn2:
    if st.button("📝 Create Notes", use_container_width=True):
        if not api_key:
            st.error("⚠️ Please configure GEMINI_API_KEY to generate notes.")
        else:
            with st.spinner("Generating structured study notes..."):
                ctx = get_deck_context(retriever, "comprehensive study notes definitions formulas key points")
                res = NotesAgent.run("create comprehensive study notes with headings and tables", ctx)
                st.session_state.notes_content = res.get("output", "")
                st.session_state.notes_type = "Comprehensive Notes"
                st.toast("📝 Comprehensive Notes generated in Tab 2: Smart Notes Studio!", icon="✨")

with col_btn3:
    if st.button("🧠 Generate Quiz Bank", use_container_width=True):
        if not api_key:
            st.error("⚠️ Please configure GEMINI_API_KEY to generate questions.")
        else:
            with st.spinner("Generating Active Recall Question Bank..."):
                ctx = get_deck_context(retriever, "important exam questions short long mcqs with options and answers")
                res = QuestionAgent.run("generate 5 short questions, 5 long questions, and 5 MCQs", ctx)
                raw_out = res.get("output", "")
                _clear_mcq_widget_states()
                st.session_state.quiz_raw = raw_out
                st.session_state.quiz_data = parse_quiz_output(raw_out)
                st.session_state.quiz_answers = {}
                st.session_state.quiz_submitted = False
                st.session_state.flashcard_index = 0
                st.session_state.flashcard_flipped = False
                st.session_state.mastered_cards = set()
                st.toast("🧠 Active Recall Lab populated in Tab 3!", icon="🎯")

# 4 Main Studio Tabs
tab_copilot, tab_notes, tab_quiz, tab_inspector = st.tabs([
    "💬 Academic Copilot",
    "📝 Smart Notes Studio",
    "🧠 Active Recall Lab",
    "🔍 Document & Evidence Inspector"
])


# ============================================================
# TAB 1: ACADEMIC COPILOT
# ============================================================
with tab_copilot:
    # Seed prompt triggers from other tabs
    pending_prompt = st.session_state.copilot_seed_prompt
    if pending_prompt:
        st.session_state.copilot_seed_prompt = None

    # Suggested Prompts (if chat is fresh)
    if not st.session_state.chat_history:
        st.markdown('<div style="margin-bottom: 12px; font-size: 0.85rem; color: #94A3B8;">Suggested questions to explore this deck:</div>', unsafe_allow_html=True)
        col_p1, col_p2, col_p3 = st.columns(3)
        with col_p1:
            if st.button("💡 Explain core methodology & principles", use_container_width=True):
                pending_prompt = "Explain the core methodology and theoretical principles discussed in these documents."
        with col_p2:
            if st.button("⚖️ Compare key algorithms and tradeoffs", use_container_width=True):
                pending_prompt = "Compare the key algorithms, techniques, and their practical tradeoffs."
        with col_p3:
            if st.button("📈 What evaluation metrics are critical?", use_container_width=True):
                pending_prompt = "What evaluation metrics and diagnostic curves are most critical and why?"

    # Display Conversational History
    for role, content in st.session_state.chat_history:
        if isinstance(content, str):
            is_assistant = "AI" in role
            with st.chat_message("assistant" if is_assistant else "user"):
                if is_assistant:
                    agent_name = role.replace("AI (", "").replace(")", "").strip()
                    st.markdown(agent_badge_html(agent_name), unsafe_allow_html=True)

                    # Extract and separate citations if present (case-insensitive regex split)
                    parts = re.split(r"sources\s+used\s*:", content, maxsplit=1, flags=re.IGNORECASE)
                    if len(parts) == 2:
                        main_body, sources_part = parts
                        st.markdown(main_body.strip())
                        sources_lines = [line.strip().lstrip("-* ") for line in sources_part.strip().split("\n") if line.strip()]
                        st.markdown(citation_chips_html(sources_lines), unsafe_allow_html=True)
                    else:
                        st.markdown(content)
                else:
                    st.markdown(content)

        elif isinstance(content, dict) and "images" in content:
            raw_paths = content.get("images", [])
            valid_paths = [p for p in raw_paths if p and os.path.exists(p)]
            if valid_paths:
                st.markdown('<div style="font-size: 0.8rem; color: #818CF8; font-weight: 600; margin: 8px 0;">🖼️ Relevant Figures Retrieved:</div>', unsafe_allow_html=True)
                cols = st.columns(min(3, len(valid_paths)))
                for i, p in enumerate(valid_paths):
                    with cols[i % 3]:
                        st.image(p, caption=os.path.basename(p), use_container_width=True)

    # Chat Input Box
    user_query = st.chat_input("Ask a grounded question about your study deck...")
    active_query = user_query or pending_prompt

    if active_query:
        if not api_key:
            st.error("⚠️ Please configure your GEMINI_API_KEY to query the Copilot.")
        else:
            with st.chat_message("user"):
                st.markdown(active_query)

            with st.chat_message("assistant"):
                with st.spinner("Grounded reasoning across study deck..."):
                    reply, st.session_state.chat_history = handle_conversation(
                        active_query,
                        retriever,
                        st.session_state.chat_history,
                        document_set_id=st.session_state.document_set_id,
                    )
                    # Check cache status across last turns
                    last_ai_roles = [r for r, _ in st.session_state.chat_history[-2:] if "AI" in r]
                    if any("AI (Cached)" in r for r in last_ai_roles):
                        st.session_state.cache_hits += 1

                # Live Token Streaming with Agent Badge and Citation Chips
                last_turn_role = st.session_state.chat_history[-1][0] if isinstance(st.session_state.chat_history[-1][1], str) else st.session_state.chat_history[-2][0]
                agent_name = last_turn_role.replace("AI (", "").replace(")", "").strip()
                st.markdown(agent_badge_html(agent_name), unsafe_allow_html=True)

                parts = re.split(r"sources\s+used\s*:", reply, maxsplit=1, flags=re.IGNORECASE)
                clean_body = parts[0].strip() if parts else reply
                st.write_stream(stream_text_chunks(clean_body))

                if len(parts) == 2:
                    sources_lines = [line.strip().lstrip("-* ") for line in parts[1].strip().split("\n") if line.strip()]
                    st.markdown(citation_chips_html(sources_lines), unsafe_allow_html=True)

            st.rerun()

    # Clear Chat History Button
    if st.session_state.chat_history:
        st.markdown('<div style="margin-top: 20px;">', unsafe_allow_html=True)
        if st.button("🧹 Clear Chat History", type="secondary"):
            st.session_state.chat_history = []
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)


# ============================================================
# TAB 2: SMART NOTES STUDIO
# ============================================================
with tab_notes:
    if not st.session_state.notes_content:
        st.markdown("""
        <div class="saas-card" style="text-align: center; padding: 40px 20px;">
            <div style="font-size: 2.5rem; margin-bottom: 12px;">📝</div>
            <div style="font-size: 1.25rem; font-weight: 700; color: #F8FAFC; margin-bottom: 8px;">No Study Notes Synthesized Yet</div>
            <div style="font-size: 0.92rem; color: #94A3B8; max-width: 500px; margin: 0 auto 20px auto;">
                Click "Create Notes" or "Summarize Deck" in the top bar to generate high-yield revision notes, definitions, and comparison tables.
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        notes_text = st.session_state.notes_content

        # Studio Control Bar
        col_ctrl1, col_ctrl2, col_ctrl3, col_ctrl4 = st.columns([3, 2, 2, 2])
        with col_ctrl1:
            st.markdown(f'<span class="hero-badge">STUDIO CANVAS: {st.session_state.notes_type.upper()}</span>', unsafe_allow_html=True)
        with col_ctrl2:
            st.download_button(
                "📥 Download (.md)",
                data=notes_text,
                file_name=f"LearnAssist_{st.session_state.notes_type.replace(' ', '_')}.md",
                mime="text/markdown",
                use_container_width=True,
            )
        with col_ctrl3:
            if st.button("📋 Copy Notes", use_container_width=True):
                escaped_text = (
                    notes_text.replace("\\", "\\\\")
                    .replace("`", "\\`")
                    .replace("$", "\\$")
                    .replace("\n", "\\n")
                    .replace("\r", "")
                    .replace('"', '\\"')
                )
                components.html(
                    f"""
                    <script>
                        navigator.clipboard.writeText("{escaped_text}");
                    </script>
                    """,
                    height=0,
                )
                st.toast("📋 Study notes copied to clipboard!", icon="✅")
        with col_ctrl4:
            if st.button("🔄 Toggle Summary/Notes", use_container_width=True):
                if st.session_state.notes_type == "Comprehensive Notes" and st.session_state.notes_summary:
                    st.session_state.notes_content = st.session_state.notes_summary
                    st.session_state.notes_type = "Executive Summary"
                else:
                    ctx = get_deck_context(retriever, "comprehensive study notes definitions formulas")
                    res = NotesAgent.run("create comprehensive study notes with headings and tables", ctx)
                    st.session_state.notes_content = res.get("output", "")
                    st.session_state.notes_type = "Comprehensive Notes"
                st.rerun()

        # Split Layout: Left Table of Contents, Right Markdown Reader
        col_toc, col_reader = st.columns([1, 3])

        with col_toc:
            st.markdown('<div class="saas-card" style="padding: 16px;">', unsafe_allow_html=True)
            st.markdown('<div class="hud-metric-label" style="margin-bottom: 10px;">📑 Table of Contents</div>', unsafe_allow_html=True)
            # Extract markdown headings
            headings = [
                line.strip().lstrip("#").strip()
                for line in notes_text.split("\n")
                if line.strip().startswith(("#", "##", "###"))
            ]
            if headings:
                for h in headings[:10]:
                    st.markdown(f'<div style="font-size: 0.8rem; color: #CBD5E1; margin-bottom: 6px; padding-left: 6px; border-left: 2px solid #6366F1;">{h}</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div style="font-size: 0.8rem; color: #64748B;">No explicit sections detected</div>', unsafe_allow_html=True)

            words_count = len(notes_text.split())
            read_time = max(1, round(words_count / 200))
            st.markdown(f'<div style="margin-top: 18px; font-size: 0.75rem; color: #94A3B8;">⏱️ {read_time} min read • {words_count} words</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col_reader:
            st.markdown('<div class="saas-card" style="padding: 28px 32px;">', unsafe_allow_html=True)
            st.markdown(notes_text)
            st.markdown('</div>', unsafe_allow_html=True)


# ============================================================
# TAB 3: ACTIVE RECALL LAB
# ============================================================
with tab_quiz:
    if not st.session_state.quiz_data:
        st.markdown("""
        <div class="saas-card" style="text-align: center; padding: 40px 20px;">
            <div style="font-size: 2.5rem; margin-bottom: 12px;">🧠</div>
            <div style="font-size: 1.25rem; font-weight: 700; color: #F8FAFC; margin-bottom: 8px;">No Active Recall Questions Generated</div>
            <div style="font-size: 0.92rem; color: #94A3B8; max-width: 500px; margin: 0 auto 20px auto;">
                Click "Generate Quiz Bank" above to populate interactive Multiple-Choice Questions with instant scoring, concept flashcards, and essay questions.
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        quiz_data = st.session_state.quiz_data
        mcqs = quiz_data.get("mcqs", [])
        flashcards = quiz_data.get("flashcards", [])
        essays = quiz_data.get("essays", [])

        # Sub-mode selection
        lab_mode = st.radio(
            "Select Lab Mode",
            options=["🎯 Interactive MCQs", "🎴 Concept Flashcards", "📝 Exam Essay Prep"],
            horizontal=True,
            label_visibility="collapsed",
        )

        # ---------------- 1. INTERACTIVE MCQS ----------------
        if "MCQs" in lab_mode:
            st.markdown('<div class="saas-card">', unsafe_allow_html=True)
            st.markdown("""
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                <div>
                    <span style="font-weight: 700; font-size: 1.1rem; color: #FFFFFF;">Interactive MCQ Assessment</span>
                    <div style="font-size: 0.8rem; color: #94A3B8;">Test your understanding with instant validation and explanations.</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            if st.session_state.quiz_submitted:
                total_mcqs = len(mcqs)
                score = sum(
                    1 for m in mcqs
                    if st.session_state.quiz_answers.get(m["id"]) == m["answer"].lower()
                )
                pct = round((score / total_mcqs) * 100) if total_mcqs > 0 else 0
                st.progress(pct / 100)
                st.markdown(
                    f'<div style="font-size: 1.15rem; font-weight: 700; color: #6EE7B7; margin: 8px 0 16px 0;">'
                    f'🏆 Score: {score} / {total_mcqs} ({pct}%)'
                    f'</div>',
                    unsafe_allow_html=True
                )
                if pct == 100:
                    st.balloons()

            for mcq in mcqs:
                qid = mcq["id"]
                with st.container(border=True):
                    st.markdown(f'<div class="mcq-question">Q{qid}. {mcq["question"]}</div>', unsafe_allow_html=True)

                    opts = mcq.get("options", {})
                    opt_keys = sorted(opts.keys())
                    opt_labels = [f"{k.upper()}) {opts[k]}" for k in opt_keys]

                    # Map back to letter
                    current_choice = st.session_state.quiz_answers.get(qid)
                    default_idx = opt_keys.index(current_choice) if current_choice in opt_keys else None

                    selected_label = st.radio(
                        f"Option for Q{qid}",
                        options=opt_labels,
                        index=default_idx,
                        key=f"mcq_radio_{qid}",
                        label_visibility="collapsed",
                        disabled=st.session_state.quiz_submitted,
                    )

                    if selected_label:
                        selected_key = selected_label[:1].lower()
                        st.session_state.quiz_answers[qid] = selected_key

                    # Show Feedback if Submitted
                    if st.session_state.quiz_submitted:
                        user_ans = st.session_state.quiz_answers.get(qid)
                        correct_ans = mcq["answer"].lower()
                        if user_ans == correct_ans:
                            st.markdown(
                                f'<div class="mcq-result-correct">✅ <b>Correct!</b> (Option {correct_ans.upper()})<br>'
                                f'<span style="color: #CBD5E1; font-size: 0.85rem;">{mcq.get("explanation", "")}</span></div>',
                                unsafe_allow_html=True
                            )
                        else:
                            st.markdown(
                                f'<div class="mcq-result-incorrect">❌ <b>Incorrect.</b> Correct answer: <b>Option ({correct_ans.upper()})</b><br>'
                                f'<span style="color: #CBD5E1; font-size: 0.85rem;">{mcq.get("explanation", "")}</span></div>',
                                unsafe_allow_html=True
                            )

            col_sub1, col_sub2 = st.columns([2, 2])
            with col_sub1:
                if not st.session_state.quiz_submitted:
                    if st.button("✅ Submit Answers & Calculate Score", type="primary", use_container_width=True):
                        st.session_state.quiz_submitted = True
                        st.rerun()
                else:
                    if st.button("🔄 Retake Quiz", use_container_width=True):
                        _clear_mcq_widget_states()
                        st.session_state.quiz_submitted = False
                        st.session_state.quiz_answers = {}
                        st.rerun()

            st.markdown('</div>', unsafe_allow_html=True)

        # ---------------- 2. CONCEPT FLASHCARDS ----------------
        elif "Flashcards" in lab_mode:
            st.markdown('<div class="saas-card">', unsafe_allow_html=True)
            if not flashcards:
                st.info("No flashcard concepts extracted. Try regenerating the question bank.")
            else:
                total_fc = len(flashcards)
                cur_idx = min(st.session_state.flashcard_index, total_fc - 1)
                card = flashcards[cur_idx]
                is_flipped = st.session_state.flashcard_flipped
                mastered_count = len(st.session_state.mastered_cards)

                # Mastery Progress
                prog_val = mastered_count / total_fc if total_fc > 0 else 0
                st.progress(prog_val)
                st.markdown(f'<div style="font-size: 0.82rem; color: #94A3B8; margin-bottom: 14px;">Mastery: {mastered_count}/{total_fc} Cards Mastered</div>', unsafe_allow_html=True)

                # Flashcard Box
                tag_cls = "flashcard-tag-back" if is_flipped else "flashcard-tag-front"
                tag_txt = "BACK • DEFINITION" if is_flipped else "FRONT • CONCEPT"
                card_text = card["back"] if is_flipped else card["front"]

                st.markdown(f"""
                <div class="flashcard-wrapper">
                    <span class="flashcard-tag {tag_cls}">{tag_txt}</span>
                    <span class="flashcard-counter">Card {cur_idx + 1} of {total_fc}</span>
                    <div class="flashcard-content">{card_text}</div>
                </div>
                """, unsafe_allow_html=True)

                # Carousel Controls
                col_c1, col_c2, col_c3, col_c4 = st.columns(4)
                with col_c1:
                    if st.button("⬅️ Previous", use_container_width=True, disabled=(cur_idx == 0)):
                        st.session_state.flashcard_index = max(0, cur_idx - 1)
                        st.session_state.flashcard_flipped = False
                        st.rerun()
                with col_c2:
                    flip_label = "🔄 Flip to Front" if is_flipped else "🔄 Flip to Reveal"
                    if st.button(flip_label, use_container_width=True, type="primary"):
                        st.session_state.flashcard_flipped = not st.session_state.flashcard_flipped
                        st.rerun()
                with col_c3:
                    if st.button("Next ➡️", use_container_width=True, disabled=(cur_idx >= total_fc - 1)):
                        st.session_state.flashcard_index = min(total_fc - 1, cur_idx + 1)
                        st.session_state.flashcard_flipped = False
                        st.rerun()
                with col_c4:
                    is_mastered = card["id"] in st.session_state.mastered_cards
                    btn_txt = "⭐ Mastered" if is_mastered else "☆ Mark Mastered"
                    if st.button(btn_txt, use_container_width=True):
                        if is_mastered:
                            st.session_state.mastered_cards.remove(card["id"])
                        else:
                            st.session_state.mastered_cards.add(card["id"])
                        st.rerun()

            st.markdown('</div>', unsafe_allow_html=True)

        # ---------------- 3. EXAM ESSAY PREP ----------------
        elif "Essay" in lab_mode:
            st.markdown('<div class="saas-card">', unsafe_allow_html=True)
            st.markdown("""
            <div style="margin-bottom: 14px;">
                <span style="font-weight: 700; font-size: 1.1rem; color: #FFFFFF;">Descriptive Exam & Essay Bank</span>
                <div style="font-size: 0.82rem; color: #94A3B8;">Comprehensive multi-part questions with grading rubrics and model evaluation criteria.</div>
            </div>
            """, unsafe_allow_html=True)

            if not essays:
                st.info("No essay questions extracted.")
            else:
                for idx, e in enumerate(essays, start=1):
                    st.markdown(f'<div class="mcq-card"><div class="mcq-question"><b>Essay Question {idx}:</b> {e["question"]}</div>', unsafe_allow_html=True)
                    with st.expander("📋 View Grading Rubric & Key Arguments"):
                        rubric = e.get("rubric", [])
                        if isinstance(rubric, list):
                            for r in rubric:
                                st.markdown(f"- {r}")
                        else:
                            st.markdown(str(rubric))

                    col_ask1, col_ask2 = st.columns([3, 1])
                    with col_ask2:
                        if st.button(f"💬 Discuss with Copilot", key=f"ask_essay_{idx}", use_container_width=True):
                            st.session_state.copilot_seed_prompt = f"How should I structure a comprehensive exam answer for: '{e['question']}'?"
                            st.toast("Prompt loaded into Academic Copilot!", icon="💬")
                    st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)


# ============================================================
# TAB 4: DOCUMENT & EVIDENCE INSPECTOR
# ============================================================
with tab_inspector:
    st.markdown("""
    <div style="margin-bottom: 14px;">
        <span style="font-weight: 700; font-size: 1.15rem; color: #FFFFFF;">Document & Evidence Inspector</span>
        <div style="font-size: 0.82rem; color: #94A3B8;">Inspect raw indexed text chunks, semantic boundaries, and extracted slide figures.</div>
    </div>
    """, unsafe_allow_html=True)

    # 1. Real-time Chunk Search
    kw = st.text_input("🔍 Filter chunks across documents by keyword...", key="chunk_kw_search")
    all_docs = get_all_indexed_documents()

    if not all_docs:
        # Fallback to active_chunks in session state if docstore is fresh
        if st.session_state.active_chunks:
            from langchain_core.documents import Document
            all_docs = [
                Document(page_content=c, metadata={"source_name": "Study Deck", "chunk_index": i})
                for i, c in enumerate(st.session_state.active_chunks, start=1)
            ]

    matching_docs = []
    if all_docs:
        if kw.strip():
            matching_docs = [d for d in all_docs if kw.lower() in d.page_content.lower()]
        else:
            matching_docs = all_docs

    st.markdown(f'<div style="font-size: 0.82rem; color: #818CF8; font-weight: 600; margin-bottom: 10px;">Showing {len(matching_docs)} of {len(all_docs)} Chunks:</div>', unsafe_allow_html=True)

    for doc in matching_docs[:12]:
        meta = getattr(doc, "metadata", {})
        src = meta.get("source_name") or meta.get("source") or "Document"
        idx = meta.get("chunk_index", "1")
        slide = meta.get("slide_number")
        slide_info = f" • Slide {slide}" if slide else ""

        st.markdown(f"""
        <div class="saas-card" style="padding: 14px 18px; margin-bottom: 10px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                <span class="citation-chip">📄 {src} • Chunk #{idx}{slide_info}</span>
                <span style="font-size: 0.72rem; color: #64748B;">{len(doc.page_content)} characters</span>
            </div>
            <div style="font-size: 0.875rem; color: #E2E8F0; line-height: 1.5; font-family: 'JetBrains Mono', monospace;">
                {doc.page_content[:450]}{'...' if len(doc.page_content) > 450 else ''}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # 2. Extracted Figures Gallery
    st.markdown('<div style="margin-top: 24px;">', unsafe_allow_html=True)
    st.markdown('<div class="hud-metric-label" style="font-size: 0.95rem; color: #F8FAFC; margin-bottom: 10px;">🖼️ Extracted Figures & Diagram Gallery</div>', unsafe_allow_html=True)

    img_dir = os.path.join(FAISS_DIR, "extracted_images")
    extracted_imgs = []
    if os.path.exists(img_dir):
        extracted_imgs = [
            os.path.join(img_dir, f)
            for f in os.listdir(img_dir)
            if os.path.splitext(f)[1].lower() in [".png", ".jpg", ".jpeg", ".webp"]
        ]

    if not extracted_imgs:
        st.markdown("""
        <div class="saas-card" style="text-align: center; padding: 24px;">
            <div style="font-size: 0.88rem; color: #94A3B8;">No figures or slide diagrams found in the current documents. Extracted diagrams from PPTX slides and images will appear here.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        gallery_cols = st.columns(min(3, len(extracted_imgs)))
        for i, img_path in enumerate(extracted_imgs):
            with gallery_cols[i % 3]:
                st.image(img_path, caption=os.path.basename(img_path), use_container_width=True)
                if st.button(f"💬 Ask Copilot about figure", key=f"ask_fig_{i}", use_container_width=True):
                    st.session_state.copilot_seed_prompt = f"Can you analyze and explain the diagram '{os.path.basename(img_path)}' extracted from the study materials?"
                    st.toast("Prompt queued in Academic Copilot!", icon="💬")

    st.markdown('</div>', unsafe_allow_html=True)
