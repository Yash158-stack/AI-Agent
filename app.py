import os
import streamlit as st
from dotenv import load_dotenv
from ingest import index_files
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
import google.generativeai as genai  # ✅ Gemini integration
from chat_engine import handle_conversation  # import your chat logic

# ---------- 1. PAGE CONFIG ----------
st.set_page_config(page_title="AI Academic Assistant", page_icon="🤖")
st.title("📚 AI Academic Assistant")
st.write("Upload your PDFs/DOCX files and ask AI about them!")

os.makedirs("uploads", exist_ok=True)
load_dotenv()

# ---------- 2. GEMINI API ----------
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    st.error("❌ Missing GEMINI_API_KEY in .env file!")
    st.stop()

genai.configure(api_key=gemini_api_key)

# ---------- 3. FILE UPLOAD ----------
uploaded_files = st.file_uploader(
    "Upload PDF/DOCX files",
    type=["pdf", "docx"],
    accept_multiple_files=True
)

saved_files = []
if uploaded_files:
    for uploaded_file in uploaded_files:
        curr_file = os.path.join("uploads", uploaded_file.name)
        with open(curr_file, "wb") as f:
            f.write(uploaded_file.read())
        saved_files.append(curr_file)
    st.success(f"✅ {len(saved_files)} files saved to 'uploads/' folder.")

# ---------- 4. INDEXING ----------
if st.button("Start Indexing"):
    if not saved_files:
        st.warning("Please upload files first!")
    else:
        progress_bar = st.progress(0)
        status_text = st.empty()

        def progress_callback(progress, message):
            progress_bar.progress(progress)
            status_text.text(message)

        with st.spinner("Indexing your documents... Please wait ⏳"):
            index_info = index_files(saved_files, progress_callback)

        # 🧹 Clear old retriever cache so FAISS reloads fresh
        st.cache_resource.clear()

        st.success(
            f"✅ Indexed {index_info['files_indexed']} files with {index_info['total_chunks']} chunks. You can now ask questions."
        )

# ---------- 5. LOAD RETRIEVER ----------
@st.cache_resource(show_spinner=False)
def load_retriever():
    """Load FAISS retriever (cached for speed)."""
    if not os.path.isdir("faiss_db"):
        st.warning("⚠️ No FAISS index found! Please index files first.")
        return None

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.load_local("faiss_db", embeddings, allow_dangerous_deserialization=True)
    return db.as_retriever(search_kwargs={"k": 4})

retriever = load_retriever()

# ---------- 6. SESSION STATE ----------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---------- 7. PROMPT TEMPLATE ----------
prompt = ChatPromptTemplate.from_template(
    """
    You are an AI assistant who answers based strictly on the given context.
    If the answer is not found in the context, reply: "I couldn't find relevant info in the uploaded documents."

    You should:
        - Maintain natural, human-like flow in conversation.
        - Use the chat history to recall details (like the user's name or previous topics).
        - Be concise — don't restate obvious things or overexplain.
        - Answer only from the provided document context when relevant.
        - If a question is unrelated to the documents, politely say so — but still respond helpfully.
        - Avoid repeating your name or previous messages unless necessary.
        - Keep your tone warm, confident, and approachable.

    Chat History:
    {chat_history}

    Context:
    {context}

    Question:
    {question}
    """
)

# ---------- 8. DEFINE GEMINI RESPONSE FUNCTION ----------
def run_gemini(prompt_text):
    try:
        model = genai.GenerativeModel("gemini-2.5-flash")  # or "gemini-pro"
        response = model.generate_content(prompt_text)
        return response.text
    except Exception as e:
        return f"⚠️ Error: {str(e)}"

# ---------- 9. CHAT SECTION ----------
if retriever is not None:
    st.markdown("---")
    st.subheader("Ask AI about your documents 🤖")

    # Streamlit chat interface
    user_query = st.chat_input("Type your question here...")

    if user_query:
        if user_query.lower() in ["exit", "quit", "bye"]:
            st.write("👋 Goodbye! Chat ended.")
        else:
            with st.spinner("Thinking... 💭"):
                response, st.session_state.chat_history = handle_conversation(
                    user_query,
                    retriever,
                    st.session_state.chat_history,
                    gemini_api_key
                )

    # Display chat history in chat bubble style
    for role, message in st.session_state.chat_history:
        if role == "You":
            st.chat_message("user").write(message)
        else:
            st.chat_message("assistant").write(message)
else:
    st.info("Please upload and index your documents first to start chatting.")