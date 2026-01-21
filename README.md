# LearnAssist – Subject Guide & Q-Bank AI Assistant

🚀 **LearnAssist** is an AI-powered study companion that helps students analyze documents, generate question banks, create notes, summarize content, and extract insights from multi-format files — all inside a clean, fast Streamlit interface.

It supports:
- 📄 **PDFs**
- 📝 **DOCX**
- 🖼️ **Images (JPG, PNG, WEBP, JPEG)**
- 📊 **PPTX**
- 🧠 **Text-based Q&A, Summaries & Notes**

LearnAssist is built using a **multi-agent architecture**, allowing smart delegation between:
- Notes Agent  
- Summary Agent  
- QA Agent  
- Question Generation Agent  
- Intent Detection Agent  
- Small-talk Agent  
- Orchestrator Agent  

---

## 🚀 Live Demo
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://yash158-stack-ai-agent-app-kvirlu.streamlit.app/)

---

## 📦 Features

### 🔍 Intelligent File Analysis
Upload **PDF, DOCX, PPTX, PNG, JPG, WEBP**, and get:
- Topic summaries  
- Question banks (MCQs, short & long answers)  
- Structured notes  
- Definitions, keywords, explanations  

---

### 🤖 Multi-Agent System
Each task is handled by a dedicated agent:
- `notes_agent.py` → Creates clean notes  
- `summary_agent.py` → Compresses long documents  
- `qa_agent.py` → Answers questions from uploaded files  
- `question_agent.py` → Generates exam-style questions  
- `intent_agent.py` → Understands the user's request  
- `orchestrator.py` → Routes queries to the right agent  

---

### 📁 Vector Storage (FAISS)
- Stores embeddings of document chunks  
- Enables fast semantic retrieval (RAG)  
- Uses HuggingFace Sentence Transformers  

---

### 🗄️ Semantic Response Caching (NEW)
- AI-generated responses are **persisted** using SQLite  
- Each user query is converted into a **vector embedding**  
- New queries are compared using **semantic similarity**  
- If a similar query already exists:
  - Cached response is returned instantly  
  - ❌ LLM is NOT called  
  - ⚡ Faster response & reduced API usage  

This creates a **shared academic knowledge base** where future users benefit from previous queries.

---

## 🔧 Tech Stack
- Python  
- Streamlit  
- Gemini (google-generativeai)  
- LangChain (core, community, text-splitters, HF embeddings)  
- Sentence Transformers  
- FAISS  
- SQLite + SQLAlchemy (Semantic Cache)  
- python-docx  
- pdfplumber / PyPDF2  
- python-pptx  
- pytesseract (OCR)  
- Pillow  
- NumPy, Pandas  

---

## 🧠 System Architecture

LearnAssist follows a hybrid AI architecture:

1. **FAISS Vector Store**
   - Stores embeddings of document chunks  
   - Used for Retrieval-Augmented Generation (RAG)  

2. **SQLite Semantic Cache**
   - Stores:
     - User queries  
     - Query embeddings  
     - AI-generated responses  
   - Prevents repeated LLM calls for similar queries  

3. **Multi-Agent Orchestrator**
   - Detects user intent  
   - Routes requests to specialized agents  
   - Combines document context, cached knowledge, and LLM reasoning  

This design ensures:
- High performance  
- Cost efficiency  
- Consistent academic explanations  

---

## 📂 Project Structure

AI-AGENT/
├── pycache/
├── .devcontainer/
├── .streamlit/
│ └── config.toml
├── agents/
│ ├── intent_agent.py
│ ├── keywords.py
│ ├── notes_agent.py
│ ├── orchestrator.py
│ ├── prompts.py
│ ├── qa_agent.py
│ ├── question_agent.py
│ ├── smalltalk_agent.py
│ └── summary_agent.py
├── faiss_db/
├── user_data/
├── venv/
├── .env
├── .gitignore
├── app.py
├── chat_engine.py
├── ingest.py
├── cache.py
├── db.py
├── requirements.txt
├── runtime.txt
└── README.md

---

## ⚙️ Setup Guide (For Users & Developers)

### 1️⃣ Clone the repository
```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

python -m venv venv
source venv/bin/activate   # Mac & Linux
venv\Scripts\activate      # Windows

pip install -r requirements.txt

GEMINI_API_KEY=your_key_here

streamlit run app.py
App runs at → http://localhost:8501

```

📊 Database & Caching Behavior

Uses a global SQLite database (learn_assist.db)
- Cached responses are shared across users
- Query similarity is determined using cosine similarity on embeddings
- On semantic cache hit:
    - Stored response is returned
    - LLM is not invoked
- This significantly reduces latency and API usage for repeated academic queries.
---

🖼 Screenshots
📌 Home Page

(Add screenshot)

📌 File Upload & Processing

(Add screenshot)

📌 AI Summary / Questions / Notes

(Add screenshot)

📌 Semantic Cache (SQLite)

(Add DB Browser screenshot showing cached queries and responses)
---
❤️ Acknowledgements

LearnAssist is developed with the goal of helping learners interact with complex academic material in a clear, intelligent, and efficient way.

It demonstrates how Retrieval-Augmented Generation and semantic caching can be combined to build scalable AI learning systems.