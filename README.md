# LearnAssist – Subject Guide & Q-Bank AI Assistant

🚀 **LearnAssist** is an AI-powered study companion that helps students analyze documents, generate question banks, create notes, summarize content, and extract insights from multi-format files — all inside a clean, fast Streamlit interface.

It supports:
- 📄 **PDFs**
- 📝 **DOCX**
- 🖼️ **Images (JPG, PNG, WEBP, JPEG)**
- 📊 **PPTX**
- 🧠 **Text-based Q&A, Summaries & Notes**

LearnAssist is built using **multi-agent architecture**, allowing smart delegation between:
- Notes Agent  
- Summary Agent  
- QA Agent  
- Question Generation Agent  
- Intent Detection Agent  
- Small-talk Agent  
- Orchestrator Agent  

---

## 🚀 Live Demo  
> *(Will be active after you deploy)*  
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/your-username/your-repo-name)

---

## 📦 Features

### 🔍 **Intelligent File Analysis**
Upload **PDF, DOCX, PPTX, PNG, JPG, WEBP**, and get:
- Topic summaries  
- Question banks (MCQs, short & long answers)  
- Structured notes  
- Explanations, keywords, definitions  

### 🤖 **Multi-Agent System**
Each task is handled by a dedicated agent:
- `notes_agent.py` → Creates clean notes  
- `summary_agent.py` → Compresses long documents  
- `qa_agent.py` → Answers questions from uploaded files  
- `question_agent.py` → Generates exam-style questions  
- `intent_agent.py` → Understands the user's request  
- `orchestrator.py` → Routes queries to the right agent  

### 📁 **Vector Storage (FAISS)**
- Fast in-memory search  
- Efficient embeddings using HuggingFace Sentence Transformers  

### 🔧 **Tech Stack**
- Streamlit  
- Python  
- Gemini (googlegenerativeai)
- LangChain (core, community, text-splitters, HF embeddings)  
- Sentence Transformers  
- FAISS  
- python-docx  
- pdfplumber / PyPDF2  
- python-pptx  
- Tesseract OCR (pytesseract)  
- Pillow  

---

# 📂 Project Structure

```
AI-AGENT/
├── __pycache__/
├── .devcontainer/
├── .streamlit/
│   └── config.toml
├── agents/
│   ├── __pycache__/
│   ├── intent_agent.py
│   ├── keywords.py
│   ├── notes_agent.py
│   ├── orchestrator.py
│   ├── prompts.py
│   ├── qa_agent.py
│   ├── question_agent.py
│   ├── smalltalk_agent.py
│   └── summary_agent.py
├── faiss_db/
├── user_data/
├── venv/
├── .env
├── .gitignore
├── app.py
├── chat_engine.py
├── ingest.py
├── packages.txt
├── requirements.txt
├── runtime.txt
└── README.md
```

---

# ⚙️ Setup Guide (For Users & Developers)

## 1️⃣ Clone the repository

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
```

---

## 2️⃣ Create & activate virtual environment

```bash
python -m venv venv
source venv/bin/activate   # Mac & Linux
venv\Scripts\activate      # Windows
```

---

## 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

## 4️⃣ Add your API keys

Create a `.env` file:

```
GROQ_API_KEY=your_key_here
```

*(Only Groq is required unless you add more LLMs)*

---

## 5️⃣ Run Streamlit app

```bash
streamlit run app.py
```

App runs at → **http://localhost:8501**

---

# 🚀 Deploy to Streamlit Cloud (1-Click)

Replace the link with your repo URL.

```
[![Deploy to Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/your-username/your-repo-name)
```

---

# 🖼 Screenshots (Add Below)

You can upload screenshots and I will embed them here:

### 📌 Home Page  
*(screenshot placeholder)*  

### 📌 File Upload + Processing  
*(screenshot placeholder)*

### 📌 Summary / Notes / Q-Bank Output  
*(screenshot placeholder)*

---


# ❤️ Acknowledgements  
LearnAssist is developed with the goal of helping learners interact with complex study material in a clear, intelligent, and efficient way.
