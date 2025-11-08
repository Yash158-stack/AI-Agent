import os
from docx import Document as DocxDocument
from PyPDF2 import PdfReader
import pdfplumber
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


def text_splitting_recursive(text):
    """Split text into overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
    )
    return splitter.split_text(text)


def extract_text_from_pdf(file_path):
    """Extract all text from a PDF file (handles scanned PDFs better)."""
    text = ""

    # First try PyPDF2
    try:
        reader = PdfReader(file_path)
        for page in reader.pages:
            if page.extract_text():
                text += page.extract_text()
    except Exception as e:
        print(f"⚠️ PyPDF2 failed: {e}")

    # Fallback to pdfplumber if PyPDF2 gives nothing
    if not text.strip():
        print(f"⚠️ No text extracted from {file_path} using PyPDF2, trying pdfplumber...")
        try:
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    text += page.extract_text() or ""
        except Exception as e:
            print(f"⚠️ pdfplumber failed: {e}")

    return text


def extract_text_from_docx(file_path):
    """Extract all text from a DOCX file, including paragraphs and tables."""
    
    doc = DocxDocument(file_path)
    all_text = []
    for paragraph in doc.paragraphs:
        # Only add non-empty lines
        if paragraph.text.strip():
            all_text.append(paragraph.text.strip())

    # Extract text from tables (rows and columns)
    for table in doc.tables:
        for row in table.rows:
            cell_texts = []
            for cell in row.cells:
                if cell.text.strip():
                    cell_texts.append(cell.text.strip())

            # Join all cell texts in a single line (like a CSV row)
            if cell_texts:
                all_text.append(" | ".join(cell_texts))
    final_text = "\n".join(all_text)
    return final_text


def index_files(file_paths, progress_callback=None):
    """
    Takes uploaded file paths, extracts text, chunks, embeds, and stores FAISS DB.
    Shows progress via optional callback function.
    """
    # Remove old FAISS folder to avoid conflicts
    if os.path.isdir("faiss_db"):
        import shutil
        shutil.rmtree("faiss_db")
    documents = []
    total_files = len(file_paths)

    for i, file in enumerate(file_paths, start=1):
        file_ext = os.path.splitext(file)[1].lower()

        if progress_callback:
            progress_callback(i / total_files / 2, f"📄 Reading file {i}/{total_files}: {os.path.basename(file)}")

        # Extract text
        text = ""
        if file_ext == ".pdf":
            text = extract_text_from_pdf(file)
        elif file_ext == ".docx":
            text = extract_text_from_docx(file)
        else:
            print(f"⚠️ Skipped unsupported file: {file}")
            continue

        if not text.strip():
            print(f"⚠️ No extractable text found in {os.path.basename(file)}")
            continue

        # Chunk text
        chunks = text_splitting_recursive(text)
        for chunk in chunks:
            documents.append(Document(page_content=chunk, metadata={"source": file}))

    if documents:
        if progress_callback:
            progress_callback(0.8, "🧠 Generating embeddings...")

        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        db = FAISS.from_documents(documents, embeddings)
        db.save_local("faiss_db")

        if progress_callback:
            progress_callback(1.0, "✅ Indexing complete!")
        print(f"✅ Indexed {len(documents)} text chunks from {len(file_paths)} file(s)")

    else:
        if progress_callback:
            progress_callback(1.0, "⚠️ No valid files to index.")
        print("⚠️ No valid documents were found.")

    return {"path": "faiss_db", "total_chunks": len(documents), "files_indexed": total_files}

