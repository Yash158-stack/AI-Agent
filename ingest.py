# ingest.py
import os
import shutil
import io
import hashlib
import json
from docx import Document as DocxDocument
try:
    from PyPDF2 import PdfReader
except ImportError:
    from pypdf import PdfReader
import pdfplumber
from PIL import Image
from pptx import Presentation
try:
    import pytesseract
except ImportError:
    pytesseract = None

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

try:
    from config import get_embeddings_model
except ImportError:
    from functools import lru_cache
    from langchain_huggingface import HuggingFaceEmbeddings

    @lru_cache(maxsize=1)
    def get_embeddings_model():
        return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".webp"]


def file_fingerprint(file_path):
    """Return stable file identity used for document-set cache scoping."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)

    stat = os.stat(file_path)
    return {
        "name": os.path.basename(file_path),
        "size": stat.st_size,
        "sha256": h.hexdigest(),
    }


def compute_document_set_id(file_paths):
    fingerprints = [file_fingerprint(p) for p in sorted(file_paths)]
    payload = json.dumps(fingerprints, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _source_id(file_path):
    return file_fingerprint(file_path)["sha256"][:16]


def _base_metadata(file_path, document_set_id):
    ext = os.path.splitext(file_path)[1].lower().lstrip(".")
    return {
        "source": file_path,
        "source_name": os.path.basename(file_path),
        "source_type": ext,
        "source_id": _source_id(file_path),
        "document_set_id": document_set_id,
    }


# ------------------------------------------------------------
# TEXT SPLITTER
# ------------------------------------------------------------
def text_splitting_recursive(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
    )
    return splitter.split_text(text)


# ------------------------------------------------------------
# PDF EXTRACTION (TEXT + SELECTIVE OCR)
# ------------------------------------------------------------
def extract_text_from_pdf(file_path):
    pages_text = []
    total_pages = 0

    # 1️⃣ Extract digital PDF text via PyPDF2
    try:
        reader = PdfReader(file_path)
        total_pages = len(reader.pages)
        for page in reader.pages:
            t = page.extract_text()
            if t and t.strip():
                pages_text.append(t.strip())
    except Exception as e:
        print(f"⚠️ PyPDF2 failed on {file_path}: {e}")

    combined = "\n\n".join(pages_text)

    # 2️⃣ If text is absent or sparse, try pdfplumber
    if not combined.strip() or (total_pages > 0 and len(combined) / total_pages < 30):
        try:
            with pdfplumber.open(file_path) as pdf:
                total_pages = len(pdf.pages)
                plumber_pages = []
                for page in pdf.pages:
                    extracted = page.extract_text() or ""
                    if extracted.strip():
                        plumber_pages.append(extracted.strip())
                if plumber_pages:
                    combined = "\n\n".join(plumber_pages)
        except Exception as e:
            print(f"⚠️ pdfplumber failed on {file_path}: {e}")

    # 3️⃣ OCR images ONLY if document text is still essentially absent (scanned PDF)
    if pytesseract and (not combined.strip() or (total_pages > 0 and len(combined) / total_pages < 30)):
        try:
            ocr_pages = []
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_ocr = []
                    for img_info in page.images:
                        w = img_info.get("width", 0)
                        h = img_info.get("height", 0)
                        # Skip small icons, decorative lines, bullet graphics
                        if w < 120 or h < 120:
                            continue

                        x0 = max(0, img_info.get("x0", 0))
                        top = max(0, img_info.get("top", 0))
                        x1 = min(page.width, x0 + w)
                        bottom = min(page.height, top + h)
                        if x1 <= x0 or bottom <= top:
                            continue

                        try:
                            cropped = page.crop((x0, top, x1, bottom)).to_image(resolution=200)
                            ocr_text = pytesseract.image_to_string(cropped.original)
                            if ocr_text.strip():
                                page_ocr.append(ocr_text.strip())
                        except Exception:
                            pass

                    if page_ocr:
                        ocr_pages.append("\n".join(page_ocr))

            if ocr_pages:
                combined = (
                    combined + "\n\n" + "\n\n".join(ocr_pages)
                    if combined
                    else "\n\n".join(ocr_pages)
                )
        except Exception:
            pass

    return combined.strip()


# ------------------------------------------------------------
# DOCX EXTRACTION (TEXT + IN-MEMORY IMAGE OCR)
# ------------------------------------------------------------
def extract_text_from_docx(file_path):
    doc = DocxDocument(file_path)
    all_text = []

    # 1️⃣ Paragraphs
    for paragraph in doc.paragraphs:
        if paragraph.text.strip():
            all_text.append(paragraph.text.strip())

    # 2️⃣ Tables
    for table in doc.tables:
        for row in table.rows:
            cell_texts = []
            for cell in row.cells:
                if cell.text.strip():
                    cell_texts.append(cell.text.strip())
            if cell_texts:
                all_text.append(" | ".join(cell_texts))

    # 3️⃣ In-memory OCR on embedded images (no temporary disk folder needed)
    for rel in doc.part._rels:
        try:
            target = doc.part._rels[rel].target_ref
            if "image" in target:
                image_part = doc.part._rels[rel].target_part
                img_bytes = image_part.blob
                img = Image.open(io.BytesIO(img_bytes))
                if pytesseract and img.width >= 120 and img.height >= 120:
                    ocr_text = pytesseract.image_to_string(img)
                    if ocr_text.strip():
                        all_text.append(ocr_text.strip())
        except Exception:
            pass

    return "\n\n".join(all_text)


# ------------------------------------------------------------
# PPTX EXTRACTION (TEXT + IMAGE OCR)
# ------------------------------------------------------------
def extract_text_from_pptx(file_path):
    prs = Presentation(file_path)
    all_text = []

    for slide in prs.slides:
        for shape in slide.shapes:
            if hasattr(shape, "text") and shape.text.strip():
                all_text.append(shape.text.strip())

    return "\n\n".join(all_text)


def extract_images_from_pptx(file_path):
    prs = Presentation(file_path)
    images = []

    for slide in prs.slides:
        for shape in slide.shapes:
            if shape.shape_type == 13:  # picture
                img = shape.image
                try:
                    pil_img = Image.open(io.BytesIO(img.blob)).convert("RGB")
                    images.append(pil_img)
                except Exception:
                    pass

    return images


# ------------------------------------------------------------
# STANDALONE IMAGE OCR
# ------------------------------------------------------------
def extract_text_from_image(file_path):
    if not pytesseract:
        return ""
    try:
        img = Image.open(file_path)
        text = pytesseract.image_to_string(img)
        return text.strip()
    except Exception as e:
        print(f"⚠️ OCR failed for {file_path}: {e}")
        return ""


# ------------------------------------------------------------
# INDEXING MAIN FUNCTION
# ------------------------------------------------------------
def index_files(file_paths, faiss_dir, progress_callback=None):
    if not file_paths:
        if progress_callback:
            progress_callback(1.0, "No files to index.")
        return {
            "path": None,
            "total_chunks": 0,
            "files_indexed": 0,
            "document_set_id": None,
            "files": [],
        }

    document_set_id = compute_document_set_id(file_paths)

    # Clean previous index
    if os.path.exists(faiss_dir):
        try:
            shutil.rmtree(faiss_dir)
        except Exception:
            pass
    os.makedirs(faiss_dir, exist_ok=True)

    documents = []
    total_files = len(file_paths)
    extracted_folder = os.path.join(faiss_dir, "extracted_images")
    os.makedirs(extracted_folder, exist_ok=True)

    for i, file in enumerate(file_paths, start=1):
        ext = os.path.splitext(file)[1].lower()

        if progress_callback:
            progress_callback(
                i / (total_files * 2),
                f"📄 Processing file {i}/{total_files}: {os.path.basename(file)}"
            )

        text = ""
        image_paths = []

        if ext == ".pdf":
            text = extract_text_from_pdf(file)

        elif ext == ".docx":
            text = extract_text_from_docx(file)

        elif ext == ".pptx":
            text = extract_text_from_pptx(file)
            pptx_images = extract_images_from_pptx(file)
            for idx, img in enumerate(pptx_images):
                if img.width >= 150 and img.height >= 150:
                    img_path = os.path.join(extracted_folder, f"{os.path.basename(file)}_img{idx}.jpg")
                    img.save(img_path)
                    image_paths.append(img_path)
                    try:
                        if pytesseract:
                            text += "\n\n" + pytesseract.image_to_string(img)
                    except Exception:
                        pass

        elif ext in IMAGE_EXTENSIONS:
            text = extract_text_from_image(file)
            img_output_path = os.path.join(extracted_folder, os.path.basename(file))
            shutil.copy(file, img_output_path)
            image_paths.append(img_output_path)

        else:
            print(f"⚠️ Unsupported file skipped: {file}")
            continue

        if not text.strip():
            print(f"⚠️ No text extracted from {os.path.basename(file)}")
            continue

        chunks = text_splitting_recursive(text)
        base_metadata = _base_metadata(file, document_set_id)

        for chunk_index, chunk in enumerate(chunks, start=1):
            doc_metadata = {
                **base_metadata,
                "chunk_index": chunk_index,
                "chunk_id": f"{base_metadata['source_id']}:{chunk_index}",
            }
            if image_paths:
                doc_metadata["image_paths"] = image_paths

            documents.append(
                Document(page_content=chunk, metadata=doc_metadata)
            )

    if not documents:
        if progress_callback:
            progress_callback(1.0, "⚠️ No valid text could be extracted from uploaded files.")
        return {
            "path": None,
            "total_chunks": 0,
            "files_indexed": 0,
            "document_set_id": document_set_id,
            "files": [os.path.basename(p) for p in file_paths],
        }

    if progress_callback:
        progress_callback(0.9, "🧠 Generating embeddings...")

    # Shared embeddings singleton
    embeddings = get_embeddings_model()
    db = FAISS.from_documents(documents, embeddings)

    save_path = os.path.abspath(faiss_dir)
    db.save_local(save_path)

    if progress_callback:
        progress_callback(1.0, "✅ Indexing complete!")

    print(f"✅ Indexed {len(documents)} chunks from {len(file_paths)} files.")

    return {
        "path": save_path,
        "total_chunks": len(documents),
        "files_indexed": total_files,
        "document_set_id": document_set_id,
        "files": [os.path.basename(p) for p in file_paths],
    }
