import glob
from docx import Document as DocxDocument
from PyPDF2 import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


def text_splitting_recursive(text):
    """
    Split text into overlapping chunks using RecursiveCharacterTextSplitter.

    Args:
        text (str): The input text to be divided into smaller parts.

    Returns:
        list[str]: A list containing text chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
    )
    chunks = text_splitter.split_text(text)
    return chunks


def pdfreader(file_list, category):
    """
    Extract text from PDF files, split into chunks, and create Document objects.

    Args:
        file_list (list[str]): List of file paths for the PDF documents.
        category (str): Category label to assign as metadata.

    Returns:
        list[Document]: List of chunked Document objects created from PDFs.
    """
    chunks = []
    chunk_document = []
    for file in file_list:
        text = ""
        print(f"Accessing information from: {file} \n")
        reader = PdfReader(file)
        page_count = len(reader.pages)
        for i in range(page_count):
            page = reader.pages[i]
            extracted_text = page.extract_text()
            text = text + extracted_text
        chunks = text_splitting_recursive(text)
        for chunk in chunks:
            doc = Document(
                page_content=chunk,
                metadata={"source": file, "category": category},
            )
            chunk_document.append(doc)
    return chunk_document


def docxreader(file_list, category):
    """
    Extract text from DOCX files, split into chunks, and create Document objects.

    Args:
        file_list (list[str]): List of file paths for the DOCX documents.
        category (str): Category label to assign as metadata.

    Returns:
        list[Document]: List of chunked Document objects created from DOCX files.
    """
    chunks = []
    chunk_document = []
    for file in file_list:
        text = ""
        print(f"Accessing information...\n")
        reader = DocxDocument(file)
        for para in reader.paragraphs:
            extracted_text = para.text
            text = text + extracted_text
        chunks = text_splitting_recursive(text)
        for chunk in chunks:
            doc = Document(
                page_content=chunk,
                metadata={"source": file, "category": category},
            )
            chunk_document.append(doc)
    return chunk_document


if __name__ == "__main__":
    """
    Extract data from PDFs and DOCX files in multiple categories, generate embeddings,
    and store the vector index in a FAISS database.

    Workflow:
        1. Scan folders for PDF and DOCX files.
        2. Extract and chunk text from each file.
        3. Create document embeddings using HuggingFace.
        4. Store embeddings locally using FAISS.
    """
    category = ["textbooks", "notes", "questions"]
    final_chunk_list = []

    for i in category:
        file_list_pdf = glob.glob(f"data/{i}/**/*.pdf", recursive=True)
        pdf_no = len(file_list_pdf)
        file_list_docx = glob.glob(f"data/{i}/**/*.docx", recursive=True)
        docx_no = len(file_list_docx)

        pdf_chunks = []
        docx_chunks = []
        if pdf_no != 0:
            pdf_chunks = pdfreader(file_list_pdf, i)
        if docx_no != 0:
            docx_chunks = docxreader(file_list_docx, i)

        final_chunk_list.extend(pdf_chunks)
        final_chunk_list.extend(docx_chunks)

    if final_chunk_list != []:
        embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        database = FAISS.from_documents(
            documents=final_chunk_list,
            embedding=embedding,
        )
        database.save_local("faiss_db")
        print("All the Index of chunks are stored in the memory.")
    else:
        print("No chunks available to index.")
