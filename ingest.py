import glob
from docx import Document as DocxDocument
from PyPDF2 import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


"""
Text splitting used  is :
Recursive text splitting
"""


def text_splitting_recusive(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
    )
    chunks = text_splitter.split_text(text)
    return chunks


"""
This function takes all the pdf available in the data folder and extract the information from them.
"""


def pdfreader(file_list):
    chunks = []
    chunk_document = []
    for file in file_list:
        text = ""
        print(f"Accessing information from: {file} \n")
        reader = PdfReader(file)
        page_count = len(reader.pages)
        for i in range(page_count):
            page = reader.pages[i]
            text = text + page.extract_text()
        chunks = text_splitting_recusive(text)
        for chunk in chunks:
            doc = Document(
                page_content=chunk,
                metadata={"source": file},
            )
            chunk_document.append(doc)
    return chunk_document


"""
This function extracts text from the word (docx) document
"""


def docxreader(file_list):
    chunks = []
    chunk_document = []
    for file in file_list:
        text = ""
        print(f"Accessing information from: {file}\n")
        reader = DocxDocument(file)
        for para in reader.paragraphs:
            text = text + para.text
        chunks = text_splitting_recusive(text)
        for chunk in chunks:
            doc = Document(
                page_content=chunk,
                metadata={"source": file},
            )
            chunk_document.append(doc)
    return chunk_document


if __name__ == "__main__":

    file_list_pdf = glob.glob("data/*.pdf")
    pdf_no = len(file_list_pdf)
    file_list_docx = glob.glob("data/*.docx")
    docx_no = len(file_list_docx)

    pdf_chunks = []
    docx_chunks = []
    if pdf_no != 0:
        pdf_chunks = pdfreader(file_list_pdf)
    if docx_no != 0:
        docx_chunks = docxreader(file_list_docx)

    final_chunks_list = []
    final_chunks_list.extend(pdf_chunks)
    final_chunks_list.extend(docx_chunks)

    if final_chunks_list != []:
        embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        database = FAISS.from_documents(
            documents=final_chunks_list,
            embedding=embedding,
        )
        database.save_local("faiss_database")

        print("All the Index of chunks are stored in the memory.")
    else:
        print("No chunks to index.")
