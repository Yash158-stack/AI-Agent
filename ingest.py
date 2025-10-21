import glob
from docx import Document
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings


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
Also provide metadata such as Author, Subject and Title
"""


def pdfreader(file_list):
    for file in file_list:
        text = ""
        print(f"Accessing information from: {file} \n")
        reader = PdfReader(file)
        page_count = len(reader.pages)
        for i in range(page_count):
            page = reader.pages[i]
            text = text + page.extract_text()
        chunks = text_splitting_recusive(text)
        print(f"Text Chunks are:{chunks}")
        meta = reader.metadata
        print("Meta Data of the PDF is:\n")
        print(f"Author: {meta.author or 'Not Available'}")
        print(f"Subject:  {meta.subject or 'Not Available'}")
        print(f"Title: {meta.title or 'Not Available'}\n\n")


"""
This function extracts text from the word (docx) document
also prints metadata such as title and author
"""


def docxreader(file_list):
    for file in file_list:
        text = ""
        print(f"Accessing information from: {file}\n")
        reader = Document(file)
        for para in reader.paragraphs:
            text = text + para.text
        chunks = text_splitting_recusive(text)
        print(f"Text Chunks are:{chunks}")
        print("Meta Data of the Doc is :\n")
        print(f"Author: {reader.core_properties.author or 'Not Available'}")
        print(f"Title: {reader.core_properties.title or 'Not Available'}\n\n")


if __name__ == "__main__":

    file_list_pdf = glob.glob("data/*.pdf")
    pdf_no = len(file_list_pdf)
    file_list_docx = glob.glob("data/*.docx")
    docx_no = len(file_list_docx)

    if pdf_no != 0:
        pdfreader(file_list_pdf)
    if docx_no != 0:
        docxreader(file_list_docx)
