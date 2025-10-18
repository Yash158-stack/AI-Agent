import glob
from docx import Document
from PyPDF2 import PdfReader

"""
This function takes all the pdf available in the data folder and extract the information from them.
Also provide metadata such as Author, Subject and Title
"""


def pdfreader(file_list):
    for file in file_list:
        print(f"Accessing information from: {file} \n")
        reader = PdfReader(file)
        page_count = len(reader.pages)
        print("Extracted Information from the Page is:\n")
        for i in range(page_count):
            page = reader.pages[i]
            print("Page No.", i + 1)
            print(page.extract_text())
        
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
        print(f"Accessing information from: {file}\n")
        reader = Document(file)
        print("Extracted Information from the Docx is:\n")
        for para in reader.paragraphs:
            print(para.text)
        
        print("Meta Data of the Doc is :\n")
        print(f"Author: {reader.core_properties.author or 'Not Available'}")
        print(f"Title: {reader.core_properties.title or 'Not Available'}\n\n")


file_list_pdf = glob.glob("data/*.pdf")
file_list_docx = glob.glob("data/*.docx")

if len(file_list_pdf) != 0:
    pdfreader(file_list_pdf)
elif len(file_list_docx) != 0:
    docxreader(file_list_docx)
