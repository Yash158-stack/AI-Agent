import glob
from PyPDF2 import PdfReader

"""
This function takes all the pdf available in the data folder and extract the information from them.
Also provide metadata such as Author, Subject and Title
"""


def pdfreader(file_name):
    for file in file_name:
        print(f"Accessing information from: {file} \n")
        reader = PdfReader(file)
        page_count = len(reader.pages)
        print("Extracted Information from the Page is:\n")
        for i in range(page_count):
            page = reader.pages[i]
            print("Page No.", i + 1)
            print(page.extract_text())
        meta = reader.metadata
        print("Meta Data related to the PDF is:\n ")
        print(f"Author: {meta.author}")
        print(f"Subject:  {meta.subject}")
        print(f"Title: \n\n{meta.title}")


file_list_pdf = glob.glob("data/*.pdf")

if len(file_list_pdf) != 0:
    pdfreader(file_list_pdf)
