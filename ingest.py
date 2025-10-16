import os
from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import FAISS

data_path = "data/"
faiss_db = "vectorstores/db_faiss"


def create_documents():

    print("Loading documents...")
    loader = DirectoryLoader(
        data_path,
        glob="**/*",
        use_multithreading=True,
        show_progress=True,
    )

    documents = loader.load()
    print(f"Loaded {len(documents)} documents.")
    return documents


def split_text(documents):
    print("Splitting documents into smaller chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
    )
    texts = text_splitter.split_documents(documents)
    print(f"Split into {len(texts)} chunks.")
    return texts


def create_vector_store(texts):

    print("Creating embeddings and vector store...")

    embedding_model = HuggingFaceInstructEmbeddings(
        model_name="hkunlp/instructor-large",
        model_kwargs={"device": "cpu"},
    )

    db = FAISS.from_documents(texts, embedding_model)

    db.save_local(faiss_db)
    print(f"Vector store created and saved at {faiss_db}")
    return db


if __name__ == "__main__":

    docs = create_documents()

    chunks = split_text(docs)

    create_vector_store(chunks)
