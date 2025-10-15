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
    """
    Creates embeddings for the text chunks and stores them in a FAISS vector store.
    """
    print("Creating embeddings and vector store...")

    # Initialize the embedding model.
    # Using a popular open-source model from Hugging Face.
    embedding_model = HuggingFaceInstructEmbeddings(
        model_name="hkunlp/instructor-large",
        model_kwargs={"device": "cpu"},  # Use CPU for embedding
    )

    # Create the FAISS vector store from the text chunks and embeddings
    # This step can take a while depending on the number of documents
    db = FAISS.from_documents(texts, embedding_model)

    # Save the vector store locally
    db.save_local(faiss_db)
    print(f"Vector store created and saved at {faiss_db}")
    return db


# --- Main Execution Block ---
if __name__ == "__main__":
    # Step 1: Load the documents
    docs = create_documents()

    # Step 2: Split the documents into chunks
    chunks = split_text(docs)

    # Step 3 & 4: Create embeddings and store them
    create_vector_store(chunks)
