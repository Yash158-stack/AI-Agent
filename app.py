from langchain_huggingface import HuggingFaceEmbeddings
import os
import streamlit as st
from langchain_community.vectorstores import FAISS


@st.cache_resource
def load_retriever():
    if not os.path.isdir("faiss_db"):
        st.error(
            "Possible reasons for the error:\n"
            "1. The ingest.py script has not been executed.\n"
            "2. No PDF or DOCX files were found in the 'data' subfolders.\n"
            "> Solution : Run ingest.py or Add your PDFs and DOCXs files for reference.",
            icon="🚨",
        )
        return None

    embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    db = FAISS.load_local(
        folder_path="faiss_db",
        embeddings=embedding,
        allow_dangerous_deserialization=True,
    )

    retriever = db.as_retriever(search_kwargs={"k": 4})

    return retriever


st.set_page_config(
    page_title="AI Academic Assistant",
    page_icon="🤖",
)
st.header("Hello, User!")
st.title("AI Academic Assistant📚")
query = st.text_input("Ask AI")

retriever = load_retriever()

if query != "" and retriever != "":
    st.spinner(
        text="Working on it... 🫡",
        show_time=True,
    )

    result = retriever.invoke(query)

    st.write("Solution:\n")

    for content in result:
        st.write(content.page_content)
        source = content.metadata["source"]
        category = content.metadata["category"]
        st.caption(f"Source : {source}\nCategory : {category}")
