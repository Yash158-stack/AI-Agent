from langchain_huggingface import HuggingFaceEmbeddings
import os
import streamlit as st
from langchain_community.vectorstores import FAISS


st.set_page_config(
    page_title="AI Academic Assistant",
    page_icon="🤖",
)
st.header("Hello, User!")
st.title("AI Academic Assistant📚")
query = st.text_input("Ask AI")
