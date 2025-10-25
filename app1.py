import streamlit as st

st.set_page_config(layout="wide")


st.title("Subject Guide and Question Bank AI Assistant")

st.markdown("Upload your subject guide and question bank documents, then ask any question related to them. The AI assistant will provide answers based on the content of the uploaded documents.")

uploaded_files = st.file_uploader("Upload the files here" , type=["pdf", "docx"], accept_multiple_files=True)
if uploaded_files:
    st.info(f"Successfully uploaded {len(uploaded_files)} files. Processing...")
st.markdown("---")
user_query = st.text_input("What would you like to know about the documents?" ,placeholder="Type your question here")

if st.button(":green[Get Answer]"):
    if user_query:
        # Placeholder for the logic to get the answer from the documents
        answer = "This is a placeholder answer based on the documents."
        st.write(f"Answer: {answer}")
    else:
        st.error("Please enter a question")

with st.spinner("Retrieving and generating response..."): ...

with st.expander("Show Source Document Details"): st.write("chunk_content")
st.info("Developed by - Team Infinite Loopers")
st.markdown("The retrieved answer is: :green[accurate] and :red[detailed].")

