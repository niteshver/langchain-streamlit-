import streamlit as st
from PyPDF2 import PdfReader
from langchain_openai import ChatOpenAI
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import os

st.title("📄 Chat with PDF")

# ---------- SESSION STATE INIT ----------
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# ---------- FILE UPLOAD ----------
file_uploader = st.file_uploader("Upload your PDF file", type=["pdf"])

if file_uploader and st.session_state.vector_store is None:
    reader = PdfReader(file_uploader)
    pages = []

    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)

    raw_text = "\n".join(pages)

    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = text_splitter.split_text(raw_text)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    st.session_state.vector_store = FAISS.from_texts(chunks, embeddings)

    st.success("PDF processed successfully!")

# ---------- QUERY INPUT (ALWAYS VISIBLE) ----------
query = st.text_input("Ask something about the PDF")

# ---------- ANSWER ----------
if query and st.session_state.vector_store:
    docs = st.session_state.vector_store.similarity_search(query, k=3)
    context = "\n\n".join([doc.page_content for doc in docs])

    llm = ChatOpenAI(
        base_url="https://api.perplexity.ai",
        api_key=os.environ["PERPLEXITY_API_KEY"],
        model="llama-3.1-sonar-small-128k-online"
    )

    response = llm.invoke(
        f"""
        Answer using ONLY the context below.
        If not found, say "Not found in document".

        Context:
        {context}

        Question:
        {query}
        """
    )

    st.write(response.content)
