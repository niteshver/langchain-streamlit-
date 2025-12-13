import streamlit as st
from PyPDF2 import PdfReader

from langchain_openai import ChatOpenAI
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

st.title("📄 Chat with PDF (Perplexity)")

raw_text = ""
vector_store = None   

file_uploader = st.file_uploader("Upload your PDF file", type=["pdf"])

if file_uploader:
    reader = PdfReader(file_uploader)
    pages = []

    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)

    raw_text = "\n".join(pages)
    st.success("PDF Loaded Successfully!")

    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = text_splitter.split_text(raw_text)

    # ✅ LOCAL EMBEDDINGS (NO OPENAI)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = FAISS.from_texts(chunks, embeddings)

query = st.text_input("Ask something about the PDF")

if query and vector_store:
    docs = vector_store.similarity_search(query, k=3)
    context = "\n\n".join([doc.page_content for doc in docs])

    # ✅ PERPLEXITY LLM (OpenAI-compatible)
    llm = ChatOpenAI(
        base_url="https://api.perplexity.ai",
        api_key=st.secrets["PERPLEXITY_API_KEY"],  # or env var
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
