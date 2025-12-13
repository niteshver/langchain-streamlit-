import streamlit as st
from PyPDF2 import PdfReader
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from PyPDF2 import PdfReader
from langchain_community.vectorstores import FAISS
st.title("File Uploader")

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

    text_split = text_splitter.split_text(raw_text)
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_texts(text_split, embeddings)

query = st.text_input("Ask something about pdf")

if query and vector_store:
    docs = vector_store.similarity_search(query, k=3)
    llm = ChatOpenAI(model="gpt-4o-mini")

    response = llm.invoke(
        f"Answer the question using only this information:\n{docs}\n\nQuestion: {query}"
    )

    st.write(response.content)


    