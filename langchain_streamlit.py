# from langchain_community.document_loaders import TextLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import FAISS


# from langchain_ollama import OllamaEmbeddings


# loader = TextLoader("/Users/niteshv1520/Desktop/testing_data.txt")
# raw_docs = loader.load()  

# splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
# documents = splitter.split_documents(raw_docs)  

# embeddings = OllamaEmbeddings(base_url="http://localhost:11434", model="nomic-embed-text:latest")


# db = FAISS.from_documents(documents=documents, embedding=embeddings)


# db.save_local("faiss_index") 


# query = "Who is Ishita?"
# results = db.similarity_search(query, k=4)
# for i, doc in enumerate(results, 1):
#     print(f"Result {i}:\n{doc.page_content}\n---\n")

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings

# loader = TextLoader("/Users/niteshv1520/Desktop/testing_data.txt")
# docs = loader.load()

# splits = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20).split_documents(docs)

# emb = OllamaEmbeddings(model="nomic-embed-text")
# db = FAISS.from_documents(splits, emb)

# db.save_local("faiss_index")

# results = db.similarity_search("Who is Ishita?", k=1)
# print(results)       

"""trying for chat with pdf"""



from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from PyPDF2 import PdfReader
from langchain_community.vectorstores import FAISS
# # from langchain.llms import OpenAI
# # from langchain.chains import RetrievalQA

# # --------- CONFIG: paste your key here ----------
# # OPENAI_API_KEY = "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"
# from langchain_community.llms import Ollama


# # 1) Read PDF
# pdf_path = "/Users/niteshv1520/Desktop/yolo7paper.pdf"   # change to your filename
# reader = PdfReader(pdf_path)

# raw_text = []
# for page in reader.pages:          # PdfReader.pages (not .page)
#     text = page.extract_text()
#     if text:
#         raw_text.append(text)

# raw_text = "\n".join(raw_text)

# if not raw_text.strip():
#     raise RuntimeError(f"No text extracted from {pdf_path}. Check that the PDF contains selectable text (not scanned images).")

# # 2) Split text into chunks
# text_splitter = CharacterTextSplitter(
#     separator="\n",
#     chunk_size=500,
#     chunk_overlap=200,
#     length_function=len
# )
# texts = text_splitter.split_text(raw_text)

# # 3) Create embeddings (passing API key directly)
# embeddings = OpenAIEmbeddings()

# # 4) Build FAISS vectorstore from texts
# vectorstore = FAISS.from_texts(texts, embeddings)

# # 5) Create retriever and RetrievalQA chain
# retriever = vectorstore.as_retriever(search_kwargs={"k": 4})  # return top 4 chunks
# llm = Ollama(model="llama3") # configure as needed

# qa = RetrievalQA.from_chain_type(
#     llm=llm,
#     chain_type="stuff",       # or "map_reduce" / "refine"
#     retriever=retriever,
#     return_source_documents=True
# )

# # 6) Ask a question
# query = "Who is the author?"
# result = qa({"query": query})

# # result can contain keys: 'result' (answer) and 'source_documents'
# answer = result.get("result") or result.get("answer") or result
# print("=== Answer ===")
# print(answer)
# print()

# # Print small snippets from source docs (if present)
# src_docs = result.get("source_documents") or []
# if src_docs:
#     print("=== Source snippets ===")
#     for i, d in enumerate(src_docs, 1):
#         txt = d.page_content if hasattr(d, "page_content") else str(d)
#         snippet = txt[:400].replace("\n", " ").strip()
#         print(f"[{i}] {snippet}...\n")
# else:
#     print("No source documents returned.")

import streamlit as st
from PyPDF2 import PdfReader

st.title("File Uploader")

file_uploader = st.file_uploader("Upload your PDF file", type=["pdf"])

if file_uploader:
    reader = PdfReader(file_uploader)

    raw_text_list = []   # keep list of pages

    for page in reader.pages:
        text = page.extract_text()
        if text:
            raw_text_list.append(text)

    # Convert to one string AFTER the loop
    raw_text = "\n".join(raw_text_list)

    st.success("PDF Loaded Successfully!")
    st.write(raw_text)

if raw_text:
    text_splitter = CharacterTextSplitter(separator="\n",chunk_size = 500, chunk_overlap = 100,len_function = len)
    text_split = text_splitter.split_text(raw_text)
    embeddings = OpenAIEmbeddings()
    vectore_store = FAISS.from_text(text_split,embeddings)

from langchain_openai import ChatOpenAI
query = st.text_input("Ask something about pdf")
if query:
    docs = vectore_store.similarity_search(query, k = 3)
    llm = ChatOpenAI(model = "gpt-4o-mini")
    response = llm.invoke(
    f"Answer the question using only this information: {docs}\nQuestion: {query}"
)
st.write(response)

    