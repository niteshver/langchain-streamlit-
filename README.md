# langchain-streamlit

python version 3.14
revision streamlit 


# simple chat bot
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS


from langchain_ollama import OllamaEmbeddings


loader = TextLoader("/Users/niteshv1520/Desktop/testing_data.txt")
raw_docs = loader.load()  

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
documents = splitter.split_documents(raw_docs)  

embeddings = OllamaEmbeddings(base_url="http://localhost:11434", model="nomic-embed-text:latest")


db = FAISS.from_documents(documents=documents, embedding=embeddings)


db.save_local("faiss_index") 


query = "Who is Ishita?"
results = db.similarity_search(query, k=4)
for i, doc in enumerate(results, 1):
    print(f"Result {i}:\n{doc.page_content}\n---\n")

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings

loader = TextLoader("/Users/niteshv1520/Desktop/testing_data.txt")
docs = loader.load()

splits = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20).split_documents(docs)

emb = OllamaEmbeddings(model="nomic-embed-text")
db = FAISS.from_documents(splits, emb)

db.save_local("faiss_index")

results = db.similarity_search("Who is Ishita?", k=1)
print(results)       

# streamlit testing
import streamlit as st
st.title("Welcome to LLM Journey")
st.write('''
         Hi,What are u doing''')

select_box = st.selectbox("U like:", ["Ishita","Ur Ex","Both"])

st.success("You choice is good")

if st.button("Click me"):
    st.write("Ur senseis good")

check_box = st.checkbox("Do u like Ishita??")
if check_box:
    st.write("U are a true lover")


relation_ship = st.radio("Relationship StatusL",
                         ["Single","Committed","Bich Ka"])
st.info(f"U ar {relation_ship} now")
if relation_ship =="Single":
    st.balloons()
else:
    st.snow()

uploaded_file = st.file_uploader("Ypload ur pic")
if uploaded_file:
    st.image(uploaded_file)