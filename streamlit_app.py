import os, shutil, glob, hashlib
from git import Repo

import streamlit as st
from google.generativeai import embed_content
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.docstore.document import Document

# Sidebar: clone/update docs
st.sidebar.title("Configuration")
repo_url  = st.sidebar.text_input("Docs repo URL", st.secrets.google.repo_url)
branch    = st.sidebar.text_input("Branch",     st.secrets.google.branch)
clone_dir = "gs_docs"
if st.sidebar.button("Refresh docs"):
    if os.path.exists(clone_dir): shutil.rmtree(clone_dir)
    Repo.clone_from(repo_url, clone_dir, branch=branch)
    st.sidebar.success("Docs refreshed")

# Helpers
from langchain.vectorstores import FAISS

@st.cache_resource
def init_chain(docs):
    os.environ["GOOGLE_API_KEY"] = st.secrets.google.api_key
    client = chromadb.Client(chromadb.config.Settings(allow_reset=True, anonymized_telemetry=False, chroma_db_impl="duckdb"))
    coll   = client.get_or_create_collection("gs_docs")
    
    texts = []
    metadatas = []
    
    for path in docs:
        h = file_hash(path)
        if coll.get(where={"hash": h})["ids"]:
            continue
        coll.delete(where={"path": path})
        text = open(path, "r", errors="ignore").read()
        texts.append(text)
        metadatas.append({"path": path})
        print(f"Loaded content from {path}")  # Debugging line

    # Debugging: Check if texts are loaded
    print("Loaded texts:", texts)
    
    # Embedding function
    embed_fn = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    embeddings = embed_fn.embed_documents(texts)
    
    # Debugging: Check if embeddings are correctly returned
    print("Embeddings shape:", embeddings.shape)

    # Creating FAISS vector store
    if texts:
        vectorstore = FAISS.from_texts(texts=texts, embedding=embed_fn, metadatas=metadatas)
    else:
        raise ValueError("No texts to process.")
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    llm = GoogleGenerativeAI(model="gemma-3-1b-it")
    
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)

# UI
st.title("🔎 Godspeed Docs RAG")
q = st.text_input("Ask a question about Godspeed:")
if st.button("Run RAG") and q:
    docs = get_doc_files(clone_dir)
    qa   = init_chain(docs)
    with st.spinner("Thinking..."):
        res = qa(q)
    st.markdown("### Answer")
    st.write(res["result"])
    st.markdown("---\n#### Sources")
    for src in res["source_documents"]:
        st.write(f"- `{src.metadata['path']}`")
