import os
import glob
import hashlib
import shutil
from git import Repo

import streamlit as st
from google.generativeai import embed_content
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# Sidebar: clone/update docs
st.sidebar.title("Configuration")
repo_url  = st.sidebar.text_input("Docs repo URL", "https://github.com/godspeedsystems/gs-documentation")
branch    = st.sidebar.text_input("Branch",     "main")
clone_dir = "gs_docs"

if st.sidebar.button("Refresh docs"):
    if os.path.exists(clone_dir):
        shutil.rmtree(clone_dir)
    Repo.clone_from(repo_url, clone_dir, branch=branch)
    st.sidebar.success("Docs refreshed")

# Helper: file listing
@st.cache_data
def get_doc_files(base_dir):
    patterns = ["**/*.md", "**/*.mdx", "**/*.txt", "**/*.rst", "**/*.json", "**/*.yaml", "**/*.yml"]
    files = []
    for p in patterns:
        files.extend(glob.glob(os.path.join(base_dir, p), recursive=True))
    return files

# Helper: file hash
def file_hash(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

# Init chain with FAISS
@st.cache_resource
def init_chain(docs):
    os.environ["GOOGLE_API_KEY"] = st.secrets["api_key"]

    texts = []
    metadatas = []

    for path in docs:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        if not content.strip():
            continue
        texts.append(content)
        metadatas.append({"path": path})

    embed_fn = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_texts(texts=texts, embedding=embed_fn, metadatas=metadatas)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    llm = GoogleGenerativeAI(model="gemma-1.1-7b-it")  # Change if needed
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)

# UI
st.title("🔎 Godspeed Docs RAG")
q = st.text_input("Ask a question about Godspeed:")

if st.button("Run RAG") and q:
    docs = get_doc_files(clone_dir)
    if not docs:
        st.error("No documentation files found. Please refresh docs.")
    else:
        qa = init_chain(docs)
        with st.spinner("Thinking..."):
            res = qa(q)
        st.markdown("### Answer")
        st.write(res["result"])

        st.markdown("---\n#### Sources")
        for src in res["source_documents"]:
            st.write(f"- `{src.metadata['path']}`")
