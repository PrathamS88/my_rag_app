import os, shutil, glob, hashlib
from git import Repo

import streamlit as st
from google.generativeai import embed_content
import chromadb
import chromadb.config

from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings

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
@st.cache_data
def get_doc_files(base_dir):
    patterns = ["**/*.md","**/*.mdx","**/*.txt","**/*.rst","**/*.json","**/*.yaml","**/*.yml"]
    files=[]
    for p in patterns:
        files += glob.glob(f"{base_dir}/{p}", recursive=True)
    return files

def file_hash(path):
    with open(path,"rb") as f: return hashlib.md5(f.read()).hexdigest()

@st.cache_resource
def init_chain(docs):
    os.environ["GOOGLE_API_KEY"] = st.secrets.google.api_key

    # ✅ Use DuckDB + Parquet instead of SQLite
    client = chromadb.Client(
        chromadb.config.Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=".chroma_store",  # Persist to avoid recomputation
            anonymized_telemetry=False
        )
    )

    # ✅ Remove old broken cache if needed
    if os.path.exists(".chroma_store_broken"):
        shutil.rmtree(".chroma_store_broken")

    coll = client.get_or_create_collection("gs_docs")

    for path in docs:
        h = file_hash(path)
        if coll.get(where={"hash": h})["ids"]:
            continue
        coll.delete(where={"path": path})
        text = open(path, "r", errors="ignore").read()
        emb = embed_content(
            model="models/embedding-001",
            content=text,
            task_type="retrieval_document"
        )["embedding"]
        coll.add(
            documents=[text],
            embeddings=[emb],
            metadatas=[{"path": path, "hash": h}],
            ids=[h]
        )

    embed_fn = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # ✅ Reuse client with DuckDB backend
    vs = Chroma(
        collection_name="gs_docs",
        embedding_function=embed_fn,
        persist_directory=".chroma_store",
        client=client
    )

    retr = vs.as_retriever(search_kwargs={"k": 5})
    llm = GoogleGenerativeAI(model="gemma-3-1b-it")
    return RetrievalQA.from_chain_type(llm=llm, retriever=retr, return_source_documents=True)

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
