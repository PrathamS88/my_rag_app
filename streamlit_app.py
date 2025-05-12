import os, shutil, glob, hashlib
from git import Repo
import streamlit as st

from google.generativeai import embed_content
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# === Sidebar Configuration ===
st.sidebar.title("Configuration")
repo_url = st.sidebar.text_input("Docs repo URL", "https://github.com/godspeedsystems/gs-documentation")
branch   = st.sidebar.text_input("Branch", "main")
clone_dir = "gs_docs"

# === Clone or Refresh Documentation ===
if st.sidebar.button("Refresh docs"):
    if os.path.exists(clone_dir):
        shutil.rmtree(clone_dir)
    Repo.clone_from(repo_url, clone_dir, branch=branch)
    st.sidebar.success("Docs refreshed!")

# === Helper: Find Relevant Doc Files ===
def get_doc_files(base_dir):
    patterns = ["**/*.md", "**/*.mdx", "**/*.txt", "**/*.rst", "**/*.json", "**/*.yaml", "**/*.yml"]
    files = []
    for pattern in patterns:
        files.extend(glob.glob(os.path.join(base_dir, pattern), recursive=True))
    return files

# === Helper: File Hashing ===
def file_hash(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

# === Build RAG Chain ===
@st.cache_resource
def init_chain(docs):
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

    texts = []
    metadatas = []

    for path in docs:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        texts.append(content)
        metadatas.append({"path": path})

    # Embeddings
    embed_fn = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_texts(texts=texts, embedding=embed_fn, metadatas=metadatas)

    # Retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    # LLM
    llm = GoogleGenerativeAI(model="gemma-1.1-7b-it")  # or gemma-3-1b-it depending on access
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)

# === Streamlit UI ===
st.title("🔍 Godspeed Docs RAG")

query = st.text_input("Ask a question about Godspeed:")

if st.button("Run RAG") and query:
    docs = get_doc_files(clone_dir)

    if not docs:
        st.error("No documentation files found. Please refresh docs.")
        st.stop()

    with st.spinner("Building RAG chain and thinking..."):
        qa = init_chain(docs)
        result = qa(query)

    st.markdown("### ✅ Answer")
    st.write(result["result"])

    st.markdown("---\n#### 📚 Sources")
    for doc in result["source_documents"]:
        st.write(f"- `{doc.metadata['path']}`")
