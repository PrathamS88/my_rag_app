import os
import shutil
import glob
import streamlit as st
from git import Repo
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# === Set Google API Key from Streamlit secrets ===
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

# === Helper: Find Relevant Doc Files ===
def get_doc_files(base_dir):
    patterns = ["**/*.md", "**/*.mdx", "**/*.txt", "**/*.rst", "**/*.json", "**/*.yaml", "**/*.yml"]
    files = []
    if not os.path.exists(base_dir):
        return []
    for pattern in patterns:
        files.extend(glob.glob(os.path.join(base_dir, pattern), recursive=True))
    return [f for f in files if os.path.isfile(f)]

# === Sidebar Configuration ===
st.sidebar.title("Configuration")
default_repo_url = st.secrets.get("repo_url", "https://github.com/godspeedsystems/gs-documentation")
default_branch   = st.secrets.get("branch", "main")

repo_url = st.sidebar.text_input("Docs repo URL", default_repo_url)
branch   = st.sidebar.text_input("Branch", default_branch)
clone_dir = "/tmp/gs_docs"  # Use /tmp for Streamlit Cloud compatibility

# === Clone or Refresh Documentation ===
def clone_repo():
    if os.path.exists(clone_dir):
        shutil.rmtree(clone_dir)
    Repo.clone_from(repo_url, clone_dir, branch=branch)
    st.sidebar.success("Docs refreshed!")

if st.sidebar.button("🔄 Refresh docs"):
    with st.spinner("Cloning repo..."):
        try:
            clone_repo()
        except Exception as e:
            st.sidebar.error(f"Cloning failed: {e}")

# === Auto-clone on first run if directory does not exist ===
if not os.path.exists(clone_dir):
    with st.spinner("Cloning repo for the first time..."):
        try:
            clone_repo()
        except Exception as e:
            st.error(f"Initial cloning failed: {e}")
            st.stop()

# === Build RAG Chain ===
@st.cache_resource
def init_chain(docs):
    texts = []
    metadatas = []
    for path in docs:
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            texts.append(content)
            metadatas.append({"path": path})
        except Exception as e:
            st.warning(f"Could not read {path}: {e}")

    if not texts:
        raise ValueError("No readable documentation files found.")

    embed_fn = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_texts(texts=texts, embedding=embed_fn, metadatas=metadatas)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    llm = GoogleGenerativeAI(model="gemma-3-1b-it")
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)

# === Streamlit UI ===
st.title("🔍 Godspeed Docs RAG")

docs = get_doc_files(clone_dir)
if not docs:
    st.error("No documentation files found. Please refresh docs.")
    st.write(f"Checked directory: {clone_dir}")
    st.write("Directory contents:", os.listdir(clone_dir) if os.path.exists(clone_dir) else "Directory does not exist.")
    st.stop()

query = st.text_input("Ask a question about Godspeed:")

if st.button("Run RAG") and query:
    with st.spinner("Building RAG chain and thinking..."):
        try:
            qa = init_chain(docs)
            result = qa(query)
        except Exception as e:
            st.error(f"RAG failed: {e}")
            st.stop()
    st.markdown("### ✅ Answer")
    st.write(result["result"])
    st.markdown("---\n#### 📚 Sources")
    for doc in result["source_documents"]:
        st.write(f"- `{doc.metadata['path']}`")
