import os
import shutil
import glob
import hashlib
from git import Repo
import streamlit as st
from google.generativeai import embed_content
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# === Configuration ===
REPO_URL = "https://github.com/godspeedsystems/gs-documentation"
BRANCH = "main"
CLONE_DIR = "/tmp/gs_docs"  # Changed to temp directory for cloud compatibility

# === Sidebar ===
st.sidebar.title("Configuration")
repo_url = st.sidebar.text_input("Docs repo URL", REPO_URL)
branch = st.sidebar.text_input("Branch", BRANCH)

# === Clone/Refresh Docs with Error Handling ===
def clone_repo():
    try:
        if os.path.exists(CLONE_DIR):
            shutil.rmtree(CLONE_DIR)
            st.sidebar.write(f"Removed existing directory: {CLONE_DIR}")
            
        Repo.clone_from(repo_url, CLONE_DIR, branch=branch)
        st.sidebar.success(f"Cloned {len(get_doc_files(CLONE_DIR))} files successfully!")
        
        # Debug output
        st.sidebar.write("First 5 files found:")
        for f in get_doc_files(CLONE_DIR)[:5]:
            st.sidebar.write(f" - {f}")
            
    except Exception as e:
        st.sidebar.error(f"Cloning failed: {str(e)}")
        raise

if st.sidebar.button("🔄 Refresh docs"):
    clone_repo()

# === Auto-clone on First Run ===
if not os.path.exists(CLONE_DIR):
    with st.spinner("First-time repository cloning..."):
        clone_repo()

# === File Handling ===
def get_doc_files(base_dir):
    patterns = ["**/*.md", "**/*.mdx", "**/*.txt"]
    files = []
    
    if not os.path.exists(base_dir):
        return files  # Return empty if directory doesn't exist
        
    for pattern in patterns:
        full_pattern = os.path.join(base_dir, pattern)
        try:
            files.extend(glob.glob(full_pattern, recursive=True))
        except Exception as e:
            st.error(f"Error searching {full_pattern}: {str(e)}")
            
    return [f for f in files if os.path.isfile(f)]  # Filter out directories

# === RAG Initialization ===
@st.cache_resource
def init_chain():
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
    
    docs = get_doc_files(CLONE_DIR)
    if not docs:
        raise FileNotFoundError(f"No documents found in {CLONE_DIR}")
        
    # Debug file list
    st.write(f"Found {len(docs)} files:")
    for d in docs[:3]:
        st.write(f"- {d}")
    if len(docs) > 3:
        st.write(f"- ...and {len(docs)-3} more")

    texts = []
    metadatas = []
    
    for path in docs:
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            texts.append(content)
            metadatas.append({"path": path})
        except Exception as e:
            st.error(f"Error reading {path}: {str(e)}")

    # Initialize embeddings and vector store
    embed_fn = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_texts(
        texts=texts, 
        embedding=embed_fn, 
        metadatas=metadatas
    )
    
    return RetrievalQA.from_chain_type(
        llm=GoogleGenerativeAI(model="gemini-pro"),
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
        return_source_documents=True
    )

# === Main UI ===
st.title("🔍 Godspeed Docs RAG")

try:
    qa = init_chain()  # Will fail here if no docs found
except Exception as e:
    st.error(f"Initialization failed: {str(e)}")
    st.stop()

query = st.text_input("Ask a question about Godspeed:")
if query and st.button("Search"):
    try:
        result = qa(query)
        st.markdown("### ✅ Answer")
        st.write(result["result"])

        st.markdown("---\n#### 📚 Sources")
        for doc in result["source_documents"]:
            st.write(f"- `{doc.metadata['path']}`")
            
    except Exception as e:
        st.error(f"Query failed: {str(e)}")
