# 🔍 Godspeed Docs RAG App

This is a **Streamlit app** that performs **Retrieval-Augmented Generation (RAG)** over the [Godspeed documentation](https://github.com/godspeedsystems/gs-documentation), enabling you to ask natural language questions and get smart, source-grounded answers using **Gemini** by Google.

---

## 🚀 Features

- ✅ Automatically clones or refreshes the latest Godspeed documentation from GitHub.
- 🧠 Embeds documentation using Google Gemini Embeddings (`embedding-001`).
- 🔎 Semantic document search using FAISS vector store.
- 🤖 Answers questions via Gemini's LLM (`gemini-1.5-flash`) using LangChain's RetrievalQA.
- 📚 Displays source file paths that contributed to the answer.
- 🔐 API key and repo settings via Streamlit Secrets for secure configuration.

---

## 🧱 Tech Stack

| Tool        | Purpose                                      |
|-------------|----------------------------------------------|
| [Streamlit](https://streamlit.io)       | UI Framework for the app                   |
| [GitPython](https://gitpython.readthedocs.io/)  | Git-based cloning of repo                  |
| [LangChain](https://www.langchain.com/) | Building the RAG pipeline                  |
| [FAISS](https://github.com/facebookresearch/faiss)         | Fast vector similarity search              |
| [Gemini API](https://ai.google.dev/)    | Google's LLM and Embeddings                |

---


---

## 🧠 How It Works

1. **Clone Docs**: The app clones the Godspeed docs repo (or any repo you input).
2. **Find Files**: Searches for Markdown, text, and structured doc files.
3. **Embed**: Loads the text and creates vector embeddings using Google's embedding model.
4. **RAG Chain**: A `RetrievalQA` chain uses Gemini (via LangChain) to answer questions.
5. **Source Attribution**: The app shows which files were used to answer the query.

---

## 🛠️ Setup

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/godspeed-docs-rag.git
cd godspeed-docs-rag
```
### 2. Install Dependencies
```bash
pip install -r requirements.txt
```
### 3. Add Streamlit Secrets
```toml
GOOGLE_API_KEY = "your-google-api-key"
repo_url = "https://github.com/godspeedsystems/gs-documentation"
branch = "main"
```
💡 Make sure you have access to the Gemini API and enabled it via Google Cloud Console(Currently the deployed app has my Gemini api).
## ▶️ Run the App 
```
https://myragapp-9w6xz4mncdksge82pjyezk.streamlit.app/#sources
```
Once it launches in your browser:

Use the sidebar to change the GitHub repo/branch if needed.

Click "🔄 Refresh docs" to fetch and index docs.

Enter your question in the main input and click "Run RAG".

Get your answer + source file references.

## 🙋 About
This app is to explore RAG using Godspeed documentation. It showcases how LLMs can be augmented with real documentation using open-source tools like LangChain, FAISS, and Streamlit.


