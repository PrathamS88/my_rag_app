# 🔍 Godspeed Docs RAG

This project is a **Retrieval-Augmented Generation (RAG)** application built with **Streamlit**, **LangChain**, **Google Gemini**, and **FAISS**, designed to help users ask natural language questions about the [Godspeed documentation](https://github.com/godspeedsystems/gs-documentation) and receive relevant, AI-generated answers along with source references.

---

## 🚀 Features

- 🔄 Clone and refresh the Godspeed documentation from any GitHub repository
- 🔍 Search across `.md`, `.txt`, `.yaml`, `.json`, `.rst`, and more
- ⚡ Uses **Google Gemini (Gemini 1.5 Flash)** for answering questions
- 🧠 Employs **GoogleGenerativeAIEmbeddings** for semantic search
- 📚 Shows actual source document paths for every answer
- 🖥️ Clean UI built using **Streamlit**

---

## 🧱 Tech Stack

- [Streamlit](https://streamlit.io/) - For the web UI
- [LangChain](https://www.langchain.com/) - To build the RAG pipeline
- [FAISS](https://github.com/facebookresearch/faiss) - For vector similarity search
- [Google Generative AI](https://ai.google.dev/) - For embeddings and LLM responses
- [GitPython](https://github.com/gitpython-developers/GitPython) - For cloning documentation repos

---

## 📦 Installation

1. **Clone this repo:**

   ```bash
   git clone https://github.com/YOUR_USERNAME/gs-docs-rag.git
   cd gs-docs-rag
# my_rag_app
