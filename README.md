# 🧠 PDF RAG Chatbot

A **Retrieval-Augmented Generation (RAG)** chatbot built with **Flask** that allows users to upload PDFs, extract text, create embeddings, and interact conversationally using an LLM.

---

## 🚀 Features
- PDF text extraction
- Chunking and vector embeddings using FAISS
- Context-aware Q&A through RAG pipeline
- Flask web interface with file upload
- Dockerized for easy deployment
- Optional CI/CD to **Google Cloud Run**

---

## 🧰 Tech Stack
- **Python 3.11+**
- **Flask**
- **FAISS**
- **Hugging Face Transformers**
- **Google Generative AI / Gemini**
- **Docker + Cloud Run**

---

## 🧑‍💻 Local Setup

### 1. Clone and install dependencies
```bash
git clone https://github.com/<your-username>/pdf-rag-chatbot.git
cd pdf-rag-chatbot
python -m venv .venv
source .venv/bin/activate    # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
