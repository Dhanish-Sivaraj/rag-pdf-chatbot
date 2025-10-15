# 🧠 PDF RAG Chatbot

A **Retrieval-Augmented Generation (RAG)** chatbot built with **Flask** that allows users to upload PDFs, extract text, create embeddings, and interact conversationally using an LLM.

## 🚀 Features
- PDF text extraction
- Chunking and vector embeddings using FAISS
- Context-aware Q&A through RAG pipeline
- Flask web interface with file upload
- Dockerized for easy deployment
- Optional CI/CD to **Google Cloud Run**

## 🧰 Tech Stack
- **Python 3.11+**
- **Flask**
- **FAISS**
- **Hugging Face Transformers**
- **Google Generative AI / Gemini**
- **Docker + Cloud Run**

## 📂 Folder structure:
```bash
your-folder-name/
├── app.py           # Flask application
├── requirements.txt # Python dependencies
├── Dockerfile       # Docker configuration
├── templates/
│ └── index.html     # HTML templates for Flask
└── utils/
└── rag_utils.py     # Utility functions (RAG pipeline)
```

## 🧑‍💻 Local Setup

### 1. Clone and install dependencies
```bash
git clone https://github.com/Dhanish-Sivaraj/pdf-rag-chatbot.git
cd pdf-rag-chatbot
python -m venv .venv
source .venv/bin/activate    # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### 2. Set environment variables
Copy `.env.example` → `.env` and update values:
```bash
PROJECT_ID=your-project-id
BUCKET_NAME=your-bucket-name
SECRET_KEY=your-secret-key
LOCATION=your-region
```

### 3. Run locally
```bash
python app.py
```
Then visit [http://localhost:5000](http://localhost:5000)

## 🐳 Docker Usage
```bash
docker build -t pdf-rag-chatbot .
docker run -p 8080:8080 pdf-rag-chatbot
```

## ☁️ Deploy to Google Cloud Run
1. Enable Cloud Run and Artifact Registry in your GCP project.
2. Configure GitHub Secrets:
   - `GCP_SERVICE_ACCOUNT_KEY`
   - `GCP_PROJECT`
   - `GCP_REGION`
   - `CLOUD_RUN_SERVICE`
3. Push to main branch — deployment triggers automatically.

## ⚙️ Security
🚫 Do **not** commit API keys, JSON credentials, or private `.env` files.
All secrets go in GitHub → Settings → Secrets → Actions.
