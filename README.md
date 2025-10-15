# 🧠 PDF RAG Chatbot

A **Retrieval-Augmented Generation (RAG)** chatbot built with **Flask** that allows users to upload PDFs, extract text, create embeddings, and interact conversationally using an LLM. Ideal for knowledge retrieval from documents in an interactive way.

---

## 🚀 Features
- Upload PDFs and extract text automatically
- Chunking and vector embeddings using **FAISS**
- Context-aware Q&A through a **RAG pipeline**
- Web interface built with **Flask**
- Dockerized for easy deployment
- Optional CI/CD integration with **Google Cloud Run**

---

## 🧰 Tech Stack
- **Python 3.11+**
- **Flask** (web framework)
- **FAISS** (vector embeddings & similarity search)
- **Hugging Face Transformers**
- **Google Generative AI / Gemini**
- **Docker + Cloud Run**

---

## 📂 Folder Structure
```bash
your-folder-name/
├── app.py                # Flask application
├── requirements.txt      # Python dependencies
├── Dockerfile            # Docker configuration
├── templates/
│   └── index.html        # HTML templates for Flask
└── utils/
    └── rag_utils.py      # Utility functions for RAG pipeline
```
🧑‍💻 Local Setup
1. Clone the repository
```bash
git clone https://github.com/Dhanish-Sivaraj/pdf-rag-chatbot.git
cd your-folder-name
```
2. Create and activate virtual environment
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```
3. Install dependencies
```bash
pip install -r requirements.txt
```
4. Set environment variables
```bash
PROJECT_ID=your-gcp-project-id
BUCKET_NAME=your-bucket-name
SECRET_KEY=your-secret-key
LOCATION=your-region
```
5. Run locally
```bash
python app.py
Visit http://localhost:5000 to interact with the chatbot.
```

🐳 Docker Usage
```bash
docker build -t pdf-rag-chatbot .
docker run -p 8080:8080 pdf-rag-chatbot
The app will be available at http://localhost:8080.
```

☁️ Deploy to Google Cloud Run
1. Enable required GCP services
```bash
gcloud services enable run.googleapis.com
gcloud services enable artifactregistry.googleapis.com
gcloud services enable storage.googleapis.com
gcloud services enable secretmanager.googleapis.com
gcloud services enable iam.googleapis.com
```
2. Authenticate with GCP
```bash
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
```
3. Build & push Docker image to Artifact Registry
```bash
# Create repository if not exists
gcloud artifacts repositories create pdf-rag-chatbot-repo \
    --repository-format=docker \
    --location=YOUR_REGION

# Build Docker image
docker build -t YOUR_REGION-docker.pkg.dev/YOUR_PROJECT_ID/pdf-rag-chatbot-repo/pdf-rag-chatbot:latest .

# Push image
docker push YOUR_REGION-docker.pkg.dev/YOUR_PROJECT_ID/pdf-rag-chatbot-repo/pdf-rag-chatbot:latest
```
4. Deploy to Cloud Run
```bash
gcloud run deploy pdf-rag-chatbot \
    --image YOUR_REGION-docker.pkg.dev/YOUR_PROJECT_ID/pdf-rag-chatbot-repo/pdf-rag-chatbot:latest \
    --platform managed \
    --region YOUR_REGION \
    --allow-unauthenticated
```
5. Access your app
After deployment, Cloud Run provides a public URL to interact with your chatbot.

🔐 Security
🚫 Do not commit API keys, JSON credentials, or .env files.
All secrets should go into GitHub → Settings → Secrets → Actions or Google Secret Manager.
