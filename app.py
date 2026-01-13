# app.py - Complete single-file solution for Render
import os
import uuid
import tempfile
import re
import logging
import traceback
import json
import numpy as np
import faiss
import torch
from datetime import datetime, timedelta
from flask import Flask, render_template, request, jsonify, session
from werkzeug.utils import secure_filename
import pypdf
from transformers import AutoTokenizer, AutoModel
import google.generativeai as genai
from typing import List, Tuple

# ==================== CONFIGURATION ====================
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Environment Variables for Render
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
SECRET_KEY = os.environ.get("SECRET_KEY", "your-render-secret-key-change-in-production")
PORT = int(os.environ.get("PORT", 10000))  # Render uses port 10000

logger.info(f"🚀 Initializing PDF RAG Chatbot for Render")
logger.info(f"📡 Port: {PORT}, Gemini API Key configured: {bool(GEMINI_API_KEY)}")

# Flask App Configuration
app = Flask(__name__)
app.secret_key = SECRET_KEY
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=1)  # Session timeout

# ==================== UTILITY FUNCTIONS ====================
def clean_text(text: str) -> str:
    """
    Clean text from PDF extraction issues
    """
    if not text:
        return ""
    
    # Fix encoding issues
    text = (text.replace('ﬁ', 'fi')
                .replace('ﬂ', 'fl')
                .replace('ﬀ', 'ff')
                .replace('ﬃ', 'ffi')
                .replace('ﬄ', 'ffl'))
    
    # Fix common broken patterns
    text = re.sub(r'\b([A-Z]) ([A-Z]) ([A-Z])\b', r'\1\2\3', text)  # L L M -> LLM
    text = re.sub(r'\b([A-Z]) ([A-Z])\b', r'\1\2', text)  # N L -> NL
    
    # Fix broken words
    text = re.sub(r'\b(\w{2,}) (\w{1,2})\b', r'\1\2', text)
    text = re.sub(r'\b(\w{1,2}) (\w{2,})\b', r'\1\2', text)
    
    # Fix specific common broken patterns
    common_fixes = [
        (r'\b(\w+)have\b', r'\1 have'),
        (r'\b(\w+)a\b', r'\1 a'),
        (r'\b(\w+)of\b', r'\1 of'),
        (r'\b(\w+)in\b', r'\1 in'),
        (r'\b(\w+)on\b', r'\1 on'),
        (r'\b(\w+)to\b', r'\1 to'),
        (r'\b(\w+)is\b', r'\1 is'),
        (r'\b(\w+)are\b', r'\1 are'),
        (r'\b(\w+)and\b', r'\1 and'),
    ]
    
    for pattern, replacement in common_fixes:
        text = re.sub(pattern, replacement, text)
    
    # Normalize all whitespace
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r' +', ' ', text)
    
    return text.strip()

# ==================== EMBEDDING MODEL ====================
class EnhancedEmbeddingModel:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        logger.info(f"Loading embedding model: {model_name}")
        self.model_name = model_name
        self.vector_size = 384
        self.model = None
        self.tokenizer = None
        logger.info("✅ Embedding model initialized!")

    def _ensure_model_loaded(self):
        if self.model is None:
            try:
                logger.info(f"🔄 Loading model weights for {self.model_name}...")
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModel.from_pretrained(self.model_name)
                logger.info("✅ Embedding model loaded successfully!")
            except Exception as e:
                logger.error(f"❌ Failed to load embedding model: {e}")
                raise RuntimeError(f"Model loading failed: {e}")

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def embed(self, text: str) -> np.ndarray:
        self._ensure_model_loaded()
        try:
            if not text or not text.strip():
                logger.warning("Empty text provided for embedding")
                return np.zeros(self.vector_size)
            
            clean_text_str = clean_text(' '.join(text.split()[:300]))
            
            inputs = self.tokenizer(
                clean_text_str, 
                return_tensors='pt', 
                truncation=True, 
                max_length=512, 
                padding=True
            )
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            embeddings = self._mean_pooling(outputs, inputs['attention_mask'])
            embeddings = embeddings.squeeze().numpy()
            
            norm = np.linalg.norm(embeddings)
            if norm > 0:
                embeddings = embeddings / norm
                
            return embeddings.astype('float32')
            
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            return np.zeros(self.vector_size)

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        if not texts:
            logger.warning("Empty text list provided for batch embedding")
            return np.array([])
        
        embeddings = []
        for text in texts:
            embedding = self.embed(text)
            embeddings.append(embedding)
            
        return np.array(embeddings)

# ==================== RAG SYSTEM ====================
class AdvancedRAGSystem:
    def __init__(self):
        logger.info("Initializing Advanced RAG System")
        self.embedder = EnhancedEmbeddingModel()
        self.index = None
        self.chunks = []
        self.current_pdf_name = None
        logger.info("✅ RAG System initialized successfully!")

    def create_embeddings(self, chunks: List[str]) -> np.ndarray:
        logger.info(f"Creating embeddings for {len(chunks)} chunks")
        if not chunks:
            raise ValueError("Cannot create embeddings: chunks list is empty")
        
        embeddings = self.embedder.embed_batch(chunks)
        logger.info(f"✅ Successfully created {len(embeddings)} embeddings")
        return embeddings

    def build_index(self, embeddings: np.ndarray):
        if len(embeddings) == 0:
            raise ValueError("No embeddings available to build index")
        
        logger.info(f"Building FAISS index with {embeddings.shape[1]} dimensions")
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        logger.info(f"✅ FAISS index built with {self.index.ntotal} vectors")

    def load_documents_from_storage(self, pdf_name: str, chunks: List[str], embeddings: List[List[float]]):
        logger.info(f"Loading documents for {pdf_name} from storage")
        self.current_pdf_name = pdf_name
        self.chunks = chunks
        embedding_array = np.array(embeddings).astype('float32')
        self.build_index(embedding_array)
        logger.info(f"✅ Loaded {len(chunks)} chunks for {pdf_name}")

    def add_new_documents(self, pdf_name: str, chunks: List[str]):
        logger.info(f"Adding new documents for {pdf_name}")
        self.current_pdf_name = pdf_name
        self.chunks = chunks
        
        if not chunks:
            raise ValueError("No text chunks to process")
        
        logger.info(f"Processing {len(chunks)} chunks")
        embeddings = self.create_embeddings(chunks)
        self.build_index(embeddings)
        logger.info(f"✅ Added {len(chunks)} documents to RAG system")
        return embeddings.tolist()

    def search(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        if self.index is None or not self.chunks:
            logger.warning("No documents loaded for search")
            return []
        
        try:
            logger.info(f"Searching for: '{query}'")
            
            query_embedding = self.embedder.embed(query).astype('float32')
            query_embedding = query_embedding.reshape(1, -1)
            faiss.normalize_L2(query_embedding)
            
            search_k = min(k * 5, len(self.chunks))
            if search_k == 0:
                return []
            
            similarities, indices = self.index.search(query_embedding, search_k)
            
            results = []
            for similarity, idx in zip(similarities[0], indices[0]):
                if idx < len(self.chunks) and similarity > 0.05:
                    clean_chunk = clean_text(self.chunks[idx])
                    
                    if len(clean_chunk.split()) < 5:
                        continue
                        
                    results.append((clean_chunk, float(similarity)))
            
            results.sort(key=lambda x: x[1], reverse=True)
            
            if results:
                logger.info(f"✅ Found {len(results)} relevant chunks")
            else:
                logger.info("❌ No relevant chunks found")
                
            return results[:k]
            
        except Exception as e:
            logger.error(f"Search failed for query '{query}': {e}")
            return []

    def clear_documents(self):
        logger.info("Clearing all documents from RAG system")
        self.index = None
        self.chunks = []
        self.current_pdf_name = None
        logger.info("✅ Documents cleared successfully")

# ==================== GEMINI LLM CLIENT ====================
class GeminiLLMClient:
    def __init__(self):
        logger.info("🚀 Initializing Gemini LLM Client")
        self.model = None
        self.configured = False
        
        try:
            api_key = os.environ.get("GEMINI_API_KEY")
            if not api_key:
                logger.warning("❌ GEMINI_API_KEY not found in environment variables")
                logger.info("💡 Please set GEMINI_API_KEY environment variable")
                return
            
            logger.info(f"🔑 Found Gemini API Key (first 10 chars): {api_key[:10]}...")
                
            genai.configure(api_key=api_key)
            
            model_priority = [
                'gemini-2.5-flash',
                'gemini-2.5-pro', 
                'gemini-2.5-flash-lite',
                'models/gemini-2.5-flash',
                'models/gemini-2.5-pro',
                'models/gemini-2.5-flash-lite'
            ]
            
            for model_name in model_priority:
                try:
                    logger.info(f"🔄 Trying to load model: {model_name}")
                    self.model = genai.GenerativeModel(model_name)
                    
                    test_response = self.model.generate_content("Say 'TEST' in one word.")
                    if test_response and test_response.text:
                        logger.info(f"✅ Successfully initialized model: {model_name}")
                        break
                    else:
                        logger.warning(f"❌ Model {model_name} test failed - no response")
                        self.model = None
                        
                except Exception as model_error:
                    logger.warning(f"❌ Model {model_name} failed: {str(model_error)[:100]}...")
                    self.model = None
                    continue
            
            if self.model is None:
                logger.error("❌ All Gemini models failed to initialize")
                return
            
            self.configured = True
            logger.info("✅ Gemini LLM client initialized successfully!")
            
        except Exception as e:
            logger.error(f"❌ Gemini initialization failed: {str(e)}")
            logger.error(traceback.format_exc())
            self.configured = False

    def _get_general_response(self, query: str) -> str:
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['hi', 'hello', 'hey']):
            return "👋 Hello! I'm your AI PDF assistant powered by Google Gemini. I can read and answer questions about your documents!"
        elif 'thank' in query_lower:
            return "💫 You're welcome! I'm happy to help."
        elif any(word in query_lower for word in ['bye', 'goodbye']):
            return "👋 Goodbye!"
        elif 'who are you' in query_lower:
            return "🤖 I'm an AI-powered PDF assistant using Google Gemini to provide intelligent answers!"
        elif 'help' in query_lower or 'what can you do' in query_lower:
            return "🤖 I can read PDFs and provide intelligent summaries and answers using advanced AI!"
        else:
            return ""

    def generate_response(self, query: str, context: str, pdf_name: str) -> str:
        general_response = self._get_general_response(query)
        if general_response:
            return general_response
        
        if not pdf_name:
            return "📚 Please upload a PDF file first!"
        
        if self.configured and self.model:
            try:
                return self._generate_with_gemini(query, context, pdf_name)
            except Exception as e:
                logger.error(f"❌ Gemini generation error: {e}")
                return self._fallback_response(query, pdf_name)
        else:
            logger.warning("🔧 Gemini not configured, using fallback response")
            return self._fallback_response(query, pdf_name)
    
    def _generate_with_gemini(self, query: str, context: str, pdf_name: str) -> str:
        try:
            logger.info("🎯 Using Gemini for response generation")
            
            if not context or not context.strip():
                return f"""🤔 I searched through **{pdf_name}** but couldn't find specific information about '{query}'.

**Suggestions:**
- Try rephrasing your question
- Ask about specific topics mentioned in the document
- Check if the PDF contains the information you're looking for

The document might discuss related concepts but not directly answer your specific question."""
            
            prompt = f"""
            You are an AI assistant that answers questions based on the provided document context.
            
            DOCUMENT CONTEXT FROM "{pdf_name}":
            {context}
            
            USER QUESTION: {query}
            
            IMPORTANT INSTRUCTIONS:
            1. Answer based primarily on the document context above
            2. If the context doesn't directly answer the question, but contains related information, explain what the document DOES say about related topics
            3. Be honest about what information is and isn't in the document
            4. If the context is insufficient, you can provide general knowledge but clearly state this
            5. Use bullet points if helpful for organization
            6. Format your response to be readable and well-structured
            
            Please provide your answer:
            """
            
            logger.info(f"📝 Sending prompt to Gemini (context: {len(context)} chars, query: {len(query)} chars)")
            
            response = self.model.generate_content(prompt)
            
            if not response or not response.text:
                logger.error("❌ Gemini returned empty response")
                return self._fallback_response(query, pdf_name)
            
            formatted_response = f"**Based on {pdf_name}**:\n\n{response.text}"
            
            logger.info(f"✅ Gemini response generated: {len(response.text)} characters")
            return formatted_response
            
        except Exception as e:
            logger.error(f"❌ Gemini generation failed: {e}")
            raise
    
    def _fallback_response(self, query: str, pdf_name: str) -> str:
        return f"**Based on {pdf_name}**:\n\nI found relevant content in the document but couldn't generate a detailed AI response. The document contains information related to '{query}'.\n\n🔧 *Note: AI response generation is currently unavailable.*"

# Initialize components
rag_system = AdvancedRAGSystem()
llm_client = GeminiLLMClient()

# ==================== PDF PROCESSING FUNCTIONS ====================
class RenderStorage:
    def __init__(self, base_folder=None):
        self.base_folder = base_folder or tempfile.gettempdir()
        self.documents = {}
        os.makedirs(self.base_folder, exist_ok=True)
        
    def save_file(self, file, filename):
        filepath = os.path.join(self.base_folder, secure_filename(filename))
        file.save(filepath)
        logger.info(f"💾 Saved file: {filename} to {filepath}")
        return filepath
    
    def save_json(self, data, filename):
        filepath = os.path.join(self.base_folder, secure_filename(filename))
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        logger.info(f"💾 Saved JSON: {filename}")
        return filepath
    
    def load_json(self, filename):
        filepath = os.path.join(self.base_folder, secure_filename(filename))
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        logger.warning(f"⚠️ File not found: {filename}")
        return None
    
    def file_exists(self, filename):
        filepath = os.path.join(self.base_folder, secure_filename(filename))
        return os.path.exists(filepath)
    
    def list_files(self, extension=None):
        files = []
        for f in os.listdir(self.base_folder):
            if extension and not f.endswith(extension):
                continue
            files.append(f)
        return files
    
    def delete_file(self, filename):
        filepath = os.path.join(self.base_folder, secure_filename(filename))
        if os.path.exists(filepath):
            os.remove(filepath)
            logger.info(f"🗑️ Deleted file: {filename}")
            return True
        return False

storage = RenderStorage()
processed_pdfs = {}

def extract_text_from_pdf(pdf_path):
    try:
        logger.info(f"📄 Extracting text from: {pdf_path}")
        text = ""
        
        with open(pdf_path, "rb") as file:
            pdf_reader = pypdf.PdfReader(file)
            total_pages = len(pdf_reader.pages)
            
            logger.info(f"🔢 Processing {total_pages} pages...")
            
            for page_num in range(total_pages):
                page = pdf_reader.pages[page_num]
                page_text = page.extract_text()
                
                if page_text and page_text.strip():
                    page_text = clean_text(page_text)
                    text += page_text + "\n"
                
                if (page_num + 1) % 10 == 0:
                    logger.info(f"📖 Processed {page_num + 1}/{total_pages} pages")
        
        if not text.strip():
            raise Exception("No text content extracted from PDF")
        
        logger.info(f"✅ Extracted {len(text)} characters from {total_pages} pages")
        return text.strip()
        
    except Exception as e:
        logger.error(f"❌ PDF extraction failed: {e}")
        logger.error(traceback.format_exc())
        raise Exception(f"PDF processing error: {str(e)}")

def chunk_text(text, chunk_size=600):
    if not text:
        logger.warning("Empty text provided for chunking")
        return []
    
    text = clean_text(text)
    logger.info(f"✂️ Chunking text of {len(text)} characters...")
    
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip() and len(p.strip()) > 50]
    
    chunks = []
    for i, paragraph in enumerate(paragraphs):
        if len(paragraph) <= chunk_size:
            chunks.append(paragraph)
        else:
            sentences = re.split(r'(?<=[.!?])\s+', paragraph)
            
            current_chunk = ""
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                    
                if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = sentence
                else:
                    current_chunk += " " + sentence if current_chunk else sentence
            
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
    
    if len(chunks) < 3:
        logger.info("🔄 Using word-based chunking fallback")
        words = text.split()
        chunks = []
        current_chunk = ""
        
        for word in words:
            if len(current_chunk) + len(word) > chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = word
            else:
                current_chunk += " " + word if current_chunk else word
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
    
    logger.info(f"✅ Created {len(chunks)} chunks")
    return chunks

def get_base_filename(pdf_name):
    return os.path.splitext(pdf_name)[0]

def is_pdf_processed(pdf_name):
    base_name = get_base_filename(pdf_name)
    
    chunks_exists = storage.file_exists(f"{base_name}_chunks.json")
    embeddings_exists = storage.file_exists(f"{base_name}_embeddings.json")
    
    processed = chunks_exists and embeddings_exists
    logger.info(f"🔍 PDF {pdf_name} processed: {processed}")
    
    return processed

def process_new_pdf(pdf_file, pdf_name):
    base_name = get_base_filename(pdf_name)
    
    logger.info(f"🔄 Starting to process PDF: {pdf_name}")
    
    temp_pdf_path = storage.save_file(pdf_file, pdf_name)
    
    try:
        logger.info("📖 Step 1: Extracting text from PDF...")
        text = extract_text_from_pdf(temp_pdf_path)
        
        logger.info("✂️ Step 2: Creating text chunks...")
        chunks = chunk_text(text)
        storage.save_json(chunks, f"{base_name}_chunks.json")
        
        logger.info("🧠 Step 3: Generating embeddings...")
        embeddings = rag_system.add_new_documents(pdf_name, chunks)
        storage.save_json(embeddings, f"{base_name}_embeddings.json")
        
        session_pdf_key = f"pdf_{pdf_name}"
        processed_pdfs[session_pdf_key] = {
            'name': pdf_name,
            'chunks_count': len(chunks),
            'processed_at': datetime.utcnow().isoformat()
        }
        
        logger.info(f"✅ Successfully processed new PDF: {pdf_name} with {len(chunks)} chunks")
        return len(chunks)
        
    except Exception as e:
        logger.error(f"❌ PDF processing failed for {pdf_name}: {e}")
        logger.error(traceback.format_exc())
        raise
    finally:
        if os.path.exists(temp_pdf_path):
            os.remove(temp_pdf_path)
            logger.debug(f"🧹 Cleaned up temp PDF: {temp_pdf_path}")

def load_processed_pdf(pdf_name):
    base_name = get_base_filename(pdf_name)
    
    try:
        logger.info(f"📥 Loading processed PDF: {pdf_name}")
        
        chunks = storage.load_json(f"{base_name}_chunks.json")
        embeddings = storage.load_json(f"{base_name}_embeddings.json")
        
        if chunks and embeddings:
            rag_system.load_documents_from_storage(pdf_name, chunks, embeddings)
            logger.info(f"✅ Successfully loaded processed PDF: {pdf_name} with {len(chunks)} chunks")
            return len(chunks)
        else:
            raise Exception(f"Missing chunks or embeddings for {pdf_name}")
            
    except Exception as e:
        logger.error(f"❌ Failed to load processed PDF {pdf_name}: {e}")
        logger.error(traceback.format_exc())
        raise Exception(f"Failed to load processed PDF: {str(e)}")

def list_available_pdfs():
    try:
        chunk_files = [f for f in storage.list_files() if f.endswith('_chunks.json')]
        pdfs = []
        
        for chunk_file in chunk_files:
            if '_chunks.json' in chunk_file:
                pdf_name = chunk_file.replace('_chunks.json', '') + '.pdf'
                pdfs.append(pdf_name)
        
        logger.info(f"📚 Found {len(pdfs)} processed PDFs in storage")
        return pdfs
        
    except Exception as e:
        logger.error(f"❌ Failed to list PDFs: {e}")
        return []

# ==================== FLASK ROUTES ====================
@app.before_request
def make_session_permanent():
    session.permanent = True

@app.errorhandler(404)
def not_found(error):
    logger.warning(f"404 Error: {request.url}")
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(413)
def too_large(error):
    logger.warning(f"413 Error: File too large")
    return jsonify({"error": "File too large. Maximum size is 16MB"}), 413

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"500 Internal Server Error: {error}")
    logger.error(traceback.format_exc())
    return jsonify({"error": "Internal server error"}), 500

@app.route("/")
def index():
    try:
        pdfs = list_available_pdfs()
        
        if 'session_id' not in session:
            session['session_id'] = str(uuid.uuid4())
            session['current_pdf'] = None
            session['has_documents'] = False
        
        current_pdf = rag_system.current_pdf_name
        has_documents = len(rag_system.chunks) > 0
        
        logger.info(f"🏠 Serving index page - {len(pdfs)} PDFs available, current PDF: {current_pdf}")
        
        return render_template("index.html", 
                             pdfs=pdfs, 
                             clients_ok=True,
                             current_pdf=current_pdf,
                             has_documents=has_documents)
                             
    except Exception as e:
        logger.error(f"❌ Index route error: {e}")
        logger.error(traceback.format_exc())
        return "Error loading page", 500

@app.route("/upload", methods=["POST"])
def upload_pdf():
    try:
        if 'pdf' not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        pdf_file = request.files['pdf']
        if pdf_file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        if not pdf_file.filename.lower().endswith('.pdf'):
            return jsonify({"error": "Only PDF files are allowed"}), 400
        
        pdf_name = secure_filename(pdf_file.filename)
        
        logger.info(f"📤 Processing new PDF upload: {pdf_name}")
        
        chunk_count = process_new_pdf(pdf_file, pdf_name)
        
        session['current_pdf'] = pdf_name
        session['has_documents'] = True
        
        return jsonify({
            "success": f"PDF '{pdf_name}' uploaded and processed successfully! Ready for chatting.",
            "pdf_name": pdf_name,
            "chunks": chunk_count
        })
        
    except Exception as e:
        logger.error(f"❌ Upload error: {e}")
        return jsonify({"error": f"Upload failed: {str(e)}"}), 500

@app.route("/load", methods=["POST"])
def load_existing_pdf():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
            
        pdf_name = data.get("pdf_name")
        if not pdf_name:
            return jsonify({"error": "PDF name required"}), 400
        
        logger.info(f"📥 Attempting to load PDF: {pdf_name}")
        
        if not is_pdf_processed(pdf_name):
            logger.error(f"❌ PDF not processed: {pdf_name}")
            return jsonify({"error": f"PDF '{pdf_name}' is not processed yet. Please upload it first."}), 400
        
        logger.info(f"✅ PDF {pdf_name} is processed, loading...")
        
        chunk_count = load_processed_pdf(pdf_name)
        
        session['current_pdf'] = pdf_name
        session['has_documents'] = True
        
        logger.info(f"✅ Successfully loaded PDF {pdf_name} with {chunk_count} chunks")
        
        return jsonify({
            "success": f"PDF '{pdf_name}' loaded successfully! Ready for chatting.",
            "pdf_name": pdf_name,
            "chunks": chunk_count
        })
        
    except Exception as e:
        logger.error(f"❌ Load error: {e}")
        return jsonify({"error": f"Load failed: {str(e)}"}), 500

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
            
        user_input = data.get("message", "").strip()
        
        if not user_input:
            return jsonify({"error": "Empty message"}), 400
        
        current_pdf_name = rag_system.current_pdf_name
        
        logger.info(f"💬 Chat request - PDF: {current_pdf_name}, Query: '{user_input}'")
        
        relevant_chunks = []
        context = ""
        
        if rag_system.chunks and current_pdf_name:
            relevant_chunks = rag_system.search(user_input, k=8)
            logger.info(f"🔍 Found {len(relevant_chunks)} relevant chunks")
            
            if relevant_chunks:
                context_chunks = [chunk for chunk, score in relevant_chunks]
                context = "\n\n".join(context_chunks)
                logger.info(f"📝 Context length: {len(context)} characters")
        
        response = llm_client.generate_response(user_input, context, current_pdf_name)
        logger.info(f"🤖 Generated response length: {len(response)}")
        
        return jsonify({
            "response": response,
            "pdf_name": current_pdf_name,
            "relevant_chunks_count": len(relevant_chunks)
        })
        
    except Exception as e:
        logger.error(f"❌ Chat error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({"error": f"Chat processing failed: {str(e)}"}), 500

@app.route("/clear", methods=["POST"])
def clear_documents():
    try:
        current_pdf = rag_system.current_pdf_name
        rag_system.clear_documents()
        
        session['current_pdf'] = None
        session['has_documents'] = False
        
        logger.info(f"🧹 Cleared documents for PDF: {current_pdf}")
        
        return jsonify({
            "success": "Current PDF cleared from memory",
            "cleared_pdf": current_pdf
        })
    except Exception as e:
        logger.error(f"❌ Clear documents error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/health")
def health():
    try:
        rag_healthy = rag_system is not None
        rag_chunks_loaded = len(rag_system.chunks) if rag_system else 0
        
        llm_healthy = llm_client is not None and hasattr(llm_client, 'configured') and llm_client.configured
        
        storage_healthy = True
        
        if rag_healthy and llm_healthy:
            status = "healthy"
        elif rag_healthy:
            status = "degraded"
        else:
            status = "unhealthy"
        
        return jsonify({
            "status": status,
            "timestamp": datetime.utcnow().isoformat(),
            "deployment": "Render",
            "storage_healthy": storage_healthy,
            "rag_system_healthy": rag_healthy,
            "llm_client_healthy": llm_healthy,
            "current_pdf": rag_system.current_pdf_name if rag_system else None,
            "rag_chunks_loaded": rag_chunks_loaded,
            "available_pdfs": len(list_available_pdfs()),
            "version": "2.0.0-render",
            "environment": os.environ.get("FLASK_ENV", "production")
        })
    except Exception as e:
        logger.error(f"❌ Health check failed: {e}")
        return jsonify({"status": "unhealthy", "error": str(e)}), 500

if __name__ == "__main__":
    logger.info(f"🚀 Starting PDF RAG Chatbot on Render (port: {PORT})")
    logger.info(f"🔑 Gemini API Key configured: {bool(GEMINI_API_KEY)}")
    logger.info(f"📁 Temporary storage: {storage.base_folder}")
    app.run(host="0.0.0.0", port=PORT, debug=os.environ.get("FLASK_ENV") == "development")
