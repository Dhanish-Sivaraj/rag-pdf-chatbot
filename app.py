import os
import uuid
import tempfile
import re
import logging
import traceback
import json
import base64
import pickle
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

# Use /data for persistent storage if available, otherwise /tmp
STORAGE_PATH = '/data' if os.path.exists('/data') else tempfile.gettempdir()

logger.info(f"🚀 Initializing PDF RAG Chatbot for Render")
logger.info(f"📡 Port: {PORT}, Storage: {STORAGE_PATH}, Gemini API Key: {bool(GEMINI_API_KEY)}")

# Flask App Configuration
app = Flask(__name__)
app.secret_key = SECRET_KEY
app.config['UPLOAD_FOLDER'] = STORAGE_PATH
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=2)  # Extended session
app.config['SESSION_TYPE'] = 'filesystem'

# Ensure storage directory exists
os.makedirs(STORAGE_PATH, exist_ok=True)

# ==================== HYBRID STORAGE SYSTEM ====================
class HybridStorage:
    """
    Hybrid storage system for Render:
    1. Uses session for active PDFs (survives within session)
    2. Uses file storage when possible (survives restarts if /data mounted)
    3. Falls back gracefully
    """
    
    def __init__(self, base_path=None):
        self.base_path = base_path or STORAGE_PATH
        os.makedirs(self.base_path, exist_ok=True)
        logger.info(f"💾 HybridStorage initialized at: {self.base_path}")
        
    def save_pdf_temporary(self, pdf_file, pdf_name):
        """Save PDF temporarily for processing"""
        temp_path = os.path.join(self.base_path, secure_filename(pdf_name))
        pdf_file.save(temp_path)
        logger.info(f"📥 PDF saved temporarily: {temp_path}")
        return temp_path
        
    def save_chunks(self, pdf_name, chunks):
        """Save chunks using multiple strategies"""
        base_name = get_base_filename(pdf_name)
        
        # Strategy 1: Save to file (persistent if /data exists)
        try:
            file_path = os.path.join(self.base_path, f"{base_name}_chunks.json")
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(chunks, f, indent=2)
            logger.info(f"💾 Chunks saved to file: {file_path}")
        except Exception as e:
            logger.warning(f"⚠️ Could not save chunks to file: {e}")
            
        # Strategy 2: Save to session (for active use)
        try:
            session_key = f"chunks_{base_name}"
            # Store only first 50 chunks in session to avoid size limits
            session_chunks = chunks[:50] if len(chunks) > 50 else chunks
            session[session_key] = json.dumps(session_chunks)
            logger.info(f"💾 {len(session_chunks)} chunks saved to session")
        except Exception as e:
            logger.warning(f"⚠️ Could not save chunks to session: {e}")
            
        # Strategy 3: Save metadata
        self.save_metadata(pdf_name, {'chunks_count': len(chunks), 'saved_at': datetime.utcnow().isoformat()})
        
        return len(chunks)
        
    def load_chunks(self, pdf_name):
        """Load chunks with fallback strategies"""
        base_name = get_base_filename(pdf_name)
        
        # Try file storage first
        try:
            file_path = os.path.join(self.base_path, f"{base_name}_chunks.json")
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    chunks = json.load(f)
                logger.info(f"📤 Loaded {len(chunks)} chunks from file: {file_path}")
                return chunks
        except Exception as e:
            logger.warning(f"⚠️ Could not load chunks from file: {e}")
            
        # Try session storage
        try:
            session_key = f"chunks_{base_name}"
            if session_key in session:
                chunks = json.loads(session[session_key])
                logger.info(f"📤 Loaded {len(chunks)} chunks from session")
                return chunks
        except Exception as e:
            logger.warning(f"⚠️ Could not load chunks from session: {e}")
            
        return None
        
    def save_embeddings(self, pdf_name, embeddings):
        """Save embeddings - store only metadata in session, full in file"""
        base_name = get_base_filename(pdf_name)
        
        # Save full embeddings to file
        try:
            file_path = os.path.join(self.base_path, f"{base_name}_embeddings.json")
            # Convert numpy arrays to lists for JSON
            embeddings_list = [emb.tolist() if isinstance(emb, np.ndarray) else emb for emb in embeddings]
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(embeddings_list, f)
            logger.info(f"💾 Embeddings saved to file: {file_path}")
        except Exception as e:
            logger.warning(f"⚠️ Could not save embeddings to file: {e}")
            
        # Save only metadata to session
        try:
            session[f"embeddings_{base_name}"] = True
        except Exception as e:
            logger.warning(f"⚠️ Could not save embeddings metadata to session: {e}")
            
    def load_embeddings(self, pdf_name):
        """Load embeddings from file"""
        base_name = get_base_filename(pdf_name)
        
        try:
            file_path = os.path.join(self.base_path, f"{base_name}_embeddings.json")
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    embeddings = json.load(f)
                logger.info(f"📤 Loaded embeddings from file: {file_path}")
                return embeddings
        except Exception as e:
            logger.warning(f"⚠️ Could not load embeddings: {e}")
            
        return None
        
    def save_metadata(self, pdf_name, metadata):
        """Save PDF metadata"""
        base_name = get_base_filename(pdf_name)
        
        # Update available PDFs list
        available = self.get_available_pdfs()
        if pdf_name not in available:
            available.append(pdf_name)
            session['available_pdfs'] = available
            
        # Save individual metadata
        session[f"meta_{base_name}"] = json.dumps(metadata)
        
    def get_metadata(self, pdf_name):
        """Get PDF metadata"""
        base_name = get_base_filename(pdf_name)
        meta_key = f"meta_{base_name}"
        if meta_key in session:
            return json.loads(session[meta_key])
        return None
        
    def get_available_pdfs(self):
        """Get list of available PDFs"""
        return session.get('available_pdfs', [])
        
    def has_pdf(self, pdf_name):
        """Check if PDF is available"""
        # Check file storage
        base_name = get_base_filename(pdf_name)
        chunks_file = os.path.exists(os.path.join(self.base_path, f"{base_name}_chunks.json"))
        embeddings_file = os.path.exists(os.path.join(self.base_path, f"{base_name}_embeddings.json"))
        
        # Check session
        chunks_session = f"chunks_{base_name}" in session
        embeddings_session = f"embeddings_{base_name}" in session
        
        return chunks_file or chunks_session
        
    def cleanup(self, pdf_name):
        """Clean up temporary files"""
        try:
            base_name = get_base_filename(pdf_name)
            
            # Remove from available PDFs
            available = self.get_available_pdfs()
            if pdf_name in available:
                available.remove(pdf_name)
                session['available_pdfs'] = available
                
            # Remove session data
            for key in ['chunks_', 'embeddings_', 'meta_']:
                session_key = f"{key}{base_name}"
                if session_key in session:
                    session.pop(session_key)
                    
            logger.info(f"🧹 Cleaned up storage for: {pdf_name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Cleanup failed: {e}")
            return False

# Initialize storage
storage = HybridStorage()

# ==================== UTILITY FUNCTIONS ====================
def clean_text(text: str) -> str:
    """Clean text from PDF extraction issues"""
    if not text:
        return ""
    
    # Fix encoding issues
    text = (text.replace('ﬁ', 'fi')
                .replace('ﬂ', 'fl')
                .replace('ﬀ', 'ff')
                .replace('ﬃ', 'ffi')
                .replace('ﬄ', 'ffl'))
    
    # Fix common broken patterns
    text = re.sub(r'\b([A-Z]) ([A-Z]) ([A-Z])\b', r'\1\2\3', text)
    text = re.sub(r'\b([A-Z]) ([A-Z])\b', r'\1\2', text)
    
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
    
    # Normalize whitespace
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
                logger.info(f"🔄 Loading model weights...")
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModel.from_pretrained(self.model_name)
                logger.info("✅ Embedding model loaded!")
            except Exception as e:
                logger.error(f"❌ Failed to load model: {e}")
                raise

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
            logger.error(f"Embedding failed: {e}")
            return np.zeros(self.vector_size)

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        if not texts:
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
        logger.info("✅ RAG System initialized!")

    def create_embeddings(self, chunks: List[str]) -> np.ndarray:
        logger.info(f"Creating embeddings for {len(chunks)} chunks")
        if not chunks:
            raise ValueError("No chunks to process")
        
        embeddings = self.embedder.embed_batch(chunks)
        logger.info(f"✅ Created {len(embeddings)} embeddings")
        return embeddings

    def build_index(self, embeddings: np.ndarray):
        if len(embeddings) == 0:
            raise ValueError("No embeddings available")
        
        logger.info(f"Building FAISS index with {embeddings.shape[1]} dimensions")
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        logger.info(f"✅ FAISS index built with {self.index.ntotal} vectors")

    def load_documents_from_storage(self, pdf_name: str, chunks: List[str], embeddings: List[List[float]]):
        logger.info(f"Loading documents for {pdf_name}")
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
        
        # Return embeddings as list for storage
        return embeddings.tolist()

    def search(self, query: str, k: int = 8) -> List[Tuple[str, float]]:
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
        logger.info("✅ Documents cleared")

# ==================== GEMINI LLM CLIENT ====================
class GeminiLLMClient:
    def __init__(self):
        logger.info("🚀 Initializing Gemini LLM Client")
        self.model = None
        self.configured = False
        
        try:
            api_key = os.environ.get("GEMINI_API_KEY")
            if not api_key:
                logger.warning("❌ GEMINI_API_KEY not found")
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
                    logger.info(f"🔄 Trying model: {model_name}")
                    self.model = genai.GenerativeModel(model_name)
                    
                    test_response = self.model.generate_content("Say 'TEST' in one word.")
                    if test_response and test_response.text:
                        logger.info(f"✅ Model loaded: {model_name}")
                        break
                    self.model = None
                        
                except Exception:
                    continue
            
            if self.model is None:
                logger.error("❌ All Gemini models failed")
                return
            
            self.configured = True
            logger.info("✅ Gemini LLM client initialized!")
            
        except Exception as e:
            logger.error(f"❌ Gemini initialization failed: {e}")
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
        elif 'help' in query_lower:
            return "🤖 I can read PDFs and provide intelligent summaries and answers!"
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
                logger.error(f"❌ Gemini error: {e}")
                return self._fallback_response(query, pdf_name)
        else:
            return self._fallback_response(query, pdf_name)
    
    def _generate_with_gemini(self, query: str, context: str, pdf_name: str) -> str:
        try:
            if not context or not context.strip():
                return f"""🤔 I searched through **{pdf_name}** but couldn't find specific information about '{query}'.

**Suggestions:**
- Try rephrasing your question
- Ask about specific topics mentioned in the document
- Check if the PDF contains the information you're looking for"""

            prompt = f"""
            You are an AI assistant that answers questions based on the provided document context.
            
            DOCUMENT CONTEXT FROM "{pdf_name}":
            {context}
            
            USER QUESTION: {query}
            
            IMPORTANT INSTRUCTIONS:
            1. Answer based primarily on the document context
            2. If the context doesn't directly answer the question, explain what the document DOES say about related topics
            3. Be honest about what information is and isn't in the document
            4. Use bullet points if helpful for organization
            
            Please provide your answer:
            """
            
            response = self.model.generate_content(prompt)
            
            if not response or not response.text:
                return self._fallback_response(query, pdf_name)
            
            return f"**Based on {pdf_name}**:\n\n{response.text}"
            
        except Exception as e:
            logger.error(f"❌ Gemini generation failed: {e}")
            raise
    
    def _fallback_response(self, query: str, pdf_name: str) -> str:
        return f"**Based on {pdf_name}**:\n\nI found relevant content but couldn't generate a detailed AI response. The document contains information related to '{query}'.\n\n🔧 *Note: AI response generation is currently unavailable.*"

# Initialize components
rag_system = AdvancedRAGSystem()
llm_client = GeminiLLMClient()

# ==================== PDF PROCESSING FUNCTIONS ====================
def get_base_filename(pdf_name):
    """Get base filename without extension"""
    return os.path.splitext(pdf_name)[0]

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF file"""
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
            raise Exception("No text content extracted")
        
        logger.info(f"✅ Extracted {len(text)} characters from {total_pages} pages")
        return text.strip()
        
    except Exception as e:
        logger.error(f"❌ PDF extraction failed: {e}")
        raise Exception(f"PDF processing error: {str(e)}")

def chunk_text(text, chunk_size=600):
    """Split text into coherent chunks"""
    if not text:
        return []
    
    text = clean_text(text)
    logger.info(f"✂️ Chunking text of {len(text)} characters...")
    
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip() and len(p.strip()) > 50]
    
    chunks = []
    for paragraph in paragraphs:
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

def process_new_pdf(pdf_file, pdf_name):
    """Process a new PDF: extract text, create chunks, generate embeddings"""
    logger.info(f"🔄 Starting to process PDF: {pdf_name}")
    
    # Save PDF temporarily for processing
    temp_pdf_path = storage.save_pdf_temporary(pdf_file, pdf_name)
    
    try:
        # 1. Extract text
        logger.info("📖 Step 1: Extracting text from PDF...")
        text = extract_text_from_pdf(temp_pdf_path)
        
        # 2. Create chunks
        logger.info("✂️ Step 2: Creating text chunks...")
        chunks = chunk_text(text)
        
        # Save chunks using hybrid storage
        storage.save_chunks(pdf_name, chunks)
        
        # 3. Generate embeddings
        logger.info("🧠 Step 3: Generating embeddings...")
        embeddings = rag_system.add_new_documents(pdf_name, chunks)
        
        # Save embeddings
        storage.save_embeddings(pdf_name, embeddings)
        
        # Save metadata
        storage.save_metadata(pdf_name, {
            'chunks_count': len(chunks),
            'processed_at': datetime.utcnow().isoformat(),
            'size_mb': len(text) / (1024 * 1024)
        })
        
        # Update session
        session['current_pdf'] = pdf_name
        session['has_documents'] = True
        
        logger.info(f"✅ Successfully processed PDF: {pdf_name} with {len(chunks)} chunks")
        return len(chunks)
        
    except Exception as e:
        logger.error(f"❌ PDF processing failed: {e}")
        raise
    finally:
        # Cleanup temp PDF file
        if os.path.exists(temp_pdf_path):
            os.remove(temp_pdf_path)
            logger.debug(f"🧹 Cleaned up temp PDF")

def load_processed_pdf(pdf_name):
    """Load already processed PDF"""
    try:
        logger.info(f"📥 Loading PDF: {pdf_name}")
        
        # Load chunks
        chunks = storage.load_chunks(pdf_name)
        if not chunks:
            raise Exception(f"No chunks found for {pdf_name}")
        
        # Load embeddings
        embeddings = storage.load_embeddings(pdf_name)
        if not embeddings:
            raise Exception(f"No embeddings found for {pdf_name}")
        
        # Load into RAG system
        rag_system.load_documents_from_storage(pdf_name, chunks, embeddings)
        
        # Update session
        session['current_pdf'] = pdf_name
        session['has_documents'] = True
        
        logger.info(f"✅ Successfully loaded PDF: {pdf_name} with {len(chunks)} chunks")
        return len(chunks)
            
    except Exception as e:
        logger.error(f"❌ Failed to load PDF: {e}")
        raise

def list_available_pdfs():
    """List PDFs available in storage"""
    try:
        # Get from session storage
        pdfs = storage.get_available_pdfs()
        
        # Also check file system
        for file in os.listdir(storage.base_path):
            if file.endswith('_chunks.json'):
                pdf_name = file.replace('_chunks.json', '') + '.pdf'
                if pdf_name not in pdfs:
                    pdfs.append(pdf_name)
        
        logger.info(f"📚 Found {len(pdfs)} PDFs in storage")
        return sorted(pdfs)
        
    except Exception as e:
        logger.error(f"❌ Failed to list PDFs: {e}")
        return []

# ==================== FLASK ROUTES ====================
@app.before_request
def make_session_permanent():
    session.permanent = True

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(413)
def too_large(error):
    return jsonify({"error": "File too large. Maximum size is 16MB"}), 413

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"500 Error: {error}")
    return jsonify({"error": "Internal server error"}), 500

@app.route("/")
def index():
    """Main page - serve the chat interface"""
    try:
        pdfs = list_available_pdfs()
        
        # Initialize session
        if 'session_id' not in session:
            session['session_id'] = str(uuid.uuid4())
            session['current_pdf'] = None
            session['has_documents'] = False
        
        current_pdf = rag_system.current_pdf_name
        has_documents = len(rag_system.chunks) > 0
        
        # Add storage info
        storage_info = {
            'path': storage.base_path,
            'type': 'persistent' if storage.base_path == '/data' else 'temporary',
            'exists': os.path.exists(storage.base_path)
        }
        
        logger.info(f"🏠 Serving index page - {len(pdfs)} PDFs available")
        
        return render_template("index.html", 
                             pdfs=pdfs, 
                             clients_ok=True,
                             current_pdf=current_pdf,
                             has_documents=has_documents,
                             storage_info=storage_info)
                             
    except Exception as e:
        logger.error(f"❌ Index route error: {e}")
        return "Error loading page", 500

@app.route("/upload", methods=["POST"])
def upload_pdf():
    """Handle PDF upload and processing"""
    try:
        if 'pdf' not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        pdf_file = request.files['pdf']
        if pdf_file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        if not pdf_file.filename.lower().endswith('.pdf'):
            return jsonify({"error": "Only PDF files are allowed"}), 400
        
        pdf_name = secure_filename(pdf_file.filename)
        
        logger.info(f"📤 Processing upload: {pdf_name}")
        
        chunk_count = process_new_pdf(pdf_file, pdf_name)
        
        return jsonify({
            "success": f"PDF '{pdf_name}' processed successfully! Ready for chatting.",
            "pdf_name": pdf_name,
            "chunks": chunk_count,
            "storage_type": "persistent" if storage.base_path == '/data' else "session/temporary",
            "warning": "Note: Files are stored in session and will be lost when the app restarts or session expires." 
                       if storage.base_path != '/data' else ""
        })
        
    except Exception as e:
        logger.error(f"❌ Upload error: {e}")
        return jsonify({"error": f"Upload failed: {str(e)}"}), 500

@app.route("/load", methods=["POST"])
def load_existing_pdf():
    """Load existing PDF from storage"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
            
        pdf_name = data.get("pdf_name")
        if not pdf_name:
            return jsonify({"error": "PDF name required"}), 400
        
        logger.info(f"📥 Attempting to load PDF: {pdf_name}")
        
        # Check if PDF exists
        if not storage.has_pdf(pdf_name):
            return jsonify({"error": f"PDF '{pdf_name}' not found. Please upload it first."}), 404
        
        # Load the PDF
        chunk_count = load_processed_pdf(pdf_name)
        
        return jsonify({
            "success": f"PDF '{pdf_name}' loaded successfully!",
            "pdf_name": pdf_name,
            "chunks": chunk_count
        })
        
    except Exception as e:
        logger.error(f"❌ Load error: {e}")
        return jsonify({"error": f"Load failed: {str(e)}"}), 500

@app.route("/chat", methods=["POST"])
def chat():
    """Handle chat messages with RAG"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
            
        user_input = data.get("message", "").strip()
        
        if not user_input:
            return jsonify({"error": "Empty message"}), 400
        
        current_pdf_name = rag_system.current_pdf_name
        
        logger.info(f"💬 Chat - PDF: {current_pdf_name}, Query: '{user_input}'")
        
        # Search for relevant chunks
        relevant_chunks = []
        context = ""
        
        if rag_system.chunks and current_pdf_name:
            relevant_chunks = rag_system.search(user_input, k=8)
            logger.info(f"🔍 Found {len(relevant_chunks)} relevant chunks")
            
            if relevant_chunks:
                context_chunks = [chunk for chunk, score in relevant_chunks]
                context = "\n\n".join(context_chunks)
        
        # Generate response
        response = llm_client.generate_response(user_input, context, current_pdf_name)
        
        return jsonify({
            "response": response,
            "pdf_name": current_pdf_name,
            "relevant_chunks_count": len(relevant_chunks)
        })
        
    except Exception as e:
        logger.error(f"❌ Chat error: {e}")
        return jsonify({"error": f"Chat processing failed: {str(e)}"}), 500

@app.route("/clear", methods=["POST"])
def clear_documents():
    """Clear current documents from RAG system"""
    try:
        current_pdf = rag_system.current_pdf_name
        
        # Clear from RAG system
        rag_system.clear_documents()
        
        # Clear from session
        session['current_pdf'] = None
        session['has_documents'] = False
        
        logger.info(f"🧹 Cleared documents for PDF: {current_pdf}")
        
        return jsonify({
            "success": "Current PDF cleared from memory",
            "cleared_pdf": current_pdf
        })
    except Exception as e:
        logger.error(f"❌ Clear error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/clear-storage", methods=["POST"])
def clear_storage():
    """Clear all stored PDFs (development only)"""
    try:
        # Only allow in development
        if os.environ.get("FLASK_ENV") != "development":
            return jsonify({"error": "Not available in production"}), 403
        
        cleared = []
        for pdf_name in storage.get_available_pdfs():
            if storage.cleanup(pdf_name):
                cleared.append(pdf_name)
        
        # Clear RAG system
        rag_system.clear_documents()
        
        # Clear session
        session.clear()
        
        return jsonify({
            "success": f"Cleared {len(cleared)} PDFs",
            "cleared": cleared
        })
        
    except Exception as e:
        logger.error(f"❌ Clear storage error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/health")
def health():
    """Health check endpoint"""
    try:
        rag_healthy = rag_system is not None
        llm_healthy = llm_client.configured
        
        status = "healthy" if rag_healthy and llm_healthy else "degraded" if rag_healthy else "unhealthy"
        
        return jsonify({
            "status": status,
            "timestamp": datetime.utcnow().isoformat(),
            "deployment": "Render",
            "rag_system": rag_healthy,
            "llm_client": llm_healthy,
            "current_pdf": rag_system.current_pdf_name,
            "chunks_loaded": len(rag_system.chunks),
            "available_pdfs": len(list_available_pdfs()),
            "storage": {
                "path": storage.base_path,
                "type": "persistent" if storage.base_path == '/data' else "temporary",
                "files": len(os.listdir(storage.base_path)) if os.path.exists(storage.base_path) else 0
            },
            "session": {
                "id": session.get('session_id'),
                "has_documents": session.get('has_documents', False)
            }
        })
    except Exception as e:
        logger.error(f"❌ Health check failed: {e}")
        return jsonify({"status": "unhealthy", "error": str(e)}), 500

@app.route("/storage-info")
def storage_info():
    """Get storage information"""
    try:
        tmp_files = []
        data_files = []
        
        if os.path.exists('/tmp'):
            tmp_files = [f for f in os.listdir('/tmp') if f.endswith(('.json', '.pdf'))][:10]
        
        if os.path.exists('/data'):
            data_files = os.listdir('/data')
        
        return jsonify({
            "storage_path": storage.base_path,
            "tmp_files": tmp_files,
            "data_files": data_files,
            "session_keys": list(session.keys()),
            "available_pdfs": storage.get_available_pdfs(),
            "storage_type": "persistent" if storage.base_path == '/data' else "temporary",
            "warning": "Using temporary storage - files will be lost on app restart" 
                       if storage.base_path != '/data' else "Using persistent storage"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    logger.info(f"🚀 Starting PDF RAG Chatbot on Render")
    logger.info(f"📡 Port: {PORT}, Storage: {storage.base_path}")
    logger.info(f"🔑 Gemini API: {llm_client.configured}")
    app.run(host="0.0.0.0", port=PORT, debug=os.environ.get("FLASK_ENV") == "development")
