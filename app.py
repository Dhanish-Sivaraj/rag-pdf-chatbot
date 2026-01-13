import os
import uuid
import tempfile
import re
import logging
import traceback
import json
from datetime import datetime, timedelta
from flask import Flask, render_template, request, jsonify, session
from werkzeug.utils import secure_filename
import pypdf
from utils.rag_utils import RAGSystem, LLMClient

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

# Initialize components with lazy loading
logger.info("🔄 Initializing RAG system and LLM client...")
rag_system = RAGSystem()
llm_client = LLMClient()

# Simple local storage management for Render
class RenderStorage:
    def __init__(self, base_folder=None):
        self.base_folder = base_folder or tempfile.gettempdir()
        self.documents = {}
        os.makedirs(self.base_folder, exist_ok=True)
        
    def save_file(self, file, filename):
        """Save uploaded file to temporary storage"""
        filepath = os.path.join(self.base_folder, secure_filename(filename))
        file.save(filepath)
        logger.info(f"💾 Saved file: {filename} to {filepath}")
        return filepath
    
    def save_json(self, data, filename):
        """Save JSON data to temporary storage"""
        filepath = os.path.join(self.base_folder, secure_filename(filename))
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        logger.info(f"💾 Saved JSON: {filename}")
        return filepath
    
    def load_json(self, filename):
        """Load JSON data from temporary storage"""
        filepath = os.path.join(self.base_folder, secure_filename(filename))
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        logger.warning(f"⚠️ File not found: {filename}")
        return None
    
    def file_exists(self, filename):
        """Check if file exists in storage"""
        filepath = os.path.join(self.base_folder, secure_filename(filename))
        return os.path.exists(filepath)
    
    def list_files(self, extension=None):
        """List files in storage"""
        files = []
        for f in os.listdir(self.base_folder):
            if extension and not f.endswith(extension):
                continue
            files.append(f)
        return files
    
    def delete_file(self, filename):
        """Delete file from storage"""
        filepath = os.path.join(self.base_folder, secure_filename(filename))
        if os.path.exists(filepath):
            os.remove(filepath)
            logger.info(f"🗑️ Deleted file: {filename}")
            return True
        return False

# Initialize storage
storage = RenderStorage()

# Track currently processed PDFs in memory (session-based)
processed_pdfs = {}

# ----------------- Helper Functions -----------------
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

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF file with enhanced cleaning and error handling"""
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
                    # Apply text cleaning during extraction
                    page_text = clean_text(page_text)
                    
                    text += page_text + "\n"
                
                if (page_num + 1) % 10 == 0:  # Log progress every 10 pages
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
    """Split text into coherent chunks with intelligent paragraph handling"""
    if not text:
        logger.warning("Empty text provided for chunking")
        return []
    
    # Apply text cleaning before chunking
    text = clean_text(text)
    
    logger.info(f"✂️ Chunking text of {len(text)} characters...")
    
    # try paragraph-based chunking (most coherent)
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip() and len(p.strip()) > 50]
    
    chunks = []
    for i, paragraph in enumerate(paragraphs):
        if len(paragraph) <= chunk_size:
            chunks.append(paragraph)
        else:
            # Split long paragraphs by sentences
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
    
    # Fallback: word-based chunking for texts without clear paragraphs
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
    """Get base filename without extension"""
    return os.path.splitext(pdf_name)[0]

def is_pdf_processed(pdf_name):
    """Check if PDF is already processed (has chunks and embeddings)"""
    base_name = get_base_filename(pdf_name)
    
    chunks_exists = storage.file_exists(f"{base_name}_chunks.json")
    embeddings_exists = storage.file_exists(f"{base_name}_embeddings.json")
    
    processed = chunks_exists and embeddings_exists
    logger.info(f"🔍 PDF {pdf_name} processed: {processed}")
    
    return processed

def process_new_pdf(pdf_file, pdf_name):
    """Process a new PDF: extract text, create chunks, generate embeddings"""
    base_name = get_base_filename(pdf_name)
    
    logger.info(f"🔄 Starting to process PDF: {pdf_name}")
    
    # Save file temporarily
    temp_pdf_path = storage.save_file(pdf_file, pdf_name)
    
    try:
        # 1. Extract text from PDF
        logger.info("📖 Step 1: Extracting text from PDF...")
        text = extract_text_from_pdf(temp_pdf_path)
        
        # 2. Create chunks and save to storage
        logger.info("✂️ Step 2: Creating text chunks...")
        chunks = chunk_text(text)
        storage.save_json(chunks, f"{base_name}_chunks.json")
        
        # 3. Generate embeddings and save to storage
        logger.info("🧠 Step 3: Generating embeddings...")
        embeddings = rag_system.add_new_documents(pdf_name, chunks)
        storage.save_json(embeddings, f"{base_name}_embeddings.json")
        
        # Store in memory for this session
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
        # Cleanup temp PDF file
        if os.path.exists(temp_pdf_path):
            os.remove(temp_pdf_path)
            logger.debug(f"🧹 Cleaned up temp PDF: {temp_pdf_path}")

def load_processed_pdf(pdf_name):
    """Load already processed PDF from storage with error handling"""
    base_name = get_base_filename(pdf_name)
    
    try:
        logger.info(f"📥 Loading processed PDF: {pdf_name}")
        
        # Download chunks and embeddings
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
    """List PDFs available in storage"""
    try:
        # Get all .json chunk files
        chunk_files = [f for f in storage.list_files() if f.endswith('_chunks.json')]
        pdfs = []
        
        for chunk_file in chunk_files:
            # Extract PDF name from chunk file name
            # Format: {basename}_chunks.json
            if '_chunks.json' in chunk_file:
                pdf_name = chunk_file.replace('_chunks.json', '') + '.pdf'
                pdfs.append(pdf_name)
        
        logger.info(f"📚 Found {len(pdfs)} processed PDFs in storage")
        return pdfs
        
    except Exception as e:
        logger.error(f"❌ Failed to list PDFs: {e}")
        return []

# ----------------- Session Management -----------------
@app.before_request
def make_session_permanent():
    """Make session permanent and set timeout"""
    session.permanent = True

# ----------------- Error Handlers -----------------
@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    logger.warning(f"404 Error: {request.url}")
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(413)
def too_large(error):
    """Handle file too large errors"""
    logger.warning(f"413 Error: File too large")
    return jsonify({"error": "File too large. Maximum size is 16MB"}), 413

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors with detailed logging"""
    logger.error(f"500 Internal Server Error: {error}")
    logger.error(traceback.format_exc())
    return jsonify({"error": "Internal server error"}), 500

# ----------------- Routes -----------------
@app.route("/")
def index():
    """Main page - serve the chat interface"""
    try:
        pdfs = list_available_pdfs()
        
        # Initialize session with security
        if 'session_id' not in session:
            session['session_id'] = str(uuid.uuid4())
            session['current_pdf'] = None
            session['has_documents'] = False
        
        # Get current PDF from RAG system, not session (more reliable)
        current_pdf = rag_system.current_pdf_name
        has_documents = len(rag_system.chunks) > 0 if rag_system else False
        
        logger.info(f"🏠 Serving index page - {len(pdfs)} PDFs available, current PDF: {current_pdf}")
        
        return render_template("index.html", 
                             pdfs=pdfs, 
                             clients_ok=True,  # Always true for Render
                             current_pdf=current_pdf,
                             has_documents=has_documents)
                             
    except Exception as e:
        logger.error(f"❌ Index route error: {e}")
        logger.error(traceback.format_exc())
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
        
        # Secure the filename
        pdf_name = secure_filename(pdf_file.filename)
        
        logger.info(f"📤 Processing new PDF upload: {pdf_name}")
        
        # Process the PDF
        chunk_count = process_new_pdf(pdf_file, pdf_name)
        
        # Update session
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
    """Load existing PDF from storage"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
            
        pdf_name = data.get("pdf_name")
        if not pdf_name:
            return jsonify({"error": "PDF name required"}), 400
        
        logger.info(f"📥 Attempting to load PDF: {pdf_name}")
        
        # Check if PDF is already processed
        if not is_pdf_processed(pdf_name):
            logger.error(f"❌ PDF not processed: {pdf_name}")
            return jsonify({"error": f"PDF '{pdf_name}' is not processed yet. Please upload it first."}), 400
        
        logger.info(f"✅ PDF {pdf_name} is processed, loading...")
        
        # Load the processed PDF
        chunk_count = load_processed_pdf(pdf_name)
        
        # Update session
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
    """Handle chat messages with RAG - main conversation endpoint"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
            
        user_input = data.get("message", "").strip()
        
        if not user_input:
            return jsonify({"error": "Empty message"}), 400
        
        current_pdf_name = rag_system.current_pdf_name
        
        logger.info(f"💬 Chat request - PDF: {current_pdf_name}, Query: '{user_input}'")
        
        # Search for relevant chunks if we have a PDF loaded
        relevant_chunks = []
        context = ""
        
        if rag_system.chunks and current_pdf_name:
            # Search for more chunks to get better context
            relevant_chunks = rag_system.search(user_input, k=8)
            logger.info(f"🔍 Found {len(relevant_chunks)} relevant chunks")
            
            if relevant_chunks:
                context_chunks = [chunk for chunk, score in relevant_chunks]
                context = "\n\n".join(context_chunks)
                logger.info(f"📝 Context length: {len(context)} characters")
        
        # Generate response using LLM 
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
    """Clear current documents from RAG system"""
    try:
        current_pdf = rag_system.current_pdf_name
        
        # Clear from RAG system
        rag_system.clear_documents()
        
        # Clear session
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

@app.route("/clear-storage", methods=["POST"])
def clear_storage():
    """Clear all stored PDFs from temporary storage (for development)"""
    try:
        # Only allow in development or with authentication in production
        if os.environ.get("FLASK_ENV") != "development":
            return jsonify({"error": "This endpoint is only available in development mode"}), 403
        
        cleared_files = 0
        all_files = storage.list_files()
        
        for filename in all_files:
            if storage.delete_file(filename):
                cleared_files += 1
        
        # Clear RAG system
        rag_system.clear_documents()
        
        # Clear session
        session.clear()
        
        logger.info(f"🧹 Cleared {cleared_files} files from storage")
        
        return jsonify({
            "success": f"Cleared {cleared_files} files from storage",
            "cleared_count": cleared_files
        })
        
    except Exception as e:
        logger.error(f"❌ Clear storage error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/health")
def health():
    """Comprehensive health check endpoint"""
    try:
        # Check RAG system status
        rag_healthy = rag_system is not None
        rag_chunks_loaded = len(rag_system.chunks) if rag_system else 0
        
        # Check LLM client status
        llm_healthy = llm_client is not None and hasattr(llm_client, 'configured') and llm_client.configured
        
        # Check storage status
        storage_healthy = True  # Local storage is always available
        
        # Determine overall status
        if rag_healthy and llm_healthy:
            status = "healthy"
        elif rag_healthy:
            status = "degraded"  # LLM might be disabled
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

@app.route("/status")
def status():
    """Detailed status endpoint"""
    try:
        pdfs = list_available_pdfs()
        
        return jsonify({
            "status": "operational",
            "timestamp": datetime.utcnow().isoformat(),
            "rag_system": {
                "initialized": rag_system is not None,
                "current_pdf": rag_system.current_pdf_name if rag_system else None,
                "chunks_loaded": len(rag_system.chunks) if rag_system else 0,
                "index_built": rag_system.index is not None if rag_system else False
            },
            "llm_client": {
                "initialized": llm_client is not None,
                "configured": llm_client.configured if llm_client else False,
                "gemini_available": bool(GEMINI_API_KEY)
            },
            "storage": {
                "type": "local_temporary",
                "base_folder": storage.base_folder,
                "files_count": len(storage.list_files()),
                "available_pdfs": pdfs
            },
            "session": {
                "session_id": session.get('session_id'),
                "current_pdf": session.get('current_pdf'),
                "has_documents": session.get('has_documents', False)
            },
            "limits": {
                "max_upload_size": "16MB",
                "chunk_size": "600 characters",
                "session_lifetime": "1 hour"
            }
        })
    except Exception as e:
        logger.error(f"❌ Status endpoint error: {e}")
        return jsonify({"error": str(e)}), 500

# Test endpoint to verify the system is working
@app.route("/test", methods=["GET"])
def test_endpoint():
    """Test endpoint to verify system functionality"""
    return jsonify({
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat(),
        "rag_system_ready": rag_system is not None,
        "llm_client_ready": llm_client is not None and hasattr(llm_client, 'configured') and llm_client.configured,
        "current_pdf": rag_system.current_pdf_name if rag_system else None,
        "chunks_loaded": len(rag_system.chunks) if rag_system else 0,
        "message": "PDF RAG Chatbot is running correctly on Render!",
        "deployment": "Render",
        "port": PORT
    })

@app.route("/api/info")
def api_info():
    """API information endpoint"""
    return jsonify({
        "name": "PDF RAG Chatbot",
        "version": "2.0.0",
        "description": "Retrieval-Augmented Generation chatbot for PDF documents",
        "deployment": "Render",
        "endpoints": {
            "GET /": "Main interface",
            "POST /upload": "Upload and process PDF",
            "POST /load": "Load existing PDF",
            "POST /chat": "Chat with loaded PDF",
            "POST /clear": "Clear current PDF",
            "GET /health": "Health check",
            "GET /status": "Detailed status",
            "GET /test": "Test endpoint"
        },
        "features": [
            "PDF text extraction",
            "Intelligent text chunking",
            "FAISS vector embeddings",
            "Semantic similarity search",
            "Google Gemini AI responses",
            "Session-based document management"
        ]
    })

if __name__ == "__main__":
    logger.info(f"🚀 Starting PDF RAG Chatbot on Render (port: {PORT})")
    logger.info(f"🔑 Gemini API Key configured: {bool(GEMINI_API_KEY)}")
    logger.info(f"📁 Temporary storage: {storage.base_folder}")
    app.run(host="0.0.0.0", port=PORT, debug=os.environ.get("FLASK_ENV") == "development")
