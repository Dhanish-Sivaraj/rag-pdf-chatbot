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
from google.cloud import storage
import pypdf
from utils.rag_utils import RAGSystem, LLMClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Environment Variables
PROJECT_ID = os.environ.get("PROJECT_ID", "your-project-id")
REGION = os.environ.get("REGION", "your-region")
BUCKET_NAME = os.environ.get("BUCKET_NAME", "your-bucket-name")
SECRET_KEY = os.environ.get("SECRET_KEY", "your-secret-key")

logger.info(f"🚀 Initializing PDF RAG Chatbot")
logger.info(f"📁 Project: {PROJECT_ID}, Region: {REGION}, Bucket: {BUCKET_NAME}")

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

# Cloud Storage Client Initialization
storage_client = None
bucket = None
clients_initialized = False

try:
    logger.info("☁️ Initializing Google Cloud Storage...")
    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET_NAME)
    
    if not bucket.exists():
        logger.info(f"📦 Creating bucket: {BUCKET_NAME} in {REGION}")
        bucket = storage_client.create_bucket(BUCKET_NAME, location=REGION)
        logger.info(f"✅ Bucket {BUCKET_NAME} created successfully")
    else:
        logger.info(f"✅ Bucket {BUCKET_NAME} already exists")
    
    clients_initialized = True
    logger.info("✅ Cloud Storage initialized successfully")
    
except Exception as e:
    logger.error(f"❌ Storage initialization failed: {e}")
    clients_initialized = False

# ----------------- Helper Functions -----------------
def upload_to_bucket(local_path, dest_folder, dest_filename):
    """Upload file to Google Cloud Storage bucket"""
    if not clients_initialized:
        raise Exception("Cloud storage not available")
    
    blob_name = f"{dest_folder}/{dest_filename}"
    try:
        blob = bucket.blob(blob_name)
        blob.upload_from_filename(local_path)
        logger.info(f"📤 Uploaded to: {blob_name}")
        return blob_name
    except Exception as e:
        logger.error(f"❌ Upload failed for {blob_name}: {e}")
        raise

def list_pdfs():
    """List all PDF files from the pdfs folder in storage"""
    if not clients_initialized:
        logger.warning("Cloud storage not available for listing PDFs")
        return []
    
    try:
        blobs = bucket.list_blobs(prefix="pdfs/")
        pdfs = []
        for blob in blobs:
            if blob.name.endswith('.pdf') and blob.name != "pdfs/":
                pdf_name = os.path.basename(blob.name)
                pdfs.append(pdf_name)
        
        logger.info(f"📚 Found {len(pdfs)} PDFs in storage")
        return pdfs
    except Exception as e:
        logger.error(f"❌ Failed to list PDFs: {e}")
        return []

def file_exists(folder, filename):
    """Check if a file exists in the specified folder in storage"""
    if not clients_initialized:
        return False
    
    try:
        blob_name = f"{folder}/{filename}"
        blob = bucket.blob(blob_name)
        exists = blob.exists()
        logger.debug(f"🔍 File {blob_name} exists: {exists}")
        return exists
    except Exception as e:
        logger.error(f"❌ Error checking file existence {folder}/{filename}: {e}")
        return False

def download_from_bucket(folder, filename, local_path):
    """Download file from bucket to local path"""
    if not clients_initialized:
        raise Exception("Cloud storage not available")
    
    blob_name = f"{folder}/{filename}"
    try:
        blob = bucket.blob(blob_name)
        blob.download_to_filename(local_path)
        logger.info(f"📥 Downloaded {blob_name} to {local_path}")
    except Exception as e:
        logger.error(f"❌ Download failed for {blob_name}: {e}")
        raise

def upload_json_to_bucket(data, folder, filename):
    """Upload JSON data to bucket"""
    if not clients_initialized:
        raise Exception("Cloud storage not available")
    
    blob_name = f"{folder}/{filename}"
    try:
        blob = bucket.blob(blob_name)
        blob.upload_from_string(json.dumps(data, indent=2), content_type='application/json')
        logger.info(f"📤 Uploaded JSON to: {blob_name}")
    except Exception as e:
        logger.error(f"❌ JSON upload failed for {blob_name}: {e}")
        raise

def download_json_from_bucket(folder, filename):
    """Download JSON data from bucket"""
    if not clients_initialized:
        raise Exception("Cloud storage not available")
    
    blob_name = f"{folder}/{filename}"
    try:
        blob = bucket.blob(blob_name)
        if blob.exists():
            data = blob.download_as_string()
            logger.info(f"📥 Downloaded JSON from: {blob_name}")
            return json.loads(data)
        else:
            logger.warning(f"⚠️ JSON file not found: {blob_name}")
            return None
    except Exception as e:
        logger.error(f"❌ JSON download failed for {blob_name}: {e}")
        return None

def clean_text(text: str) -> str:
    """
    Clean text from PDF extraction issues - standalone function to avoid circular imports
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
    """Check if PDF is already processed (has text, chunks, and embeddings)"""
    base_name = get_base_filename(pdf_name)
    
    text_exists = file_exists("texts", f"{base_name}_text.txt")
    chunks_exists = file_exists("chunks", f"{base_name}_chunks.json")
    embeddings_exists = file_exists("embeddings", f"{base_name}_embeddings.json")
    
    processed = text_exists and chunks_exists and embeddings_exists
    logger.info(f"🔍 PDF {pdf_name} processed: {processed}")
    
    return processed

def process_new_pdf(pdf_file, pdf_name):
    """Process a new PDF: extract text, create chunks, generate embeddings"""
    base_name = get_base_filename(pdf_name)
    
    logger.info(f"🔄 Starting to process PDF: {pdf_name}")
    
    # Save file temporarily
    temp_pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], pdf_name)
    pdf_file.save(temp_pdf_path)
    
    try:
        # 1. Upload PDF to pdfs folder
        logger.info("📤 Step 1: Uploading PDF to Cloud Storage...")
        upload_to_bucket(temp_pdf_path, "pdfs", pdf_name)
        
        # 2. Extract text and upload to texts folder
        logger.info("📖 Step 2: Extracting text from PDF...")
        text = extract_text_from_pdf(temp_pdf_path)
        
        text_file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{base_name}_text.txt")
        with open(text_file_path, "w", encoding="utf-8") as f:
            f.write(text)
        upload_to_bucket(text_file_path, "texts", f"{base_name}_text.txt")
        
        # 3. Create chunks and upload to chunks folder
        logger.info("✂️ Step 3: Creating text chunks...")
        chunks = chunk_text(text)
        upload_json_to_bucket(chunks, "chunks", f"{base_name}_chunks.json")
        
        # 4. Generate embeddings and upload to embeddings folder
        logger.info("🧠 Step 4: Generating embeddings...")
        embeddings = rag_system.add_new_documents(pdf_name, chunks)
        upload_json_to_bucket(embeddings, "embeddings", f"{base_name}_embeddings.json")
        
        logger.info(f"✅ Successfully processed new PDF: {pdf_name} with {len(chunks)} chunks")
        return len(chunks)
        
    except Exception as e:
        logger.error(f"❌ PDF processing failed for {pdf_name}: {e}")
        logger.error(traceback.format_exc())
        raise
    finally:
        # Cleanup temp files
        if os.path.exists(temp_pdf_path):
            os.remove(temp_pdf_path)
            logger.debug(f"🧹 Cleaned up temp PDF: {temp_pdf_path}")
        
        text_file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{base_name}_text.txt")
        if os.path.exists(text_file_path):
            os.remove(text_file_path)
            logger.debug(f"🧹 Cleaned up temp text file: {text_file_path}")

def load_processed_pdf(pdf_name):
    """Load already processed PDF from storage with error handling"""
    base_name = get_base_filename(pdf_name)
    
    try:
        logger.info(f"📥 Loading processed PDF: {pdf_name}")
        
        # Download chunks and embeddings
        chunks = download_json_from_bucket("chunks", f"{base_name}_chunks.json")
        embeddings = download_json_from_bucket("embeddings", f"{base_name}_embeddings.json")
        
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
        pdfs = list_pdfs() if clients_initialized else []
        
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
                             clients_ok=clients_initialized,
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
        if not clients_initialized:
            return jsonify({"error": "Cloud storage unavailable"}), 500
            
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
            
        pdf_name = data.get("pdf_name")
        if not pdf_name:
            return jsonify({"error": "PDF name required"}), 400
        
        logger.info(f"📥 Attempting to load PDF: {pdf_name}")
        
        # Check if PDF exists
        if not file_exists("pdfs", pdf_name):
            logger.error(f"❌ PDF not found in storage: {pdf_name}")
            return jsonify({"error": f"PDF '{pdf_name}' not found in storage"}), 404
        
        # Check if PDF is already processed
        if not is_pdf_processed(pdf_name):
            logger.error(f"❌ PDF not processed: {pdf_name}")
            return jsonify({"error": f"PDF '{pdf_name}' is not processed yet"}), 400
        
        logger.info(f"✅ PDF {pdf_name} exists and is processed, loading...")
        
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
            relevant_chunks = rag_system.search(user_input, k=8)  # Increased from 5 to 8
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
        rag_system.clear_documents()
        
        # Clear session
        session['current_pdf'] = None
        session['has_documents'] = False
        
        logger.info(f"🧹 Cleared documents for PDF: {current_pdf}")
        
        return jsonify({"success": "Current PDF cleared from memory"})
    except Exception as e:
        logger.error(f"❌ Clear documents error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/health")
def health():
    """Comprehensive health check endpoint"""
    try:
        # Check storage connectivity
        storage_healthy = clients_initialized and bucket.exists()
        
        # Check RAG system status
        rag_healthy = rag_system is not None
        rag_chunks_loaded = len(rag_system.chunks) if rag_system else 0
        
        # Check LLM client status
        llm_healthy = llm_client is not None and hasattr(llm_client, 'configured') and llm_client.configured
        
        # Determine overall status
        if storage_healthy and rag_healthy and llm_healthy:
            status = "healthy"
        elif storage_healthy and rag_healthy:
            status = "degraded"  # LLM might be disabled intentionally
        else:
            status = "unhealthy"
        
        return jsonify({
            "status": status,
            "timestamp": datetime.utcnow().isoformat(),
            "project": PROJECT_ID,
            "storage_healthy": storage_healthy,
            "rag_system_healthy": rag_healthy,
            "llm_client_healthy": llm_healthy,
            "clients_initialized": clients_initialized,
            "bucket": BUCKET_NAME,
            "current_pdf": rag_system.current_pdf_name if rag_system else None,
            "rag_chunks_loaded": rag_chunks_loaded,
            "version": "1.0.0"
        })
    except Exception as e:
        logger.error(f"❌ Health check failed: {e}")
        return jsonify({"status": "unhealthy", "error": str(e)}), 500

# Test endpoint to verify the system is working
@app.route("/test", methods=["GET"])
def test_endpoint():
    """Test endpoint to verify system functionality"""
    return jsonify({
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat(),
        "rag_system_ready": rag_system is not None,
        "llm_client_ready": llm_client is not None and hasattr(llm_client, 'configured') and llm_client.configured,
        "storage_ready": clients_initialized,
        "current_pdf": rag_system.current_pdf_name if rag_system else None,
        "chunks_loaded": len(rag_system.chunks) if rag_system else 0,
        "message": "PDF RAG Chatbot is running correctly!"
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    logger.info(f"🚀 Starting PDF RAG Chatbot on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)
