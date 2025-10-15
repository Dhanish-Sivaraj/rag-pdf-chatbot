import os
import numpy as np
import faiss
from typing import List, Tuple
import logging
import re
import torch
from transformers import AutoTokenizer, AutoModel
import google.generativeai as genai
import traceback

logger = logging.getLogger(__name__)

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

class EnhancedEmbeddingModel:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the embedding model with lazy loading for memory efficiency.
        """
        logger.info(f"Loading embedding model: {model_name}")
        self.model_name = model_name
        self.vector_size = 384
        self.model = None
        self.tokenizer = None
        logger.info("✅ Embedding model initialized!")

    def _ensure_model_loaded(self):
        """Ensure the model is loaded only when first used to save memory"""
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
        """
        Apply mean pooling to get sentence embeddings from token embeddings.
        """
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def embed(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text string.
        """
        self._ensure_model_loaded()
        try:
            if not text or not text.strip():
                logger.warning("Empty text provided for embedding")
                return np.zeros(self.vector_size)
            
            # Clean and truncate text before embedding
            clean_text_str = clean_text(' '.join(text.split()[:300]))
            
            # Tokenize the input text
            inputs = self.tokenizer(
                clean_text_str, 
                return_tensors='pt', 
                truncation=True, 
                max_length=512, 
                padding=True
            )
            
            # Generate embeddings without tracking gradients
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Apply mean pooling and convert to numpy
            embeddings = self._mean_pooling(outputs, inputs['attention_mask'])
            embeddings = embeddings.squeeze().numpy()
            
            # Normalize the embeddings
            norm = np.linalg.norm(embeddings)
            if norm > 0:
                embeddings = embeddings / norm
                
            return embeddings.astype('float32')
            
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            return np.zeros(self.vector_size)

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a batch of texts.
        """
        if not texts:
            logger.warning("Empty text list provided for batch embedding")
            return np.array([])
        
        embeddings = []
        for text in texts:
            embedding = self.embed(text)
            embeddings.append(embedding)
            
        return np.array(embeddings)

    def _clean_text(self, text: str) -> str:
        """
        Clean text extracted from PDF to fix common formatting issues.
        Uses the standalone clean_text function.
        """
        return clean_text(text)


class AdvancedRAGSystem:
    """
    Advanced RAG system for document retrieval and question answering.
    """
    
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
            
            # Generate query embedding
            query_embedding = self.embedder.embed(query).astype('float32')
            query_embedding = query_embedding.reshape(1, -1)
            faiss.normalize_L2(query_embedding)
            
            # Search for more results initially to get better matches
            search_k = min(k * 5, len(self.chunks))
            if search_k == 0:
                return []
            
            # Perform similarity search
            similarities, indices = self.index.search(query_embedding, search_k)
            
            # Process results with better filtering
            results = []
            for similarity, idx in zip(similarities[0], indices[0]):
                if idx < len(self.chunks) and similarity > 0.05:  # Lower threshold to get more results
                    clean_chunk = clean_text(self.chunks[idx])
                    
                    # Skip very short chunks that are unlikely to be useful
                    if len(clean_chunk.split()) < 5:
                        continue
                        
                    results.append((clean_chunk, float(similarity)))
            
            # Sort by similarity score (descending)
            results.sort(key=lambda x: x[1], reverse=True)
            
            # Log the top results for debugging
            if results:
                logger.info(f"✅ Found {len(results)} relevant chunks")
                for i, (chunk, score) in enumerate(results[:3]):
                    logger.info(f"   Top {i+1}: score={score:.3f}, text='{chunk[:100]}...'")
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


class GeminiLLMClient:
    """
    LLM client using Google Gemini for high-quality responses.
    """
    
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
            
            # List available models for debugging
            try:
                models = genai.list_models()
                logger.info(f"📋 Available Gemini models: {[model.name for model in models]}")
            except Exception as model_list_error:
                logger.warning(f"Could not list models: {model_list_error}")
            
            # Try available models in order of preference
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
                    
                    # Test the model with a simple prompt
                    test_response = self.model.generate_content("Say 'TEST' in one word.")
                    if test_response and test_response.text:
                        logger.info(f"✅ Successfully initialized model: {model_name}")
                        logger.info(f"🧪 Model test response: {test_response.text}")
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
        """
        Handle general conversational queries.
        """
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
        """
        Generate response using Gemini API with the document context.
        """
        # Handle general queries first
        general_response = self._get_general_response(query)
        if general_response:
            return general_response
        
        # No PDF loaded
        if not pdf_name:
            return "📚 Please upload a PDF file first!"
        
        # Use Gemini if available
        if self.configured and self.model:
            try:
                return self._generate_with_gemini(query, context, pdf_name)
            except Exception as e:
                logger.error(f"❌ Gemini generation error: {e}")
                logger.error(traceback.format_exc())
                return self._fallback_response(query, pdf_name)
        else:
            logger.warning("🔧 Gemini not configured, using fallback response")
            return self._fallback_response(query, pdf_name)
    
    def _generate_with_gemini(self, query: str, context: str, pdf_name: str) -> str:
        """
        Generate response using Gemini API.
        """
        try:
            logger.info("🎯 Using Gemini for response generation")
            
            # No context found - provide a more helpful response
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
            
            # Format the response nicely
            formatted_response = f"**Based on {pdf_name}**:\n\n{response.text}"
            
            logger.info(f"✅ Gemini response generated: {len(response.text)} characters")
            return formatted_response
            
        except Exception as e:
            logger.error(f"❌ Gemini generation failed: {e}")
            logger.error(traceback.format_exc())
            raise  # Re-raise to be handled by the caller
    
    def _fallback_response(self, query: str, pdf_name: str) -> str:
        """
        Simple fallback response when Gemini is not available.
        """
        return f"**Based on {pdf_name}**:\n\nI found relevant content in the document but couldn't generate a detailed AI response. The document contains information related to '{query}'.\n\n🔧 *Note: AI response generation is currently unavailable.*"

    def _clean_text(self, text: str) -> str:
        """
        Clean text from PDF extraction issues.
        Uses the standalone clean_text function.
        """
        return clean_text(text)

# For backward compatibility
RAGSystem = AdvancedRAGSystem
LLMClient = GeminiLLMClient

# Export the main classes
__all__ = ['RAGSystem', 'LLMClient', 'AdvancedRAGSystem', 'GeminiLLMClient', 'EnhancedEmbeddingModel', 'clean_text']