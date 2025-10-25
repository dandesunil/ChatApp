from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS, Chroma
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from config import *
from services.llm import get_embedding_model,get_hf_llm

# ============================================
# 1. DOCUMENT LOADING
# ============================================

class DocumentLoader:
    """Load documents from various sources"""
    
    @staticmethod
    def load_pdf(file_path: str=DEFAULT_PDF):
        """Load PDF documents"""
        loader = PyPDFLoader(file_path)
        return loader.load()
    

# ============================================
# 2. CHUNKING STRATEGIES
# ============================================

class ChunkingStrategy:
    """Different chunking strategies for document splitting"""
    
    @staticmethod
    def recursive_character_split(documents, chunk_size=CHUNK_SIZE, chunk_overlap=OVERLAP):
        """
        Recursive character splitting - RECOMMENDED for most use cases
        Tries to split on paragraphs, then sentences, then words
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        return text_splitter.split_documents(documents)

# ============================================
# 3. EMBEDDING MODELS
# ============================================

class EmbeddingModels:
    """Different HuggingFace embedding model configurations"""
    

    @staticmethod
    def get_bge_base_embeddings():
        """
        - 768 dimensions
        Good balance of performance and speed
        """        
        return get_embedding_model()
    
# ============================================
# 4. VECTOR STORES
# ============================================

class VectorStoreManager:
    """Manage different vector store implementations"""
    
    
    @staticmethod
    def create_chroma_store(documents, embeddings, persist_directory="ChormaDB"):
        """
        Chroma Vector Store - Persistent storage
        Good for: Production, persistent data
        """
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=persist_directory
        )
        return vectorstore
    
    @staticmethod
    def load_chroma_store(embeddings, persist_directory="ChormaDB"):
        """Load existing Chroma store"""
        return Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings
        )

    
# ============================================
# 5. RETRIEVER CONFIGURATIONS
# ============================================

class RetrieverConfig:
    """Different retriever configurations"""
    
    @staticmethod
    def similarity_retriever(vectorstore, k=4):
        """Basic similarity search retriever"""
        return vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k}
        )
    
    @staticmethod
    def mmr_retriever(vectorstore, k=4, fetch_k=20, lambda_mult=0.5):
        """
        Maximum Marginal Relevance (MMR) retriever
        Balances relevance and diversity
        """
        return vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": k,
                "fetch_k": fetch_k,
                "lambda_mult": lambda_mult
            }
        )
    
    @staticmethod
    def similarity_score_threshold_retriever(vectorstore, k=4, score_threshold=0.5):
        """Retriever with similarity score threshold"""
        return vectorstore.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": k,
                "score_threshold": score_threshold
            }
        )
