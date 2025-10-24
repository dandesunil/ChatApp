import os
from fastapi import FastAPI, Request, UploadFile, File
from typing import List
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
import pdfplumber
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from config import *
from services.llm import get_embedding_model

def read_text_file(path: str) -> str:
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        return f.read()

def recursive_chunking(documents: List[Document], chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP) -> List[Document]:
    """
    Recursively chunk the documents into smaller pieces.
    """
    # Create a recursive splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,       # maximum size of each chunk
        chunk_overlap=overlap,     # overlap between chunks to preserve context
        length_function=len,   # function used to count characters
        separators=["\n\n", "\n", ".", "!", "?", " ", ""],  # split hierarchy
    )
    return text_splitter.split_documents(documents)
def ingest_document(path: str, chroma_dir: str = CHROMA_DIR) -> None:
    os.makedirs(chroma_dir, exist_ok=True)
    loader = PyPDFLoader(path)
    documents = loader.load()  # returns list[Document]
    embed = get_embedding_model(EMBEDDING_MODEL)
    db = Chroma.from_documents(documents, embedding=embed, persist_directory=chroma_dir)
    db.persist()
    