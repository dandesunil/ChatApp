"""
FastAPI PDF RAG service (single-file)
- Upload PDFs
- Extract text/images/tables/code blocks (uses Unstructured or pdfplumber/fitz fallback)
- Clean & multi-stage chunking (preserve tables/images/code)
- Create embeddings (OpenAI or HuggingFace)
- Store/retrieve from FAISS (persistent)
- /upload -> process PDF and build/update vector index
- /search -> semantic search with optional retrieval-augmented answer

Usage:
1. pip install -r requirements.txt
2. export OPENAI_API_KEY=...
3. uvicorn fastapi_pdf_rag:app --host 0.0.0.0 --port 8000

Notes:
- This file intentionally keeps implementation in one place for clarity. In prod, split into modules.
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import os
import uuid
import shutil
import tempfile
import json
from pathlib import Path

# Document processing
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# Embeddings and vectorstores
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

# For table conversion fallback
import pdfplumber
import fitz  # PyMuPDF

# Simple QA chain
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

# concurrency
from concurrent.futures import ThreadPoolExecutor

# -----------------------------
# CONFIG
# -----------------------------
DATA_DIR = Path("./data")
INDEX_DIR = DATA_DIR / "indices"
PDF_DIR = DATA_DIR / "pdfs"
TMP_DIR = DATA_DIR / "tmp"

for d in [DATA_DIR, INDEX_DIR, PDF_DIR, TMP_DIR]:
    d.mkdir(parents=True, exist_ok=True)

DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 120

# Use thread executor for CPU-bound extraction
executor = ThreadPoolExecutor(max_workers=3)

app = FastAPI(title="PDF RAG Service")

# -----------------------------
# Helpers
# -----------------------------

def save_upload_file(upload_file: UploadFile, destination: Path) -> Path:
    with destination.open("wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)
    return destination


def load_with_unstructured(path: str) -> List[Document]:
    try:
        loader = UnstructuredPDFLoader(path, strategy="hi_res")
        docs = loader.load()
        return docs
    except Exception as e:
        # fallback
        return []


def fallback_pdfplumber(path: str) -> List[Document]:
    docs: List[Document] = []
    try:
        with pdfplumber.open(path) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""
                # simple: include raw text and page number metadata
                meta = {"page": i, "source": Path(path).name}
                docs.append(Document(page_content=text, metadata=meta))
    except Exception:
        pass
    return docs


def extract_images_with_fitz(path: str) -> List[Document]:
    docs: List[Document] = []
    try:
        doc = fitz.open(path)
        for i in range(len(doc)):
            page = doc[i]
            image_list = page.get_images(full=True)
            for img_index, img in enumerate(image_list, start=1):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                # Save to temp and attach a simple caption metadata
                tmp_name = TMP_DIR / f"{Path(path).stem}_p{i}_img{img_index}.png"
                with open(tmp_name, "wb") as f:
                    f.write(image_bytes)
                caption = f"[IMAGE]{tmp_name.name} (extracted from page {i})"
                meta = {"page": i, "category": "image", "source": Path(path).name, "image_path": str(tmp_name)}
                docs.append(Document(page_content=caption, metadata=meta))
    except Exception:
        pass
    return docs


def clean_and_classify(docs: List[Document]) -> List[Document]:
    cleaned: List[Document] = []
    for d in docs:
        text = d.page_content or ""
        meta = dict(d.metadata) if d.metadata else {}
        # heuristic: detect code blocks
        if "```" in text or "def " in text[:200] or "class " in text[:200]:
            meta["category"] = meta.get("category", "code")
            # ensure code fences
            if not text.strip().startswith("```"):
                text = "```\n" + text.strip() + "\n```"
            cleaned.append(Document(page_content=text, metadata=meta))
            continue
        # heuristic: table detection (common markers)
        if "\t" in text or "|" in text[:300] or "Column" in text[:200]:
            meta["category"] = meta.get("category", "table")
            # convert to markdown-ish wrapper
            text = "[TABLE]\n" + text.strip() + "\n[/TABLE]"
            cleaned.append(Document(page_content=text, metadata=meta))
            continue
        # images already handled separately
        if meta.get("category") == "image":
            cleaned.append(Document(page_content=text, metadata=meta))
            continue

        # otherwise normal paragraph
        meta["category"] = meta.get("category", "text")
        cleaned.append(Document(page_content=text.strip(), metadata=meta))
    return cleaned


def chunk_documents(docs: List[Document], chunk_size=DEFAULT_CHUNK_SIZE, chunk_overlap=DEFAULT_CHUNK_OVERLAP) -> List[Document]:
    # Use RecursiveCharacterTextSplitter but favor not splitting inside code blocks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n```", "\n## ", "\n\n", "\n", " ", ""]
    )
    return splitter.split_documents(docs)


def get_embedding_model(use_openai=True):
    if use_openai:
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY not set in environment")
        return OpenAIEmbeddings(model="text-embedding-3-small")
    else:
        # example open-source
        return HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh")


def save_faiss(index: FAISS, name: str):
    path = INDEX_DIR / name
    path.mkdir(parents=True, exist_ok=True)
    index.save_local(str(path))


def load_faiss(name: str, embedding) -> Optional[FAISS]:
    path = INDEX_DIR / name
    if not path.exists():
        return None
    return FAISS.load_local(str(path), embedding)

# -----------------------------
# Processing pipeline
# -----------------------------

def process_pdf_build_index(pdf_path: str, index_name: str, use_openai=True):
    """Full pipeline: extract -> clean -> chunk -> embed -> save"""
    # 1. Try unstructured loader
    docs = load_with_unstructured(pdf_path)

    # 2. Fallbacks if empty
    if not docs:
        docs = fallback_pdfplumber(pdf_path)

    # 3. Images
    images = extract_images_with_fitz(pdf_path)
    if images:
        docs.extend(images)

    # 4. Clean/classify
    docs = clean_and_classify(docs)

    # 5. Chunk
    chunks = chunk_documents(docs)

    # 6. Embedding
    embedding = get_embedding_model(use_openai=use_openai)

    # 7. Vector DB
    vectorstore = FAISS.from_documents(chunks, embedding)

    # save vectorstore
    save_faiss(vectorstore, index_name)

    # save metadata mapping (simple)
    meta_path = INDEX_DIR / index_name / "meta.json"
    meta = {"source_pdf": Path(pdf_path).name, "chunks": len(chunks)}
    meta_path.write_text(json.dumps(meta))

    return {"index_name": index_name, "chunks": len(chunks)}

# -----------------------------
# API models
# -----------------------------

class UploadResponse(BaseModel):
    index_name: str
    chunks: int

class SearchRequest(BaseModel):
    index_name: str
    query: str
    k: Optional[int] = 4
    use_qa: Optional[bool] = False

# -----------------------------
# Endpoints
# -----------------------------

@app.post("/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...), background_tasks: BackgroundTasks = None, use_openai: bool = True):
    # save file
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    uid = uuid.uuid4().hex
    saved = PDF_DIR / f"{uid}_{file.filename}"
    save_upload_file(file, saved)

    index_name = f"idx_{uid}"

    # run processing in background if provided (note: this only schedules the task in FastAPI; the user asked for a runnable example)
    if background_tasks is not None:
        background_tasks.add_task(process_pdf_build_index, str(saved), index_name, use_openai)
        # return early indicating index build scheduled
        return JSONResponse(status_code=202, content={"index_name": index_name, "chunks": -1})

    # otherwise run synchronously
    result = process_pdf_build_index(str(saved), index_name, use_openai)
    return result


@app.post("/search")
async def search(req: SearchRequest):
    embedding = get_embedding_model(use_openai=True)
    vs = load_faiss(req.index_name, embedding)
    if vs is None:
        raise HTTPException(status_code=404, detail="Index not found")

    if req.use_qa:
        # use OpenAI LLM for simple retrieval QA
        llm = OpenAI(temperature=0)
        qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vs.as_retriever(search_kwargs={"k": req.k}))
        answer = qa.run(req.query)
        return {"answer": answer}
    else:
        docs = vs.similarity_search(req.query, k=req.k)
        return {"results": [ {"content": d.page_content, "metadata": d.metadata} for d in docs ]}


@app.get("/indices")
async def list_indices():
    items = []
    for p in INDEX_DIR.iterdir():
        if p.is_dir():
            meta = p / "meta.json"
            info = {"name": p.name}
            if meta.exists():
                try:
                    info.update(json.loads(meta.read_text()))
                except Exception:
                    pass
            items.append(info)
    return items

@app.get("/index/{name}/download")
async def download_index(name: str):
    path = INDEX_DIR / name
    if not path.exists():
        raise HTTPException(status_code=404, detail="Index not found")
    # bundle directory into a zip
    zip_path = TMP_DIR / f"{name}.zip"
    shutil.make_archive(str(zip_path).replace('.zip',''), 'zip', path)
    return JSONResponse({"download_path": str(zip_path)})

# -----------------------------
# Simple CLI helpers for local testing
# -----------------------------

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdf', help='Path to pdf to build index')
    parser.add_argument('--name', help='Index name')
    args = parser.parse_args()
    if args.pdf and args.name:
        print('Processing...')
        res = process_pdf_build_index(args.pdf, args.name)
        print('Done:', res)
    else:
        print('Run uvicorn to start the service')
