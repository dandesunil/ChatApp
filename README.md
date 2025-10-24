# RAG Chat Repo (Python + Starlite + LangChain + Chroma)

## 🧠 Problem Statement
Build a **Retrieval-Augmented Generation (RAG)** chat interface that allows users to:
1. Upload their own documents (text, PDF, etc.).
2. Automatically chunk, embed, and store the documents in a local vector database (ChromaDB).
3. Interact conversationally with the uploaded document using an open-source LLM.
4. Run completely **offline** or with **free open-source models**.

This solves the problem of enabling **private, document-specific conversational AI** without relying on proprietary APIs or paid models.

---

## 🧰 Tech Stack

| Component | Library / Tool | Purpose |
|------------|----------------|----------|
| **Frontend UI** | [Starlite Templates + Vanilla JS] | Lightweight web UI for upload and chat. |
| **Backend Framework** | [Starlite](https://starlite-api.github.io/starlite/) | Fast ASGI framework; modern alternative to FastAPI with great async support and templating. |
| **LLM Orchestration** | [LangChain](https://www.langchain.com/) | Provides standardized chains, retrievers, and memory for RAG pipelines. |
| **Vector Store** | [ChromaDB](https://www.trychroma.com/) | Local, simple, and fast vector database to store document embeddings. |
| **Embedding Model** | [Sentence-Transformers (all-MiniLM-L6-v2)] | Creates compact, high-quality embeddings for semantic retrieval. |
| **LLM Model** | [HuggingFace Transformers - flan-t5-small] | Open-source lightweight model for question answering and summarization. |
| **Package Manager** | [uv](https://github.com/astral-sh/uv) | Fast modern Python environment and dependency manager. |

### ⚙️ Why These Were Chosen
- **Starlite**: Lightweight and performant compared to FastAPI, easy templating for UI.
- **LangChain**: Handles RAG orchestration elegantly (retrieval, memory, LLM integration).
- **ChromaDB**: Simple to use, no server setup, ideal for local vector storage.
- **Sentence-Transformers**: High accuracy sentence-level embeddings for search.
- **flan-t5-small**: CPU-friendly open model with good reasoning and summarization ability.
- **uv**: Speeds up installs and isolates environments cleanly.

### 🧩 Possible Alternatives
| Function | Alternatives |
|-----------|---------------|
| Web Framework | **FastAPI**|
| Vector DB | **FAISS**, **Milvus**, **Weaviate**, **Pinecone** (managed) |
| Embeddings | **InstructorEmbeddings**, **E5 models**, **OpenAI embeddings** (paid) |
| LLMs | **Mistral**, **Llama 2**, **Phi-2**, **Mixtral**, or **llama.cpp** (for local inference) |
| Orchestration | **LlamaIndex**, **Haystack**, **LangGraph** |

---

## ⚙️ Setup Instructions

### 1️⃣ Prerequisites
- Python 3.12+
- (Optional) [uv](https://github.com/astral-sh/uv) for fast environment setup.

### 2️⃣ Clone the repo
```bash
git clone https://github.com/dandesunil/ChatApp.git
cd ChatApp
```

### 3️⃣ Create and activate environment
Using **uv**:
```bash
uv venv
source .venv/bin/activate
```
Or using traditional Python:
```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
```

### 4️⃣ Install dependencies
```bash
uv sync
```

### 5️⃣ Run the application
```bash
uvicorn main:app --reload
```

The app will be available at:  
👉 **http://localhost:8000**

---

## 🧪 How It Works
1. User uploads a text file from the web UI.
2. File is stored under the `/code` folder.
3. `ingest_document()` runs:
   - Reads file content
   - Splits into overlapping chunks
   - Generates embeddings using SentenceTransformers
   - Stores them in local ChromaDB
4. When user sends a chat query:
   - System retrieves the top-k relevant chunks
   - Combines them into a context
   - Passes both query + context to `flan-t5-small`
   - Returns an answer via LangChain ConversationalRetrievalChain

---

## 🚀 Next Steps
- Add **PDF/DOCX loaders** with LangChain’s `PyPDFLoader` or `UnstructuredLoader`.
- Integrate **LlamaCpp** for local Llama-based inference.
- Enable **async ingestion** and progress tracking.
- Add **user-based namespaces** for vector isolation.
- Deploy on **Docker + Azure App Service** or **Render**.

---

## 💡 Example Interaction
1. Upload: `finance_report.txt`
2. Ask: “What was the total revenue last quarter?”
3. Model responds based on content in the document — no internet or external API required.

---

## 🧾 License
MIT License — free to use, modify, and distribute.

---

**Author:** Sunil Kumar  
**Version:** 1.0  
**Last Updated:** October 2025

