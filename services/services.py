
from services.generate_embeddings import *
from services.rag_pipeline import *
def build_embeddings():
    """
    Build complete RAG system
    Returns:
        RAGPipeline instance
    """
    
    # 1. Load documents
    print("Loading documents...")
    loader = DocumentLoader()
    documents = loader.load_pdf()
    
    print(f"Loaded {len(documents)} documents")
    
    # 2. Chunk documents
    print("Chunking documents...")
    chunks = ChunkingStrategy.recursive_character_split(
        documents,
        chunk_size=400,
        chunk_overlap=50
    )
    print(f"Created {len(chunks)} chunks")
    
    # 3. Create embeddings
    print("Creating embeddings...")
    embeddings = EmbeddingModels.get_bge_base_embeddings()
    
    # 4. Create vector store
    print("Building vector store...")
    VectorStoreManager.create_chroma_store(
        chunks,
        embeddings,
        persist_directory="ChromaDB"
    )
        
    return "Embeddings and Vector Store created successfully."



def rag_pipeline(query: str):
    """
    Build complete RAG system
    Returns:
        RAGPipeline instance
    """
    try:
        # 1 Load Embedding model
        embeddings = EmbeddingModels.get_bge_base_embeddings()
        # 1 Load chroma vector store
        vectorstore  = VectorStoreManager.load_chroma_store(
            embeddings,
            persist_directory="ChromaDB"
        )
        # 5. Create retriever
        print("Configuring retriever...")
        retriever = RetrieverConfig.mmr_retriever(vectorstore, k=4)
        
        # 6. Setup LLM (using a small HuggingFace model)
        print("Loading LLM...")
        llm = get_hf_llm()
        
        # 7. Create RAG pipeline
        print("Building RAG pipeline...")
        rag_pipeline = RAGPipeline(llm, retriever)
        
        print("RAG system ready!")
        result = rag_pipeline.query(query)
    
        # print(f"\nQuestion: {query}")
        # print(f"\nAnswer: {result['answer']}")
        # print(f"\nSources: {len(result['source_documents'])} documents")
        return result
    except Exception as e:
        print("Error building RAG pipeline:", str(e))
        return None
