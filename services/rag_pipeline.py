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
# 6. RAG PIPELINE
# ============================================

class RAGPipeline:
    """Complete RAG pipeline implementation"""
    
    def __init__(self, llm, retriever, prompt_template=None):
        self.llm = llm
        self.retriever = retriever
        self.prompt_template = prompt_template or self._get_default_prompt()
    
    def _get_default_prompt(self):
        """Default RAG prompt template"""
        # template = """Use the following pieces of context to answer the question at the end.
        #     If you don't know the answer, just say that you don't know, don't try to make up an answer.
        #     Use three sentences maximum and keep the answer as concise as possible.

        #     Context: {context}

        #     Question: {question}

        #     Helpful Answer:"""
        template = """
            Use the following pieces of context to answer the question at the end.
            If you don't know the answer, just say that you don't know, don't try to make up an answer.
            Use two sentences maximum and keep the answer as concise as possible. 

            However, if the user asks a general question or says a greeting, respond politely and naturally.
            Examples:
            Human: Hi
            AI Assistant: Hello! How can I help you today?

            Human: How are you?
            AI Assistant: I'm doing well, thank you! How about you?

            Context:
            {context}

            Human: {question}
            AI Assistant:
            """
        
        return PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
    
    def create_qa_chain(self):
        """Create RetrievalQA chain"""
        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": self.prompt_template}
        )
    
    def query(self, question: str):
        """Execute query and return answer with sources"""
        qa_chain = self.create_qa_chain()
        result = qa_chain({"query": question})
        
        return {
            "answer": result["result"],
            "source_documents": result["source_documents"]
        }
