from langchain.vectorstores import Chroma
from services.llm import *
from typing import List
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationStringBufferMemory,ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage, SystemMessage
from langchain.chains import LLMChain, RetrievalQA
from config import *


class Answerer:
    def __init__(self, chroma_dir: str = CHROMA_DIR):
        self.chroma_dir = chroma_dir
        self.embed = get_embedding_model(EMBEDDING_MODEL)
        self.db = Chroma(persist_directory=self.chroma_dir, embedding_function=self.embed)            
        self.retriever = self.db.as_retriever(search_kwargs={"k": K_RETRIEVE})
        self.llm = get_hf_llm()
        # self.memory = ConversationStringBufferMemory(memory_key="chat_history", return_messages=False,output_key='answer')     
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,   # returns list of HumanMessage / AIMessage
            output_key="answer"
        ) 

        # Build the chain with custom prompt
        # self.chain = ConversationalRetrievalChain.from_llm(
        #     llm=self.llm,
        #     retriever=self.retriever,
        #     memory=self.memory,
        # )
    def retrieve_chunks(self, query: str, top_k: int = TOP_K) -> List[str]:
        """
        Retrieve chunks from the vector store.
        """
        results = self.db.similarity_search(query, k=top_k*4)
        unique_results = []
        seen_contect=[]
        for res in results:
            if res.page_content not in seen_contect:
                unique_results.append(res)
                seen_contect.append(res.page_content)
            if len(unique_results) >= top_k:
                break        
        return unique_results
   
    def semantic_retrieval(self, query: str) -> str:
        try:

            documents = self.retrieve_chunks(query, TOP_K)
            prompt_template = PROMPT_TEMPLATE


            CUSTOM_PROMPT = PromptTemplate(
                input_variables=["context", "chat_history", "question"],
                template=prompt_template
            )        
            vectorstore = self.db.from_documents(documents, self.embed)
            retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})

            # --- 6️⃣ Create ConversationalRetrievalChain ---
            qa_chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=retriever,
                memory=self.memory,
                combine_docs_chain_kwargs={"prompt": CUSTOM_PROMPT},
                return_source_documents=True,
                output_key="answer" 
            )
            response = qa_chain({"question": query})

            print("Answer:", response["answer"])
            print("Sources:", response["source_documents"])
            return response["answer"]
        except Exception as e:
            return f"Error during retrieval: {str(e)}"

    def answer(self, query: str) -> str:
        # return self.semantic_retrieval(query)
        return self.refined_retrieval(query)
    
    def refined_retrieval(self, query: str) -> str:
        try:
            # Main prompt (first pass)
            QUESTION_PROMPT = PromptTemplate(
                input_variables=["context", "chat_history", "question"],
                template=PROMPT_TEMPLATE
            )

            # Refinement prompt (for subsequent docs)
            REFINE_PROMPT = PromptTemplate(
                input_variables=["existing_answer", "context", "chat_history", "question"],
                template=REFINED_PROMPT_TEMPLATE
            )
            documents = self.retrieve_chunks(query, TOP_K)
            vectorstore = self.db.from_documents(documents, self.embed)
            retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})
            qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="refine",
                retriever=retriever,
                chain_type_kwargs={
                    "question_prompt": QUESTION_PROMPT,
                    "refine_prompt": REFINE_PROMPT
                },
                output_key="answer"
            )
            response = qa_chain({"question": query})

            print("Answer:", response["answer"])
            print("Sources:", response["source_documents"])
            return response["answer"]
        except Exception as e:
                return f"Error during retrieval: {str(e)}"

