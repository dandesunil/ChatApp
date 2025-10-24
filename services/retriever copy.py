from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings
from services.llm import get_hf_llm
from typing import List
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

class Answerer:
    def __init__(self, chroma_dir: str = './chroma_db'):
        self.chroma_dir = chroma_dir
        self.embed = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        self.db = Chroma(persist_directory=self.chroma_dir, embedding_function=self.embed)
        self.retriever = self.db.as_retriever(search_kwargs={"k": 4})
        self.llm = get_hf_llm()
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.chain = ConversationalRetrievalChain.from_llm(self.llm, self.retriever, memory=self.memory)

    
    def semantic_retrieval(self, query: str,top_k=3) -> List[str]:
        """
        Perform semantic retrieval on the database
        given a query.
        """
        results = self.db.similarity_search(query, k=top_k*2)
        unique_results = []
        seen_contents = set()
        for res in results:
            content = res.page_content
            if content not in seen_contents:
                unique_results.append(content)
                seen_contents.add(content)
            if len(unique_results) >= top_k:
                break

        prompt = f"Assistant, responding as a friendly AI, in the context of a conversation about a document. Context: {"\n\n".join([doc.page_content for doc in unique_results])}."
        prompt = prompt + f"Please generate a response to the following query: {query}."
        prompt = prompt + f"If query is general knowledge or a greeting, respond accordingly."
        prompt = prompt + f"If the query does not refer to any specific information from the document, respond with a greeting or answer a general knowledge question. If no information can be found, respond with 'I don't know'."
        return self.chain.invoke(prompt)
    
    async def answer(self, query: str) -> str:
        print("Vector Store Data:")
        print(self.db)
        return self.semantic_retrieval(query)
