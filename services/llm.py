from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM,AutoModelForCausalLM
from langchain.llms import HuggingFacePipeline
from langchain.embeddings import HuggingFaceEmbeddings
from config import *

def get_hf_llm(model_name: str = LLM_MODEL) -> HuggingFacePipeline:
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=HUGGINGFACE_API_KEY)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",  # use GPU if available
        torch_dtype="auto",
        use_auth_token=HUGGINGFACE_API_KEY
    )
    llm_pipeline = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512)    
    return HuggingFacePipeline(pipeline=llm_pipeline)
def get_embedding_model(model_name: str = EMBEDDING_MODEL):
    # Ensure clean directory if embedding dim mismatch   
    return HuggingFaceEmbeddings(model_name=model_name)