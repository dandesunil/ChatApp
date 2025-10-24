from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from langchain.llms import HuggingFacePipeline
from config import *

def get_hf_llm(model_name: str = LLM_MODEL) -> HuggingFacePipeline:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, truncation=True, max_length=256)
    return HuggingFacePipeline(pipeline=pipe)
