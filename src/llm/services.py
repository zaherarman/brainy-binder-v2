from langchain_ollama import ChatOllama, OllamaEmbeddings
from src.config import settings

def get_llm():
    return ChatOllama(
        model=settings.llm_model_name,
        temperature=settings.llm_temp,
        base_url=settings.llm_base_url,
    )

def get_embedder():
    return OllamaEmbeddings(
        model=settings.llm_model_name,
        temperature=settings.llm_temp,
        base_url=settings.llm_base_url,
    )