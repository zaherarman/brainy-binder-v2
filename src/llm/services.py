# from langchain_ollama import ChatOllama, OllamaEmbeddings
from src.config import settings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

def get_llm():
    # if settings.llm_provider == "ollama":
    #     return ChatOllama(
    #         model=settings.llm_model_name,
    #         temperature=settings.llm_temp,
    #         base_url=settings.llm_base_url,
    #     )

    if settings.LLM_PROVIDER == "openai_compatible":
        return ChatOpenAI(
            model=settings.LLM_MODEL_NAME,
            temperature=settings.LLM_TEMP,
            base_url=settings.LLM_BASE_URL,
            api_key=settings.AZURE_API_KEY,
            max_tokens=settings.LLM_MAX_TOKENS,
            timeout=settings.LLM_TIMEOUT,
        )

def get_embedder():
    # if settings.embedding_provider == "ollama":
    #     return OllamaEmbeddings(
    #         model=settings.embedding_model_name,
    #         base_url=settings.embedding_base_url or settings.llm_base_url,
    #     )

    if settings.EMBEDDING_PROVIDER == "openai_compatible":
        return OpenAIEmbeddings(
            model=settings.EMBEDDING_MODEL_NAME,
            base_url=settings.EMBEDDING_BASE_URL or settings.LLM_BASE_URL,
            api_key=settings.EMBEDDING_API_KEY or settings.AZURE_API_KEY,
        )
    
    if settings.EMBEDDING_PROVIDER == "sentence_transformers":
        return HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL_NAME
        )
