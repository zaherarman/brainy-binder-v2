import dotenv
import os 

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from typing import Optional

BASE_DIR = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file= BASE_DIR / ".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore")
    
    DATA_DIR: Path = BASE_DIR / "data"

    LLM_PROVIDER: str = "openai_compatible"
    LLM_BASE_URL: str
    LLM_MODEL_NAME:  str
    LLM_TEMP: float = Field(0.7, ge=0.0, le=2.0)
    LLM_MAX_TOKENS: int = Field (2048, gt=0)
    LLM_TIMEOUT: int = Field(120, gt=0)

    EMBEDDING_PROVIDER: str = "sentence_transformers" 
    EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"
    EMBEDDING_BASE_URL: Optional[str] = None
    EMBEDDING_API_KEY: Optional[str] = None
    
    TOP_K: int = Field(5, gt=0)
    CHUNK_SIZE: int = Field(1000, gt=0) # Max size of chunks in charactors
    CHUNK_OVERLAP: int = Field(300, ge=0)

    OCR_PROVIDER: Optional[str] = None
    AZURE_MISTRAL_OCR_MODEL: str

    AZURE_API_KEY: str
    
    NEO4J_URI: str
    NEO4J_USERNAME: str
    NEO4J_PASSWORD: str
    NEO4J_DATABASE: str
    
settings = Settings()