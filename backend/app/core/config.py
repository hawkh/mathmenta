"""
Configuration management for Math Mentor FastAPI backend.
"""
import os
from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Application
    APP_NAME: str = "Math Mentor API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    API_V1_STR: str = "/api/v1"

    # API Keys
    ANTHROPIC_API_KEY: str = ""
    OPENAI_API_KEY: str = ""

    # Ollama Configuration
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    USE_OLLAMA: bool = False
    DEFAULT_MODEL: str = "llama3.3"
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

    # Database
    DATABASE_URL: str = "sqlite+aiosqlite:///./mathmentor.db"

    # Vector Store
    VECTOR_STORE_DIR: str = "vector_store"
    KNOWLEDGE_BASE_DIR: str = "knowledge_base"

    # RAG Settings
    TOP_K_RETRIEVAL: int = 5
    RAG_INITIAL_K: int = 20  # For hybrid retrieval
    RAG_FINAL_K: int = 5

    # Confidence Thresholds
    OCR_CONFIDENCE_THRESHOLD: float = 0.75
    ASR_CONFIDENCE_THRESHOLD: float = 0.75
    VERIFIER_CONFIDENCE_THRESHOLD: float = 0.70

    # Reranker Settings
    RERANKER_MODEL: str = "fast"  # fast, balanced, accurate
    RERANKER_USE_GPU: bool = False

    # CORS
    CORS_ORIGINS: list[str] = ["*"]
    CORS_ALLOW_CREDENTIALS: bool = True
    CORS_ALLOW_METHODS: list[str] = ["*"]
    CORS_ALLOW_HEADERS: list[str] = ["*"]

    # Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = 60

    # Chunking
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50

    class Config:
        env_file = ".env"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
