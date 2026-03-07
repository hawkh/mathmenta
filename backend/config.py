"""
Configuration for Math Mentor Backend
"""
import os
from pydantic_settings import BaseSettings
from typing import Optional
import secrets


class Settings(BaseSettings):
    """Application settings."""
    
    # Application
    APP_NAME: str = "Math Mentor API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True
    ENVIRONMENT: str = "development"  # development, staging, production
    
    # API
    API_PREFIX: str = "/api/v1"
    ALLOWED_HOSTS: list = ["*"]
    CORS_ORIGINS: list = ["http://localhost:3000", "http://localhost:8000"]
    
    # Database
    DATABASE_URL: str = "sqlite+aiosqlite:///./mathmentor.db"
    
    # Security
    SECRET_KEY: str = os.getenv("SECRET_KEY", secrets.token_urlsafe(32))
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7  # 1 week
    
    # Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = 20  # Requests per minute per IP
    RATE_LIMIT_PER_HOUR: int = 100  # Requests per hour per IP
    
    # LLM
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    ANTHROPIC_MODEL: str = "claude-3-sonnet-20240229"
    LLM_TEMPERATURE: float = 0.3  # Lower for more deterministic math
    LLM_MAX_TOKENS: int = 2048
    
    # RAG
    VECTOR_STORE_PATH: str = "./vector_store"
    KNOWLEDGE_BASE_PATH: str = "./knowledge_base"
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    RAG_TOP_K: int = 5
    RAG_INITIAL_K: int = 20  # For reranking
    
    # Symbolic Math
    SYMBOLIC_TIMEOUT: float = 5.0  # Seconds
    
    # File Upload
    MAX_UPLOAD_SIZE: int = 10 * 1024 * 1024  # 10MB
    UPLOAD_DIR: str = "./uploads"
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"  # json or console
    
    # Monitoring
    ENABLE_METRICS: bool = True
    METRICS_ENDPOINT: str = "/metrics"
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get settings instance."""
    return settings
