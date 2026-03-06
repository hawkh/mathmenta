"""
Configuration management for Math Mentor.
Loads environment variables and provides centralized settings.
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """Centralized configuration for the Math Mentor system."""
    
    # API Keys
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    
    # Model Settings
    DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "claude-sonnet-4-20250514")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    
    # Confidence Thresholds for HITL
    OCR_CONFIDENCE_THRESHOLD = float(os.getenv("OCR_CONFIDENCE_THRESHOLD", "0.75"))
    ASR_CONFIDENCE_THRESHOLD = float(os.getenv("ASR_CONFIDENCE_THRESHOLD", "0.75"))
    VERIFIER_CONFIDENCE_THRESHOLD = float(os.getenv("VERIFIER_CONFIDENCE_THRESHOLD", "0.70"))
    
    # Paths
    KNOWLEDGE_BASE_DIR = "knowledge_base"
    VECTOR_STORE_DIR = "vector_store"
    MEMORY_STORE_FILE = os.path.join("memory", "memory_store.json")
    
    # RAG Settings
    TOP_K_RETRIEVAL = 3
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50
    
    @classmethod
    def validate(cls) -> bool:
        """Validate that required configuration is present."""
        if not cls.ANTHROPIC_API_KEY:
            raise ValueError("ANTHROPIC_API_KEY is required. Please set it in .env file.")
        return True
    
    @classmethod
    def is_openai_available(cls) -> bool:
        """Check if OpenAI API key is configured for Whisper."""
        return bool(cls.OPENAI_API_KEY)
