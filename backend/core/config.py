"""
Application configuration settings
"""

from typing import List
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    # API Settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = True
    SECRET_KEY: str = "your-secret-key-change-in-production"
    
    # CORS
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8000",
    ]
    
    # Database
    DATABASE_URL: str = "postgresql://postgres:postgres@localhost:5432/sers_insight"
    
    # Redis (for caching)
    REDIS_URL: str = "redis://localhost:6379/0"
    
    # File Storage
    UPLOAD_DIR: str = "uploads"
    RESULTS_DIR: str = "results"
    MODELS_DIR: str = "models"
    MAX_UPLOAD_SIZE: int = 50 * 1024 * 1024  # 50MB
    
    # AI Integration (OpenRouter)
    OPENROUTER_API_KEY: str = ""
    DEFAULT_AI_MODEL: str = "anthropic/claude-sonnet-4.5"
    
    # Processing
    MAX_SPECTRUM_POINTS: int = 10000
    DEFAULT_BASELINE_LAMBDA: float = 1e5
    DEFAULT_SMOOTHING_WINDOW: int = 11
    DEFAULT_SMOOTHING_POLYORDER: int = 3
    
    class Config:
        env_file = ".env"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()


settings = get_settings()
