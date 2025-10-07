"""
Configuration Management
"""
from pydantic_settings import BaseSettings
from typing import List
import os

class Settings(BaseSettings):
    # Application
    APP_NAME: str = "CCTV Video Analysis"
    VERSION: str = "1.0.0"
    DEBUG: bool = False
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 4
    
    # Paths
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    UPLOAD_DIR: str = "data/videos/uploads"
    OUTPUT_DIR: str = "data/videos/outputs"
    MODELS_DIR: str = "data/models"
    LOGS_DIR: str = "logs"
    
    # Model Configuration
    MODEL_PATH: str = "yolov8n.pt"
    CONF_THRESHOLD: float = 0.25
    IOU_THRESHOLD: float = 0.45
    DEVICE: str = "auto"
    
    # Video Processing
    SKIP_FRAMES: int = 0
    MAX_WORKERS: int = 4
    
    # Security
    ALLOWED_ORIGINS: List[str] = ["*"]
    API_KEY: str = "your-secret-api-key"
    
    # Database
    DATABASE_URL: str = "sqlite:///./cctv_analysis.db"
    
    # Redis (for caching)
    REDIS_URL: str = "redis://localhost:6379"
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()