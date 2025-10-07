"""
Logging Configuration
"""
from loguru import logger
import sys
from pathlib import Path

def setup_logger():
    """Setup application logger"""
    # Remove default handler
    logger.remove()
    
    # Console handler
    logger.add(
        sys.stdout,
        colorize=True,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
        level="INFO"
    )
    
    # File handler
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    logger.add(
        log_dir / "app_{time:YYYY-MM-DD}.log",
        rotation="500 MB",
        retention="10 days",
        compression="zip",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function} - {message}",
        level="DEBUG"
    )
    
    return logger

def get_logger(name: str):
    """Get logger instance"""
    return logger.bind(name=name)