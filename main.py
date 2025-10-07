"""
CCTV Video Analysis System - Main Application
"""
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.api.routes import router
from backend.api.websocket import websocket_router
from backend.utils.config import settings
from backend.utils.logger import setup_logger

# Setup logger
logger = setup_logger()

# Initialize FastAPI app
app = FastAPI(
    title="CCTV Video Analysis API",
    description="AI-powered video detection and analysis system",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(router, prefix="/api/v1")
app.include_router(websocket_router, prefix="/ws")

@app.on_event("startup")
async def startup_event():
    logger.info("🚀 Starting CCTV Video Analysis System")
    # Initialize models, database connections, etc.

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("🛑 Shutting down CCTV Video Analysis System")
    # Cleanup resources

@app.get("/")
async def root():
    return {
        "message": "CCTV Video Analysis API",
        "version": "1.0.0",
        "status": "active"
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        workers=settings.WORKERS
    )