"""
API Routes for CCTV Video Analysis
"""
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from typing import Optional
import cv2
import numpy as np
from backend.models.detector import ObjectDetectionModel
from backend.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()

# Initialize detector
detector = ObjectDetectionModel("yolov8n.pt", confidence_threshold=0.25)

@router.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    """Upload and process video"""
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        
        # Process video
        logger.info(f"Processing video: {file.filename}")
        
        return {
            "success": True,
            "message": "Video uploaded successfully",
            "filename": file.filename
        }
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/detect/frame")
async def detect_frame(file: UploadFile = File(...)):
    """Detect objects in a single frame"""
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        detections = detector.detect_objects(frame)
        
        return {
            "success": True,
            "detections": detections,
            "count": len(detections)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": detector.model is not None}