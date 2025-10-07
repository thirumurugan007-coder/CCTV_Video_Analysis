"""
Video Processing Service for CCTV Analysis
"""
import cv2
import numpy as np
from typing import Generator, Optional, Dict, List, Callable
from pathlib import Path
import asyncio
from concurrent.futures import ThreadPoolExecutor
from backend.utils.logger import get_logger

logger = get_logger(__name__)

class VideoProcessor:
    """
    Process video streams for object detection and analysis
    """
    
    def __init__(self, detector, tracker=None, skip_frames: int = 0, max_workers: int = 4):
        """
        Initialize video processor
        
        Args:
            detector: Object detector instance
            tracker: Object tracker instance
            skip_frames: Number of frames to skip between processing
            max_workers: Max number of worker threads
        """
        self.detector = detector
        self.tracker = tracker
        self.skip_frames = skip_frames
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
    def process_video(self, video_source: str, output_path: Optional[str] = None, callback: Optional[Callable] = None, detect_classes: Optional[List[int]] = None) -> Dict:
        """
        Process entire video file
        
        Args:
            video_source: Path to video file or stream URL
            output_path: Path to save annotated video
            callback: Callback function for each frame
            detect_classes: Specific classes to detect
            
        Returns:
            Processing results and statistics
        """
        logger.info(f"📹 Processing video: {video_source}")
        
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            logger.error(f"❌ Failed to open video: {video_source}")
            return {"success": False, "error": "Failed to open video"}
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"Video: {width}x{height} @ {fps}fps, {total_frames} frames")
        
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        stats = {
            "total_frames": total_frames,
            "processed_frames": 0,
            "total_detections": 0,
            "detections_by_class": {},
            "processing_time": 0
        }
        
        frame_count = 0
        
        try:
            import time
            start_time = time.time()
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                if self.skip_frames > 0 and frame_count % (self.skip_frames + 1) != 0:
                    continue
                
                detections = self.detector.detect_objects(frame)
                
                if detections:
                    stats["processed_frames"] += 1
                    stats["total_detections"] += len(detections)
                    
                    for det in detections:
                        class_name = det.get("class", "unknown")
                        stats["detections_by_class"][class_name] = stats["detections_by_class"].get(class_name, 0) + 1
                    
                    self.detector.visualize_detections(frame, detections)
                    
                    if writer:
                        writer.write(frame)
                    
                    if callback:
                        callback(frame_count, frame, detections)
                
                if frame_count % 100 == 0:
                    progress = (frame_count / total_frames) * 100
                    logger.info(f"Progress: {progress:.1f}% ({frame_count}/{total_frames})")
            
            stats["processing_time"] = time.time() - start_time
            stats["success"] = True
            
            logger.info(f"✅ Video processing complete in {stats['processing_time']:.2f}s")
            
        except Exception as e:
            logger.error(f"❌ Error processing video: {e}")
            stats["success"] = False
            stats["error"] = str(e)
            
        finally:
            cap.release()
            if writer:
                writer.release()
        
        return stats
    
    def process_stream_generator(self, video_source: str, detect_classes: Optional[List[int]] = None) -> Generator:
        """
        Process video stream and yield frames for real-time streaming
        """
        cap = cv2.VideoCapture(video_source)
        
        if not cap.isOpened():
            logger.error(f"Failed to open stream: {video_source}")
            return
        
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                if self.skip_frames > 0 and frame_count % (self.skip_frames + 1) != 0:
                    continue
                
                detections = self.detector.detect_objects(frame)
                
                if detections:
                    self.detector.visualize_detections(frame, detections)
                    
                    yield {
                        "frame": frame,
                        "detections": detections,
                        "frame_number": frame_count
                    }
                    
        finally:
            cap.release()
    
    async def process_video_async(self, video_source: str, output_path: Optional[str] = None, detect_classes: Optional[List[int]] = None) -> Dict:
        """
        Process video asynchronously
        """
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            self.executor,
            self.process_video,
            video_source,
            output_path,
            None,
            detect_classes
        )
        return result