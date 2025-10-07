import cv2
import numpy as np
from yolov8 import YOLO  # Assuming YOLOv8 is installed
from collections import deque

class ObjectDetectionModel:
    def __init__(self, model_path, confidence_threshold=0.5):
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.trackers = {}
        self.track_id = 0

    def detect_objects(self, frame):
        results = self.model.predict(frame)

        detections = []
        for result in results:
            if result['confidence'] > self.confidence_threshold:
                detections.append(result)

        return detections

    def track_objects(self, detections):
        for detection in detections:
            x, y, w, h = detection['bbox']
            self.track_id += 1
            tracker = cv2.TrackerKCF_create()
            tracker.init(frame, (x, y, w, h))
            self.trackers[self.track_id] = tracker

    def visualize_detections(self, frame, detections):
        for detection in detections:
            x, y, w, h = detection['bbox']
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, f"{detection['class']} {detection['confidence']:.2f}", 
                        (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    def process_frame(self, frame):
        try:
            detections = self.detect_objects(frame)
            self.track_objects(detections)
            self.visualize_detections(frame, detections)
        except Exception as e:
            print(f"Error processing frame: {e}")

if __name__ == "__main__":
    model_path = "path/to/yolov8_model.pt"  # Update with the actual model path
    video_source = "path/to/video.mp4"  # Update with the actual video path
    cap = cv2.VideoCapture(video_source)

    detector = ObjectDetectionModel(model_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        detector.process_frame(frame)
        cv2.imshow("Detections", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()