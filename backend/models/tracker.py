"""
Object Tracking Module
"""
import cv2
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple

class ObjectTracker:
    """Track detected objects across frames"""
    
    def __init__(self, max_disappeared=50):
        self.next_object_id = 0
        self.objects = {}
        self.disappeared = defaultdict(int)
        self.max_disappeared = max_disappeared
        
    def register(self, centroid):
        """Register a new object"""
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1
        
    def deregister(self, object_id):
        """Remove an object"""
        del self.objects[object_id]
        del self.disappeared[object_id]
        
    def update(self, detections):
        """Update tracked objects"""
        if len(detections) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects
            
        input_centroids = np.array([self._get_centroid(det) for det in detections])
        
        if len(self.objects) == 0:
            for centroid in input_centroids:
                self.register(centroid)
        else:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())
            
            D = self._compute_distance_matrix(object_centroids, input_centroids)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            
            used_rows = set()
            used_cols = set()
            
            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                    
                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]
                self.disappeared[object_id] = 0
                
                used_rows.add(row)
                used_cols.add(col)
                
            unused_rows = set(range(0, D.shape[0])).difference(used_rows)
            unused_cols = set(range(0, D.shape[1])).difference(used_cols)
            
            for row in unused_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
                    
            for col in unused_cols:
                self.register(input_centroids[col])
                
        return self.objects
    
    def _get_centroid(self, detection):
        """Calculate centroid from bounding box"""
        bbox = detection['bbox']
        return [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
    
    def _compute_distance_matrix(self, centroids_a, centroids_b):
        """Compute Euclidean distance matrix"""
        return np.linalg.norm(np.array(centroids_a)[:, np.newaxis] - np.array(centroids_b), axis=2)