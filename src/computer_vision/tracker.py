import numpy as np
from scipy.optimize import linear_sum_assignment

class DebrisTracker:
    def __init__(self):
        self.tracks = {}
        self.next_id = 0
        self.max_age = 10
    
    def update(self, detections):
        if len(self.tracks) == 0:
            for det in detections:
                self.create_new_track(det)
            return self.tracks
        
        cost_matrix = self.compute_cost_matrix(detections)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        matched_detections = set()
        matched_tracks = set()
        
        for row, col in zip(row_ind, col_ind):
            if cost_matrix[row, col] < 0.7:
                track_id = list(self.tracks.keys())[row]
                self.tracks[track_id]['bbox'] = detections[col]
                self.tracks[track_id]['age'] = 0
                matched_detections.add(col)
                matched_tracks.add(row)
        
        for i, det in enumerate(detections):
            if i not in matched_detections:
                self.create_new_track(det)
        
        self.remove_old_tracks()
        return self.tracks
    
    def compute_cost_matrix(self, detections):
        cost_matrix = np.zeros((len(self.tracks), len(detections)))
        for i, track in enumerate(self.tracks.values()):
            for j, det in enumerate(detections):
                cost_matrix[i, j] = self.iou(track['bbox'], det)
        return 1 - cost_matrix
    
    def iou(self, box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
    def create_new_track(self, bbox):
        self.tracks[self.next_id] = {
            'bbox': bbox,
            'age': 0,
            'positions': [((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)]
        }
        self.next_id += 1
    
    def remove_old_tracks(self):
        to_remove = []
        for track_id, track in self.tracks.items():
            track['age'] += 1
            if track['age'] > self.max_age:
                to_remove.append(track_id)
        
        for track_id in to_remove:
            del self.tracks[track_id]