"""
ByteTrack: Multi-Object Tracking for Soccer Players and Ball.
Provides persistent track IDs across frames and handles partial occlusions
using a two-stage association strategy with high and low confidence detections.
"""
from enum import Enum
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
from scipy.optimize import linear_sum_assignment


class TrackState(Enum):
    NEW = 0
    TRACKED = 1
    LOST = 2
    REMOVED = 3


class KalmanBoxTracker:
    """
    Kalman filter for tracking bounding boxes in image space [x_center, y_center, aspect_ratio, height].
    """
    count = 0

    def __init__(self, bbox: np.ndarray, score: float = 1.0, team_id: int = 0):
        # bbox: [x1, y1, x2, y2]
        self.state = np.zeros(7, dtype=np.float64)  # [xc, yc, s, h, vxc, vyc, vs]
        w = max(1e-4, bbox[2] - bbox[0])
        h = max(1e-4, bbox[3] - bbox[1])
        xc = bbox[0] + w * 0.5
        yc = bbox[1] + h * 0.5
        s = w * h

        self.state[0] = xc
        self.state[1] = yc
        self.state[2] = s
        self.state[3] = h

        self.P = np.eye(7, dtype=np.float64) * 10.0
        self.P[4:, 4:] *= 100.0  # High uncertainty for unobserved velocities

        self.Q = np.eye(7, dtype=np.float64)
        self.Q[4:, 4:] *= 0.01

        self.R = np.eye(4, dtype=np.float64) * 1.0

        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history: List[np.ndarray] = []
        self.hits = 1
        self.hit_streak = 1
        self.age = 0
        self.score = score
        self.team_id = team_id
        self.state_enum = TrackState.NEW

    def predict(self) -> np.ndarray:
        # Motion model: constant velocity
        F = np.eye(7, dtype=np.float64)
        F[0, 4] = 1.0
        F[1, 5] = 1.0
        F[2, 6] = 1.0

        self.state = F @ self.state
        self.P = F @ self.P @ F.T + self.Q

        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(self.get_bbox())
        return self.get_bbox()

    def update(self, bbox: np.ndarray, score: float = 1.0, team_id: Optional[int] = None):
        self.time_since_update = 0
        self.history.clear()
        self.hits += 1
        self.hit_streak += 1
        self.score = score
        if team_id is not None:
            self.team_id = team_id

        w = max(1e-4, bbox[2] - bbox[0])
        h = max(1e-4, bbox[3] - bbox[1])
        xc = bbox[0] + w * 0.5
        yc = bbox[1] + h * 0.5
        s = w * h
        z = np.array([xc, yc, s, h], dtype=np.float64)

        H = np.zeros((4, 7), dtype=np.float64)
        H[0, 0] = 1.0
        H[1, 1] = 1.0
        H[2, 2] = 1.0
        H[3, 3] = 1.0

        y = z - H @ self.state
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.state = self.state + K @ y
        self.P = (np.eye(7) - K @ H) @ self.P
        self.state_enum = TrackState.TRACKED

    def get_bbox(self) -> np.ndarray:
        xc, yc, s, h = self.state[0], self.state[1], self.state[2], self.state[3]
        if s <= 0 or h <= 0:
            return np.zeros(4, dtype=np.float64)
        w = s / h
        return np.array([xc - w * 0.5, yc - h * 0.5, xc + w * 0.5, yc + h * 0.5], dtype=np.float64)


def box_iou_batch(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Computes pairwise IoU between two sets of bounding boxes."""
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float64)

    # boxes_a: [N, 4], boxes_b: [M, 4]
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])

    iw = np.maximum(0.0, np.minimum(boxes_a[:, 2:3], boxes_b[:, 2].T) - np.maximum(boxes_a[:, 0:1], boxes_b[:, 0].T))
    ih = np.maximum(0.0, np.minimum(boxes_a[:, 3:4], boxes_b[:, 3].T) - np.maximum(boxes_a[:, 1:2], boxes_b[:, 1].T))
    inter = iw * ih
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.maximum(union, 1e-6)


class ByteTracker:
    """
    Robust Two-Stage Multi-Object Tracker for Soccer Players.
    Retains low-confidence detections across partial occlusions.
    """
    def __init__(self, 
                 track_thresh: float = 0.45, 
                 high_thresh: float = 0.50, 
                 match_thresh: float = 0.70, 
                 max_time_lost: int = 30):
        self.track_thresh = track_thresh
        self.high_thresh = high_thresh
        self.match_thresh = match_thresh
        self.max_time_lost = max_time_lost
        
        self.tracked_tracks: List[KalmanBoxTracker] = []
        self.lost_tracks: List[KalmanBoxTracker] = []
        self.frame_id = 0

    def reset(self):
        KalmanBoxTracker.count = 0
        self.tracked_tracks.clear()
        self.lost_tracks.clear()
        self.frame_id = 0

    def update(self, 
               detections: np.ndarray, 
               scores: np.ndarray, 
               team_ids: Optional[np.ndarray] = None) -> List[Tuple[int, float, float, float, float, int]]:
        """
        Updates tracker with new frame detections.
        detections: np.ndarray of shape [N, 4] with [x1, y1, x2, y2]
        scores: np.ndarray of shape [N]
        team_ids: np.ndarray of shape [N] (0 = Team A, 1 = Team B, -1 = Unassigned/Ref)
        Returns:
            List of (track_id, center_x, center_y, width, height, team_id)
        """
        self.frame_id += 1
        if team_ids is None:
            team_ids = np.zeros(len(detections), dtype=np.int32)

        # 1. Predict next position for all existing tracks
        for track in self.tracked_tracks:
            track.predict()
        for track in self.lost_tracks:
            track.predict()

        if len(detections) == 0:
            # Mark all as lost or remove
            for track in self.tracked_tracks:
                track.state_enum = TrackState.LOST
            self.lost_tracks.extend(self.tracked_tracks)
            self.tracked_tracks = []
            self.lost_tracks = [t for t in self.lost_tracks if t.time_since_update <= self.max_time_lost]
            return []

        # Split detections into high and low confidence groups
        high_mask = scores >= self.high_thresh
        low_mask = (scores >= 0.15) & (scores < self.high_thresh)

        dets_high = detections[high_mask]
        scores_high = scores[high_mask]
        teams_high = team_ids[high_mask]

        dets_low = detections[low_mask]
        scores_low = scores[low_mask]
        teams_low = team_ids[low_mask]

        # -------------------------------------------------------------
        # Stage 1: Associate High-Confidence Detections with Tracked Tracks
        # -------------------------------------------------------------
        pool_tracks = self.tracked_tracks + self.lost_tracks
        track_boxes = np.array([t.get_bbox() for t in pool_tracks]) if pool_tracks else np.empty((0, 4))

        matched_tracks_idx, unmatched_tracks_idx, unmatched_dets_high_idx = self._associate(
            track_boxes, dets_high, iou_threshold=self.match_thresh
        )

        new_tracked: List[KalmanBoxTracker] = []
        for t_idx, d_idx in matched_tracks_idx:
            track = pool_tracks[t_idx]
            track.update(dets_high[d_idx], score=scores_high[d_idx], team_id=int(teams_high[d_idx]))
            new_tracked.append(track)
            if track in self.lost_tracks:
                self.lost_tracks.remove(track)

        # -------------------------------------------------------------
        # Stage 2: Associate Low-Confidence Detections with Remaining Tracks
        # (This rescues players behind defenders or motion-blurred objects!)
        # -------------------------------------------------------------
        unmatched_tracks = [pool_tracks[i] for i in unmatched_tracks_idx if pool_tracks[i] not in new_tracked]
        unmatched_track_boxes = np.array([t.get_bbox() for t in unmatched_tracks]) if unmatched_tracks else np.empty((0, 4))

        matched_low_idx, unmatched_low_tracks_idx, _ = self._associate(
            unmatched_track_boxes, dets_low, iou_threshold=0.45
        )

        for t_idx, d_idx in matched_low_idx:
            track = unmatched_tracks[t_idx]
            track.update(dets_low[d_idx], score=scores_low[d_idx], team_id=int(teams_low[d_idx]))
            new_tracked.append(track)
            if track in self.lost_tracks:
                self.lost_tracks.remove(track)

        # Tracks that couldn't be matched in either stage
        for idx in unmatched_low_tracks_idx:
            track = unmatched_tracks[idx]
            if track not in self.lost_tracks:
                track.state_enum = TrackState.LOST
                self.lost_tracks.append(track)

        # -------------------------------------------------------------
        # Stage 3: Initiate New Tracks from Unmatched High-Confidence Detections
        # -------------------------------------------------------------
        for d_idx in unmatched_dets_high_idx:
            if scores_high[d_idx] >= self.track_thresh:
                new_t = KalmanBoxTracker(dets_high[d_idx], score=scores_high[d_idx], team_id=int(teams_high[d_idx]))
                new_t.state_enum = TrackState.TRACKED
                new_tracked.append(new_t)

        # Filter out tracks that have been lost for too long
        self.lost_tracks = [t for t in self.lost_tracks if t.time_since_update <= self.max_time_lost]
        self.tracked_tracks = new_tracked

        # Compile active output tracks
        output = []
        for t in self.tracked_tracks:
            if t.hits >= 2 or self.frame_id <= 3:
                box = t.get_bbox()
                w = max(0.001, float(box[2] - box[0]))
                h = max(0.001, float(box[3] - box[1]))
                cx = float(box[0] + w * 0.5)
                cy = float(box[1] + h * 0.5)
                output.append((t.id, cx, cy, w, h, t.team_id))

        return output

    def _associate(self, 
                   track_boxes: np.ndarray, 
                   det_boxes: np.ndarray, 
                   iou_threshold: float = 0.5) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """Matches tracks to detections using Hungarian bipartite matching on IoU matrix."""
        if len(track_boxes) == 0 or len(det_boxes) == 0:
            return [], list(range(len(track_boxes))), list(range(len(det_boxes)))

        ious = box_iou_batch(track_boxes, det_boxes)
        # Cost matrix: minimize cost -> 1.0 - iou
        cost_matrix = 1.0 - ious

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matched: List[Tuple[int, int]] = []
        unmatched_tracks = set(range(len(track_boxes)))
        unmatched_dets = set(range(len(det_boxes)))

        for r, c in zip(row_ind, col_ind):
            if ious[r, c] >= iou_threshold:
                matched.append((r, c))
                unmatched_tracks.discard(r)
                unmatched_dets.discard(c)

        return matched, sorted(list(unmatched_tracks)), sorted(list(unmatched_dets))
