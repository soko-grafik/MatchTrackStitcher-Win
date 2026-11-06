"""
Test Suite for Phase B: Robust Multi-Object Tracking (ByteTrack) & 2D Tactical Radar Minimap.
"""
import os
import sys

# Ensure root folder is in python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import cv2

from matchtrack.bytetrack import ByteTracker, KalmanBoxTracker
from matchtrack.radar_minimap import SoccerRadarRenderer
from matchtrack.ai_tracker import AIBroadcastTracker, BroadcastConfig

from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QPainter, QImage
from PySide6.QtCore import QRectF, QPointF


def test_bytetrack_association():
    tracker = ByteTracker(track_thresh=0.40, high_thresh=0.50, match_thresh=0.60)
    
    # Frame 1: 3 players detected
    f1_boxes = np.array([
        [0.20, 0.40, 0.24, 0.50],
        [0.50, 0.45, 0.54, 0.55],
        [0.80, 0.35, 0.84, 0.45]
    ], dtype=np.float32)
    f1_scores = np.array([0.85, 0.90, 0.75], dtype=np.float32)
    f1_teams = np.array([0, 1, 2], dtype=np.int32)

    tracks1 = tracker.update(f1_boxes, f1_scores, f1_teams)
    assert len(tracks1) == 3, f"Expected 3 tracks in frame 1, got {len(tracks1)}"
    track_ids_1 = {t[0] for t in tracks1}

    # Frame 2: Slight movement, one low confidence detection (partial occlusion)
    f2_boxes = np.array([
        [0.205, 0.402, 0.245, 0.502],
        [0.505, 0.452, 0.545, 0.552], # Low confidence
        [0.805, 0.352, 0.845, 0.452]
    ], dtype=np.float32)
    f2_scores = np.array([0.88, 0.35, 0.80], dtype=np.float32)
    f2_teams = np.array([0, 1, 2], dtype=np.int32)

    tracks2 = tracker.update(f2_boxes, f2_scores, f2_teams)
    assert len(tracks2) == 3, f"Expected ByteTrack to maintain 3 tracks through stage-2 association, got {len(tracks2)}"
    track_ids_2 = {t[0] for t in tracks2}
    
    # Track IDs should remain identical
    assert track_ids_1 == track_ids_2, f"Track IDs diverged: {track_ids_1} vs {track_ids_2}"


def test_radar_opencv_rendering():
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    players = [(0.25, 0.50, 0), (0.75, 0.50, 1), (0.50, 0.50, 2)]
    ball = (0.52, 0.48)
    cam_span = (0.35, 0.65)

    positions = ["top_right", "bottom_right", "bottom_center", "top_left", "bottom_left"]
    for pos in positions:
        f_copy = frame.copy()
        out = SoccerRadarRenderer.draw_radar_opencv(
            f_copy,
            players=players,
            ball_pos=ball,
            cam_span=cam_span,
            radar_w=320,
            pos=pos,
            opacity=0.85
        )
        assert out is not None
        assert out.shape == (720, 1280, 3)

    os.makedirs("tests/output", exist_ok=True)
    cv2.imwrite("tests/output/test_radar_opencv_phase_b.jpg", f_copy)
    assert os.path.exists("tests/output/test_radar_opencv_phase_b.jpg")


def test_radar_qpainter_rendering():
    app = QApplication.instance() or QApplication(sys.argv)
    
    img = QImage(1280, 720, QImage.Format_ARGB32_Premultiplied)
    img.fill(0)

    painter = QPainter(img)
    try:
        players = [(0.25, 0.50, 0), (0.75, 0.50, 1), (0.50, 0.50, -1)]
        ball = (0.52, 0.48)
        cam_span = (0.35, 0.65)

        SoccerRadarRenderer.paint_radar_qt(
            painter=painter,
            widget_w=1280,
            widget_h=720,
            players=players,
            ball_norm=ball,
            cam_span=cam_span,
            radar_w=320,
            pos="top_right",
            opacity=0.88
        )
    finally:
        painter.end()

    os.makedirs("tests/output", exist_ok=True)
    img.save("tests/output/test_radar_qt_phase_b.png")
    assert os.path.exists("tests/output/test_radar_qt_phase_b.png")


def test_ai_tracker_radar_properties():
    cfg = BroadcastConfig(enable_radar=True, radar_width=320, radar_position="top_right", radar_opacity=0.85)
    tracker = AIBroadcastTracker(cfg)

    assert isinstance(tracker.active_players, list)
    assert tracker.current_ball_norm is None
    span = tracker.get_camera_span()
    assert len(span) == 2
    assert 0.0 <= span[0] < span[1] <= 1.0


if __name__ == "__main__":
    print("Running Phase B Tracking & Radar tests...")
    test_bytetrack_association()
    print("✓ test_bytetrack_association passed")
    test_radar_opencv_rendering()
    print("✓ test_radar_opencv_rendering passed")
    test_radar_qpainter_rendering()
    print("✓ test_radar_qpainter_rendering passed")
    test_ai_tracker_radar_properties()
    print("✓ test_ai_tracker_radar_properties passed")
    print("All Phase B tests passed perfectly!")
