import sys
import os
import numpy as np
import cv2

# Set path to project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from matchtrack.ai_detector import (
    DetectionBox,
    OffsideAnalyzer,
    YOLODetectorBackend,
    RTDETRBackend,
    YOLOPoseBackend,
    SKELETON_CONNECTIONS
)
from matchtrack.ai_tracker import AIBroadcastTracker, BroadcastConfig
from matchtrack.radar_minimap import SoccerRadarRenderer


def test_detection_box_properties():
    box = DetectionBox(bbox=(0.2, 0.3, 0.4, 0.7), conf=0.88, cls_id=0, cls_name="player", team_id=1)
    cx, cy = box.center
    assert abs(cx - 0.3) < 1e-4, f"Expected cx=0.3, got {cx}"
    assert abs(cy - 0.5) < 1e-4, f"Expected cy=0.5, got {cy}"
    assert abs(box.width - 0.2) < 1e-4
    assert abs(box.height - 0.4) < 1e-4
    print("✓ test_detection_box_properties passed")


def test_offside_analyzer():
    # Fewer than 4 players -> None
    assert OffsideAnalyzer.compute_offside_line([]) is None
    
    # Right-side attack (ball at x=0.75):
    # Team 1 (red) is defending at right goal (x > 0.6)
    # Team 0 (blue) is attacking
    kpts_p1 = np.zeros((17, 3), dtype=np.float32)
    kpts_p1[15] = [0.82, 0.60, 0.9] # Left ankle
    kpts_p1[16] = [0.84, 0.60, 0.9] # Right ankle (deepest defender)

    kpts_p2 = np.zeros((17, 3), dtype=np.float32)
    kpts_p2[15] = [0.77, 0.55, 0.9]
    kpts_p2[16] = [0.78, 0.55, 0.9] # 2nd last defender ankle at x=0.78

    players = [
        # Attacking team 0
        DetectionBox(bbox=(0.60, 0.40, 0.65, 0.70), conf=0.9, cls_id=0, cls_name="player", team_id=0),
        DetectionBox(bbox=(0.72, 0.45, 0.76, 0.75), conf=0.9, cls_id=0, cls_name="player", team_id=0),
        # Defending team 1
        DetectionBox(bbox=(0.80, 0.40, 0.85, 0.70), conf=0.9, cls_id=0, cls_name="player", team_id=1, keypoints=kpts_p1),
        DetectionBox(bbox=(0.75, 0.40, 0.79, 0.70), conf=0.9, cls_id=0, cls_name="player", team_id=1, keypoints=kpts_p2),
    ]

    res = OffsideAnalyzer.compute_offside_line(players, ball_pos=(0.75, 0.50))
    assert res is not None, "Expected offside line to be detected"
    off_x, off_team, label = res
    assert off_team == 1, f"Expected defending team 1, got {off_team}"
    assert abs(off_x - 0.78) < 1e-3, f"Expected offside x=0.78 from ankle keypoint, got {off_x}"
    assert "Team Rot" in label
    print("✓ test_offside_analyzer passed")


def test_backends_and_tracker():
    cfg = BroadcastConfig(
        ai_backend="yolo11",
        enable_offside_line=True,
        enable_skeletons=True
    )
    tracker = AIBroadcastTracker(cfg)
    assert tracker.detector_backend is not None
    assert "YOLO" in tracker.detector_backend.name

    # Switch to pose backend
    ok_pose = tracker.set_backend("pose")
    assert ok_pose
    assert "Pose" in tracker.detector_backend.name

    # Switch to rtdetr backend
    ok_rtdetr = tracker.set_backend("rtdetr")
    assert ok_rtdetr
    assert "DETR" in tracker.detector_backend.name

    # Switch back to yolo11
    ok_yolo = tracker.set_backend("yolo11")
    assert ok_yolo
    print("✓ test_backends_and_tracker passed")


def test_radar_offside_rendering():
    dummy = np.full((720, 1280, 3), 40, dtype=np.uint8)
    players = [(0.2, 0.3, 0), (0.4, 0.5, 1), (0.7, 0.4, 1), (0.8, 0.6, 1)]
    ball = (0.5, 0.5)
    cam_span = (0.3, 0.7)
    offside = (0.72, 1, "🚩 Abseitslinie (Team Rot)")

    rendered = SoccerRadarRenderer.draw_radar_opencv(
        dummy,
        players,
        ball,
        cam_span,
        radar_w=320,
        pos="top_right",
        opacity=0.85,
        offside_line=offside
    )
    assert rendered is not None
    assert rendered.shape == (720, 1280, 3)
    print("✓ test_radar_offside_rendering passed")


if __name__ == "__main__":
    test_detection_box_properties()
    test_offside_analyzer()
    test_backends_and_tracker()
    test_radar_offside_rendering()
    print("🎉 All Phase C tests passed successfully!")

