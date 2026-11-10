"""
Unit & Integration Tests for Advanced Ball & Sideline Filtering:
1. White jersey torso exclusion (prevents players with white kits from being tracked as balls)
2. Static pitch artifact & leaf suppression (suppresses stationary leaves, divots, pitch marks)
3. Sideline coach & staff exclusion (prevents sideline coaches/staff from affecting tracking)
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from matchtrack.ai_detector import DetectionBox
from matchtrack.ball_filter import (
    BallPlausibilityFilter,
    StaticPitchArtifactSuppressor,
    is_player_on_pitch,
    is_point_in_pitch_polygon_with_margin,
    SidelineStaffClassifier
)
from matchtrack.ai_tracker import AIBroadcastTracker, BroadcastConfig


def test_anti_jersey_torso_exclusion():
    """Verifies that ball detections located on player torsos/jerseys are strictly rejected."""
    # Player bounding box: [x1, y1, x2, y2]
    # Height = 0.20, Width = 0.05
    player = DetectionBox(
        bbox=(0.40, 0.40, 0.45, 0.60),
        conf=0.90,
        cls_id=0,
        cls_name="player",
        team_id=0
    )
    players = [player]

    # 1. Candidate on chest / upper torso (rel_y = (0.48 - 0.40) / 0.20 = 0.40 -> 40% torso)
    assert BallPlausibilityFilter.is_inside_player_torso(0.42, 0.48, players) is True

    # 2. Candidate on belly / shorts (rel_y = (0.54 - 0.40) / 0.20 = 0.70 -> 70% shorts)
    assert BallPlausibilityFilter.is_inside_player_torso(0.42, 0.54, players) is True

    # 3. Candidate at feet on grass (rel_y = (0.59 - 0.40) / 0.20 = 0.95 -> 95% feet)
    assert BallPlausibilityFilter.is_inside_player_torso(0.42, 0.59, players) is False

    # 4. Candidate above head (rel_y = (0.39 - 0.40) / 0.20 = -0.05 -> aerial duel)
    assert BallPlausibilityFilter.is_inside_player_torso(0.42, 0.39, players) is False

    # 5. Candidate in open space beside player
    assert BallPlausibilityFilter.is_inside_player_torso(0.55, 0.50, players) is False


def test_ball_aspect_ratio_and_size():
    """Verifies that non-spherical shapes (stripes/lines) and oversized patches are rejected."""
    # Good round ball: 12x12 px normalized
    assert BallPlausibilityFilter.validate_aspect_ratio_and_size(0.50, 0.50, 0.51, 0.51, median_player_height=0.15) is True

    # Elongated vertical stripe (e.g. jersey sleeve): 4x20 px
    assert BallPlausibilityFilter.validate_aspect_ratio_and_size(0.50, 0.50, 0.504, 0.520, median_player_height=0.15) is False

    # Oversized patch: 50% of player height
    assert BallPlausibilityFilter.validate_aspect_ratio_and_size(0.50, 0.50, 0.58, 0.58, median_player_height=0.15) is False


def test_static_leaf_suppressor():
    """Verifies that stationary leaves/divots are blacklisted after remaining static without player contact."""
    suppressor = StaticPitchArtifactSuppressor(static_threshold_frames=15, suppression_radius=0.015)

    # Static candidate (leaf) at (0.60, 0.45)
    leaf_cand = DetectionBox(bbox=(0.595, 0.445, 0.605, 0.455), conf=0.35, cls_id=32, cls_name="ball")
    
    # Active players far away
    player_feet = [(0.25, 0.50), (0.30, 0.55)]

    # Feed candidate over 20 consecutive frames without movement
    for frame_idx in range(25):
        valid_cands, suppressed = suppressor.update_and_filter([leaf_cand], player_feet, frame_idx)
        if frame_idx < 14:
            assert len(valid_cands) == 1
        else:
            # At 15 stationary frames without player, it MUST be suppressed!
            assert len(valid_cands) == 0
            assert len(suppressed) >= 1

    # In frame 30, the leaf is still detected by AI, but suppressor must filter it out
    valid_cands, _ = suppressor.update_and_filter([leaf_cand], player_feet, 30)
    assert len(valid_cands) == 0

    # Meanwhile, a real moving ball arriving at (0.40, 0.50) MUST be accepted!
    real_ball = DetectionBox(bbox=(0.395, 0.495, 0.405, 0.505), conf=0.75, cls_id=32, cls_name="ball")
    valid_cands, _ = suppressor.update_and_filter([real_ball], player_feet, 31)
    assert len(valid_cands) == 1
    assert valid_cands[0] == real_ball


def test_sideline_coach_exclusion():
    """Verifies that sideline staff and coaches standing outside/on touchlines are excluded."""
    # 6-point pitch polygon:
    # Top touchline at y ~ 0.15, Bottom touchline at y ~ 0.85, Left at x ~ 0.10, Right at x ~ 0.90
    corners = [
        [0.10, 0.15], [0.50, 0.12], [0.90, 0.15],
        [0.90, 0.85], [0.50, 0.88], [0.10, 0.85]
    ]

    # 1. Active player on the pitch
    # Center = (0.50, 0.50), Feet = (0.50, 0.56) -> Clearly on pitch
    active_player = (0.48, 0.44, 0.52, 0.56)
    assert is_player_on_pitch(active_player, corners, margin_pct=1.0) is True

    # 2. Coach standing outside near touchline
    # Feet at y2 = 0.92 (touchline is at 0.88), Center at cy = 0.83 (inside polygon if tested naively)
    coach_near_sideline = (0.48, 0.74, 0.52, 0.92)
    assert is_player_on_pitch(coach_near_sideline, corners, margin_pct=1.0) is False

    # 3. Coach standing outside far touchline (behind far sideline)
    # Feet at y2 = 0.10 (touchline is at 0.12), Center at cy = 0.05
    coach_far_sideline = (0.48, 0.00, 0.52, 0.10)
    assert is_player_on_pitch(coach_far_sideline, corners, margin_pct=1.0) is False


def test_end_to_end_tracker_filtering():
    """End-to-end integration test with AIBroadcastTracker filtering white jersey and sideline coach."""
    config = BroadcastConfig(
        enable_anti_jersey_filter=True,
        enable_static_leaf_filter=True,
        enable_sideline_filter=True,
        pitch_margin_pct=1.0,
        pitch_corners=[
            [0.10, 0.15], [0.50, 0.12], [0.90, 0.15],
            [0.90, 0.85], [0.50, 0.88], [0.10, 0.85]
        ]
    )
    tracker = AIBroadcastTracker(config)

    # Synthetic panorama frame (384x108 for fast test)
    dummy_frame = np.zeros((108, 384, 3), dtype=np.uint8)

    # 1. On-pitch player wearing white jersey
    player_on_pitch = DetectionBox(
        bbox=(0.48, 0.40, 0.52, 0.60),
        conf=0.90,
        cls_id=0,
        cls_name="player",
        team_id=0
    )

    # 2. Sideline coach outside bottom touchline (feet at 0.93 > 0.88)
    coach_on_sideline = DetectionBox(
        bbox=(0.50, 0.75, 0.54, 0.93),
        conf=0.88,
        cls_id=0,
        cls_name="player",
        team_id=1
    )

    # 3. False ball on player's white shirt (rel_y = (0.48 - 0.40) / 0.20 = 0.40)
    white_shirt_ball = DetectionBox(
        bbox=(0.495, 0.475, 0.505, 0.485),
        conf=0.35,
        cls_id=32,
        cls_name="ball",
        team_id=-1
    )

    # 4. Real ball near player's feet
    real_ball = DetectionBox(
        bbox=(0.495, 0.590, 0.505, 0.600),
        conf=0.85,
        cls_id=32,
        cls_name="ball",
        team_id=-1
    )

    raw_players = [player_on_pitch, coach_on_sideline]
    raw_balls = [white_shirt_ball, real_ball]

    valid_players, valid_balls = tracker._filter_players_and_balls(
        raw_players, raw_balls, config.pitch_corners, dummy_frame, f_idx=1
    )

    # Sideline coach MUST be filtered out! Only player on pitch remains!
    assert len(valid_players) == 1
    assert valid_players[0] == player_on_pitch

    # White shirt ball MUST be filtered out! Only real ball at feet remains!
    assert len(valid_balls) == 1
    assert valid_balls[0] == real_ball


if __name__ == "__main__":
    test_anti_jersey_torso_exclusion()
    print("✓ test_anti_jersey_torso_exclusion passed")
    test_ball_aspect_ratio_and_size()
    print("✓ test_ball_aspect_ratio_and_size passed")
    test_static_leaf_suppressor()
    print("✓ test_static_leaf_suppressor passed")
    test_sideline_coach_exclusion()
    print("✓ test_sideline_coach_exclusion passed")
    test_end_to_end_tracker_filtering()
    print("✓ test_end_to_end_tracker_filtering passed")
    print("🎉 All ball and sideline filter tests passed successfully!")
