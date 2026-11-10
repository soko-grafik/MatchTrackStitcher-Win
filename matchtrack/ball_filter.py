"""
Advanced Ball & Sideline Filtering Engine for MatchTrack-Stitcher.
Provides multi-stage spatial, kinematic, and temporal verification to eliminate:
1. White jersey / clothing false positives (torso/shorts intersection & synchronous motion)
2. Static pitch artifacts (leaves, divots, trash, pitch spots that never move)
3. Sideline persons (trainers, coaches, substitutes, spectators outside or along the boundary)
4. Non-spherical or oversized ball detection anomalies
"""
import math
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import cv2

from .logger import get_logger

logger = get_logger("ball_filter")


@dataclass
class StaticPitchSpot:
    """Tracks a candidate stationary position across time to identify static artifacts."""
    x: float
    y: float
    stationary_frames: int = 1
    last_seen_frame: int = 0
    is_suppressed: bool = False
    max_conf: float = 0.0


class StaticPitchArtifactSuppressor:
    """
    Identifies and suppresses static pitch artifacts (leaves, divots, white marks, trash).
    A real soccer ball in open play is in continuous motion (rolling, bouncing, kicked).
    If a detection stays at the exact same location (displacement < 0.004) for > 20 frames (~0.7s)
    without any player standing directly over it to play a restart, it is blacklisted as a static artifact.
    """
    def __init__(self, 
                 static_threshold_frames: int = 20, 
                 suppression_radius: float = 0.015,
                 max_inactive_frames: int = 150):
        self.spots: Dict[int, StaticPitchSpot] = {}
        self.next_spot_id: int = 0
        self.static_threshold_frames = static_threshold_frames
        self.suppression_radius = suppression_radius
        self.max_inactive_frames = max_inactive_frames

    def reset(self):
        self.spots.clear()
        self.next_spot_id = 0

    def update_and_filter(self, 
                          candidates: List[Any], 
                          player_feet_positions: List[Tuple[float, float]],
                          frame_idx: int) -> Tuple[List[Any], List[Tuple[float, float]]]:
        """
        Updates static candidate tracking and filters out blacklisted stationary artifacts.
        Returns: (valid_candidates, list_of_suppressed_spot_positions)
        """
        # 1. Age and prune old inactive spots
        prune_ids = [
            sid for sid, s in self.spots.items() 
            if (frame_idx - s.last_seen_frame) > self.max_inactive_frames
        ]
        for sid in prune_ids:
            del self.spots[sid]

        # 2. Associate each candidate with tracked stationary spots
        valid_candidates = []
        matched_spot_ids = set()

        for cand in candidates:
            # cand can be DetectionBox or (bx, by, bconf, ...)
            if hasattr(cand, "center"):
                cx, cy = cand.center
                conf = cand.conf
            else:
                cx, cy = cand[0], cand[1]
                conf = cand[2] if len(cand) > 2 else 0.5

            # Check if this candidate falls inside an already suppressed spot
            is_blacklisted = False
            for sid, spot in self.spots.items():
                if spot.is_suppressed:
                    d = math.hypot(cx - spot.x, cy - spot.y)
                    if d <= self.suppression_radius:
                        # Refresh last seen frame
                        spot.last_seen_frame = frame_idx
                        is_blacklisted = True
                        break

            if is_blacklisted:
                continue

            # Find matching spot for motion tracking
            matched_sid = None
            min_d = 0.005 # ~5 pixels normalized movement threshold
            for sid, spot in self.spots.items():
                if sid in matched_spot_ids:
                    continue
                d = math.hypot(cx - spot.x, cy - spot.y)
                if d < min_d:
                    min_d = d
                    matched_sid = sid

            if matched_sid is not None:
                spot = self.spots[matched_sid]
                spot.stationary_frames += 1
                spot.last_seen_frame = frame_idx
                spot.max_conf = max(spot.max_conf, conf)
                matched_spot_ids.add(matched_sid)

                # Check if spot has reached stationary threshold
                if spot.stationary_frames >= self.static_threshold_frames:
                    # Check if any player's feet are within kicking distance (< 0.045)
                    player_nearby = False
                    if player_feet_positions:
                        min_p_dist = min(math.hypot(cx - px, cy - py) for px, py in player_feet_positions)
                        if min_p_dist < 0.045:
                            player_nearby = True

                    # If no player is nearby, or if it has been stationary for > 60 frames (~2s)
                    # even with players around, it is a static artifact (leaf, divot, turf mark)
                    if not player_nearby or spot.stationary_frames >= 60:
                        spot.is_suppressed = True
                        continue # Suppress this candidate
            else:
                # Register new prospective stationary spot
                sid = self.next_spot_id
                self.next_spot_id += 1
                self.spots[sid] = StaticPitchSpot(
                    x=cx,
                    y=cy,
                    stationary_frames=1,
                    last_seen_frame=frame_idx,
                    is_suppressed=False,
                    max_conf=conf
                )
                matched_spot_ids.add(sid)

            valid_candidates.append(cand)

        suppressed_positions = [
            (s.x, s.y) for s in self.spots.values() if s.is_suppressed
        ]
        return valid_candidates, suppressed_positions


class BallPlausibilityFilter:
    """
    Multi-stage verification for soccer ball detections:
    1. Rejects candidates located inside player jerseys/torso/shorts (white kits).
    2. Rejects candidates with non-spherical aspect ratios (jersey stripes, line fragments).
    3. Rejects candidates oversized relative to nearby players.
    4. Validates color/brightness contrast for low-confidence detections (brown leaves vs white ball).
    """

    @staticmethod
    def is_inside_player_torso(bx: float, by: float, players: List[Any]) -> bool:
        """
        Determines whether candidate (bx, by) falls inside any player's upper body / jersey area.
        Human anatomical proportions in bounding box:
        - 0.00 to 0.12: Head / Neck (aerial duels, valid)
        - 0.12 to 0.82: Torso, Chest, Belly, Hips, Shorts (JERSEY ZONE - INVALID FOR BALL)
        - 0.82 to 1.00: Legs, Ankles, Feet / Grass (dribble / ground pass, valid)
        """
        for p in players:
            if hasattr(p, "bbox"):
                px1, py1, px2, py2 = p.bbox
            else:
                px1, py1, px2, py2 = p[0], p[1], p[2], p[3]

            pw = px2 - px1
            ph = py2 - py1
            if pw <= 1e-4 or ph <= 1e-4:
                continue

            # Check horizontal alignment with slight 5% margin
            if (px1 - 0.05 * pw) <= bx <= (px2 + 0.05 * pw):
                rel_y = (by - py1) / ph
                if 0.12 <= rel_y <= 0.82:
                    return True # Inside shirt / shorts

        return False

    @staticmethod
    def validate_aspect_ratio_and_size(bx1: float, by1: float, bx2: float, by2: float, 
                                       median_player_height: Optional[float] = None) -> bool:
        """
        Validates that the ball candidate is approximately circular and physiologically sized.
        """
        bw = max(1e-4, bx2 - bx1)
        bh = max(1e-4, by2 - by1)
        aspect_ratio = bw / bh

        # Soccer ball is spherical; allowing for motion blur between 0.50 and 1.85
        if aspect_ratio < 0.45 or aspect_ratio > 2.00:
            return False

        # If median player height is known, ball height should not exceed ~32% of player height
        # (FIFA ball = 22cm, player = 180cm -> ball is ~12% of height; with blur max 30%)
        if median_player_height is not None and median_player_height > 0.01:
            if bh > (median_player_height * 0.35):
                return False

        # Absolute upper limit in normalized coordinates (a ball is never > 4.5% of total frame height)
        if bh > 0.045:
            return False

        return True

    @staticmethod
    def validate_patch_color(frame_bgr: np.ndarray, 
                             norm_bbox: Tuple[float, float, float, float],
                             min_brightness: int = 110) -> bool:
        """
        Validates that candidate patch is sufficiently bright/high-contrast (soccer ball)
        rather than dull/dark olive/brown (dead leaf, mud divot).
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return True

        h, w = frame_bgr.shape[:2]
        x1 = max(0, min(w - 1, int(round(norm_bbox[0] * w))))
        y1 = max(0, min(h - 1, int(round(norm_bbox[1] * h))))
        x2 = max(x1 + 2, min(w, int(round(norm_bbox[2] * w))))
        y2 = max(y1 + 2, min(h, int(round(norm_bbox[3] * h))))

        patch = frame_bgr[y1:y2, x1:x2]
        if patch.size < 4:
            return True

        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        v_channel = hsv[:, :, 2]
        max_val = int(np.max(v_channel))
        mean_val = float(np.mean(v_channel))

        # A real soccer ball has bright pixels (white or high-vis neon)
        # Dead leaves/divots typically have max_val < 110 and mean < 90
        if max_val < min_brightness and mean_val < (min_brightness - 25):
            return False

        return True


def is_player_on_pitch(bbox: Tuple[float, float, float, float], 
                       corners: List[List[float]], 
                       margin_pct: float = 1.0) -> bool:
    """
    Validates that a player is genuinely standing ON the pitch turf.
    Uses the feet contact point at the bottom of the bounding box ((x1+x2)/2, y2).
    A person is on the pitch IF AND ONLY IF their feet touch the playing surface inside the boundary.
    Applies an optional inward margin (e.g. 1.0%) to strictly exclude sideline coaches, substitutes, and benches.
    """
    x1, y1, x2, y2 = bbox
    foot_x = (x1 + x2) * 0.5
    foot_y = y2  # Feet touching grass
    center_x = (x1 + x2) * 0.5
    center_y = (y1 + y2) * 0.5

    # 1. Feet must be strictly inside the pitch boundary polygon (with inward margin)
    if not is_point_in_pitch_polygon_with_margin(foot_x, foot_y, corners, margin_pct=margin_pct):
        return False

    # 2. Torso/center must also not be outside in the opposite direction
    if not is_point_in_pitch_polygon_with_margin(center_x, center_y, corners, margin_pct=-2.0):
        return False

    return True


def is_point_in_pitch_polygon_with_margin(pt_x: float, pt_y: float, 
                                          corners: List[List[float]], 
                                          margin_pct: float = 0.0) -> bool:
    """
    Ray casting point-in-polygon test with optional inward/outward margin offset.
    margin_pct > 0 insets the boundary towards the centroid (stricter pitch boundary).
    margin_pct < 0 expands the boundary outward (relaxed boundary).
    """
    if not corners or len(corners) < 3:
        return True

    poly_corners = corners
    if abs(margin_pct) > 1e-4:
        cx = sum(c[0] for c in corners) / len(corners)
        cy = sum(c[1] for c in corners) / len(corners)
        shrink_ratio = margin_pct / 100.0
        poly_corners = [
            [c[0] + (cx - c[0]) * shrink_ratio, c[1] + (cy - c[1]) * shrink_ratio]
            for c in corners
        ]

    inside = False
    n = len(poly_corners)
    for i in range(n):
        x1, y1 = poly_corners[i]
        x2, y2 = poly_corners[(i + 1) % n]
        if ((y1 > pt_y) != (y2 > pt_y)) and (pt_x < (x2 - x1) * (pt_y - y1) / (y2 - y1 + 1e-9) + x1):
            inside = not inside
    return inside


class SidelineStaffClassifier:
    """
    Detects and filters coaches, trainers, and substitutes who remain stationary
    along the sideline / technical area during the match.
    """
    def __init__(self, speed_threshold_kmh: float = 2.8, boundary_dist_threshold: float = 0.045):
        self.speed_threshold_kmh = speed_threshold_kmh
        self.boundary_dist_threshold = boundary_dist_threshold
        # {track_id: {'positions': [(x, y, f_idx)], 'sideline_frames': int}}
        self.track_history: Dict[int, Dict[str, Any]] = {}

    def reset(self):
        self.track_history.clear()

    def update_and_classify(self, 
                             active_tracks: List[Tuple[int, float, float, float, float, int]], 
                             corners: List[List[float]],
                             frame_idx: int) -> Dict[int, bool]:
        """
        Takes active ByteTrack output: [(track_id, cx, cy, w, h, team_id), ...]
        Returns dictionary {track_id: is_sideline_staff}.
        """
        result: Dict[int, bool] = {}
        if not corners:
            for t in active_tracks:
                result[t[0]] = False
            return result

        current_ids = set()
        for t in active_tracks:
            tid, cx, cy, w, h, _ = t
            current_ids.add(tid)
            foot_x = cx
            foot_y = cy + h * 0.5

            # Compute minimum distance to any segment of pitch polygon
            min_boundary_dist = self._dist_to_polygon_boundary(foot_x, foot_y, corners)

            if tid not in self.track_history:
                self.track_history[tid] = {
                    'positions': [(cx, cy, frame_idx)],
                    'sideline_frames': 0
                }
            hist = self.track_history[tid]
            hist['positions'].append((cx, cy, frame_idx))
            if len(hist['positions']) > 90: # Keep last 3 seconds
                hist['positions'].pop(0)

            # Check if standing near sideline
            if min_boundary_dist < self.boundary_dist_threshold:
                hist['sideline_frames'] += 1
            else:
                hist['sideline_frames'] = max(0, hist['sideline_frames'] - 2)

            # Check speed / range of motion over last 60-90 frames
            is_staff = False
            if hist['sideline_frames'] >= 45 and len(hist['positions']) >= 30:
                xs = [p[0] for p in hist['positions']]
                ys = [p[1] for p in hist['positions']]
                x_range = max(xs) - min(xs)
                y_range = max(ys) - min(ys)
                disp = math.hypot(x_range, y_range)
                # If total movement over 1-3 seconds is minimal (< 0.015), they are stationary staff/coach
                if disp < 0.018:
                    is_staff = True

            result[tid] = is_staff

        # Prune old tracks
        prune_ids = [tid for tid in self.track_history if tid not in current_ids]
        for tid in prune_ids:
            del self.track_history[tid]

        return result

    @staticmethod
    def _dist_to_polygon_boundary(px: float, py: float, corners: List[List[float]]) -> float:
        """Calculates Euclidean distance from (px, py) to the closest line segment of the polygon."""
        min_d = float('inf')
        n = len(corners)
        for i in range(n):
            x1, y1 = corners[i]
            x2, y2 = corners[(i + 1) % n]
            # Segment vector
            dx = x2 - x1
            dy = y2 - y1
            l2 = dx * dx + dy * dy
            if l2 < 1e-9:
                d = math.hypot(px - x1, py - y1)
            else:
                # Project point onto segment
                t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / l2))
                proj_x = x1 + t * dx
                proj_y = y1 + t * dy
                d = math.hypot(px - proj_x, py - proj_y)
            if d < min_d:
                min_d = d
        return min_d

