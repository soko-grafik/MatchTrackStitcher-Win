"""
AI-Powered Soccer Broadcast Camera Engine (YOLO11 / YOLOv8 + Player Density Clustering + Physics-Gated Ball Tracking).
Delivers broadcast-quality pure horizontal camera panning (Left-Right Follow Cam, No Zoom) with rock-solid stability.
"""
from dataclasses import dataclass, field
import os
import time
import math
import numpy as np
import cv2
from typing import Tuple, Optional, List, Dict, Any, Callable
from .paths import get_resource_path
from .bytetrack import ByteTracker
from .ai_detector import (
    DetectionBox, 
    BaseDetectorBackend, 
    YOLODetectorBackend, 
    RTDETRBackend, 
    YOLOPoseBackend, 
    OffsideAnalyzer,
    SKELETON_CONNECTIONS
)
from .tactics_metrics import (
    TacticsMetricsEngine,
    TacticsReportExporter,
    MatchTacticsSummary
)
from .ball_filter import (
    StaticPitchArtifactSuppressor,
    BallPlausibilityFilter,
    is_player_on_pitch,
    is_point_in_pitch_polygon_with_margin,
    SidelineStaffClassifier
)
from .logger import get_logger

logger = get_logger("ai_tracker")

try:
    import torch
    if "YOLO_CONFIG_DIR" not in os.environ:
        import tempfile
        candidate_dir = os.path.join(tempfile.gettempdir(), "ultralytics")
        try:
            os.makedirs(candidate_dir, exist_ok=True)
            os.environ["YOLO_CONFIG_DIR"] = candidate_dir
        except Exception:
            pass
    from ultralytics import YOLO
    HAS_YOLO = True
except Exception:
    HAS_YOLO = False


class BallKalmanTracker:
    """
    2D Kalman Filter with physics-based drag modeling and velocity clamping for soccer ball tracking.
    State vector: [x, y, vx, vy] (normalized to [0.0 ... 1.0])
    """
    def __init__(self):
        self.state = np.array([0.5, 0.50, 0.0, 0.0], dtype=np.float64)
        self.P = np.eye(4, dtype=np.float64) * 0.05
        self.Q = np.diag([1e-4, 1e-4, 5e-4, 5e-4])  # Process noise
        self.R = np.diag([2e-3, 2e-3])               # Measurement noise
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float64)
        self.missed_frames = 0
        self.max_plausible_speed = 0.045             # Max normalized distance per frame (~120 km/h at 30 fps)

    def reset(self, x: float = 0.5, y: float = 0.50):
        self.state = np.array([x, y, 0.0, 0.0], dtype=np.float64)
        self.P = np.eye(4, dtype=np.float64) * 0.05
        self.missed_frames = 0

    def predict(self, dt: float = 1.0) -> Tuple[float, float]:
        # Aerodynamic drag decay (0.95)
        dt_val = float(dt)
        A = np.eye(4, dtype=np.float64)
        A[0, 2] = dt_val
        A[1, 3] = dt_val
        A[2, 2] = 0.95
        A[3, 3] = 0.95
        self.state = A @ self.state
        self.P = A @ self.P @ A.T + self.Q
        self.missed_frames += 1
        return float(self.state[0]), float(self.state[1])

    def is_plausible_measurement(self, meas_x: float, meas_y: float, max_dist_override: Optional[float] = None) -> bool:
        """Validates if a measurement obeys physical soccer kinematics (no impossible teleportation)."""
        pred_x, pred_y = self.pos
        dist = np.sqrt((meas_x - pred_x)**2 + (meas_y - pred_y)**2)
        max_dist = max_dist_override if max_dist_override is not None else (self.max_plausible_speed * (self.missed_frames + 1) + 0.04)
        return dist <= max_dist

    def update(self, meas_x: float, meas_y: float, confidence: float = 1.0):
        z = np.array([meas_x, meas_y], dtype=np.float64)
        y = z - self.H @ self.state
        R = self.R / max(0.15, confidence)
        S = self.H @ self.P @ self.H.T + R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.state = self.state + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P
        
        # Clamp velocity to physical max
        speed = np.sqrt(self.state[2]**2 + self.state[3]**2)
        if speed > self.max_plausible_speed:
            scale = self.max_plausible_speed / speed
            self.state[2] *= scale
            self.state[3] *= scale
            
        self.missed_frames = 0

    @property
    def pos(self) -> Tuple[float, float]:
        return float(self.state[0]), float(self.state[1])

    @property
    def vel(self) -> Tuple[float, float]:
        return float(self.state[2]), float(self.state[3])

    @property
    def speed(self) -> float:
        return float(np.sqrt(self.state[2]**2 + self.state[3]**2))


def is_point_in_pitch_polygon(pt_x: float, pt_y: float, corners: List[List[float]]) -> bool:
    """Ray casting point-in-polygon test for arbitrary 4-corner quadrilateral."""
    if not corners or len(corners) < 3:
        return True
    inside = False
    n = len(corners)
    for i in range(n):
        x1, y1 = corners[i]
        x2, y2 = corners[(i + 1) % n]
        if ((y1 > pt_y) != (y2 > pt_y)) and (pt_x < (x2 - x1) * (pt_y - y1) / (y2 - y1 + 1e-9) + x1):
            inside = not inside
    return inside


def get_pitch_polygon_bbox(corners: List[List[float]]) -> Tuple[float, float, float, float]:
    """Returns (min_x, max_x, min_y, max_y) for arbitrary pitch corners."""
    if not corners or len(corners) < 4:
        return 0.0, 1.0, 0.05, 0.95
    xs = [c[0] for c in corners]
    ys = [c[1] for c in corners]
    return float(min(xs)), float(max(xs)), float(min(ys)), float(max(ys))


@dataclass
class BroadcastConfig:
    """Settings for AI Broadcast Camera with Pure Horizontal Panning and Fixed 1.0x Full-Height View."""
    enabled: bool = True
    ai_tracking: bool = True
    enable_dynamic_zoom: bool = False    # Default FALSE: Pure horizontal left-right pan, NO zoom pumping
    fixed_zoom_factor: float = 1.0       # Fixed zoom level (1.0 = Full height 16:9 framing)
    tracking_mode: str = "hybrid_fusion" # 'hybrid_fusion', 'player_density', 'ball_centric', 'smooth_tactic'
    ball_weight: float = 0.70            # Weight of soccer ball tracking (0.0 to 1.0)
    players_weight: float = 0.30         # Weight of player cluster (0.0 to 1.0)
    enable_zone_damping: bool = True     # Dynamic agility near penalty boxes/goals
    zone_boost_factor: float = 1.80      # Agility boost in penalty box zones (1.0x to 3.0x)
    min_zoom: float = 1.00               # Wide angle zoom
    max_zoom: float = 1.40               # Max zoom
    current_zoom: float = 1.00           # Current zoom level
    zoom_speed: float = 0.04             # Speed of zoom transitions
    smoothing_factor: float = 0.94       # 0.88 = responsive, 0.96 = ultra-smooth broadcast
    anticipation_lead: float = 0.15      # Gentle look-ahead in direction of ball movement
    deadband_width: float = 0.08         # Fractional deadzone (8% width) to eliminate micro-jitter
    max_pan_speed: float = 0.04          # Maximum pan speed per frame
    vertical_center_bias: float = 0.50   # Vertical framing (0.5 = center of panorama)
    
    # 6 Freely Movable Pitch Field ROI Polygon Points [TL, TC, TR, BR, BC, BL] (Normalized 0.0 - 1.0)
    # Includes Top-Center (TC) and Bottom-Center (BC) on the halfway line (Mittellinie)
    # to perfectly compensate for panorama optical distortion/curvature along the sidelines.
    pitch_corners: List[List[float]] = field(default_factory=lambda: [
        [0.0, 0.0],  # 0: Top-Left (Video-Ecke Oben-Links / TL)
        [0.5, 0.0],  # 1: Top-Center (Video-Mittellinie Oben / TC)
        [1.0, 0.0],  # 2: Top-Right (Video-Ecke Oben-Rechts / TR)
        [1.0, 1.0],  # 3: Bottom-Right (Video-Ecke Unten-Rechts / BR)
        [0.5, 1.0],  # 4: Bottom-Center (Video-Mittellinie Unten / BC)
        [0.0, 1.0]   # 5: Bottom-Left (Video-Ecke Unten-Links / BL)
    ])
    
    # 16:9 Tactical Mesh-Warp Margin (0.0% to 10.0%)
    tactical_margin: float = 0.0
    
    # Performance & Lookahead Sampling
    scan_step: int = 5                   # Step interval for offline trajectory scan (5 = 3x faster, 8 = 5x faster)
    use_fp16: bool = True                # Enable Half-precision FP16 on CUDA Tensor Cores

    # 2D Tactical Radar Minimap Overlay
    enable_radar: bool = True            # Display 2D bird's-eye pitch radar
    radar_position: str = "top_right"    # 'top_right', 'bottom_right', 'bottom_center', 'top_left', 'bottom_left'
    radar_width: int = 320               # Width of radar in pixels
    radar_opacity: float = 0.85          # Opacity of radar overlay (0.3 to 1.0)
    burn_radar_in_export: bool = False   # Whether to burn radar minimap directly into exported video

    # Phase C: AI Detector Engine & Tactical Skeleton/Offside
    ai_backend: str = "yolo11"           # "yolo11" (Ultralytics), "rtdetr" (Transformer), "pose" (YOLO-Pose)
    enable_offside_line: bool = True     # Automatic VAR Virtual Offside Line
    enable_skeletons: bool = False       # Display player 17-keypoint skeletons & body orientation

    # Phase D: Metric Analytics & Scouting (SkillCorner & TacticAI)
    enable_speed_badges: bool = False    # Display player/ball real-time speed in km/h
    enable_tactical_heatmap: bool = False # Display tactical density heatmap on radar/canvas

    # Robust Ball & Sideline Filtering (Anti-Jersey, Anti-Leaf, Sideline-Staff Filter)
    enable_anti_jersey_filter: bool = True     # Rejects ball candidates overlapping player torso/clothing
    enable_static_leaf_filter: bool = True     # Rejects static pitch spots/leaves/divots that never move
    enable_sideline_filter: bool = True        # Strictly excludes sideline coaches/staff/spectators outside pitch
    pitch_margin_pct: float = 1.0              # Inward boundary margin (0.0% to 5.0%) to cleanly reject sideline people


class AIBroadcastTracker:
    """
    AI-Powered Soccer Broadcast Camera Tracker for 32:9 Panorama -> 16:9 TV Broadcast.
    Uses Player Density Clustering (Kernel Density Estimation) + Physics-Gated Ball Tracking.
    """
    def __init__(self, config: Optional[BroadcastConfig] = None):
        self.config = config or BroadcastConfig()
        
        # Camera position and velocity (normalized [0.0 ... 1.0])
        self.cam_x: float = 0.5
        self.cam_y: float = self.config.vertical_center_bias
        self.cam_vx: float = 0.0
        self.cam_zoom: float = self.config.fixed_zoom_factor if not self.config.enable_dynamic_zoom else self.config.current_zoom
        
        # Target action state
        self.target_x: float = 0.5
        self.target_y: float = self.config.vertical_center_bias
        self.target_zoom: float = self.cam_zoom
        
        # Keyframing and inference interval
        self.frame_counter: int = 0
        self.last_target_x: Optional[float] = None
        self.last_target_y: Optional[float] = None
        self.last_target_zoom: Optional[float] = None
        self.inference_interval: int = 3
        
        # Kalman Ball Tracker
        self.kalman = BallKalmanTracker()
        self.ball_confidence: float = 0.0
        self.last_detected_ball_px: Optional[Tuple[int, int]] = None
        
        # Player Cluster & Multi-Object ByteTracker State
        self.last_player_centroid: float = 0.5
        self.bytetracker = ByteTracker(track_thresh=0.40, high_thresh=0.50, match_thresh=0.65)
        self.current_player_tracks: List[Tuple[float, float, int]] = [] # (norm_x, norm_y, team_id)

        # Robust Ball & Sideline Filtering Engines
        self.static_leaf_suppressor = StaticPitchArtifactSuppressor()
        self.sideline_classifier = SidelineStaffClassifier()
        self.suppressed_spots: List[Tuple[float, float]] = []
        self.sideline_staff_tracks: List[int] = []
        
        # Motion history for CV fallback
        self.prev_gray: Optional[np.ndarray] = None
        
        # Precomputed offline lookahead trajectory cache {frame_index: (cam_x, cam_y, cam_zoom)}
        self.precomputed_trajectory: Dict[int, Tuple[float, float, float]] = {}
        
        # AI Detector Backend & Tactical Analysis
        self.detector_backend: Optional[BaseDetectorBackend] = None
        self.current_detections: List[DetectionBox] = []
        self._current_offside_line: Optional[Tuple[float, int, str]] = None
        self.tactics_engine = TacticsMetricsEngine(self.config.pitch_corners)

        # AI Models Legacy
        self.model: Optional[Any] = None
        self.ball_model: Optional[Any] = None
        self.model_name: str = "none"
        self.device = "cuda" if (HAS_YOLO and torch.cuda.is_available()) else "cpu"
        self._init_models()

    def _init_models(self):
        """Initializes specialized soccer detection models, RT-DETR Transformer, or YOLO-Pose."""
        backend_type = getattr(self.config, 'ai_backend', 'yolo11')
        use_fp16 = (self.device == "cuda" and getattr(self.config, "use_fp16", True))

        try:
            if backend_type == "rtdetr":
                self.detector_backend = RTDETRBackend("rtdetr-l.pt")
                if not self.detector_backend.load(self.device, use_fp16):
                    logger.warning("RTDETR konnte nicht geladen werden, Fallback auf YOLO11.")
                    self.detector_backend = YOLODetectorBackend()
                    self.detector_backend.load(self.device, use_fp16)
            elif backend_type == "pose":
                self.detector_backend = YOLOPoseBackend("yolov8n-pose.pt")
                if not self.detector_backend.load(self.device, use_fp16):
                    logger.warning("YOLO-Pose konnte nicht geladen werden, Fallback auf YOLO11.")
                    self.detector_backend = YOLODetectorBackend()
                    self.detector_backend.load(self.device, use_fp16)
            else:
                self.detector_backend = YOLODetectorBackend()
                self.detector_backend.load(self.device, use_fp16)

            # Direct references for compatibility
            if isinstance(self.detector_backend, YOLODetectorBackend):
                self.model = self.detector_backend.model
                self.ball_model = self.detector_backend.ball_model
                self.model_name = self.detector_backend.player_model_name
            else:
                self.model = getattr(self.detector_backend, "model", None)
                self.ball_model = None
                self.model_name = getattr(self.detector_backend, "name", "custom")
        except Exception as e:
            logger.warning(f"Konnte AI-Backend nicht initialisieren: {e}")
            self.detector_backend = None
            self.model = None
            self.ball_model = None

    def set_backend(self, backend_name: str) -> bool:
        """Switches the active AI detection engine ('yolo11', 'rtdetr', 'pose')."""
        self.config.ai_backend = backend_name
        self._init_models()
        return self.detector_backend is not None and getattr(self.detector_backend, 'is_loaded', False)

    def reset(self):
        """Resets tracking state."""
        self.cam_x = 0.5
        self.cam_y = self.config.vertical_center_bias
        self.cam_vx = 0.0
        self.cam_zoom = self.config.fixed_zoom_factor if not self.config.enable_dynamic_zoom else self.config.current_zoom
        self.target_x = 0.5
        self.target_y = self.config.vertical_center_bias
        self.target_zoom = self.cam_zoom
        self.frame_counter = 0
        self.last_target_x = None
        self.last_target_y = None
        self.last_target_zoom = None
        self.kalman.reset(0.5, self.config.vertical_center_bias)
        self.ball_confidence = 0.0
        self.last_detected_ball_px = None
        self.last_player_centroid = 0.5
        self.bytetracker.reset()
        self.current_player_tracks.clear()
        self.static_leaf_suppressor.reset()
        self.sideline_classifier.reset()
        self.suppressed_spots.clear()
        self.sideline_staff_tracks.clear()
        self.prev_gray = None
        self.precomputed_trajectory.clear()

    def set_pitch_corners(self, corners: List[List[float]]):
        """Updates pitch polygon corners in both config and metric homography."""
        self.config.pitch_corners = corners
        if hasattr(self, 'tactics_engine') and self.tactics_engine is not None:
            self.tactics_engine.set_pitch_corners(corners)

    def get_tactics_summary(self) -> MatchTacticsSummary:
        """Returns comprehensive tactical match metrics and heatmaps."""
        return self.tactics_engine.get_summary()

    def get_player_speed(self, track_id: int) -> float:
        """Returns speed in km/h for given player track ID."""
        return self.tactics_engine.get_player_speed(track_id)

    def get_ball_speed(self) -> float:
        """Returns instantaneous ball speed in km/h."""
        return self.tactics_engine.get_ball_speed()

    def reset_tactics(self):
        """Resets accumulated tactical statistics."""
        if hasattr(self, 'tactics_engine') and self.tactics_engine is not None:
            self.tactics_engine.reset()

    def _compute_player_density_centroid(self, player_positions: List[Tuple[float, float]]) -> float:
        """
        Computes Kernel Density Estimation (KDE) weighted center of the players.
        Gives higher weight to players closely packed in the active zone and filters out isolated goalkeepers/outliers.
        """
        if not player_positions:
            return self.last_player_centroid

        xs = np.array([p[0] for p in player_positions], dtype=np.float64)
        if len(xs) == 1:
            return float(xs[0])

        if len(xs) <= 3:
            return float(np.median(xs))

        # Gaussian Kernel Density Estimation over player horizontal positions
        # Bandwidth sigma ~ 0.12 (approx. 1/8th of field width)
        sigma = 0.12
        weights = np.zeros(len(xs), dtype=np.float64)

        for i in range(len(xs)):
            diffs = xs - xs[i]
            # Sum of Gaussian proximity to all other players
            weights[i] = np.sum(np.exp(-0.5 * (diffs / sigma)**2))

        # Filter out extreme low-density outliers (isolated players far from play)
        max_w = np.max(weights)
        valid_mask = weights >= (max_w * 0.35)

        if np.any(valid_mask):
            filtered_xs = xs[valid_mask]
            filtered_ws = weights[valid_mask]
            density_centroid = np.sum(filtered_xs * filtered_ws) / np.sum(filtered_ws)
        else:
            density_centroid = np.median(xs)

        return float(density_centroid)

    def _filter_players_and_balls(self, 
                                  raw_players: List[DetectionBox], 
                                  raw_balls: List[DetectionBox], 
                                  corners: List[List[float]], 
                                  pano_frame_bgr: np.ndarray, 
                                  f_idx: int) -> Tuple[List[DetectionBox], List[DetectionBox]]:
        """
        Applies multi-stage filtering:
        1. Sideline & Coach Exclusion: Only players whose feet contact point is inside the pitch polygon.
        2. Anti-Jersey Filter: Rejects ball candidates located inside player jerseys/torso/shorts.
        3. Aspect Ratio & Relative Size: Rejects non-circular or oversized ball candidates.
        4. Static Pitch Artifact Suppressor: Rejects leaves, divots, and turf blemishes that do not move for > 20 frames.
        5. Color / Luminance Contrast: Rejects dark brown leaves or grass patches.
        """
        enable_sideline = getattr(self.config, 'enable_sideline_filter', True)
        margin = float(getattr(self.config, 'pitch_margin_pct', 1.0)) if enable_sideline else 0.0

        # 1. Sideline Filter for Players (Feet Ground Contact)
        if enable_sideline:
            valid_players = [
                p for p in raw_players 
                if is_player_on_pitch(p.bbox, corners, margin_pct=margin)
            ]
        else:
            valid_players = [
                p for p in raw_players 
                if is_point_in_pitch_polygon(p.center[0], p.center[1], corners)
            ]

        # 2. Pitch Boundary Filter for Balls
        valid_balls = [
            b for b in raw_balls 
            if is_point_in_pitch_polygon_with_margin(b.center[0], b.center[1], corners, margin_pct=0.0)
        ]

        enable_anti_jersey = getattr(self.config, 'enable_anti_jersey_filter', True)
        enable_static_leaf = getattr(self.config, 'enable_static_leaf_filter', True)

        # 3. Anti-Jersey Filter (Rejects white kit / clothing false positives)
        if enable_anti_jersey and valid_players:
            valid_balls = [
                b for b in valid_balls 
                if not BallPlausibilityFilter.is_inside_player_torso(b.center[0], b.center[1], valid_players)
            ]

        # 4. Aspect Ratio & Relative Size Filter
        p_heights = [p.bbox[3] - p.bbox[1] for p in valid_players if (p.bbox[3] - p.bbox[1]) > 0.01]
        med_ph = float(np.median(p_heights)) if p_heights else None
        valid_balls = [
            b for b in valid_balls 
            if BallPlausibilityFilter.validate_aspect_ratio_and_size(b.bbox[0], b.bbox[1], b.bbox[2], b.bbox[3], med_ph)
        ]

        # 5. Color / Luminance Contrast Filter for low-confidence detections
        valid_balls = [
            b for b in valid_balls 
            if (b.conf >= 0.40 or BallPlausibilityFilter.validate_patch_color(pano_frame_bgr, b.bbox, min_brightness=110))
        ]

        # 6. Static Pitch Artifact & Leaf Suppressor
        if enable_static_leaf:
            player_feet = [((p.bbox[0] + p.bbox[2]) * 0.5, p.bbox[3]) for p in valid_players]
            valid_balls, self.suppressed_spots = self.static_leaf_suppressor.update_and_filter(valid_balls, player_feet, f_idx)

        return valid_players, valid_balls

    def detect_action(self, pano_frame_bgr: np.ndarray, frame_index: Optional[int] = None) -> Tuple[float, float, float, bool, Optional[Tuple[int, int]]]:
        """
        Runs AI Action Detection on the stitched panorama.
        Calculates:
        1. Player Density Cluster Centroid (primary rock-solid anchor)
        2. Physics-Gated Soccer Ball Detection
        3. Smooth Target Camera Position
        Returns: (target_x, target_y, target_zoom, is_ball_found, (ball_px_x, ball_px_y))
        """
        h, w = pano_frame_bgr.shape[:2]
        f_idx = frame_index if frame_index is not None else self.frame_counter
        self.frame_counter += 1
        
        # 1. Kalman Ball Prediction
        pred_bx, pred_by = self.kalman.predict(dt=1.0)
        
        # 2. Downscale and crop to Pitch ROI Polygon bounding box for accelerated inference
        corners = getattr(self.config, 'pitch_corners', [[0.0, 0.05], [1.0, 0.05], [1.0, 0.95], [0.0, 0.95]])
        p_min_x, p_max_x, p_min_y, p_max_y = get_pitch_polygon_bbox(corners)

        infer_w = 1280
        infer_h = int(h * (infer_w / w))
        infer_img = cv2.resize(pano_frame_bgr, (infer_w, infer_h), interpolation=cv2.INTER_AREA)

        # Calculate pixel ROI within infer_img
        rx0 = max(0, int(round(p_min_x * infer_w)))
        rx1 = min(infer_w, int(round(p_max_x * infer_w)))
        ry0 = max(0, int(round(p_min_y * infer_h)))
        ry1 = min(infer_h, int(round(p_max_y * infer_h)))

        # Only crop if valid non-empty sub-box
        if (rx1 - rx0) >= 120 and (ry1 - ry0) >= 80:
            target_infer = infer_img[ry0:ry1, rx0:rx1]
            offset_x = rx0
            offset_y = ry0
        else:
            target_infer = infer_img
            offset_x = 0
            offset_y = 0

        player_positions: List[Tuple[float, float]] = []
        player_boxes: List[List[float]] = []
        player_scores: List[float] = []
        player_teams: List[int] = []
        ball_candidates: List[Tuple[float, float, float, int, int]] = [] # (norm_x, norm_y, score, px_x, px_y)

        # 3. AI Inference via Pluggable Backend (YOLO11, RT-DETR, or YOLO-Pose)
        use_fp16 = (self.device == "cuda" and getattr(self.config, "use_fp16", True))
        if self.detector_backend is not None and self.config.ai_tracking:
            try:
                raw_players, raw_balls = self.detector_backend.predict(
                    target_infer,
                    conf_thresh=0.20,
                    offset_x=offset_x,
                    offset_y=offset_y,
                    pano_w=infer_w,
                    pano_h=infer_h
                )
                valid_players, valid_balls = self._filter_players_and_balls(
                    raw_players, raw_balls, corners, pano_frame_bgr, f_idx
                )
                self.current_detections = valid_players

                for p in valid_players:
                    player_boxes.append(list(p.bbox))
                    player_scores.append(p.conf)
                    player_teams.append(p.team_id)
                    if p.cls_name != "goalkeeper":
                        player_positions.append(p.center)

                for b in valid_balls:
                    orig_px_x = int(round(b.center[0] * w))
                    orig_px_y = int(round(b.center[1] * h))
                    ball_candidates.append((b.center[0], b.center[1], b.conf, orig_px_x, orig_px_y))
            except Exception as e:
                logger.warning(f"Inferenz-Fehler im Detektor: {e}")
        elif self.model is not None and self.config.ai_tracking:
            try:
                with torch.inference_mode():
                    results = self.model.predict(
                        target_infer,
                        device=self.device,
                        half=use_fp16,
                        verbose=False,
                        imgsz=640,
                        conf=0.20
                    )
                if results and len(results) > 0:
                    boxes = results[0].boxes
                    names = self.model.model.names if hasattr(self.model, "model") and hasattr(self.model.model, "names") else {}
                    raw_p = []
                    raw_b = []
                    for box in boxes:
                        cls_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        cls_name = names.get(cls_id, "").lower()
                        xyxy = box.xyxy[0].cpu().numpy()
                        bx1 = max(0.0, min(1.0, float(xyxy[0] + offset_x) / infer_w))
                        by1 = max(0.0, min(1.0, float(xyxy[1] + offset_y) / infer_h))
                        bx2 = max(0.0, min(1.0, float(xyxy[2] + offset_x) / infer_w))
                        by2 = max(0.0, min(1.0, float(xyxy[3] + offset_y) / infer_h))

                        is_ball = (cls_name == "ball" or cls_id == 32)
                        is_player = (cls_name in ["player", "goalkeeper", "referee", "person"] or cls_id == 0)

                        if is_player:
                            raw_p.append(DetectionBox(
                                bbox=(bx1, by1, bx2, by2),
                                conf=conf,
                                cls_id=cls_id,
                                cls_name=cls_name,
                                team_id=0
                            ))
                        elif is_ball:
                            raw_b.append(DetectionBox(
                                bbox=(bx1, by1, bx2, by2),
                                conf=conf,
                                cls_id=cls_id,
                                cls_name="ball",
                                team_id=-1
                            ))

                    valid_players, valid_balls = self._filter_players_and_balls(
                        raw_p, raw_b, corners, pano_frame_bgr, f_idx
                    )
                    self.current_detections = valid_players

                    for p in valid_players:
                        player_boxes.append(list(p.bbox))
                        player_scores.append(p.conf)
                        player_teams.append(p.team_id)
                        if p.cls_name != "goalkeeper":
                            player_positions.append(p.center)

                    for b in valid_balls:
                        orig_px_x = int(round(b.center[0] * w))
                        orig_px_y = int(round(b.center[1] * h))
                        ball_candidates.append((b.center[0], b.center[1], b.conf, orig_px_x, orig_px_y))
            except Exception:
                pass

        # Multi-Object Tracking (ByteTrack) update with high & low confidence detections
        staff_map = {}
        if len(player_boxes) > 0:
            det_arr = np.array(player_boxes, dtype=np.float32)
            score_arr = np.array(player_scores, dtype=np.float32)
            team_arr = np.array(player_teams, dtype=np.int32)
            active_tracks = self.bytetracker.update(det_arr, score_arr, team_arr)
            
            # Identify stationary sideline staff (coaches/trainers)
            staff_map = self.sideline_classifier.update_and_classify(active_tracks, corners, f_idx)
            self.sideline_staff_tracks = [tid for tid, is_s in staff_map.items() if is_s]
            
            self.current_player_tracks = [
                (cx, cy, tid) for (tid, cx, cy, _, _, tid_team) in active_tracks
                if not staff_map.get(tid, False)
            ]
        else:
            active_tracks = self.bytetracker.update(np.empty((0, 4), dtype=np.float32), np.empty(0, dtype=np.float32), np.empty(0, dtype=np.int32))
            staff_map = self.sideline_classifier.update_and_classify(active_tracks, corners, f_idx)
            if active_tracks:
                self.current_player_tracks = [
                    (cx, cy, tid) for (tid, cx, cy, _, _, tid_team) in active_tracks
                    if not staff_map.get(tid, False)
                ]
            elif player_positions:
                self.current_player_tracks = [(px, py, 0) for px, py in player_positions]
            else:
                self.current_player_tracks = []

        # 4. Fallback Player Cluster via Motion / Pitch Mask if no AI players found
        if len(player_positions) == 0:
            hsv = cv2.cvtColor(infer_img, cv2.COLOR_BGR2HSV)
            lower_green = np.array([22, 18, 18])
            upper_green = np.array([92, 255, 255])
            pitch_mask = cv2.inRange(hsv, lower_green, upper_green)
            
            # Mask out non-pitch regions with exact polygon
            poly_pts_px = np.array([
                [int(round(c[0] * infer_w)), int(round(c[1] * infer_h))] for c in corners
            ], dtype=np.int32)
            poly_mask = np.zeros((infer_h, infer_w), dtype=np.uint8)
            cv2.fillPoly(poly_mask, [poly_pts_px], 255)
            pitch_mask = cv2.bitwise_and(pitch_mask, poly_mask)

            gray = cv2.cvtColor(infer_img, cv2.COLOR_BGR2GRAY)
            gray_blur = cv2.GaussianBlur(gray, (11, 11), 0)
            if self.prev_gray is not None and self.prev_gray.shape == gray_blur.shape:
                diff = cv2.absdiff(gray_blur, self.prev_gray)
                _, mmask = cv2.threshold(diff, 16, 255, cv2.THRESH_BINARY)
                mmask = cv2.bitwise_and(mmask, pitch_mask)
                moms = cv2.moments(mmask)
                if moms["m00"] > 300:
                    motion_cx = (moms["m10"] / moms["m00"]) / infer_w
                    motion_cy = (moms["m01"] / moms["m00"]) / infer_h
                    if is_point_in_pitch_polygon(motion_cx, motion_cy, corners):
                        player_positions.append((motion_cx, motion_cy))
            self.prev_gray = gray_blur

        # 5. Compute Rock-Solid Player Density Centroid
        player_cx = self._compute_player_density_centroid(player_positions)
        self.last_player_centroid = player_cx

        # 6. Physical Ball Verification & Plausibility Filter
        ball_found = False
        ball_px = None
        verified_ball: Optional[Tuple[float, float, float, int, int]] = None

        # Anti-Sticky Kalman Unlocking:
        # If Kalman was stationary (< 0.002) and play moved away (> 0.12), do not cling to the old spot
        if self.kalman.speed < 0.002 and abs(player_cx - self.kalman.pos[0]) > 0.12:
            self.ball_confidence = max(0.0, self.ball_confidence - 0.15)
            if self.ball_confidence <= 0.10:
                self.kalman.reset(player_cx, self.config.vertical_center_bias)
                pred_bx, pred_by = player_cx, self.config.vertical_center_bias

        if ball_candidates:
            # Score each candidate: Confidence + proximity to prediction + proximity to player cluster
            valid_candidates = []
            for cand in ball_candidates:
                bx, by, bconf, bpx_x, bpx_y = cand
                dist_to_players = abs(bx - player_cx)
                dist_to_kalman = math.hypot(bx - pred_bx, by - pred_by)

                # Plausibility check: Must be near play OR have physical trajectory continuity
                is_near_play = (dist_to_players < 0.35)
                is_kalman_plausible = self.kalman.is_plausible_measurement(bx, by)

                if is_near_play or is_kalman_plausible:
                    motion_dist = math.hypot(bx - self.kalman.pos[0], by - self.kalman.pos[1])
                    motion_bonus = 0.25 if motion_dist > 0.005 else -0.20 # penalize complete immobility
                    score = bconf * 1.6 - (dist_to_kalman * 1.5) - (dist_to_players * 0.4) + motion_bonus
                    valid_candidates.append((score, cand))

            if valid_candidates:
                valid_candidates.sort(key=lambda item: item[0], reverse=True)
                verified_ball = valid_candidates[0][1]

        if verified_ball is not None:
            bx, by, bconf, bpx_x, bpx_y = verified_ball
            self.kalman.update(bx, by, confidence=bconf)
            self.ball_confidence = min(1.0, self.ball_confidence + 0.35)
            self.last_detected_ball_px = (bpx_x, bpx_y)
            ball_px = (bpx_x, bpx_y)
            ball_found = True
        else:
            # Decay ball confidence smoothly
            self.ball_confidence = max(0.0, self.ball_confidence - 0.05)
            if self.kalman.missed_frames < 20 and self.ball_confidence > 0.15:
                ball_x, ball_y = self.kalman.pos
                ball_px = (int(round(ball_x * w)), int(round(ball_y * h)))

        ball_x, ball_y = self.kalman.pos
        ball_vx, _ = self.kalman.vel

        # 6.5. Compute VAR Virtual Offside Line
        if getattr(self.config, 'enable_offside_line', True) and len(self.current_detections) >= 4:
            b_norm = (ball_x, ball_y) if self.ball_confidence > 0.15 else None
            self._current_offside_line = OffsideAnalyzer.compute_offside_line(self.current_detections, ball_pos=b_norm)
        else:
            self._current_offside_line = None

        # 7. Action Center Strategy Fusion
        mode = getattr(self.config, "tracking_mode", "hybrid_fusion")
        cfg_bw = float(getattr(self.config, "ball_weight", 0.70))
        cfg_pw = float(getattr(self.config, "players_weight", 0.30))

        # Normalize weights if both are set
        total_w = cfg_bw + cfg_pw
        if total_w > 1e-4:
            norm_bw = cfg_bw / total_w
            norm_pw = cfg_pw / total_w
        else:
            norm_bw, norm_pw = 0.70, 0.30

        # Confidence blending factor: ramp up between 0.10 and 0.30 confidence
        conf_scale = float(np.clip((self.ball_confidence - 0.10) / 0.20, 0.0, 1.0))

        if mode == "player_density":
            # 100% stable player density center
            raw_target_x = player_cx
        elif mode == "ball_centric" and self.ball_confidence > 0.15:
            # Ball-centric tracking with anticipation
            b_ratio = max(norm_bw, 0.85) * conf_scale
            p_ratio = 1.0 - b_ratio
            lead = np.clip(ball_vx * self.config.anticipation_lead, -0.06, 0.06) * conf_scale
            raw_target_x = ball_x * b_ratio + player_cx * p_ratio + lead
        elif mode == "smooth_tactic":
            # Broad tactical overview (gentle ball presence)
            b_ratio = min(norm_bw, 0.30) * conf_scale
            p_ratio = 1.0 - b_ratio
            raw_target_x = player_cx * p_ratio + (ball_x if conf_scale > 0.2 else player_cx) * b_ratio
        else: # "hybrid_fusion" (Recommended Default: Target_x = Ball_x * w_ball + Player_x * w_player)
            if self.ball_confidence > 0.15:
                # Effective weighted blend scaled with detection confidence
                eff_bw = norm_bw * conf_scale
                eff_pw = 1.0 - eff_bw
                lead = np.clip(ball_vx * (self.config.anticipation_lead * 0.8), -0.05, 0.05) * conf_scale
                raw_target_x = ball_x * eff_bw + player_cx * eff_pw + lead
            else:
                # Seamless fallback to player cluster when ball is obscured / untracked
                raw_target_x = player_cx

        # 8. Pure Horizontal Tracking (Fixed Zoom & Fixed Vertical Height)
        if not getattr(self.config, "enable_dynamic_zoom", False):
            desired_zoom = getattr(self.config, "fixed_zoom_factor", 1.0)
        else:
            desired_zoom = self.config.min_zoom if self.kalman.speed > 0.012 else self.config.max_zoom

        # Target filtering
        self.target_x = self.target_x * 0.50 + raw_target_x * 0.50
        self.target_y = self.config.vertical_center_bias
        self.target_zoom = desired_zoom

        # 9. Update Real-Time Tactical Metrics (SkillCorner & TacticAI Engine)
        if hasattr(self, 'tactics_engine') and self.tactics_engine is not None:
            try:
                players_tactics = [(cx, cy, tid, trk_id) for (trk_id, cx, cy, _, _, tid) in active_tracks]
                b_norm = (verified_ball[0], verified_ball[1]) if (ball_found and verified_ball is not None) else None
                self.tactics_engine.update_frame(players_tactics, b_norm, f_idx, fps=30.0)

                # Match velocities and track IDs to current_detections
                for det in self.current_detections:
                    best_d = 0.08
                    best_id = None
                    for (trk_id, cx, cy, _, _, tid) in active_tracks:
                        d = math.hypot(det.center[0] - cx, det.center[1] - cy)
                        if d < best_d:
                            best_d = d
                            best_id = trk_id
                    if best_id is not None:
                        det.track_id = best_id
                        det.speed_kmh = round(self.tactics_engine.get_player_speed(best_id), 1)
            except Exception as e:
                logger.warning(f"Fehler bei Taktik-Metrik-Berechnung: {e}")

        return self.target_x, self.target_y, self.target_zoom, ball_found, ball_px

    def update_camera(self, target_x: float, target_y: float, target_zoom: float) -> Tuple[float, float, float]:
        """
        Critically damped spring-damper camera smoothing filter with deadband for broadcast panning.
        Guarantees smooth left-right movement with zero jitter.
        """
        diff_x = target_x - self.cam_x

        # Deadband: if target is within deadband zone of frame, camera remains still
        deadband = self.config.deadband_width
        if abs(diff_x) < deadband:
            diff_x = 0.0
        else:
            diff_x = diff_x - np.sign(diff_x) * deadband

        # Base responsiveness / spring damping
        base_responsiveness = 1.0 - np.clip(self.config.smoothing_factor, 0.70, 0.99)
        max_speed = max(0.005, self.config.max_pan_speed)

        # Dynamic Zone Damping (Strafraum-Dynamik: schnellere Reaktion in Tornähe)
        if getattr(self.config, "enable_zone_damping", True):
            boost = max(1.0, float(getattr(self.config, "zone_boost_factor", 1.80)))
            # Relative Distanz zur Spielfeld-Mitte (0.5)
            dist_from_center = abs(target_x - 0.5)
            # Ramping: 0.0 im Mittelfeld (dist <= 0.15) bis 1.0 im Strafraum (dist >= 0.38)
            zone_proximity = float(np.clip((dist_from_center - 0.15) / 0.23, 0.0, 1.0))
            zone_multiplier = 1.0 + (boost - 1.0) * (zone_proximity ** 1.5)
            responsiveness = float(np.clip(base_responsiveness * zone_multiplier, 0.005, 0.40))
            max_speed = float(np.clip(max_speed * (1.0 + (boost - 1.0) * 0.45 * zone_proximity), 0.005, 0.16))
        else:
            responsiveness = base_responsiveness

        desired_vx = diff_x * responsiveness
        
        # Strict maximum pan speed clamp
        desired_vx = np.clip(desired_vx, -max_speed, max_speed)

        # Smooth velocity integration
        self.cam_vx = self.cam_vx * 0.75 + desired_vx * 0.25
        self.cam_x += self.cam_vx

        # Zoom handling (Fixed 1.0x by default)
        if not getattr(self.config, "enable_dynamic_zoom", False):
            self.cam_zoom = getattr(self.config, "fixed_zoom_factor", 1.0)
        else:
            self.cam_zoom += (target_zoom - self.cam_zoom) * self.config.zoom_speed
            self.cam_zoom = float(np.clip(self.cam_zoom, self.config.min_zoom, self.config.max_zoom))

        # Clamp horizontal camera position so 16:9 crop never shows black borders
        crop_w_ratio, _ = self.get_crop_aspect_ratios()
        min_x = crop_w_ratio * 0.5
        max_x = 1.0 - (crop_w_ratio * 0.5)

        self.cam_x = float(np.clip(self.cam_x, min_x, max_x))
        self.cam_y = self.config.vertical_center_bias

        return self.cam_x, self.cam_y, self.cam_zoom

    @property
    def active_players(self) -> List[Tuple[float, float, int]]:
        """List of active tracked players as (norm_x, norm_y, team_id)."""
        return self.current_player_tracks

    @property
    def current_ball_norm(self) -> Optional[Tuple[float, float]]:
        """Current normalized ball coordinates (norm_x, norm_y) if tracked, else None."""
        if self.ball_confidence > 0.15:
            return (self.kalman.pos[0], self.kalman.pos[1])
        return None

    @property
    def active_detections(self) -> List[DetectionBox]:
        """List of active detection boxes (including pose keypoints and orientation)."""
        return self.current_detections

    @property
    def current_offside_line(self) -> Optional[Tuple[float, int, str]]:
        """VAR virtual offside line: (norm_x, defending_team, label)."""
        return self._current_offside_line

    def get_camera_span(self, pano_aspect: float = 32.0 / 9.0) -> Tuple[float, float]:
        """Returns normalized (x_start, x_end) coverage of current virtual camera FOV."""
        zoom = max(1.0, self.cam_zoom)
        box_w_norm = min(1.0, (1.0 / pano_aspect) * (16.0 / 9.0) / zoom)
        span_left = max(0.0, self.cam_x - box_w_norm * 0.5)
        span_right = min(1.0, self.cam_x + box_w_norm * 0.5)
        return (span_left, span_right)

    def get_crop_aspect_ratios(self) -> Tuple[float, float]:
        """
        Returns normalized (crop_width, crop_height) inside 32:9 panorama.
        At fixed 1.0x zoom: crop_w = 0.5 (50% of 32:9 width), crop_h = 1.0 (100% of height).
        """
        zoom = max(1.0, self.cam_zoom)
        crop_w = (0.5 / zoom)
        crop_h = (1.0 / zoom)
        return crop_w, crop_h

    def get_crop_rect(self, pano_width: int, pano_height: int) -> Tuple[int, int, int, int]:
        """
        Computes pixel bounding box (x, y, w, h) for 16:9 crop inside 32:9 panorama.
        """
        zoom = max(1.0, self.cam_zoom)
        
        # 16:9 crop dimensions
        box_h = int(round(pano_height / zoom))
        box_w = int(round(pano_height * (16.0 / 9.0) / zoom))

        box_w = min(pano_width, box_w)
        box_h = min(pano_height, box_h)

        center_px_x = int(round(self.cam_x * pano_width))
        center_px_y = int(round(self.cam_y * pano_height))

        x1 = center_px_x - (box_w // 2)
        y1 = center_px_y - (box_h // 2)

        x1 = max(0, min(pano_width - box_w, x1))
        y1 = max(0, min(pano_height - box_h, y1))

        return x1, y1, box_w, box_h

    def get_crop_rect_for_frame(self, frame_index: int, pano_width: int, pano_height: int) -> Tuple[int, int, int, int]:
        """
        Computes pixel bounding box (x, y, w, h) for a specific frame index
        using precomputed lookahead trajectory or live camera state.
        """
        if frame_index in self.precomputed_trajectory:
            cx, cy, zoom = self.precomputed_trajectory[frame_index]
        else:
            cx, cy, zoom = self.cam_x, self.cam_y, self.cam_zoom

        zoom = max(1.0, zoom)
        box_h = int(round(pano_height / zoom))
        box_w = int(round(pano_height * (16.0 / 9.0) / zoom))

        box_w = min(pano_width, box_w)
        box_h = min(pano_height, box_h)

        center_px_x = int(round(cx * pano_width))
        center_px_y = int(round(cy * pano_height))

        x1 = center_px_x - (box_w // 2)
        y1 = center_px_y - (box_h // 2)

        x1 = max(0, min(pano_width - box_w, x1))
        y1 = max(0, min(pano_height - box_h, y1))

        return x1, y1, box_w, box_h

    def extract_16x9_frame_with_trajectory(self, pano_frame_bgr: np.ndarray, frame_index: int, out_width: int = 1920, out_height: int = 1080) -> np.ndarray:
        """Extracts 16:9 frame using precomputed lookahead trajectory."""
        ph, pw = pano_frame_bgr.shape[:2]
        x, y, w, h = self.get_crop_rect_for_frame(frame_index, pw, ph)
        cropped = pano_frame_bgr[y:y+h, x:x+w]

        if (w, h) != (out_width, out_height):
            return cv2.resize(cropped, (out_width, out_height), interpolation=cv2.INTER_LINEAR)
        return cropped

    def extract_16x9_frame(self, pano_frame_bgr: np.ndarray, out_width: int = 1920, out_height: int = 1080, frame_index: Optional[int] = None) -> np.ndarray:
        """Extracts 16:9 smooth broadcast frame with pure horizontal tracking."""
        if frame_index is not None and frame_index in self.precomputed_trajectory:
            return self.extract_16x9_frame_with_trajectory(pano_frame_bgr, frame_index, out_width, out_height)

        ph, pw = pano_frame_bgr.shape[:2]
        self.frame_counter += 1

        # Run AI detection on keyframes
        if self.frame_counter % self.inference_interval == 0 or self.last_target_x is None:
            tx, ty, tz, _, _ = self.detect_action(pano_frame_bgr)
            self.last_target_x = tx
            self.last_target_y = ty
            self.last_target_zoom = tz

        # Update smooth camera trajectory on every frame
        self.update_camera(self.last_target_x, self.last_target_y, self.last_target_zoom)

        x, y, w, h = self.get_crop_rect(pw, ph)
        cropped = pano_frame_bgr[y:y+h, x:x+w]

        if (w, h) != (out_width, out_height):
            return cv2.resize(cropped, (out_width, out_height), interpolation=cv2.INTER_LINEAR)
        return cropped

    def generate_offline_trajectory(self, 
                                     video_source: Any, 
                                     start_frame: int = 0, 
                                     end_frame: Optional[int] = None, 
                                     step: Optional[int] = None, 
                                     lookahead_window_sec: float = 1.8, 
                                     stop_event: Optional[Any] = None,
                                     progress_callback: Optional[Callable[[int, int, float], None]] = None) -> Dict[int, Tuple[float, float, float]]:
        """
        Computes a TV-broadcast lookahead camera trajectory over a 32:9 video source.
        1. Scans action targets across keyframes with high-speed configurable step sampling.
        2. Applies bidirectional Gaussian lookahead smoothing to eliminate camera lag, overshoots, and micro-jitter.
        3. Enforces horizontal boundary limits.
        """
        filepath = video_source if isinstance(video_source, str) else getattr(video_source, "filepath", None)
        if not filepath or not os.path.exists(filepath):
            return {}

        cap = cv2.VideoCapture(filepath, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            cap = cv2.VideoCapture(filepath)
        if not cap.isOpened():
            return {}

        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if end_frame is None or end_frame > total_frames:
            end_frame = total_frames

        start_frame = max(0, start_frame)
        frames_to_scan = max(1, end_frame - start_frame)

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        self.reset()

        scan_step = step if step is not None else getattr(self.config, 'scan_step', 5)
        scan_step = max(1, min(30, int(scan_step)))

        sampled_f: List[int] = []
        raw_x: List[float] = []
        raw_y: List[float] = []
        raw_zoom: List[float] = []

        t_start = time.time()
        curr_f = start_frame

        while curr_f < end_frame:
            if stop_event and stop_event.is_set():
                cap.release()
                return {}

            if (curr_f - start_frame) % scan_step == 0:
                ret, frame = cap.read()
                if not ret:
                    break
                tx, ty, tz, _, _ = self.detect_action(frame)
                sampled_f.append(curr_f)
                raw_x.append(tx)
                raw_y.append(ty)
                raw_zoom.append(tz)
            else:
                ret = cap.grab()
                if not ret:
                    break

            curr_f += 1

            if progress_callback and ((curr_f - start_frame) % 60 == 0 or curr_f == end_frame):
                elapsed = time.time() - t_start
                cur_fps = (curr_f - start_frame) / max(elapsed, 0.001)
                progress_callback(curr_f - start_frame, frames_to_scan, cur_fps)

        cap.release()

        if len(sampled_f) == 0:
            return {}

        all_frames = np.arange(start_frame, curr_f)
        if len(sampled_f) == 1:
            interp_x = np.full(len(all_frames), raw_x[0])
            interp_y = np.full(len(all_frames), raw_y[0])
            interp_z = np.full(len(all_frames), raw_zoom[0])
        else:
            interp_x = np.interp(all_frames, sampled_f, raw_x)
            interp_y = np.interp(all_frames, sampled_f, raw_y)
            interp_z = np.interp(all_frames, sampled_f, raw_zoom)

        # Bidirectional Gaussian Lookahead Smoothing
        win_len = int(round(fps * lookahead_window_sec))
        win_len = max(5, win_len)
        if win_len % 2 == 0:
            win_len += 1
        
        sigma = win_len / 3.5
        k = np.exp(-0.5 * (np.arange(-(win_len // 2), win_len // 2 + 1) / sigma)**2)
        k /= k.sum()

        pad_len = win_len // 2
        padded_x = np.pad(interp_x, pad_len, mode='edge')
        smooth_x = np.convolve(padded_x, k, mode='valid')

        padded_y = np.pad(interp_y, pad_len, mode='edge')
        smooth_y = np.convolve(padded_y, k, mode='valid')

        padded_z = np.pad(interp_z, pad_len, mode='edge')
        smooth_z = np.convolve(padded_z, k, mode='valid')

        if not getattr(self.config, "enable_dynamic_zoom", False):
            smooth_z = np.full_like(smooth_z, getattr(self.config, "fixed_zoom_factor", 1.0))
        else:
            smooth_z = np.clip(smooth_z, self.config.min_zoom, self.config.max_zoom)

        corners = getattr(self.config, 'pitch_corners', [[0.0, 0.05], [1.0, 0.05], [1.0, 0.95], [0.0, 0.95]])
        p_min_x, p_max_x, _, _ = get_pitch_polygon_bbox(corners)

        self.precomputed_trajectory.clear()
        for idx, f_num in enumerate(all_frames):
            zoom = float(smooth_z[idx])
            crop_w_ratio = (0.5 / max(1.0, zoom))
            min_x = max(p_min_x + crop_w_ratio * 0.5, crop_w_ratio * 0.5)
            max_x = min(p_max_x - crop_w_ratio * 0.5, 1.0 - (crop_w_ratio * 0.5))
            if min_x > max_x:
                min_x = crop_w_ratio * 0.5
                max_x = 1.0 - (crop_w_ratio * 0.5)

            cam_x = float(np.clip(smooth_x[idx], min_x, max_x))
            cam_y = float(self.config.vertical_center_bias)
            self.precomputed_trajectory[int(f_num)] = (cam_x, cam_y, zoom)

        return self.precomputed_trajectory


