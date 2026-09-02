"""
AI-Powered Soccer Broadcast Camera Engine (YOLO11 / YOLOv8 + Player Density Clustering + Physics-Gated Ball Tracking).
Delivers broadcast-quality pure horizontal camera panning (Left-Right Follow Cam, No Zoom) with rock-solid stability.
"""
from dataclasses import dataclass, field
import os
import time
import numpy as np
import cv2
from typing import Tuple, Optional, List, Dict, Any, Callable
from .paths import get_resource_path

try:
    import torch
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
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
        
        # Player Cluster State
        self.last_player_centroid: float = 0.5
        
        # Motion history for CV fallback
        self.prev_gray: Optional[np.ndarray] = None
        
        # Precomputed offline lookahead trajectory cache {frame_index: (cam_x, cam_y, cam_zoom)}
        self.precomputed_trajectory: Dict[int, Tuple[float, float, float]] = {}
        
        # AI Models
        self.model: Optional[Any] = None
        self.ball_model: Optional[Any] = None
        self.model_name: str = "none"
        self.device = "cuda" if (HAS_YOLO and torch.cuda.is_available()) else "cpu"
        self._init_models()

    def _init_models(self):
        """Initializes specialized soccer detection models or general YOLO models."""
        if not HAS_YOLO:
            return

        # 1. Try to load specialized soccer models first
        soccer_player_path = get_resource_path("yolo11m_football_player.pt")
        soccer_ball_path = get_resource_path("yolo11n_football_ball.pt")
        yolo11n_path = get_resource_path("yolo11n.pt")
        yolov8n_path = get_resource_path("yolov8n.pt")

        try:
            if os.path.exists(soccer_player_path):
                self.model = YOLO(soccer_player_path)
                self.model_name = "yolo11m_football_player"
            elif os.path.exists(yolo11n_path):
                self.model = YOLO(yolo11n_path)
                self.model_name = "yolo11n"
            elif os.path.exists(yolov8n_path):
                self.model = YOLO(yolov8n_path)
                self.model_name = "yolov8n"

            # Optional dedicated ball model
            if os.path.exists(soccer_ball_path):
                self.ball_model = YOLO(soccer_ball_path)

            # Warmup inference
            if self.model is not None:
                dummy = np.zeros((320, 640, 3), dtype=np.uint8)
                with torch.inference_mode():
                    self.model.predict(dummy, device=self.device, verbose=False, imgsz=320)
                    if self.ball_model is not None:
                        self.ball_model.predict(dummy, device=self.device, verbose=False, imgsz=320)
        except Exception as e:
            print(f"Warning: Could not initialize AI tracker on {self.device}: {e}")
            self.model = None
            self.ball_model = None

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
        self.prev_gray = None
        self.precomputed_trajectory.clear()

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

    def detect_action(self, pano_frame_bgr: np.ndarray) -> Tuple[float, float, float, bool, Optional[Tuple[int, int]]]:
        """
        Runs AI Action Detection on the stitched panorama.
        Calculates:
        1. Player Density Cluster Centroid (primary rock-solid anchor)
        2. Physics-Gated Soccer Ball Detection
        3. Smooth Target Camera Position
        Returns: (target_x, target_y, target_zoom, is_ball_found, (ball_px_x, ball_px_y))
        """
        h, w = pano_frame_bgr.shape[:2]
        
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
        ball_candidates: List[Tuple[float, float, float, int, int]] = [] # (norm_x, norm_y, score, px_x, px_y)

        # 3. AI Inference (Players & Ball)
        use_fp16 = (self.device == "cuda" and getattr(self.config, "use_fp16", True))
        if self.model is not None and self.config.ai_tracking:
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
                    for box in boxes:
                        cls_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        cls_name = names.get(cls_id, "").lower()
                        xyxy = box.xyxy[0].cpu().numpy()
                        cx = float((xyxy[0] + xyxy[2]) * 0.5) + offset_x
                        cy = float((xyxy[1] + xyxy[3]) * 0.5) + offset_y

                        norm_x = cx / infer_w
                        norm_y = cy / infer_h

                        # Discard any detections outside the arbitrary 4-corner pitch polygon
                        if not is_point_in_pitch_polygon(norm_x, norm_y, corners):
                            continue

                        # Check if class is player or ball
                        is_ball = (cls_name == "ball" or cls_id == 32 or (cls_id == 0 and self.model_name == "yolo11n_football_ball"))
                        is_player = (cls_name in ["player", "goalkeeper", "referee", "person"] or (cls_id == 0 and self.model_name in ["yolo11n", "yolov8n"]))

                        if is_player:
                            if cls_name != "goalkeeper":
                                player_positions.append((norm_x, norm_y))
                        elif is_ball:
                            orig_px_x = int(round(cx * (w / infer_w)))
                            orig_px_y = int(round(cy * (h / infer_h)))
                            ball_candidates.append((norm_x, norm_y, conf, orig_px_x, orig_px_y))
            except Exception:
                pass

        # Optional: Run dedicated ball model if available
        if self.ball_model is not None and self.config.ai_tracking and len(ball_candidates) == 0:
            try:
                with torch.inference_mode():
                    b_results = self.ball_model.predict(
                        target_infer,
                        device=self.device,
                        half=use_fp16,
                        verbose=False,
                        imgsz=640,
                        conf=0.15
                    )
                if b_results and len(b_results) > 0:
                    for box in b_results[0].boxes:
                        conf = float(box.conf[0])
                        xyxy = box.xyxy[0].cpu().numpy()
                        cx = float((xyxy[0] + xyxy[2]) * 0.5) + offset_x
                        cy = float((xyxy[1] + xyxy[3]) * 0.5) + offset_y
                        norm_x = cx / infer_w
                        norm_y = cy / infer_h
                        if is_point_in_pitch_polygon(norm_x, norm_y, corners):
                            orig_px_x = int(round(cx * (w / infer_w)))
                            orig_px_y = int(round(cy * (h / infer_h)))
                            ball_candidates.append((norm_x, norm_y, conf, orig_px_x, orig_px_y))
            except Exception:
                pass

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

        if ball_candidates:
            # Score each candidate: Confidence + proximity to prediction + proximity to player cluster
            valid_candidates = []
            for cand in ball_candidates:
                bx, by, bconf, bpx_x, bpx_y = cand
                dist_to_players = abs(bx - player_cx)
                dist_to_kalman = np.sqrt((bx - pred_bx)**2 + (by - pred_by)**2)

                # Plausibility check: Must be near play OR have physical trajectory continuity
                is_near_play = (dist_to_players < 0.28)
                is_kalman_plausible = self.kalman.is_plausible_measurement(bx, by)

                if is_near_play or is_kalman_plausible:
                    score = bconf * 1.5 - (dist_to_kalman * 1.2) - (dist_to_players * 0.5)
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

        # 7. Action Center Strategy Fusion
        mode = getattr(self.config, "tracking_mode", "hybrid_fusion")

        if mode == "player_density":
            # 100% stable player density center
            raw_target_x = player_cx
        elif mode == "ball_centric" and self.ball_confidence > 0.20:
            # Ball-centric tracking with gentle anticipation
            lead = np.clip(ball_vx * self.config.anticipation_lead, -0.06, 0.06)
            raw_target_x = ball_x * 0.85 + player_cx * 0.15 + lead
        elif mode == "smooth_tactic":
            # Broad tactical overview
            raw_target_x = player_cx * 0.70 + (ball_x if self.ball_confidence > 0.2 else player_cx) * 0.30
        else: # "hybrid_fusion" (Recommended Default)
            if self.ball_confidence > 0.25:
                # 60% Ball + 40% Player Cluster + gentle lead (eliminates sudden ball jumps)
                lead = np.clip(ball_vx * (self.config.anticipation_lead * 0.8), -0.05, 0.05)
                raw_target_x = ball_x * 0.60 + player_cx * 0.40 + lead
            else:
                # When ball is not clearly tracked, seamlessly stay centered on the action cluster
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

        # Responsiveness / spring damping
        responsiveness = 1.0 - np.clip(self.config.smoothing_factor, 0.70, 0.99)
        desired_vx = diff_x * responsiveness
        
        # Strict maximum pan speed clamp
        max_speed = max(0.005, self.config.max_pan_speed)
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


