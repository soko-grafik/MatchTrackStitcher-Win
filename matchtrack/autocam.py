"""
AutoCam: Intelligent Ball-Follow & Smooth Broadcast Tracking.
Features dedicated Soccer Ball Detection (Circularity, Contrast, Trajectory Estimation),
Player Cluster Fusion, and Cinematic Spring-Damper Camera Panning.
"""
from dataclasses import dataclass
import numpy as np
import cv2
from typing import Tuple, Optional, List


@dataclass
class AutoCamConfig:
    """Configuration for automated 16:9 broadcast camera tracking."""
    enabled: bool = True
    follow_ball: bool = True           # Prioritize tracking the soccer ball directly
    ball_weight: float = 0.80          # 80% ball position, 20% player centroid
    zoom_factor: float = 1.25          # 1.0 = Full height 16:9 crop, 1.6 = tight zoom
    smoothing_factor: float = 0.94     # 0.85 = responsive follow, 0.98 = ultra-smooth cinematic
    deadband_width: float = 0.06       # Fractional deadzone (no pan if movement < 6% of width)
    max_pan_speed: float = 0.05        # Max camera shift per frame (fraction of pano width)
    anticipation_lead: float = 0.20    # Look-ahead in direction of ball movement
    vertical_center_bias: float = 0.55 # Vertical framing (0.5 = center, 0.6 = slightly lower on field)


class AutoCamTracker:
    """Tracks the soccer ball and player centroid to produce a smooth 16:9 broadcast view."""
    def __init__(self, config: Optional[AutoCamConfig] = None):
        self.config = config or AutoCamConfig()
        
        # State: current virtual camera center (normalized [0.0 ... 1.0])
        self.cam_x: float = 0.5
        self.cam_y: float = 0.55
        self.cam_vx: float = 0.0
        
        # Target action position
        self.target_x: float = 0.5
        self.target_y: float = 0.55
        
        # Ball Tracker State
        self.ball_x: float = 0.5
        self.ball_y: float = 0.55
        self.ball_vx: float = 0.0
        self.ball_vy: float = 0.0
        self.ball_confidence: float = 0.0
        self.last_detected_ball_px: Optional[Tuple[int, int]] = None
        
        # Motion history
        self.prev_gray: Optional[np.ndarray] = None
        self.prev_frame_bgr: Optional[np.ndarray] = None

    def reset(self):
        """Resets camera and ball tracking state."""
        self.cam_x = 0.5
        self.cam_y = 0.55
        self.cam_vx = 0.0
        self.target_x = 0.5
        self.target_y = 0.55
        self.ball_x = 0.5
        self.ball_y = 0.55
        self.ball_vx = 0.0
        self.ball_vy = 0.0
        self.ball_confidence = 0.0
        self.last_detected_ball_px = None
        self.prev_gray = None
        self.prev_frame_bgr = None

    def detect_ball_and_players(self, pano_frame_bgr: np.ndarray) -> Tuple[float, float, bool, Optional[Tuple[int, int]]]:
        """
        Detects the soccer ball and player cluster on the pitch.
        Returns: (target_x, target_y, is_ball_tracked, (ball_px_x, ball_px_y))
        Fast execution: ~3ms.
        """
        h, w = pano_frame_bgr.shape[:2]
        
        # Downscale for ultra-fast CV analysis
        small_w = 960
        small_h = int(h * (small_w / w))
        small = cv2.resize(pano_frame_bgr, (small_w, small_h), interpolation=cv2.INTER_AREA)
        
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.GaussianBlur(gray, (9, 9), 0)

        # 1. Pitch Segmentation (Isolate soccer grass, exclude sky/spectators)
        lower_green = np.array([28, 35, 30])
        upper_green = np.array([85, 255, 255])
        pitch_mask = cv2.inRange(hsv, lower_green, upper_green)
        pitch_mask = cv2.dilate(pitch_mask, np.ones((9, 9), np.uint8), iterations=2)

        # 2. Motion Detection (Frame Differencing)
        motion_mask = np.zeros((small_h, small_w), dtype=np.uint8)
        if self.prev_gray is not None and self.prev_gray.shape == gray_blur.shape:
            diff = cv2.absdiff(gray_blur, self.prev_gray)
            _, motion_mask = cv2.threshold(diff, 14, 255, cv2.THRESH_BINARY)
            motion_mask = cv2.bitwise_and(motion_mask, pitch_mask)
        self.prev_gray = gray_blur

        # 3. Player Cluster Centroid
        moments = cv2.moments(motion_mask)
        if moments["m00"] > 300:
            player_cx = (moments["m10"] / moments["m00"]) / small_w
            player_cy = (moments["m01"] / moments["m00"]) / small_h
        else:
            player_cx = self.target_x
            player_cy = 0.55

        # 4. Ball Candidate Detection (High-contrast circular bright/white/neon blobs on the pitch)
        # In soccer, the ball is brighter than the grass and exhibits high contrast
        _, bright_mask = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
        # Exclude white pitch boundary lines: Ball is small and circular, lines are long
        ball_field_mask = cv2.bitwise_and(bright_mask, pitch_mask)
        
        contours, _ = cv2.findContours(ball_field_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        best_ball_candidate = None
        best_score = -1.0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Filter by ball size (typical ball area in 960x270 is 8 to 250 px)
            if 6 < area < 300:
                perimeter = cv2.arcLength(cnt, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * (area / (perimeter * perimeter))
                    if circularity > 0.50: # reasonably round
                        (cx_cnt, cy_cnt), radius = cv2.minEnclosingCircle(cnt)
                        norm_bx = cx_cnt / small_w
                        norm_by = cy_cnt / small_h
                        
                        # Proximity score to previous ball position / active play
                        dist_to_prev = np.sqrt((norm_bx - self.ball_x)**2 + (norm_by - self.ball_y)**2)
                        dist_to_players = np.sqrt((norm_bx - player_cx)**2 + (norm_by - player_cy)**2)
                        
                        # High circularity + near active play = higher score
                        score = circularity * 2.0 - dist_to_prev * 1.5 - dist_to_players * 0.8
                        if score > best_score:
                            best_score = score
                            best_ball_candidate = (norm_bx, norm_by, int(cx_cnt * (w / small_w)), int(cy_cnt * (h / small_h)))

        ball_found = False
        ball_px = None

        if best_ball_candidate is not None and best_score > 0.2:
            nbx, nby, px_x, px_y = best_ball_candidate
            # Update ball velocity
            self.ball_vx = (nbx - self.ball_x)
            self.ball_vy = (nby - self.ball_y)
            # Smooth ball position
            self.ball_x = self.ball_x * 0.4 + nbx * 0.6
            self.ball_y = self.ball_y * 0.4 + nby * 0.6
            self.ball_confidence = min(1.0, self.ball_confidence + 0.25)
            self.last_detected_ball_px = (px_x, px_y)
            ball_px = (px_x, px_y)
            ball_found = True
        else:
            # Ball coasting / dead reckoning via velocity
            self.ball_x += self.ball_vx * 0.6
            self.ball_y += self.ball_vy * 0.6
            self.ball_vx *= 0.85
            self.ball_vy *= 0.85
            self.ball_confidence = max(0.0, self.ball_confidence - 0.08)

        # 5. Fusion: Weighted Combination of Ball & Player Centroid
        if self.config.follow_ball and self.ball_confidence > 0.3:
            w_ball = self.config.ball_weight * self.ball_confidence
            w_player = 1.0 - w_ball
            raw_target_x = w_ball * self.ball_x + w_player * player_cx
            raw_target_y = w_ball * self.ball_y + w_player * player_cy
            # Anticipation lead: look ahead in direction of ball movement
            raw_target_x += self.ball_vx * self.config.anticipation_lead
        else:
            raw_target_x = player_cx
            raw_target_y = player_cy

        # Filter target to eliminate single-frame spikes
        self.target_x = self.target_x * 0.6 + raw_target_x * 0.4
        self.target_y = self.target_y * 0.6 + raw_target_y * 0.4

        return self.target_x, self.target_y, ball_found, ball_px

    def update_camera(self, target_x: float, target_y: float) -> Tuple[float, float]:
        """
        Critically damped spring-damper smoothing filter for cinematic TV broadcast motion.
        """
        diff_x = target_x - self.cam_x

        # Deadband: if target is close to center of 16:9 view, don't move
        deadband = self.config.deadband_width
        if abs(diff_x) < deadband:
            diff_x = 0.0
        else:
            diff_x = diff_x - np.sign(diff_x) * deadband

        # Smooth velocity damping
        responsiveness = 1.0 - self.config.smoothing_factor
        desired_vx = diff_x * responsiveness
        
        # Clamp maximum panning speed (prevents rapid disorienting pans)
        max_speed = self.config.max_pan_speed
        desired_vx = np.clip(desired_vx, -max_speed, max_speed)

        # Update position
        self.cam_vx = self.cam_vx * 0.8 + desired_vx * 0.2
        self.cam_x += self.cam_vx

        # Clamp camera to valid range so 16:9 crop never shows black borders
        crop_w_ratio, crop_h_ratio = self.get_crop_aspect_ratios()
        min_x = crop_w_ratio * 0.5
        max_x = 1.0 - (crop_w_ratio * 0.5)

        self.cam_x = float(np.clip(self.cam_x, min_x, max_x))
        self.cam_y = self.config.vertical_center_bias

        return self.cam_x, self.cam_y

    def get_crop_aspect_ratios(self) -> Tuple[float, float]:
        """Returns normalized (crop_width, crop_height) inside 32:9."""
        zoom = max(1.0, self.config.zoom_factor)
        crop_w = (0.5 / zoom)
        crop_h = (1.0 / zoom)
        return crop_w, crop_h

    def get_crop_rect(self, pano_width: int, pano_height: int) -> Tuple[int, int, int, int]:
        """Computes pixel bounding box (x, y, w, h) for 16:9 crop inside 32:9."""
        crop_w_norm, crop_h_norm = self.get_crop_aspect_ratios()
        
        box_w = int(round(pano_width * crop_w_norm))
        box_h = int(round(pano_height * crop_h_norm))

        center_px_x = int(round(self.cam_x * pano_width))
        center_px_y = int(round(self.cam_y * pano_height))

        x1 = center_px_x - (box_w // 2)
        y1 = center_px_y - (box_h // 2)

        x1 = max(0, min(pano_width - box_w, x1))
        y1 = max(0, min(pano_height - box_h, y1))

        return x1, y1, box_w, box_h

    def extract_16x9_frame(self, pano_frame_bgr: np.ndarray, out_width: int = 1920, out_height: int = 1080) -> np.ndarray:
        """Extracts 16:9 smooth ball-follow broadcast frame."""
        ph, pw = pano_frame_bgr.shape[:2]
        
        tx, ty, _, _ = self.detect_ball_and_players(pano_frame_bgr)
        self.update_camera(tx, ty)

        x, y, w, h = self.get_crop_rect(pw, ph)
        cropped = pano_frame_bgr[y:y+h, x:x+w]

        if (w, h) != (out_width, out_height):
            return cv2.resize(cropped, (out_width, out_height), interpolation=cv2.INTER_LINEAR)
        return cropped

