"""
Modular AI Detection & Pose Architecture for MatchTrack.
Supports pluggable backends:
1. YOLO11 / YOLOv8: Ultralytics CNN for high-speed live inference.
2. RF-DETR / RT-DETR: Transformer-based Deformable Attention (Apache 2.0 licensed).
3. YOLO-Pose: 17-keypoint skeleton detector for body orientation and VAR virtual offside lines.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
import os
import math
import numpy as np
import cv2

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from ultralytics import YOLO, RTDETR
    HAS_ULTRALYTICS = True
except ImportError:
    HAS_ULTRALYTICS = False

from .paths import get_resource_path
from .logger import get_logger
from .ball_filter import BallPlausibilityFilter

logger = get_logger("detector")

# Standard COCO 17 Keypoints
# 0: nose, 1: l_eye, 2: r_eye, 3: l_ear, 4: r_ear
# 5: l_shoulder, 6: r_shoulder, 7: l_elbow, 8: r_elbow, 9: l_wrist, 10: r_wrist
# 11: l_hip, 12: r_hip, 13: l_knee, 14: r_knee, 15: l_ankle, 16: r_ankle
SKELETON_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),    # Head
    (5, 6),                            # Shoulders
    (5, 7), (7, 9),                    # Left Arm
    (6, 8), (8, 10),                   # Right Arm
    (11, 12),                          # Hips
    (5, 11), (6, 12),                  # Torso flanks
    (11, 13), (13, 15),                # Left Leg
    (12, 14), (14, 16)                 # Right Leg
]


@dataclass
class DetectionBox:
    """Standardized detection container for all AI backends."""
    bbox: Tuple[float, float, float, float]  # [x1, y1, x2, y2] normalized (0.0 to 1.0)
    conf: float
    cls_id: int
    cls_name: str
    team_id: int = 0                         # 0 = Team A (blue), 1 = Team B (red), 2 = Referee
    keypoints: Optional[np.ndarray] = None   # Shape (17, 3): [norm_x, norm_y, conf]
    heading_deg: Optional[float] = None      # Estimated body orientation in degrees (0 = right, 90 = down, etc.)
    track_id: Optional[int] = None           # Continuous Multi-Object Tracking ID
    speed_kmh: Optional[float] = None        # Metric speed in km/h from TacticsMetricsEngine

    @property
    def center(self) -> Tuple[float, float]:
        cx = (self.bbox[0] + self.bbox[2]) * 0.5
        cy = (self.bbox[1] + self.bbox[3]) * 0.5
        return (cx, cy)

    @property
    def width(self) -> float:
        return max(1e-4, self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> float:
        return max(1e-4, self.bbox[3] - self.bbox[1])


class BaseDetectorBackend(ABC):
    """Abstract interface for detection backends."""
    def __init__(self, name: str):
        self.name = name
        self.device = "cpu"
        self.use_fp16 = False
        self.is_loaded = False

    @abstractmethod
    def load(self, device: str = "cpu", use_fp16: bool = True) -> bool:
        """Loads weights onto target device (cuda / cpu)."""
        pass

    @abstractmethod
    def predict(self, 
                img_bgr: np.ndarray, 
                conf_thresh: float = 0.20, 
                offset_x: int = 0, 
                offset_y: int = 0, 
                pano_w: int = 1920, 
                pano_h: int = 1080,
                crop_scale: float = 1.0) -> Tuple[List[DetectionBox], List[DetectionBox]]:
        """
        Runs inference on sub-crop and returns (players, balls) in full-frame normalized coords.
        crop_scale: Scale factor if img_bgr was resized from (pano_crop) before inference.
        """
        pass


class YOLODetectorBackend(BaseDetectorBackend):
    """Ultralytics YOLO11 / YOLOv8 CNN Backend."""
    def __init__(self, player_model: str = "yolo11m_football_player.pt", ball_model: str = "yolo11n_football_ball.pt"):
        super().__init__("YOLO11")
        self.player_model_name = player_model
        self.ball_model_name = ball_model
        self.model: Optional[Any] = None
        self.ball_model: Optional[Any] = None

    def load(self, device: str = "cpu", use_fp16: bool = True) -> bool:
        if not HAS_ULTRALYTICS:
            logger.error("Ultralytics ist nicht installiert.")
            return False

        self.device = device
        self.use_fp16 = (device == "cuda" and use_fp16)

        # 1. Main player / referee detector
        candidates = [
            self.player_model_name,
            get_resource_path(self.player_model_name),
            "yolo11n.pt",
            get_resource_path("yolo11n.pt"),
            "yolov8n.pt",
            get_resource_path("yolov8n.pt")
        ]
        for c in candidates:
            if os.path.exists(c):
                try:
                    self.model = YOLO(c)
                    logger.info(f"YOLO-Backend erfolgreich geladen: {c} auf {self.device}")
                    break
                except Exception as e:
                    logger.warning(f"Fehler beim Laden von {c}: {e}")

        # 2. Dedicated ball detector
        ball_candidates = [
            self.ball_model_name,
            get_resource_path(self.ball_model_name)
        ]
        for bc in ball_candidates:
            if os.path.exists(bc):
                try:
                    self.ball_model = YOLO(bc)
                    logger.info(f"YOLO-Ball-Modell erfolgreich geladen: {bc}")
                    break
                except Exception:
                    pass

        self.is_loaded = (self.model is not None)
        return self.is_loaded

    def predict(self, 
                img_bgr: np.ndarray, 
                conf_thresh: float = 0.20, 
                offset_x: int = 0, 
                offset_y: int = 0, 
                pano_w: int = 1920, 
                pano_h: int = 1080,
                crop_scale: float = 1.0) -> Tuple[List[DetectionBox], List[DetectionBox]]:
        players: List[DetectionBox] = []
        balls: List[DetectionBox] = []

        if not self.is_loaded or self.model is None:
            return players, balls

        cs = max(1e-6, float(crop_scale))

        try:
            with torch.inference_mode():
                results = self.model.predict(
                    img_bgr,
                    device=self.device,
                    half=self.use_fp16,
                    verbose=False,
                    imgsz=1280,
                    conf=conf_thresh
                )
            if results and len(results) > 0:
                res = results[0]
                names = res.names if hasattr(res, "names") else {}
                ih, iw = img_bgr.shape[:2]

                for box in res.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    cls_name = names.get(cls_id, "").lower()
                    xyxy = box.xyxy[0].cpu().numpy()

                    # Convert to normalized coordinates of full panorama
                    bx1 = max(0.0, min(1.0, float(xyxy[0] / cs + offset_x) / pano_w))
                    by1 = max(0.0, min(1.0, float(xyxy[1] / cs + offset_y) / pano_h))
                    bx2 = max(0.0, min(1.0, float(xyxy[2] / cs + offset_x) / pano_w))
                    by2 = max(0.0, min(1.0, float(xyxy[3] / cs + offset_y) / pano_h))

                    is_ball = ("ball" in cls_name or (cls_id == 32 and cls_name in ["sports ball", "ball", ""]))
                    is_player = (cls_name in ["player", "goalkeeper", "referee", "person"]) or (cls_id == 0 and cls_name == "person")

                    if is_ball:
                        balls.append(DetectionBox(
                            bbox=(bx1, by1, bx2, by2),
                            conf=conf,
                            cls_id=cls_id,
                            cls_name="ball",
                            team_id=-1
                        ))
                    elif is_player:
                        team_id = 2 if cls_name == "referee" else _classify_torso_color(img_bgr, xyxy)
                        players.append(DetectionBox(
                            bbox=(bx1, by1, bx2, by2),
                            conf=conf,
                            cls_id=cls_id,
                            cls_name=cls_name,
                            team_id=team_id
                        ))

            # Dedicated specialized ball model inference to enhance small-ball recall
            if self.ball_model is not None:
                with torch.inference_mode():
                    b_res = self.ball_model.predict(
                        img_bgr,
                        device=self.device,
                        half=self.use_fp16,
                        verbose=False,
                        imgsz=1280,
                        conf=0.15
                    )
                if b_res and len(b_res) > 0:
                    for box in b_res[0].boxes:
                        conf = float(box.conf[0])
                        xyxy = box.xyxy[0].cpu().numpy()
                        bx1 = max(0.0, min(1.0, float(xyxy[0] / cs + offset_x) / pano_w))
                        by1 = max(0.0, min(1.0, float(xyxy[1] / cs + offset_y) / pano_h))
                        bx2 = max(0.0, min(1.0, float(xyxy[2] / cs + offset_x) / pano_w))
                        by2 = max(0.0, min(1.0, float(xyxy[3] / cs + offset_y) / pano_h))

                        # Deduplicate with existing balls (keep higher confidence)
                        bcx = (bx1 + bx2) * 0.5
                        bcy = (by1 + by2) * 0.5
                        duplicate = False
                        for eb in balls:
                            if math.hypot(bcx - eb.center[0], bcy - eb.center[1]) < 0.025:
                                if conf > eb.conf:
                                    eb.conf = conf
                                    eb.bbox = (bx1, by1, bx2, by2)
                                duplicate = True
                                break
                        if not duplicate:
                            balls.append(DetectionBox(
                                bbox=(bx1, by1, bx2, by2),
                                conf=conf,
                                cls_id=32,
                                cls_name="ball",
                                team_id=-1
                            ))
        except Exception as e:
            logger.warning(f"Fehler bei YOLO-Inferenz: {e}")

        # Early Aspect Ratio & Shape Filter
        if balls:
            frame_ar = float(pano_w) / max(1.0, float(pano_h))
            balls = [
                b for b in balls 
                if BallPlausibilityFilter.validate_aspect_ratio_and_size(
                    b.bbox[0], b.bbox[1], b.bbox[2], b.bbox[3], frame_aspect_ratio=frame_ar
                )
            ]

        return players, balls


class RTDETRBackend(BaseDetectorBackend):
    """Real-Time Detection Transformer Backend (Apache 2.0)."""
    def __init__(self, model_path: str = "rtdetr-l.pt"):
        super().__init__("RF-DETR Transformer")
        self.model_path = model_path
        self.model: Optional[Any] = None

    def load(self, device: str = "cpu", use_fp16: bool = True) -> bool:
        if not HAS_ULTRALYTICS:
            logger.error("Ultralytics RTDETR nicht verfügbar.")
            return False

        self.device = device
        self.use_fp16 = (device == "cuda" and use_fp16)

        candidates = [
            self.model_path,
            get_resource_path(self.model_path),
            "rtdetr-resnet50.pt"
        ]
        for c in candidates:
            if os.path.exists(c):
                try:
                    self.model = RTDETR(c)
                    logger.info(f"RT-DETR Transformer erfolgreich geladen: {c} auf {self.device}")
                    self.is_loaded = True
                    return True
                except Exception as e:
                    logger.warning(f"Konnte RTDETR {c} nicht laden: {e}")

        # If not present locally, attempt default load
        try:
            self.model = RTDETR(self.model_path)
            self.is_loaded = True
            return True
        except Exception as e:
            logger.error(f"RTDETR Backend nicht geladen: {e}")
            return False

    def predict(self, 
                img_bgr: np.ndarray, 
                conf_thresh: float = 0.20, 
                offset_x: int = 0, 
                offset_y: int = 0, 
                pano_w: int = 1920, 
                pano_h: int = 1080,
                crop_scale: float = 1.0) -> Tuple[List[DetectionBox], List[DetectionBox]]:
        players: List[DetectionBox] = []
        balls: List[DetectionBox] = []

        if not self.is_loaded or self.model is None:
            return players, balls

        cs = max(1e-6, float(crop_scale))

        try:
            with torch.inference_mode():
                results = self.model.predict(
                    img_bgr,
                    device=self.device,
                    half=self.use_fp16,
                    verbose=False,
                    imgsz=1280,
                    conf=conf_thresh
                )
            if results and len(results) > 0:
                res = results[0]
                names = res.names if hasattr(res, "names") else {}

                for box in res.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    cls_name = names.get(cls_id, "").lower()
                    xyxy = box.xyxy[0].cpu().numpy()

                    bx1 = max(0.0, min(1.0, float(xyxy[0] / cs + offset_x) / pano_w))
                    by1 = max(0.0, min(1.0, float(xyxy[1] / cs + offset_y) / pano_h))
                    bx2 = max(0.0, min(1.0, float(xyxy[2] / cs + offset_x) / pano_w))
                    by2 = max(0.0, min(1.0, float(xyxy[3] / cs + offset_y) / pano_h))

                    is_ball = (cls_name in ["sports ball", "ball"] or cls_id == 32)
                    is_player = (cls_name in ["person", "player", "referee"] or cls_id == 0)

                    if is_player:
                        team_id = _classify_torso_color(img_bgr, xyxy)
                        players.append(DetectionBox(
                            bbox=(bx1, by1, bx2, by2),
                            conf=conf,
                            cls_id=cls_id,
                            cls_name="player",
                            team_id=team_id
                        ))
                    elif is_ball:
                        balls.append(DetectionBox(
                            bbox=(bx1, by1, bx2, by2),
                            conf=conf,
                            cls_id=cls_id,
                            cls_name="ball",
                            team_id=-1
                        ))
        except Exception as e:
            logger.warning(f"Fehler bei RTDETR-Inferenz: {e}")

        # Early Aspect Ratio & Shape Filter
        if balls:
            frame_ar = float(pano_w) / max(1.0, float(pano_h))
            balls = [
                b for b in balls 
                if BallPlausibilityFilter.validate_aspect_ratio_and_size(
                    b.bbox[0], b.bbox[1], b.bbox[2], b.bbox[3], frame_aspect_ratio=frame_ar
                )
            ]

        return players, balls


class YOLOPoseBackend(BaseDetectorBackend):
    """
    YOLO-Pose Skeleton & Keypoint Backend.
    Extracts 17 anatomical keypoints per player and computes body orientation
    and extremities for VAR Virtual Offside line detection.
    """
    def __init__(self, model_path: str = "yolov8n-pose.pt"):
        super().__init__("YOLO-Pose Skelett")
        self.model_path = model_path
        self.model: Optional[Any] = None

    def load(self, device: str = "cpu", use_fp16: bool = True) -> bool:
        if not HAS_ULTRALYTICS:
            logger.error("Ultralytics YOLO-Pose nicht verfügbar.")
            return False

        self.device = device
        self.use_fp16 = (device == "cuda" and use_fp16)

        candidates = [
            self.model_path,
            get_resource_path(self.model_path),
            "yolo11n-pose.pt",
            get_resource_path("yolo11n-pose.pt")
        ]
        for c in candidates:
            if os.path.exists(c):
                try:
                    self.model = YOLO(c)
                    logger.info(f"YOLO-Pose Backend erfolgreich geladen: {c} auf {self.device}")
                    self.is_loaded = True
                    return True
                except Exception as e:
                    logger.warning(f"Konnte Pose-Modell {c} nicht laden: {e}")

        try:
            self.model = YOLO(self.model_path)
            self.is_loaded = True
            return True
        except Exception as e:
            logger.error(f"YOLO-Pose nicht geladen: {e}")
            return False

    def predict(self, 
                img_bgr: np.ndarray, 
                conf_thresh: float = 0.20, 
                offset_x: int = 0, 
                offset_y: int = 0, 
                pano_w: int = 1920, 
                pano_h: int = 1080,
                crop_scale: float = 1.0) -> Tuple[List[DetectionBox], List[DetectionBox]]:
        players: List[DetectionBox] = []
        balls: List[DetectionBox] = []

        if not self.is_loaded or self.model is None:
            return players, balls

        cs = max(1e-6, float(crop_scale))

        try:
            with torch.inference_mode():
                results = self.model.predict(
                    img_bgr,
                    device=self.device,
                    half=self.use_fp16,
                    verbose=False,
                    imgsz=1280,
                    conf=conf_thresh
                )
            if results and len(results) > 0:
                res = results[0]
                has_kpts = hasattr(res, "keypoints") and res.keypoints is not None

                for i, box in enumerate(res.boxes):
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    xyxy = box.xyxy[0].cpu().numpy()

                    bx1 = max(0.0, min(1.0, float(xyxy[0] / cs + offset_x) / pano_w))
                    by1 = max(0.0, min(1.0, float(xyxy[1] / cs + offset_y) / pano_h))
                    bx2 = max(0.0, min(1.0, float(xyxy[2] / cs + offset_x) / pano_w))
                    by2 = max(0.0, min(1.0, float(xyxy[3] / cs + offset_y) / pano_h))

                    # Process 17 keypoints if available
                    norm_kpts = None
                    heading_angle = None
                    if has_kpts and len(res.keypoints) > i:
                        raw_kpts = res.keypoints[i].data[0].cpu().numpy() # (17, 3) -> [x, y, conf]
                        norm_kpts = np.zeros_like(raw_kpts)
                        norm_kpts[:, 0] = np.clip((raw_kpts[:, 0] / cs + offset_x) / pano_w, 0.0, 1.0)
                        norm_kpts[:, 1] = np.clip((raw_kpts[:, 1] / cs + offset_y) / pano_h, 0.0, 1.0)
                        norm_kpts[:, 2] = raw_kpts[:, 2] # confidence

                        # Estimate body orientation from left shoulder (5) and right shoulder (6)
                        ls = norm_kpts[5]
                        rs = norm_kpts[6]
                        if ls[2] > 0.3 and rs[2] > 0.3:
                            sdx = rs[0] - ls[0]
                            sdy = rs[1] - ls[1]
                            # Perpendicular to shoulder axis is the facing direction
                            heading_angle = math.degrees(math.atan2(-sdx, sdy))

                    team_id = _classify_torso_color(img_bgr, xyxy)
                    players.append(DetectionBox(
                        bbox=(bx1, by1, bx2, by2),
                        conf=conf,
                        cls_id=0,
                        cls_name="player",
                        team_id=team_id,
                        keypoints=norm_kpts,
                        heading_deg=heading_angle
                    ))
        except Exception as e:
            logger.warning(f"Fehler bei YOLO-Pose-Inferenz: {e}")

        return players, balls


def _classify_torso_color(img_bgr: np.ndarray, xyxy: np.ndarray) -> int:
    """Fast sampling of upper-torso BGR color for Team A / Team B separation."""
    try:
        ih, iw = img_bgr.shape[:2]
        px1 = max(0, min(iw - 1, int(round(xyxy[0]))))
        py1 = max(0, min(ih - 1, int(round(xyxy[1]))))
        px2 = max(px1 + 1, min(iw, int(round(xyxy[2]))))
        py2 = max(py1 + 1, min(ih, int(round(xyxy[3]))))
        bh = py2 - py1
        bw = px2 - px1
        if bh >= 6 and bw >= 4:
            ty1 = py1 + int(bh * 0.25)
            ty2 = py1 + int(bh * 0.55)
            tx1 = px1 + int(bw * 0.20)
            tx2 = px1 + int(bw * 0.80)
            torso = img_bgr[ty1:ty2, tx1:tx2]
            if torso.size > 0:
                b_mean = float(np.mean(torso[:, :, 0]))
                r_mean = float(np.mean(torso[:, :, 2]))
                return 1 if r_mean > b_mean else 0
    except Exception:
        pass
    return 0


class OffsideAnalyzer:
    """
    FIFA-Standard Virtual Offside Line (VAR) Calculator.
    Identifies the second-last defender and calculates the exact reference line
    using feet, limbs, and torso boundaries.
    """
    @staticmethod
    def compute_offside_line(
        players: List[DetectionBox],
        ball_pos: Optional[Tuple[float, float]] = None
    ) -> Optional[Tuple[float, int, str]]:
        """
        Calculates the virtual offside line:
        Returns: (offside_norm_x, defending_team_id, description_label) or None
        """
        if len(players) < 4:
            return None

        # Filter out referees
        field_players = [p for p in players if p.team_id in (0, 1)]
        if len(field_players) < 4:
            return None

        # Determine action focus side (0 = Left Half attack, 1 = Right Half attack)
        if ball_pos is not None:
            action_x = ball_pos[0]
        else:
            action_x = sum(p.center[0] for p in field_players) / len(field_players)

        # If action is on right half ($x \ge 0.5$), attack is towards right goal (defending team is rightmost)
        if action_x >= 0.5:
            # Defending team is the team that has the furthest right players
            team_0_xs = [p.bbox[2] for p in field_players if p.team_id == 0]
            team_1_xs = [p.bbox[2] for p in field_players if p.team_id == 1]
            if not team_0_xs or not team_1_xs:
                return None
            
            defending_team = 1 if max(team_1_xs) >= max(team_0_xs) else 0
            defenders = [p for p in field_players if p.team_id == defending_team]
            if len(defenders) < 2:
                return None

            # Sort defenders descending by their furthest-right point (x2 or right foot)
            def get_effective_max_x(p: DetectionBox) -> float:
                if p.keypoints is not None:
                    # Feet ankles: index 15 (L) and 16 (R)
                    valid_feet = [p.keypoints[idx, 0] for idx in (15, 16) if p.keypoints[idx, 2] > 0.25]
                    if valid_feet:
                        return max(valid_feet)
                return p.bbox[2]

            defenders.sort(key=get_effective_max_x, reverse=True)
            # The second-last defender determines the offside line
            second_last = defenders[1]
            offside_x = float(get_effective_max_x(second_last))
            team_str = "Team Rot" if defending_team == 1 else "Team Blau"
            return (offside_x, defending_team, f"🚩 Abseitslinie ({team_str})")

        else: # Attack is towards left goal ($x < 0.5$)
            team_0_xs = [p.bbox[0] for p in field_players if p.team_id == 0]
            team_1_xs = [p.bbox[0] for p in field_players if p.team_id == 1]
            if not team_0_xs or not team_1_xs:
                return None

            defending_team = 1 if min(team_1_xs) <= min(team_0_xs) else 0
            defenders = [p for p in field_players if p.team_id == defending_team]
            if len(defenders) < 2:
                return None

            def get_effective_min_x(p: DetectionBox) -> float:
                if p.keypoints is not None:
                    valid_feet = [p.keypoints[idx, 0] for idx in (15, 16) if p.keypoints[idx, 2] > 0.25]
                    if valid_feet:
                        return min(valid_feet)
                return p.bbox[0]

            defenders.sort(key=get_effective_min_x, reverse=False)
            second_last = defenders[1]
            offside_x = float(get_effective_min_x(second_last))
            team_str = "Team Rot" if defending_team == 1 else "Team Blau"
            return (offside_x, defending_team, f"🚩 Abseitslinie ({team_str})")

