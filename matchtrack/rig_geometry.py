"""
Rig Geometry & 3D Projection Engine for MatchTrack-Stitcher.
Handles 3D camera pose transformations, rig pitch leveling, and cylindrical/panini projections.
"""
from dataclasses import dataclass, field
import json
import numpy as np
from typing import Tuple, Optional, Dict, Any
from .camera_model import CameraIntrinsics, CAMERA_PRESETS


@dataclass
class CameraPose:
    """Extrinsic 3D pose of a single camera relative to the rig center and 4-corner perspective pin offsets."""
    yaw: float = 0.0    # Degrees (horizontal pan: negative = left, positive = right)
    pitch: float = -15.0 # Degrees (tilt: negative = tilted down)
    roll: float = 0.0   # Degrees (rotation around optical axis)
    # Translation baseline (meters, optional for near-field parallax, usually negligible for soccer pitch)
    tx: float = 0.0
    ty: float = 0.0
    tz: float = 0.0
    # 4 Corner pins: [TL, TR, BR, BL] normalized [0.0 ... 1.0]
    # Default: [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
    corners: list = field(default_factory=lambda: [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])

    def is_corners_modified(self) -> bool:
        if not self.corners or len(self.corners) != 4:
            return False
        default = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
        for (cx, cy), (dx, dy) in zip(self.corners, default):
            if abs(cx - dx) > 1e-4 or abs(cy - dy) > 1e-4:
                return True
        return False

    def reset_corners(self):
        self.corners = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]

    def get_homography_matrix(self, width: float, height: float) -> Optional[np.ndarray]:
        """
        Computes the perspective homography transform matrix mapping canonical quad [0,w]x[0,h] to distorted corners.
        """
        import cv2
        if not self.is_corners_modified():
            return None
        src_pts = np.array([
            [0.0, 0.0],
            [width, 0.0],
            [width, height],
            [0.0, height]
        ], dtype=np.float32)
        dst_pts = np.array([
            [self.corners[0][0] * width, self.corners[0][1] * height],
            [self.corners[1][0] * width, self.corners[1][1] * height],
            [self.corners[2][0] * width, self.corners[2][1] * height],
            [self.corners[3][0] * width, self.corners[3][1] * height]
        ], dtype=np.float32)
        return cv2.getPerspectiveTransform(src_pts, dst_pts)

    def rotation_matrix(self) -> np.ndarray:
        """Computes 3x3 rotation matrix R = Rz(roll) * Rx(pitch) * Ry(yaw)."""
        rad_y = np.radians(self.yaw)
        rad_p = np.radians(self.pitch)
        rad_r = np.radians(self.roll)

        # Yaw (around Y axis)
        Ry = np.array([
            [np.cos(rad_y), 0, np.sin(rad_y)],
            [0, 1, 0],
            [-np.sin(rad_y), 0, np.cos(rad_y)]
        ], dtype=np.float64)

        # Pitch (around X axis)
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(rad_p), -np.sin(rad_p)],
            [0, np.sin(rad_p), np.cos(rad_p)]
        ], dtype=np.float64)

        # Roll (around Z axis)
        Rz = np.array([
            [np.cos(rad_r), -np.sin(rad_r), 0],
            [np.sin(rad_r), np.cos(rad_r), 0],
            [0, 0, 1]
        ], dtype=np.float64)

        return Rz @ Rx @ Ry

    def to_dict(self) -> Dict[str, Any]:
        return {
            "yaw": self.yaw,
            "pitch": self.pitch,
            "roll": self.roll,
            "tx": self.tx,
            "ty": self.ty,
            "tz": self.tz,
            "corners": [list(c) for c in self.corners]
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CameraPose':
        corners = data.get("corners", [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        return cls(
            yaw=data.get("yaw", 0.0),
            pitch=data.get("pitch", -15.0),
            roll=data.get("roll", 0.0),
            tx=data.get("tx", 0.0),
            ty=data.get("ty", 0.0),
            tz=data.get("tz", 0.0),
            corners=[list(c) for c in corners]
        )


@dataclass
class RigConfiguration:
    """Full Dual-Camera Rig setup with calibration parameters."""
    name: str = "DJI Action 4 Dual-Rig 2.7K Dewarp (80 deg, 15 deg tilt)"
    # Left & Right camera intrinsics
    left_camera: CameraIntrinsics = field(default_factory=lambda: CAMERA_PRESETS["DJI_Action4_2_7K_Dewarp"])
    right_camera: CameraIntrinsics = field(default_factory=lambda: CAMERA_PRESETS["DJI_Action4_2_7K_Dewarp"])
    
    # Left & Right camera extrinsics (Left ~ -40 deg, Right ~ +40 deg, both tilted -15 deg)
    left_pose: CameraPose = field(default_factory=lambda: CameraPose(yaw=-40.0, pitch=-15.0, roll=0.0, tx=-0.04))
    right_pose: CameraPose = field(default_factory=lambda: CameraPose(yaw=40.0, pitch=-15.0, roll=0.0, tx=0.04))
    
    # Global Rig leveling & framing adjustments
    global_pitch_correction: float = 15.0  # Counteracts the -15° downward tilt to level the horizon/lines
    global_roll_correction: float = 0.0   # Horizon tilt fine-tuning
    global_yaw_center: float = 0.0        # Panning center offset
    
    # Panorama projection settings
    pano_hfov: float = 145.0              # Horizontal field of view of output 32:9 panorama (in degrees)
    projection_type: str = "cylindrical"  # 'cylindrical', 'pannini', or 'rectilinear'
    
    # Overlap blending & Auto-Crop parameters
    blend_width_deg: float = 8.0          # Angular width of seam transition (degrees)
    vertical_crop_offset: float = 0.12    # -2.0 (top) to +2.0 (bottom) vertical shift (0.12 gives optimal near-field corner coverage)
    horizontal_squeeze: float = 1.0       # 0.60 to 1.60x uniform anamorphic width squeeze factor (brings corners into view)
    auto_crop_lir: bool = True            # Largest Interior Rectangle (removes black curved borders)
    lir_safety_margin: float = 0.005      # 0.5% safety margin (preserves field corners)

    def has_corner_pins(self) -> bool:
        """Returns True if either left or right camera has custom corner pinning."""
        return self.left_pose.is_corners_modified() or self.right_pose.is_corners_modified()

    def reset_all_corners(self):
        """Resets 4-corner distortion on both cameras."""
        self.left_pose.reset_corners()
        self.right_pose.reset_corners()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "left_camera": self.left_camera.to_dict(),
            "right_camera": self.right_camera.to_dict(),
            "left_pose": self.left_pose.to_dict(),
            "right_pose": self.right_pose.to_dict(),
            "global_pitch_correction": self.global_pitch_correction,
            "global_roll_correction": self.global_roll_correction,
            "global_yaw_center": self.global_yaw_center,
            "pano_hfov": self.pano_hfov,
            "projection_type": self.projection_type,
            "blend_width_deg": self.blend_width_deg,
            "vertical_crop_offset": self.vertical_crop_offset,
            "horizontal_squeeze": self.horizontal_squeeze,
            "auto_crop_lir": self.auto_crop_lir,
            "lir_safety_margin": self.lir_safety_margin
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RigConfiguration':
        left_cam = CameraIntrinsics.from_dict(data["left_camera"])
        right_cam = CameraIntrinsics.from_dict(data["right_camera"])
        left_pose = CameraPose.from_dict(data["left_pose"])
        right_pose = CameraPose.from_dict(data["right_pose"])
        
        return cls(
            name=data.get("name", "Rig Config"),
            left_camera=left_cam,
            right_camera=right_cam,
            left_pose=left_pose,
            right_pose=right_pose,
            global_pitch_correction=data.get("global_pitch_correction", 15.0),
            global_roll_correction=data.get("global_roll_correction", 0.0),
            global_yaw_center=data.get("global_yaw_center", 0.0),
            pano_hfov=data.get("pano_hfov", 145.0),
            projection_type=data.get("projection_type", "cylindrical"),
            blend_width_deg=data.get("blend_width_deg", 8.0),
            vertical_crop_offset=data.get("vertical_crop_offset", 0.12),
            horizontal_squeeze=data.get("horizontal_squeeze", 1.0),
            auto_crop_lir=data.get("auto_crop_lir", True),
            lir_safety_margin=data.get("lir_safety_margin", 0.005)
        )

    def save_to_json(self, filepath: str):
        """Saves configuration to a JSON file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=4, ensure_ascii=False)

    @classmethod
    def load_from_json(cls, filepath: str) -> 'RigConfiguration':
        """Loads configuration from a JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_dict(data)
