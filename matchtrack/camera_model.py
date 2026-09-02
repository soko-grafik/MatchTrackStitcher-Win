"""
Camera Model & Lens Calibration for DJI Osmo Action 4.
Supports Fisheye (Kannala-Brandt / Equidistant), Standard Brown-Conrady, and custom calibration.
"""
from dataclasses import dataclass, field
import json
import numpy as np
import cv2
from typing import Tuple, Optional, Dict, Any


@dataclass
class CameraIntrinsics:
    """Intrinsic camera parameters and distortion coefficients."""
    name: str = "DJI Action 4 2.7K Standard (Dewarp)"
    sensor_width: float = 9.6  # mm
    sensor_height: float = 7.2 # mm
    image_width: int = 2720
    image_height: int = 1530
    # Horizontal Field of View in degrees (Action 4 Standard/Dewarp mode is approx 88-94 deg)
    hfov_deg: float = 92.0
    # Focal length in pixels (auto-calculated from hfov_deg if fx is 0 or via update_fov)
    fx: float = 1315.0
    fy: float = 1315.0
    cx: float = 1360.0
    cy: float = 765.0
    # Fisheye / Radial distortion coefficients
    k1: float = 0.0
    k2: float = 0.0
    k3: float = 0.0
    k4: float = 0.0
    model_type: str = "pinhole" # 'pinhole' for Dewarp, 'fisheye' for Wide/Ultrawide

    def __post_init__(self):
        if (self.image_width != 2720 or self.image_height != 1530) and (self.cx == 1360.0 and self.cy == 765.0):
            self.cx = self.image_width * 0.5
            self.cy = self.image_height * 0.5
            self.set_fov(self.hfov_deg)

    def set_fov(self, hfov_degrees: float):
        """Updates the focal length to match the given horizontal FOV in degrees."""
        self.hfov_deg = max(45.0, min(160.0, hfov_degrees))
        rad = np.radians(self.hfov_deg * 0.5)
        self.fx = (self.image_width * 0.5) / np.tan(rad)
        self.fy = self.fx

    def set_resolution(self, width: int, height: int):
        """Adapts intrinsics to actual video resolution while preserving FOV."""
        self.image_width = width
        self.image_height = height
        self.cx = width * 0.5
        self.cy = height * 0.5
        self.set_fov(self.hfov_deg)

    @property
    def camera_matrix(self) -> np.ndarray:
        return np.array([
            [self.fx, 0.0, self.cx],
            [0.0, self.fy, self.cy],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)

    @property
    def dist_coeffs(self) -> np.ndarray:
        return np.array([self.k1, self.k2, self.k3, self.k4], dtype=np.float64)

    def scale_to_resolution(self, width: int, height: int) -> 'CameraIntrinsics':
        """Returns a new CameraIntrinsics scaled to a different resolution (e.g. for preview)."""
        scale_x = width / self.image_width
        scale_y = height / self.image_height
        return CameraIntrinsics(
            name=self.name,
            sensor_width=self.sensor_width,
            sensor_height=self.sensor_height,
            image_width=width,
            image_height=height,
            fx=self.fx * scale_x,
            fy=self.fy * scale_y,
            cx=self.cx * scale_x,
            cy=self.cy * scale_y,
            k1=self.k1,
            k2=self.k2,
            k3=self.k3,
            k4=self.k4,
            model_type=self.model_type
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "sensor_width": self.sensor_width,
            "sensor_height": self.sensor_height,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "fx": self.fx,
            "fy": self.fy,
            "cx": self.cx,
            "cy": self.cy,
            "k1": self.k1,
            "k2": self.k2,
            "k3": self.k3,
            "k4": self.k4,
            "model_type": self.model_type
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CameraIntrinsics':
        return cls(**data)


# Predefined lens presets for DJI Action 4
CAMERA_PRESETS = {
    "DJI_Action4_1080p_Dewarp": CameraIntrinsics(
        name="DJI Action 4 1080p Standard/Dewarp (16:9 60fps)",
        image_width=1920,
        image_height=1080,
        hfov_deg=92.0,
        fx=928.0,
        fy=928.0,
        cx=960.0,
        cy=540.0,
        k1=0.0,
        k2=0.0,
        k3=0.0,
        k4=0.0,
        model_type="pinhole"
    ),
    "DJI_Action4_2_7K_Dewarp": CameraIntrinsics(
        name="DJI Action 4 2.7K Standard/Dewarp (16:9)",
        image_width=2720,
        image_height=1530,
        hfov_deg=92.0,
        fx=1315.0,
        fy=1315.0,
        cx=1360.0,
        cy=765.0,
        k1=0.0,
        k2=0.0,
        k3=0.0,
        k4=0.0,
        model_type="pinhole"
    ),
    "DJI_Action4_4K_Dewarp": CameraIntrinsics(
        name="DJI Action 4 4K Standard/Dewarp (16:9)",
        image_width=3840,
        image_height=2160,
        hfov_deg=92.0,
        fx=1856.0,
        fy=1856.0,
        cx=1920.0,
        cy=1080.0,
        k1=0.0,
        k2=0.0,
        k3=0.0,
        k4=0.0,
        model_type="pinhole"
    ),
    "DJI_Action4_Wide_4K": CameraIntrinsics(
        name="DJI Action 4 Wide 4K (16:9)",
        image_width=3840,
        image_height=2160,
        hfov_deg=115.0,
        fx=1550.0,
        fy=1550.0,
        cx=1920.0,
        cy=1080.0,
        k1=-0.008,
        k2=0.003,
        k3=-0.0008,
        k4=0.0001,
        model_type="fisheye"
    ),
    "DJI_Action4_Ultrawide_4K": CameraIntrinsics(
        name="DJI Action 4 Ultrawide 4K (16:9)",
        image_width=3840,
        image_height=2160,
        hfov_deg=130.0,
        fx=1350.0,
        fy=1350.0,
        cx=1920.0,
        cy=1080.0,
        k1=-0.012,
        k2=0.006,
        k3=-0.018,
        k4=0.0003,
        model_type="fisheye"
    ),
    "DJI_Action4_Ultrawide_4K_4_3": CameraIntrinsics(
        name="DJI Action 4 Ultrawide 4K (4:3)",
        image_width=3840,
        image_height=2880,
        fx=1650.0,
        fy=1650.0,
        cx=1920.0,
        cy=1440.0,
        k1=-0.012,
        k2=0.006,
        k3=-0.0018,
        k4=0.0003,
        model_type="fisheye"
    )
}
