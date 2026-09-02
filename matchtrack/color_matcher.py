"""
Color & Exposure Matching for Dual-Camera Stitching.
Calculates luminance and RGB gains in the overlap region to eliminate exposure jumps across the seam.
"""
import numpy as np
import cv2
from typing import Tuple


class ColorExposureMatcher:
    """Matches color/gain between two camera views across the seam."""
    def __init__(self, smoothing_factor: float = 0.9):
        self.smoothing_factor = smoothing_factor
        self.smoothed_gain_l = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        self.smoothed_gain_r = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        self.initialized = False

    def compute_gains(self, remapped_left: np.ndarray, remapped_right: np.ndarray, overlap_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes per-channel RGB gain multipliers from the overlap region.
        """
        if not np.any(overlap_mask):
            return np.ones(3, dtype=np.float32), np.ones(3, dtype=np.float32)

        # Mean color in overlap
        mean_l = np.mean(remapped_left[overlap_mask], axis=0).astype(np.float32) # [B, G, R]
        mean_r = np.mean(remapped_right[overlap_mask], axis=0).astype(np.float32)

        mean_l = np.maximum(mean_l, 1.0)
        mean_r = np.maximum(mean_r, 1.0)

        # Target mean is the average of both
        target_mean = (mean_l + mean_r) * 0.5
        gain_l = target_mean / mean_l
        gain_r = target_mean / mean_r

        # Limit gains to prevent over-amplification
        gain_l = np.clip(gain_l, 0.7, 1.3)
        gain_r = np.clip(gain_r, 0.7, 1.3)

        if not self.initialized:
            self.smoothed_gain_l = gain_l
            self.smoothed_gain_r = gain_r
            self.initialized = True
        else:
            self.smoothed_gain_l = self.smoothing_factor * self.smoothed_gain_l + (1.0 - self.smoothing_factor) * gain_l
            self.smoothed_gain_r = self.smoothing_factor * self.smoothed_gain_r + (1.0 - self.smoothing_factor) * gain_r

        return self.smoothed_gain_l, self.smoothed_gain_r
