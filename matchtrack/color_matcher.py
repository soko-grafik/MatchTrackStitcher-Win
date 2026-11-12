"""
Color & Exposure Matching for Dual-Camera Stitching.
Calculates luminance and RGB gains in the overlap region to eliminate exposure and white-balance jumps across the seam.
"""
import threading
import numpy as np
import cv2
from typing import Tuple, Optional


class ColorExposureMatcher:
    """
    Matches color, white balance, and exposure between Left and Right cameras
    in the overlap region to completely eliminate the visible seam edge in the middle.
    """
    def __init__(self, enabled: bool = True, strength: float = 1.0, smoothing_factor: float = 0.90):
        self.enabled = bool(enabled)
        self.strength = float(np.clip(strength, 0.0, 1.0))
        self.smoothing_factor = float(np.clip(smoothing_factor, 0.0, 0.99))
        
        self.smoothed_gain_l = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        self.smoothed_gain_r = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        self.lut_l: Optional[np.ndarray] = None
        self.lut_r: Optional[np.ndarray] = None
        self.initialized = False
        self._lock = threading.Lock()
        self._rebuild_luts(self.smoothed_gain_l, self.smoothed_gain_r)

    def reset(self):
        """Resets smoothed gains so the matcher can adapt immediately to new videos/settings."""
        with self._lock:
            self.smoothed_gain_l = np.array([1.0, 1.0, 1.0], dtype=np.float32)
            self.smoothed_gain_r = np.array([1.0, 1.0, 1.0], dtype=np.float32)
            self.initialized = False
            self._rebuild_luts(self.smoothed_gain_l, self.smoothed_gain_r)

    def set_enabled(self, enabled: bool):
        self.enabled = bool(enabled)

    def set_strength(self, strength: float):
        self.strength = float(np.clip(strength, 0.0, 1.0))
        with self._lock:
            self._rebuild_luts(self.smoothed_gain_l, self.smoothed_gain_r)

    def _rebuild_luts(self, gl: np.ndarray, gr: np.ndarray):
        """Builds fast 256-element uint8 lookup tables for cv2.LUT."""
        # Blend gain towards 1.0 according to strength
        effective_gl = 1.0 + (gl - 1.0) * self.strength
        effective_gr = 1.0 + (gr - 1.0) * self.strength
        
        lut_l = np.zeros((1, 256, 3), dtype=np.uint8)
        lut_r = np.zeros((1, 256, 3), dtype=np.uint8)
        for c in range(3):
            lut_l[0, :, c] = np.clip(np.arange(256) * effective_gl[c], 0, 255).astype(np.uint8)
            lut_r[0, :, c] = np.clip(np.arange(256) * effective_gr[c], 0, 255).astype(np.uint8)
        self.lut_l = lut_l
        self.lut_r = lut_r

    def calibrate(self, frame_l: np.ndarray, frame_r: np.ndarray, luts: any, strength: float = 1.0) -> dict:
        """
        Explicitly calibrates color, white-balance and exposure between Left and Right cameras
        using the overlap region of the supplied frames. Immediately activates calibrated LUTs.
        """
        if frame_l is None or frame_r is None or luts is None:
            return {"success": False, "message": "Frames oder LUTs fehlen"}

        self.enabled = True
        self.strength = float(np.clip(strength, 0.0, 1.0))

        # Remap overlap
        if getattr(luts, 'seam_x_start', None) is not None and getattr(luts, 'map_xl_sub', None) is not None:
            x0 = luts.seam_x_start
            x1 = luts.seam_x_end
            if x1 <= x0:
                return {"success": False, "message": "Ungültiger Nahtstellen-Bereich"}
            rem_l_sub = cv2.remap(frame_l, luts.map_xl_sub, luts.map_yl_sub, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            rem_r_sub = cv2.remap(frame_r, luts.map_xr_sub, luts.map_yr_sub, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            sub_l = rem_l_sub[:, x0:x1:2][::4]
            sub_r = rem_r_sub[:, :(x1 - x0):2][::4]
            sub_wl = luts.weight_seam_l[::4, ::2, 0]
            sub_wr = luts.weight_seam_r[::4, ::2, 0]
        else:
            rem_l = cv2.remap(frame_l, luts.map_x_left, luts.map_y_left, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            rem_r = cv2.remap(frame_r, luts.map_x_right, luts.map_y_right, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            sub_l = rem_l[::4, ::4]
            sub_r = rem_r[::4, ::4]
            sub_wl = luts.weight_left[::4, ::4, 0]
            sub_wr = luts.weight_right[::4, ::4, 0]

        valid_mask = (sub_wl > 0.05) & (sub_wr > 0.05) & (sub_l.any(axis=-1)) & (sub_r.any(axis=-1))
        if np.count_nonzero(valid_mask) < 30:
            return {"success": False, "message": "Zu wenige Überlappungspixel für Farbabgleich"}

        mean_l = np.mean(sub_l[valid_mask], axis=0).astype(np.float32)
        mean_r = np.mean(sub_r[valid_mask], axis=0).astype(np.float32)

        if mean_l.min() < 1.0 or mean_r.min() < 1.0:
            return {"success": False, "message": "Mittelwerte zu dunkel"}

        target_mean = (mean_l + mean_r) * 0.5
        raw_gain_l = np.clip(target_mean / np.maximum(mean_l, 1.0), 0.55, 1.80)
        raw_gain_r = np.clip(target_mean / np.maximum(mean_r, 1.0), 0.55, 1.80)

        with self._lock:
            self.smoothed_gain_l = raw_gain_l
            self.smoothed_gain_r = raw_gain_r
            self.initialized = True
            self._rebuild_luts(self.smoothed_gain_l, self.smoothed_gain_r)

        return {
            "success": True,
            "gain_l": raw_gain_l.tolist(),
            "gain_r": raw_gain_r.tolist(),
            "mean_l": mean_l.tolist(),
            "mean_r": mean_r.tolist(),
            "target_mean": target_mean.tolist()
        }

    def update_gains_from_sub(self, rem_l_sub: np.ndarray, rem_r_sub: np.ndarray, luts: any):
        """
        Calculates per-channel gains from the partitioned overlap strip and updates LUTs.
        Optimized to run in < 0.5 ms via spatial subsampling.
        """
        if not self.enabled or luts.seam_x_start is None or luts.seam_x_end is None:
            return

        x0 = luts.seam_x_start
        x1 = luts.seam_x_end
        if x1 <= x0:
            return

        overlap_l = rem_l_sub[:, x0:x1]
        overlap_r = rem_r_sub[:, :(x1 - x0)]

        w_l = luts.weight_seam_l
        w_r = luts.weight_seam_r
        if w_l is None or w_r is None:
            return

        # Subsample for speed (every 4th row, every 2nd col)
        sub_l = overlap_l[::4, ::2]
        sub_r = overlap_r[::4, ::2]
        sub_wl = w_l[::4, ::2, 0]
        sub_wr = w_r[::4, ::2, 0]

        valid_mask = (sub_wl > 0.05) & (sub_wr > 0.05) & (sub_l.any(axis=-1)) & (sub_r.any(axis=-1))
        if np.count_nonzero(valid_mask) < 40:
            return

        mean_l = np.mean(sub_l[valid_mask], axis=0).astype(np.float32)
        mean_r = np.mean(sub_r[valid_mask], axis=0).astype(np.float32)

        if mean_l.min() < 2.0 or mean_r.min() < 2.0:
            return

        target_mean = (mean_l + mean_r) * 0.5
        raw_gain_l = np.clip(target_mean / np.maximum(mean_l, 1.0), 0.65, 1.45)
        raw_gain_r = np.clip(target_mean / np.maximum(mean_r, 1.0), 0.65, 1.45)

        with self._lock:
            if not self.initialized:
                self.smoothed_gain_l = raw_gain_l
                self.smoothed_gain_r = raw_gain_r
                self.initialized = True
            else:
                alpha = self.smoothing_factor
                self.smoothed_gain_l = alpha * self.smoothed_gain_l + (1.0 - alpha) * raw_gain_l
                self.smoothed_gain_r = alpha * self.smoothed_gain_r + (1.0 - alpha) * raw_gain_r
                
            self._rebuild_luts(self.smoothed_gain_l, self.smoothed_gain_r)

    def update_gains_from_full(self, rem_l: np.ndarray, rem_r: np.ndarray, luts: any):
        """Fallback for unpartitioned LUTs."""
        if not self.enabled:
            return
        w_l = luts.weight_left
        w_r = luts.weight_right
        valid_mask = (w_l[::4, ::4, 0] > 0.05) & (w_r[::4, ::4, 0] > 0.05) & (rem_l[::4, ::4].any(axis=-1)) & (rem_r[::4, ::4].any(axis=-1))
        if np.count_nonzero(valid_mask) < 40:
            return

        mean_l = np.mean(rem_l[::4, ::4][valid_mask], axis=0).astype(np.float32)
        mean_r = np.mean(rem_r[::4, ::4][valid_mask], axis=0).astype(np.float32)
        if mean_l.min() < 2.0 or mean_r.min() < 2.0:
            return

        target_mean = (mean_l + mean_r) * 0.5
        raw_gain_l = np.clip(target_mean / np.maximum(mean_l, 1.0), 0.65, 1.45)
        raw_gain_r = np.clip(target_mean / np.maximum(mean_r, 1.0), 0.65, 1.45)

        with self._lock:
            if not self.initialized:
                self.smoothed_gain_l = raw_gain_l
                self.smoothed_gain_r = raw_gain_r
                self.initialized = True
            else:
                alpha = self.smoothing_factor
                self.smoothed_gain_l = alpha * self.smoothed_gain_l + (1.0 - alpha) * raw_gain_l
                self.smoothed_gain_r = alpha * self.smoothed_gain_r + (1.0 - alpha) * raw_gain_r
            self._rebuild_luts(self.smoothed_gain_l, self.smoothed_gain_r)

    def apply_match(self, rem_l: np.ndarray, rem_r: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Applies precomputed LUTs to both remapped views in < 1 ms."""
        if not self.enabled or self.lut_l is None or self.lut_r is None:
            return rem_l, rem_r
        matched_l = cv2.LUT(rem_l, self.lut_l)
        matched_r = cv2.LUT(rem_r, self.lut_r)
        return matched_l, matched_r
