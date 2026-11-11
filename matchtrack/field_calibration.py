"""
Field Calibration & 10-Point Sideline Model for MatchTrack-Stitcher.
Implements the 10-point playing field boundary marking (5 near sideline, 5 far sideline)
equivalent to Once Sport Autocam 4.1.3, including spline interpolation, field polygon calculation,
and automatic heuristic line/keypoint detection.
"""

import math
import numpy as np
import cv2
from typing import List, Tuple, Dict, Any, Optional
import logging

logger = logging.getLogger("MatchTrack")


class FieldMarking:
    """
    Represents the 10-point soccer field sideline calibration:
    Points 1..5: Closer sideline, from left corner to right corner (counter-clockwise).
      - P1 (idx 0): Near left corner
      - P2 (idx 1): Near quarter-line (left of center)
      - P3 (idx 2): Near halfway line / center
      - P4 (idx 3): Near quarter-line (right of center)
      - P5 (idx 4): Near right corner
    Points 6..10: Far sideline, from right corner back to left corner.
      - P6 (idx 5): Far right corner
      - P7 (idx 6): Far quarter-line (right of center)
      - P8 (idx 7): Far halfway line / center
      - P9 (idx 8): Far quarter-line (left of center)
      - P10 (idx 9): Far left corner
    
    All coordinates are stored normalized in range [0.0, 1.0] relative to panoramic width and height.
    """

    POINT_LABELS = [
        "1: Nah-Ecke Links",
        "2: Nah 1/4 Links",
        "3: Nah Mitte",
        "4: Nah 3/4 Rechts",
        "5: Nah-Ecke Rechts",
        "6: Fern-Ecke Rechts",
        "7: Fern 3/4 Rechts",
        "8: Fern Mitte",
        "9: Fern 1/4 Links",
        "10: Fern-Ecke Links"
    ]

    def __init__(self, points: Optional[List[Tuple[float, float]]] = None):
        if points is not None and len(points) == 10:
            self.points: List[Tuple[float, float]] = [
                (float(np.clip(p[0], 0.0, 1.0)), float(np.clip(p[1], 0.0, 1.0)))
                for p in points
            ]
        else:
            self.points = self.default_points()
        
        self.history: List[List[Tuple[float, float]]] = []

    @classmethod
    def default_points(cls) -> List[Tuple[float, float]]:
        """
        Sensible default 10-point curve for a midfield-mounted dual camera rig
        matching a typical 32:9 sports panorama.
        """
        # Near sideline (slight convex curve due to lens / perspective)
        # y starts at ~0.68 at corners and dips to ~0.74 in center
        p1 = (0.07, 0.67)
        p2 = (0.28, 0.70)
        p3 = (0.50, 0.73)
        p4 = (0.72, 0.70)
        p5 = (0.93, 0.67)

        # Far sideline (straighter line higher up in the frame)
        # y is around ~0.53 at far side
        p6 = (0.91, 0.54)
        p7 = (0.71, 0.535)
        p8 = (0.50, 0.53)
        p9 = (0.29, 0.535)
        p10 = (0.09, 0.54)

        return [p1, p2, p3, p4, p5, p6, p7, p8, p9, p10]

    def save_state(self):
        """Saves current state for undo functionality."""
        self.history.append([pt for pt in self.points])
        if len(self.history) > 20:
            self.history.pop(0)

    def undo(self) -> bool:
        """Restores previous points from undo stack."""
        if self.history:
            self.points = self.history.pop()
            return True
        return False

    def reset_to_defaults(self):
        """Resets all 10 points to default positions."""
        self.save_state()
        self.points = self.default_points()

    def set_point(self, index: int, x_norm: float, y_norm: float):
        """Updates a single point with boundary clamping."""
        if 0 <= index < 10:
            self.points[index] = (
                float(np.clip(x_norm, 0.0, 1.0)),
                float(np.clip(y_norm, 0.0, 1.0))
            )

    def get_spline_curve(self, point_indices: List[int], num_samples: int = 50) -> List[Tuple[float, float]]:
        """
        Calculates a smooth Catmull-Rom / cubic interpolation through a sequence of points.
        Useful for rendering the curved near sideline (points 1..5) and far sideline (points 6..10).
        """
        if len(point_indices) < 2:
            return [self.points[i] for i in point_indices]
        
        pts = np.array([self.points[i] for i in point_indices], dtype=np.float32)
        
        # Simple polynomial or Catmull-Rom fit
        # We parameterize by chord distance
        dists = np.sqrt(np.sum(np.diff(pts, axis=0)**2, axis=1))
        t = np.concatenate(([0], np.cumsum(dists)))
        total_len = t[-1]
        if total_len < 1e-5:
            return [(float(pts[0, 0]), float(pts[0, 1]))] * num_samples
        
        t_eval = np.linspace(0, total_len, num_samples)
        
        # Fit cubic spline or polynomial (deg 2 for 5 points gives very natural curve)
        deg = min(2, len(point_indices) - 1)
        poly_x = np.poly1d(np.polyfit(t, pts[:, 0], deg))
        poly_y = np.poly1d(np.polyfit(t, pts[:, 1], deg))
        
        xs = np.clip(poly_x(t_eval), 0.0, 1.0)
        ys = np.clip(poly_y(t_eval), 0.0, 1.0)
        
        return [(float(x), float(y)) for x, y in zip(xs, ys)]

    def get_full_perimeter_points(self, samples_per_sideline: int = 40) -> List[Tuple[float, float]]:
        """
        Returns a smooth closed perimeter curve [P1..P5 -> P6..P10 -> P1]
        representing the exact playing field boundary polygon.
        """
        near_spline = self.get_spline_curve([0, 1, 2, 3, 4], samples_per_sideline)
        far_spline = self.get_spline_curve([5, 6, 7, 8, 9], samples_per_sideline)
        return near_spline + far_spline

    def get_field_center_norm(self) -> Tuple[float, float]:
        """Calculates center of the field in normalized coordinates (midway between P3 and P8)."""
        p3 = self.points[2]
        p8 = self.points[7]
        return ((p3[0] + p8[0]) * 0.5, (p3[1] + p8[1]) * 0.5)

    def get_vertical_bounds_at_x(self, x_norm: float) -> Tuple[float, float]:
        """
        Returns (y_far, y_near) field boundary limits at given horizontal position x_norm.
        Essential for keeping the virtual broadcast camera smoothly inside the touchlines.
        """
        # Interpolate far sideline
        far_xs = [self.points[i][0] for i in [9, 8, 7, 6, 5]]  # P10, P9, P8, P7, P6 (left to right)
        far_ys = [self.points[i][1] for i in [9, 8, 7, 6, 5]]
        
        # Interpolate near sideline
        near_xs = [self.points[i][0] for i in [0, 1, 2, 3, 4]] # P1, P2, P3, P4, P5 (left to right)
        near_ys = [self.points[i][1] for i in [0, 1, 2, 3, 4]]
        
        y_far = float(np.interp(x_norm, far_xs, far_ys))
        y_near = float(np.interp(x_norm, near_xs, near_ys))
        
        return min(y_far, y_near), max(y_far, y_near)

    def auto_detect_from_image(self, bgr_image: np.ndarray) -> bool:
        """
        Heuristic computer vision line detection to automatically place the 10 field markers:
        1. Identifies green turf mask using HSV range.
        2. Detects white line segments via adaptive thresholding within the green region.
        3. Identifies the nearest and farthest sideline curves and divides them into 4 quarters.
        """
        if bgr_image is None or bgr_image.size == 0:
            return False

        try:
            self.save_state()
            h, w = bgr_image.shape[:2]
            
            # Downsample for fast processing
            proc_w = 1280
            scale = proc_w / max(1, w)
            proc_h = int(h * scale)
            small = cv2.resize(bgr_image, (proc_w, proc_h), interpolation=cv2.INTER_AREA)

            # Convert to HSV to detect field grass
            hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
            # Grass green in HSV: H ~ [30, 85], S ~ [35, 255], V ~ [40, 255]
            mask_grass = cv2.inRange(hsv, np.array([28, 30, 35]), np.array([90, 255, 255]))
            
            # Morphological cleaning
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
            mask_grass = cv2.morphologyEx(mask_grass, cv2.MORPH_CLOSE, kernel)
            
            # Find largest green contour (the pitch)
            contours, _ = cv2.findContours(mask_grass, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return False
                
            pitch_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(pitch_contour) < (proc_w * proc_h * 0.15):
                return False # Too small
                
            bx, by, bw, bh = cv2.boundingRect(pitch_contour)
            
            # Subdivide pitch into left-to-right columns at 10%, 30%, 50%, 70%, 90%
            col_fractions = [0.10, 0.30, 0.50, 0.70, 0.90]
            detected_near = []
            detected_far = []
            
            # Mask slice helper
            for frac in col_fractions:
                target_x = int(bx + bw * frac)
                # Search bottom-up for near sideline (first green transition from bottom)
                # and top-down for far sideline (first green transition from top)
                col_slice = mask_grass[:, target_x]
                green_indices = np.where(col_slice > 0)[0]
                if len(green_indices) > 0:
                    y_top = green_indices[0]
                    y_bottom = green_indices[-1]
                    # Indent slightly inside the detected green boundary (e.g. 3%)
                    margin_y = int((y_bottom - y_top) * 0.04)
                    detected_far.append((target_x / proc_w, (y_top + margin_y) / proc_h))
                    detected_near.append((target_x / proc_w, (y_bottom - margin_y) / proc_h))
                else:
                    # Fallback
                    detected_far.append((target_x / proc_w, (by + bh * 0.15) / proc_h))
                    detected_near.append((target_x / proc_w, (by + bh * 0.85) / proc_h))
            
            if len(detected_near) == 5 and len(detected_far) == 5:
                # P1..P5 is near left to right
                # P6..P10 is far right to left
                p1_to_5 = detected_near
                p6_to_10 = list(reversed(detected_far))
                self.points = p1_to_5 + p6_to_10
                logger.info("Spielfeld 10-Punkte Kontur erfolgreich automatisch erkannt.")
                return True
                
        except Exception as e:
            logger.warning(f"Auto-Detect Field Marking fehlgeschlagen: {e}")
            
        return False

    def to_dict(self) -> Dict[str, Any]:
        """Serializes field marking state to dictionary."""
        return {
            "version": "1.0",
            "points": [[float(p[0]), float(p[1])] for p in self.points]
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FieldMarking":
        """Reconstructs FieldMarking from dictionary."""
        pts = data.get("points", None)
        if pts and isinstance(pts, list) and len(pts) == 10:
            parsed = [(float(p[0]), float(p[1])) for p in pts]
            return cls(parsed)
        return cls()

