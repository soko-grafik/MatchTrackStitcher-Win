"""
MatchTrack-Stitcher 16:9 Tactical Warp Engine.
Transforms a wide curved/trapezoidal soccer pitch defined by 6 points
(TL, TC, TR, BR, BC, BL) into a planar 16:9 full-pitch overview via piecewise dual-quad homography.
"""
from typing import List, Tuple, Optional
import numpy as np
import cv2


def generate_tactical_16x9_luts(src_w: int, 
                                src_h: int, 
                                dst_w: int, 
                                dst_h: int, 
                                pitch_corners: List[List[float]], 
                                margin_percent: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates remap LUTs (map_x, map_y) to transform a 6-point pitch polygon from a panorama
    into a standardized 16:9 frame.
    
    pitch_corners ordering:
    0: Top-Left (TL)
    1: Top-Center (TC - on halfway line)
    2: Top-Right (TR)
    3: Bottom-Right (BR)
    4: Bottom-Center (BC - on halfway line)
    5: Bottom-Left (BL)
    """
    # If 4 points given, convert to 6 points
    if len(pitch_corners) == 4:
        tl, tr, br, bl = pitch_corners
        tc = [(tl[0] + tr[0]) * 0.5, (tl[1] + tr[1]) * 0.5]
        bc = [(bl[0] + br[0]) * 0.5, (bl[1] + br[1]) * 0.5]
        corners = [tl, tc, tr, br, bc, bl]
    elif len(pitch_corners) == 6:
        corners = pitch_corners
    else:
        # Default fallback
        corners = [
            [0.0, 0.05], [0.5, 0.05], [1.0, 0.05],
            [1.0, 0.95], [0.5, 0.95], [0.0, 0.95]
        ]

    # Source pixel coordinates on panorama
    tl_s = np.array([corners[0][0] * src_w, corners[0][1] * src_h], dtype=np.float32)
    tc_s = np.array([corners[1][0] * src_w, corners[1][1] * src_h], dtype=np.float32)
    tr_s = np.array([corners[2][0] * src_w, corners[2][1] * src_h], dtype=np.float32)
    br_s = np.array([corners[3][0] * src_w, corners[3][1] * src_h], dtype=np.float32)
    bc_s = np.array([corners[4][0] * src_w, corners[4][1] * src_h], dtype=np.float32)
    bl_s = np.array([corners[5][0] * src_w, corners[5][1] * src_h], dtype=np.float32)

    # Destination pixel coordinates in 16:9 frame with optional margin
    mx = float(dst_w * (margin_percent / 100.0))
    my = float(dst_h * (margin_percent / 100.0))
    mid_x = dst_w * 0.5

    # Left half destination: [TL, TC, BC, BL]
    dst_left = np.array([
        [mx, my],
        [mid_x, my],
        [mid_x, dst_h - my],
        [mx, dst_h - my]
    ], dtype=np.float32)

    # Left half source: [TL, TC, BC, BL]
    src_left = np.array([tl_s, tc_s, bc_s, bl_s], dtype=np.float32)

    # Right half destination: [TC, TR, BR, BC]
    dst_right = np.array([
        [mid_x, my],
        [dst_w - mx, my],
        [dst_w - mx, dst_h - my],
        [mid_x, dst_h - my]
    ], dtype=np.float32)

    # Right half source: [TC, TR, BR, BC]
    src_right = np.array([tc_s, tr_s, br_s, bc_s], dtype=np.float32)

    # Homography matrices mapping Destination -> Source (for backward lookup in cv2.remap)
    H_left = cv2.getPerspectiveTransform(dst_left, src_left)
    H_right = cv2.getPerspectiveTransform(dst_right, src_right)

    # Generate grid for destination image
    grid_x, grid_y = np.meshgrid(np.arange(dst_w, dtype=np.float32), np.arange(dst_h, dtype=np.float32))
    ones = np.ones_like(grid_x)
    coords_dst = np.stack([grid_x, grid_y, ones], axis=-1)  # (dst_h, dst_w, 3)

    # Transform with H_left
    coords_l = coords_dst @ H_left.T
    map_x_l = (coords_l[..., 0] / (coords_l[..., 2] + 1e-9)).astype(np.float32)
    map_y_l = (coords_l[..., 1] / (coords_l[..., 2] + 1e-9)).astype(np.float32)

    # Transform with H_right
    coords_r = coords_dst @ H_right.T
    map_x_r = (coords_r[..., 0] / (coords_r[..., 2] + 1e-9)).astype(np.float32)
    map_y_r = (coords_r[..., 1] / (coords_r[..., 2] + 1e-9)).astype(np.float32)

    # Seamless linear blend across the center line (16px transition band)
    blend_half_w = 8.0
    weight_right = np.clip((grid_x - (mid_x - blend_half_w)) / (2.0 * blend_half_w), 0.0, 1.0)
    weight_left = 1.0 - weight_right

    map_x = (map_x_l * weight_left + map_x_r * weight_right).astype(np.float32)
    map_y = (map_y_l * weight_left + map_y_r * weight_right).astype(np.float32)

    return map_x, map_y


def generate_tactical_16x9_canvas_luts(src_w: int,
                                       src_h: int,
                                       canvas_w: int,
                                       canvas_h: int,
                                       pitch_corners: List[List[float]]) -> Tuple[np.ndarray, np.ndarray, Tuple[float, float, float, float]]:
    """
    Generates remap LUTs (map_x, map_y) to warp the source panorama across the entire wide viewport canvas,
    such that the region bounded by pitch_corners (TL, TC, TR, BR, BC, BL) maps exactly into a centered 16:9 frame.
    
    Returns: (map_x, map_y, (frame_x0, frame_y0, frame_w, frame_h))
    """
    if len(pitch_corners) == 4:
        tl, tr, br, bl = pitch_corners
        tc = [(tl[0] + tr[0]) * 0.5, (tl[1] + tr[1]) * 0.5]
        bc = [(bl[0] + br[0]) * 0.5, (bl[1] + br[1]) * 0.5]
        corners = [tl, tc, tr, br, bc, bl]
    elif len(pitch_corners) == 6:
        corners = pitch_corners
    else:
        corners = [
            [0.0, 0.05], [0.5, 0.05], [1.0, 0.05],
            [1.0, 0.95], [0.5, 0.95], [0.0, 0.95]
        ]

    # Source pixel coordinates on panorama
    tl_s = np.array([corners[0][0] * src_w, corners[0][1] * src_h], dtype=np.float32)
    tc_s = np.array([corners[1][0] * src_w, corners[1][1] * src_h], dtype=np.float32)
    tr_s = np.array([corners[2][0] * src_w, corners[2][1] * src_h], dtype=np.float32)
    br_s = np.array([corners[3][0] * src_w, corners[3][1] * src_h], dtype=np.float32)
    bc_s = np.array([corners[4][0] * src_w, corners[4][1] * src_h], dtype=np.float32)
    bl_s = np.array([corners[5][0] * src_w, corners[5][1] * src_h], dtype=np.float32)

    # 16:9 Target Frame coordinates centered in canvas
    frame_h = float(canvas_h)
    frame_w = frame_h * (16.0 / 9.0)
    frame_x0 = (float(canvas_w) - frame_w) * 0.5
    frame_x1 = frame_x0 + frame_w
    mid_x = float(canvas_w) * 0.5

    # Left half destination: [TL, TC, BC, BL] mapped to [frame_x0, 0], [mid_x, 0], [mid_x, canvas_h], [frame_x0, canvas_h]
    dst_left = np.array([
        [frame_x0, 0.0],
        [mid_x, 0.0],
        [mid_x, frame_h],
        [frame_x0, frame_h]
    ], dtype=np.float32)
    src_left = np.array([tl_s, tc_s, bc_s, bl_s], dtype=np.float32)

    # Right half destination: [TC, TR, BR, BC] mapped to [mid_x, 0], [frame_x1, 0], [frame_x1, canvas_h], [mid_x, canvas_h]
    dst_right = np.array([
        [mid_x, 0.0],
        [frame_x1, 0.0],
        [frame_x1, frame_h],
        [mid_x, frame_h]
    ], dtype=np.float32)
    src_right = np.array([tc_s, tr_s, br_s, bc_s], dtype=np.float32)

    # Homography matrices mapping Destination -> Source (for backward lookup in cv2.remap)
    H_left = cv2.getPerspectiveTransform(dst_left, src_left)
    H_right = cv2.getPerspectiveTransform(dst_right, src_right)

    # Generate grid for entire destination canvas
    grid_x, grid_y = np.meshgrid(np.arange(canvas_w, dtype=np.float32), np.arange(canvas_h, dtype=np.float32))
    ones = np.ones_like(grid_x)
    coords_dst = np.stack([grid_x, grid_y, ones], axis=-1)

    # Transform with H_left
    coords_l = coords_dst @ H_left.T
    map_x_l = (coords_l[..., 0] / (coords_l[..., 2] + 1e-9)).astype(np.float32)
    map_y_l = (coords_l[..., 1] / (coords_l[..., 2] + 1e-9)).astype(np.float32)

    # Transform with H_right
    coords_r = coords_dst @ H_right.T
    map_x_r = (coords_r[..., 0] / (coords_r[..., 2] + 1e-9)).astype(np.float32)
    map_y_r = (coords_r[..., 1] / (coords_r[..., 2] + 1e-9)).astype(np.float32)

    # Smooth linear blend across the center line (24px transition band)
    blend_half_w = 12.0
    weight_right = np.clip((grid_x - (mid_x - blend_half_w)) / (2.0 * blend_half_w), 0.0, 1.0)
    weight_left = 1.0 - weight_right

    map_x = (map_x_l * weight_left + map_x_r * weight_right).astype(np.float32)
    map_y = (map_y_l * weight_left + map_y_r * weight_right).astype(np.float32)

    return map_x, map_y, (frame_x0, 0.0, frame_w, frame_h)


def apply_tactical_16x9_warp(frame: np.ndarray, 
                             map_x: np.ndarray, 
                             map_y: np.ndarray) -> np.ndarray:
    """Applies the 16:9 tactical rectification warp via cv2.remap."""
    return cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

