"""
Look-Up Table (LUT) & Remap Map Generator.
Computes vectorized 2D pixel coordinate maps (MapX, MapY) and smooth alpha-blend weights.
Once generated, OpenCV / GPU shaders can remap 4K video frames in milliseconds.
"""
from dataclasses import dataclass
import numpy as np
import cv2
from typing import Tuple, Dict, Any, Optional
from .rig_geometry import RigConfiguration, CameraPose
from .camera_model import CameraIntrinsics


@dataclass
class RemapLUTs:
    """Container for precalculated remapping maps and blending weights."""
    out_width: int
    out_height: int
    # Left camera coordinate maps (float32 for cv2.remap)
    map_x_left: np.ndarray
    map_y_left: np.ndarray
    mask_left: np.ndarray       # boolean mask of valid pixels
    
    # Right camera coordinate maps
    map_x_right: np.ndarray
    map_y_right: np.ndarray
    mask_right: np.ndarray      # boolean mask of valid pixels
    
    # Blending weights [0.0 ... 1.0] for left and right
    weight_left: np.ndarray     # float32 [H, W, 1]
    weight_right: np.ndarray    # float32 [H, W, 1]

    # Optimized partitioned sub-maps for ultra-fast remapping
    seam_x_start: Optional[int] = None
    seam_x_end: Optional[int] = None
    map_xl_sub: Optional[np.ndarray] = None
    map_yl_sub: Optional[np.ndarray] = None
    map_xr_sub: Optional[np.ndarray] = None
    map_yr_sub: Optional[np.ndarray] = None
    weight_seam_l: Optional[np.ndarray] = None
    weight_seam_r: Optional[np.ndarray] = None
    smart_seam_path: Optional[np.ndarray] = None

    def update_smart_seam(self, frame_left: np.ndarray, frame_right: np.ndarray, rig: RigConfiguration) -> bool:
        """
        Dynamically routes the seam around obstacles (players, coaches) in the overlap zone
        using content-aware dynamic programming. Updates blending weights in-place with zero allocation.
        """
        if self.seam_x_start is None or self.seam_x_end is None or self.map_xl_sub is None or self.map_xr_sub is None:
            return False
        if frame_left is None or frame_right is None:
            return False

        x0 = self.seam_x_start
        x1 = self.seam_x_end
        sub_w = x1 - x0
        if sub_w < 8:
            return False

        try:
            # Remap sub-strips in the overlap region
            rem_l_sub = cv2.remap(
                frame_left, self.map_xl_sub, self.map_yl_sub,
                interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
            )
            rem_r_sub = cv2.remap(
                frame_right, self.map_xr_sub, self.map_yr_sub,
                interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
            )

            strip_l = rem_l_sub[:, x0:x1]
            strip_r = rem_r_sub[:, :sub_w]

            hfov_rad = np.radians(rig.pano_hfov)
            r_yaw = np.radians(rig.global_yaw_center)
            seam_center_angle = -r_yaw + np.radians(getattr(rig, 'seam_angle_offset', 0.0))
            center_x = int(self.out_width * 0.5 + (seam_center_angle / (hfov_rad * 0.5)) * (self.out_width * 0.5))
            center_x = max(x0 + 4, min(x1 - 5, center_x))

            path_full = compute_smart_seam_path(strip_l, strip_r, x0, x1, center_x)
            self.smart_seam_path = path_full

            blend_rad = np.radians(max(rig.blend_width_deg, 1.0))
            blend_px = max(6.0, float(self.out_width) * (blend_rad / max(1e-4, hfov_rad)))

            xs = np.arange(x0, x1, dtype=np.float32)[np.newaxis, :]
            centers = path_full[:, np.newaxis].astype(np.float32)

            t = (xs - (centers - blend_px * 0.5)) / blend_px
            t = np.clip(t, 0.0, 1.0)
            w_r_strip = (3.0 * t * t - 2.0 * t * t * t)[:, :, np.newaxis]
            w_l_strip = 1.0 - w_r_strip

            mask_l_sub = (self.map_xl_sub[:, x0:x1] >= 0) & (self.map_yl_sub[:, x0:x1] >= 0)
            mask_r_sub = (self.map_xr_sub[:, :sub_w] >= 0) & (self.map_yr_sub[:, :sub_w] >= 0)
            valid_both = mask_l_sub & mask_r_sub

            w_l_final = np.where(valid_both[:, :, np.newaxis], w_l_strip, np.where(mask_l_sub[:, :, np.newaxis], 1.0, 0.0))
            w_r_final = np.where(valid_both[:, :, np.newaxis], w_r_strip, np.where(mask_r_sub[:, :, np.newaxis], 1.0, 0.0))
            tot = w_l_final + w_r_final
            w_l_final = np.where(tot > 0.0, w_l_final / np.maximum(tot, 1e-5), 0.5)
            w_r_final = np.where(tot > 0.0, w_r_final / np.maximum(tot, 1e-5), 0.5)

            self.weight_seam_l = np.ascontiguousarray(w_l_final.astype(np.float32))
            self.weight_seam_r = np.ascontiguousarray(w_r_final.astype(np.float32))
            self.weight_left[:, x0:x1] = self.weight_seam_l
            self.weight_right[:, x0:x1] = self.weight_seam_r
            return True
        except Exception:
            return False


def compute_smart_seam_path(
    strip_l: np.ndarray,
    strip_r: np.ndarray,
    overlap_x0: int,
    overlap_x1: int,
    center_x: int,
    center_weight: float = 30.0
) -> np.ndarray:
    """
    Computes an optimal seam path x*(y) through the overlap region using Dynamic Programming.
    Avoids cutting through high-contrast objects (like players or coaches) by routing through
    uniform regions (such as grass).
    """
    H = strip_l.shape[0]
    w_ov = overlap_x1 - overlap_x0
    if w_ov < 4:
        return np.full(H, center_x, dtype=np.int32)

    scale_y = 270.0 / float(H) if H > 270 else 1.0
    h_small = max(30, int(H * scale_y))
    w_small = w_ov

    strip_l_small = cv2.resize(strip_l, (w_small, h_small), interpolation=cv2.INTER_AREA)
    strip_r_small = cv2.resize(strip_r, (w_small, h_small), interpolation=cv2.INTER_AREA)

    diff = np.mean(np.abs(strip_l_small.astype(np.float32) - strip_r_small.astype(np.float32)), axis=2)

    gray_l = cv2.cvtColor(strip_l_small, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(strip_r_small, cv2.COLOR_BGR2GRAY)
    edge_l = cv2.Sobel(gray_l, cv2.CV_32F, 1, 0)**2 + cv2.Sobel(gray_l, cv2.CV_32F, 0, 1)**2
    edge_r = cv2.Sobel(gray_r, cv2.CV_32F, 1, 0)**2 + cv2.Sobel(gray_r, cv2.CV_32F, 0, 1)**2
    edge_cost = np.sqrt(edge_l) + np.sqrt(edge_r)

    local_center = center_x - overlap_x0
    xs = np.arange(w_small, dtype=np.float32)
    dist_penalty = center_weight * ((xs - local_center) / float(max(1, w_small))) ** 2
    cost = diff * 1.5 + edge_cost * 0.8 + dist_penalty[np.newaxis, :]

    M = np.zeros_like(cost)
    M[0, :] = cost[0, :]
    backtrack = np.zeros((h_small, w_small), dtype=np.int32)

    for y in range(1, h_small):
        m_prev = M[y - 1, :]
        left = np.pad(m_prev[:-1], (1, 0), mode='edge')
        center = m_prev
        right = np.pad(m_prev[1:], (0, 1), mode='edge')

        stacked = np.stack([left, center, right], axis=0)
        min_idx = np.argmin(stacked, axis=0) - 1
        min_val = np.min(stacked, axis=0)

        M[y, :] = cost[y, :] + min_val
        prev_x = np.clip(np.arange(w_small) + min_idx, 0, w_small - 1)
        backtrack[y, :] = prev_x

    path_small = np.zeros(h_small, dtype=np.int32)
    best_x = int(np.argmin(M[-1, :]))
    path_small[-1] = best_x
    for y in range(h_small - 1, 0, -1):
        best_x = backtrack[y, best_x]
        path_small[y - 1] = best_x

    y_coords_orig = np.linspace(0, h_small - 1, H)
    path_orig_local = np.interp(y_coords_orig, np.arange(h_small), path_small)

    kernel_size = max(5, int(H * 0.05))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = np.ones(kernel_size) / float(kernel_size)
    padded = np.pad(path_orig_local, (kernel_size // 2, kernel_size // 2), mode='edge')
    smoothed_local = np.convolve(padded, kernel, mode='valid')[:H]

    path_full = (overlap_x0 + smoothed_local).astype(np.int32)
    return np.clip(path_full, overlap_x0, overlap_x1 - 1)


def _project_rays_to_camera(rays_global: np.ndarray, 
                             camera_pose: CameraPose, 
                             intrinsics: CameraIntrinsics,
                             r_global_level: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Projects 3D world rays into camera sensor pixel coordinates (u, v).
    rays_global: shape [H, W, 3] normalized direction vectors.
    """
    # 1. Apply global rig leveling
    # rays_level = rays_global @ r_global_level.T
    H, W, _ = rays_global.shape
    rays_flat = rays_global.reshape(-1, 3)
    rays_level = rays_flat @ r_global_level.T

    # 2. Transform into camera coordinate frame: v_c = R_cam^T * v_level
    R_cam = camera_pose.rotation_matrix()
    rays_cam = rays_level @ R_cam # since (R_cam^T @ v) == v @ R_cam

    xc = rays_cam[:, 0]
    yc = rays_cam[:, 1]
    zc = rays_cam[:, 2]

    # Valid points are strictly in front of the lens
    valid = zc > 0.05

    u = np.zeros(rays_flat.shape[0], dtype=np.float32)
    v = np.zeros(rays_flat.shape[0], dtype=np.float32)

    if intrinsics.model_type == "fisheye":
        # Kannala-Brandt / Fisheye projection
        rho = np.sqrt(xc * xc + yc * yc)
        rho_safe = np.maximum(rho, 1e-7)
        theta = np.arctan2(rho, zc)
        
        # Polynomial distortion
        th2 = theta * theta
        th4 = th2 * th2
        th6 = th4 * th2
        th8 = th4 * th4
        theta_d = theta * (1.0 + intrinsics.k1 * th2 + intrinsics.k2 * th4 + intrinsics.k3 * th6 + intrinsics.k4 * th8)
        
        scale = theta_d / rho_safe
        u_calc = intrinsics.fx * (xc * scale) + intrinsics.cx
        v_calc = intrinsics.fy * (yc * scale) + intrinsics.cy
    else:
        # Pinhole / standard rectilinear
        zc_safe = np.maximum(zc, 1e-5)
        xn = xc / zc_safe
        yn = yc / zc_safe
        r2 = xn * xn + yn * yn
        radial = 1.0 + intrinsics.k1 * r2 + intrinsics.k2 * (r2 * r2)
        u_calc = intrinsics.fx * (xn * radial) + intrinsics.cx
        v_calc = intrinsics.fy * (yn * radial) + intrinsics.cy

    u[valid] = u_calc[valid]
    v[valid] = v_calc[valid]

    # 3. Apply interactive 4-corner perspective distortion (Photoshop-style Corner Pinning)
    if camera_pose.is_corners_modified():
        w_cam = float(intrinsics.image_width)
        h_cam = float(intrinsics.image_height)
        src_pts = np.array([
            [0.0, 0.0],
            [w_cam, 0.0],
            [w_cam, h_cam],
            [0.0, h_cam]
        ], dtype=np.float32)
        dst_pts = np.array([
            [camera_pose.corners[0][0] * w_cam, camera_pose.corners[0][1] * h_cam],
            [camera_pose.corners[1][0] * w_cam, camera_pose.corners[1][1] * h_cam],
            [camera_pose.corners[2][0] * w_cam, camera_pose.corners[2][1] * h_cam],
            [camera_pose.corners[3][0] * w_cam, camera_pose.corners[3][1] * h_cam]
        ], dtype=np.float32)
        
        # H_inv maps destination distorted quad coordinates back to source sensor pixels
        try:
            h_inv = cv2.getPerspectiveTransform(dst_pts, src_pts)
            if np.any(valid):
                pts_valid = np.stack([u[valid], v[valid], np.ones(np.count_nonzero(valid), dtype=np.float32)], axis=-1)
                warped = pts_valid @ h_inv.T
                denom = np.where(np.abs(warped[:, 2]) < 1e-7, 1e-7, warped[:, 2])
                u[valid] = warped[:, 0] / denom
                v[valid] = warped[:, 1] / denom
        except Exception:
            pass

    # Clamp coordinates to ensure remap never samples outside valid image bounds
    u_clamped = np.clip(u, 0.0, float(intrinsics.image_width - 1.0))
    v_clamped = np.clip(v, 0.0, float(intrinsics.image_height - 1.0))

    # Check bounds within image dimensions
    in_bounds = valid & (u >= 0.0) & (u <= (intrinsics.image_width - 1.0)) & \
                        (v >= 0.0) & (v <= (intrinsics.image_height - 1.0))

    map_x = u_clamped.reshape(H, W).astype(np.float32)
    map_y = v_clamped.reshape(H, W).astype(np.float32)
    mask = in_bounds.reshape(H, W)

    return map_x, map_y, mask


def find_largest_interior_rectangle(combined_mask: np.ndarray, mask_l: np.ndarray, mask_r: np.ndarray, target_aspect: float = 32.0 / 9.0, safety_margin: float = 0.015) -> Tuple[int, int, int, int]:
    """
    Computes the Largest Inscribed Rectangle (LIR) with specified target aspect ratio (e.g. 32:9 or 21:10)
    that contains 100% valid camera pixels (zero black borders / arcs)
    AND guarantees BOTH left and right cameras are spanned across the center seam.
    Returns (x, y, width, height).
    """
    H, W = combined_mask.shape
    mask_u8 = combined_mask.astype(np.uint8)
    integral = cv2.integral(mask_u8) # Integral image for O(1) area validation

    best_box = (0, 0, W, H)
    max_h = int(min(H, W / target_aspect))
    min_h = int(max_h * 0.35)
    
    step_y = max(1, H // 140)
    step_x = max(1, W // 180)

    center_seam_x = W // 2
    found = False

    for h in range(max_h, min_h, -step_y):
        w = int(round(h * target_aspect))
        if w > W:
            continue
        
        target_sum = w * h
        center_x = (W - w) // 2
        # Prioritize searching symmetric boxes centered on the middle seam
        x_offsets = sorted(range(0, W - w + 1, step_x), key=lambda x: abs(x - center_x))

        for y in range(0, H - h + 1, step_y):
            for x in x_offsets:
                area_sum = integral[y + h, x + w] - integral[y, x + w] - integral[y + h, x] + integral[y, x]
                if area_sum == target_sum:
                    # Verify BOTH left and right camera feeds are present in the box
                    sub_l = mask_l[y:y+h, x:x+w]
                    sub_r = mask_r[y:y+h, x:x+w]
                    if np.any(sub_l) and np.any(sub_r):
                        best_box = (x, y, w, h)
                        found = True
                        break
            if found:
                break
        if found:
            break


    if not found:
        # Fallback to full frame if cameras are rotated too far apart
        return 0, 0, W, H

    if safety_margin > 0.0:
        bx, by, bw, bh = best_box
        mx = max(1, int(bw * safety_margin))
        my = max(1, int(bh * safety_margin))
        best_box = (bx + mx, by + my, bw - 2 * mx, bh - 2 * my)

    return best_box


# Backward compatibility alias
find_largest_interior_rectangle_32x9 = find_largest_interior_rectangle


def generate_remap_luts(
    rig: RigConfiguration,
    out_width: int,
    out_height: int,
    sample_frame_left: Optional[np.ndarray] = None,
    sample_frame_right: Optional[np.ndarray] = None
) -> RemapLUTs:
    """
    Generates high-performance remapping coordinates and seam blending weights for a target resolution.
    If rig.auto_crop_lir is True, applies Largest Interior Rectangle cropping to eliminate all black curved borders.
    Dynamically adjusts horizontal squeeze for target aspect ratios (e.g. 32:9, 21:10) to evenly compress feeds.
    """
    # 1. Meshgrid for the panorama canvas
    xs = np.linspace(0, out_width - 1, out_width, dtype=np.float32)
    ys = np.linspace(0, out_height - 1, out_height, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xs, ys)

    # 2. Cylindrical Panorama Ray Casting
    hfov_rad = np.radians(rig.pano_hfov)
    lambda_angles = (grid_x - (out_width * 0.5)) / (out_width * 0.5) * (hfov_rad * 0.5)
    
    # Cylindrical focal length
    f_pano = (out_width * 0.5) / np.tan(hfov_rad * 0.5)
    
    # Anamorphic horizontal squeeze factor adapted to target aspect ratio (8:3 for 4:3 cameras, 32:9 for 16:9 cameras, 21:10, etc.)
    aspect = float(out_width) / max(1.0, float(out_height))
    cam = getattr(rig, 'left_camera', None)
    cam_w = getattr(cam, 'image_width', 1920)
    cam_h = getattr(cam, 'image_height', 1080)
    cam_ar = float(cam_w) / max(1.0, float(cam_h))
    base_aspect = (8.0 / 3.0) if cam_ar < 1.55 else (32.0 / 9.0)
    aspect_ratio_scale = base_aspect / aspect if aspect > 0.1 else 1.0

    raw_squeeze = getattr(rig, 'horizontal_squeeze', 1.0)
    raw_squeeze = max(0.2, min(5.0, float(raw_squeeze)))
    effective_squeeze = raw_squeeze * aspect_ratio_scale
    effective_squeeze = max(0.2, min(10.0, float(effective_squeeze)))

    f_pano_v = f_pano / effective_squeeze

    # Vertical normalized height with vertical framing offset
    y_center = out_height * 0.5 + rig.vertical_crop_offset * (out_height * 0.35)
    h_heights = (grid_y - y_center) / f_pano_v

    # Apply field curvature compensation to straighten ground lines that bend due to downward camera pitch
    curv = float(getattr(rig, 'field_curvature_correction', 0.0))
    if abs(curv) > 1e-4:
        h_heights = h_heights + curv * (1.0 - np.cos(lambda_angles))

    if rig.projection_type == "cylindrical":
        Xg = np.sin(lambda_angles)
        Yg = h_heights
        Zg = np.cos(lambda_angles)
    elif rig.projection_type == "pannini":
        d = 1.0
        cos_lam = np.cos(lambda_angles)
        sin_lam = np.sin(lambda_angles)
        S = (d + 1.0) / (d + cos_lam)
        Xg = sin_lam * S
        Yg = h_heights
        Zg = (cos_lam + d) * S - d
    else: # Rectilinear
        Xg = np.tan(lambda_angles)
        Yg = h_heights
        Zg = np.ones_like(lambda_angles)

    norm = np.sqrt(Xg * Xg + Yg * Yg + Zg * Zg)
    norm = np.maximum(norm, 1e-7)
    rays_global = np.stack([Xg / norm, Yg / norm, Zg / norm], axis=-1)

    # 3. Global Rig Leveling Matrix (Counteract downward tilt and roll)
    r_pitch = np.radians(rig.global_pitch_correction)
    r_roll = np.radians(rig.global_roll_correction)
    r_yaw = np.radians(rig.global_yaw_center)

    Rx = np.array([[1, 0, 0], [0, np.cos(r_pitch), -np.sin(r_pitch)], [0, np.sin(r_pitch), np.cos(r_pitch)]], dtype=np.float64)
    Rz = np.array([[np.cos(r_roll), -np.sin(r_roll), 0], [np.sin(r_roll), np.cos(r_roll), 0], [0, 0, 1]], dtype=np.float64)
    Ry = np.array([[np.cos(r_yaw), 0, np.sin(r_yaw)], [0, 1, 0], [-np.sin(r_yaw), 0, np.cos(r_yaw)]], dtype=np.float64)
    R_global = Rz @ Rx @ Ry

    # 4. Project into Left Camera
    map_x_l, map_y_l, mask_l = _project_rays_to_camera(rays_global, rig.left_pose, rig.left_camera, R_global)

    # 4b. Right Camera with Bottom-Area Parallax / Taper Compensation
    taper_x = float(getattr(rig, 'bottom_seam_taper_x', 0.0))
    taper_y = float(getattr(rig, 'bottom_seam_taper_y', 0.0))
    if abs(taper_x) > 1e-4 or abs(taper_y) > 1e-4:
        # Normalized Y from 0.0 (top) to 1.0 (bottom)
        y_norm = np.clip(grid_y / max(1.0, float(out_height - 1)), 0.0, 1.0)
        # Top 25% is completely untouched (weight = 0.0). Below that, quadratic onset:
        w_bottom = np.maximum(0.0, (y_norm - 0.25) / 0.75) ** 2

        lam_r = lambda_angles + w_bottom * np.radians(taper_x)
        h_heights_r = h_heights + w_bottom * np.radians(taper_y) * (f_pano / max(1e-4, f_pano_v))

        if rig.projection_type == "cylindrical":
            Xg_r = np.sin(lam_r)
            Yg_r = h_heights_r
            Zg_r = np.cos(lam_r)
        elif rig.projection_type == "pannini":
            cos_lam_r = np.cos(lam_r)
            sin_lam_r = np.sin(lam_r)
            d = 1.0
            S_r = (d + 1.0) / (d + cos_lam_r)
            Xg_r = sin_lam_r * S_r
            Yg_r = h_heights_r
            Zg_r = (cos_lam_r + d) * S_r - d
        else: # Rectilinear
            Xg_r = np.tan(lam_r)
            Yg_r = h_heights_r
            Zg_r = np.ones_like(lam_r)

        norm_r = np.maximum(np.sqrt(Xg_r * Xg_r + Yg_r * Yg_r + Zg_r * Zg_r), 1e-7)
        rays_global_r = np.stack([Xg_r / norm_r, Yg_r / norm_r, Zg_r / norm_r], axis=-1)
    else:
        rays_global_r = rays_global

    map_x_r, map_y_r, mask_r = _project_rays_to_camera(rays_global_r, rig.right_pose, rig.right_camera, R_global)

    # 5. Seamless Multi-Band / Linear Feathering Blending Weights
    overlap = mask_l & mask_r
    weight_l = np.zeros((out_height, out_width, 1), dtype=np.float32)
    weight_r = np.zeros((out_height, out_width, 1), dtype=np.float32)

    # Left only
    weight_l[mask_l & (~mask_r)] = 1.0
    # Right only
    weight_r[mask_r & (~mask_l)] = 1.0

    # Overlap Region Smooth Transition (dynamically tracks yaw offset)
    if np.any(overlap):
        blend_rad = np.radians(max(rig.blend_width_deg, 1.0))
        seam_center_angle = -r_yaw + np.radians(getattr(rig, 'seam_angle_offset', 0.0))
        t = (lambda_angles - (seam_center_angle - blend_rad * 0.5)) / blend_rad
        t = np.clip(t, 0.0, 1.0)
        # Smoothstep curve: 3*t^2 - 2*t^3
        w_right_blend = (3.0 * t * t - 2.0 * t * t * t)[:, :, np.newaxis]
        w_left_blend = 1.0 - w_right_blend

        weight_l[overlap] = w_left_blend[overlap]
        weight_r[overlap] = w_right_blend[overlap]

    # Normalize weights
    total_w = weight_l + weight_r
    weight_l = np.where(total_w > 0.0, weight_l / np.maximum(total_w, 1e-5), 0.5)
    weight_r = np.where(total_w > 0.0, weight_r / np.maximum(total_w, 1e-5), 0.5)

    # 6. Largest Interior Rectangle (LIR) Auto-Crop
    if rig.auto_crop_lir:
        combined_mask = mask_l | mask_r
        if not np.all(combined_mask):
            cx, cy, cw, ch = find_largest_interior_rectangle(
                combined_mask,
                mask_l,
                mask_r,
                target_aspect=aspect, 
                safety_margin=rig.lir_safety_margin
            )
            # Only crop if valid dual-camera box was found
            if (cw, ch) != (out_width, out_height) and (cx != 0 or cy != 0):
                map_x_l_sub = map_x_l[cy:cy+ch, cx:cx+cw]
                map_y_l_sub = map_y_l[cy:cy+ch, cx:cx+cw]
                mask_l_sub = mask_l[cy:cy+ch, cx:cx+cw]

                map_x_r_sub = map_x_r[cy:cy+ch, cx:cx+cw]
                map_y_r_sub = map_y_r[cy:cy+ch, cx:cx+cw]
                mask_r_sub = mask_r[cy:cy+ch, cx:cx+cw]

                weight_l_sub = weight_l[cy:cy+ch, cx:cx+cw]
                weight_r_sub = weight_r[cy:cy+ch, cx:cx+cw]

                # Resize back to target output dimensions (e.g. 3840x1080, 5120x1440)
                map_x_l = cv2.resize(map_x_l_sub, (out_width, out_height), interpolation=cv2.INTER_LINEAR)
                map_y_l = cv2.resize(map_y_l_sub, (out_width, out_height), interpolation=cv2.INTER_LINEAR)
                mask_l = cv2.resize(mask_l_sub.astype(np.uint8), (out_width, out_height), interpolation=cv2.INTER_NEAREST).astype(bool)

                map_x_r = cv2.resize(map_x_r_sub, (out_width, out_height), interpolation=cv2.INTER_LINEAR)
                map_y_r = cv2.resize(map_y_r_sub, (out_width, out_height), interpolation=cv2.INTER_LINEAR)
                mask_r = cv2.resize(mask_r_sub.astype(np.uint8), (out_width, out_height), interpolation=cv2.INTER_NEAREST).astype(bool)

                weight_l = cv2.resize(weight_l_sub, (out_width, out_height), interpolation=cv2.INTER_LINEAR)[:, :, np.newaxis]
                weight_r = cv2.resize(weight_r_sub, (out_width, out_height), interpolation=cv2.INTER_LINEAR)[:, :, np.newaxis]

                # Ensure weight sum is 1.0
                tot_w_sub = weight_l + weight_r
                weight_l = np.where(tot_w_sub > 0.0, weight_l / np.maximum(tot_w_sub, 1e-5), 0.5)
                weight_r = np.where(tot_w_sub > 0.0, weight_r / np.maximum(tot_w_sub, 1e-5), 0.5)

    # 7. Pre-slice sub-regions for ultra-fast partitioned remap & seam blending
    col_min = np.min(weight_l[:, :, 0], axis=0)
    col_max = np.max(weight_l[:, :, 0], axis=0)
    transition_cols = np.where((col_min < 0.999) & (col_max > 0.001))[0]
    if len(transition_cols) > 0:
        seam_x_start = max(0, int(transition_cols[0]))
        seam_x_end = min(out_width, int(transition_cols[-1]) + 1)
        map_xl_sub = np.ascontiguousarray(map_x_l[:, :seam_x_end])
        map_yl_sub = np.ascontiguousarray(map_y_l[:, :seam_x_end])
        map_xr_sub = np.ascontiguousarray(map_x_r[:, seam_x_start:])
        map_yr_sub = np.ascontiguousarray(map_y_r[:, seam_x_start:])
        weight_seam_l = np.ascontiguousarray(weight_l[:, seam_x_start:seam_x_end])
        weight_seam_r = np.ascontiguousarray(weight_r[:, seam_x_start:seam_x_end])
    else:
        seam_x_start = None
        seam_x_end = None
        map_xl_sub = None
        map_yl_sub = None
        map_xr_sub = None
        map_yr_sub = None
        weight_seam_l = None
        weight_seam_r = None

    res_luts = RemapLUTs(
        out_width=out_width,
        out_height=out_height,
        map_x_left=map_x_l,
        map_y_left=map_y_l,
        mask_left=mask_l,
        map_x_right=map_x_r,
        map_y_right=map_y_r,
        mask_right=mask_r,
        weight_left=weight_l,
        weight_right=weight_r,
        seam_x_start=seam_x_start,
        seam_x_end=seam_x_end,
        map_xl_sub=map_xl_sub,
        map_yl_sub=map_yl_sub,
        map_xr_sub=map_xr_sub,
        map_yr_sub=map_yr_sub,
        weight_seam_l=weight_seam_l,
        weight_seam_r=weight_seam_r
    )
    if getattr(rig, 'smart_seam_enabled', False) and sample_frame_left is not None and sample_frame_right is not None:
        res_luts.update_smart_seam(sample_frame_left, sample_frame_right, rig)
    return res_luts


def apply_stitch(frame_left: np.ndarray, frame_right: np.ndarray, luts: RemapLUTs, color_matcher: Optional[Any] = None) -> np.ndarray:
    """
    Applies the precalculated LUTs and blends two camera frames into the final 32:9 panorama.
    Uses high-speed partitioned sub-region remapping to maximize memory bandwidth & CPU throughput.
    Optionally applies automatic color and exposure matching to eliminate the center seam line.
    """
    if luts.seam_x_start is not None and luts.seam_x_end is not None:
        x0 = luts.seam_x_start
        x1 = luts.seam_x_end
        
        rem_l_sub = cv2.remap(
            frame_left, 
            luts.map_xl_sub, 
            luts.map_yl_sub, 
            interpolation=cv2.INTER_LINEAR, 
            borderMode=cv2.BORDER_REPLICATE
        )
        
        rem_r_sub = cv2.remap(
            frame_right, 
            luts.map_xr_sub, 
            luts.map_yr_sub, 
            interpolation=cv2.INTER_LINEAR, 
            borderMode=cv2.BORDER_REPLICATE
        )
        
        # Apply automatic color and white-balance matching to eliminate seam edge
        if color_matcher is not None and getattr(color_matcher, 'enabled', False):
            color_matcher.update_gains_from_sub(rem_l_sub, rem_r_sub, luts)
            rem_l_sub, rem_r_sub = color_matcher.apply_match(rem_l_sub, rem_r_sub)

        out = np.empty((luts.out_height, luts.out_width, 3), dtype=np.uint8)
        out[:, :x0] = rem_l_sub[:, :x0]
        out[:, x1:] = rem_r_sub[:, (x1 - x0):]
        
        seam_l = rem_l_sub[:, x0:x1].astype(np.float32) * luts.weight_seam_l
        seam_r = rem_r_sub[:, :(x1 - x0)].astype(np.float32) * luts.weight_seam_r
        out[:, x0:x1] = np.clip(seam_l + seam_r, 0, 255).astype(np.uint8)
        return out
    else:
        remapped_left = cv2.remap(
            frame_left, 
            luts.map_x_left, 
            luts.map_y_left, 
            interpolation=cv2.INTER_LINEAR, 
            borderMode=cv2.BORDER_REPLICATE
        )

        remapped_right = cv2.remap(
            frame_right, 
            luts.map_x_right, 
            luts.map_y_right, 
            interpolation=cv2.INTER_LINEAR, 
            borderMode=cv2.BORDER_REPLICATE
        )

        # Apply automatic color and white-balance matching to eliminate seam edge
        if color_matcher is not None and getattr(color_matcher, 'enabled', False):
            color_matcher.update_gains_from_full(remapped_left, remapped_right, luts)
            remapped_left, remapped_right = color_matcher.apply_match(remapped_left, remapped_right)

        blended = (remapped_left.astype(np.float32) * luts.weight_left + 
                   remapped_right.astype(np.float32) * luts.weight_right)
        
        return np.clip(blended, 0, 255).astype(np.uint8)


