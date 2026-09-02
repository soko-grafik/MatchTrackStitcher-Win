"""
Automated Stitch Calibration & 3D Rig Optimizer for MatchTrack-Stitcher.
Uses Computer Vision (SIFT / AKAZE feature matching), RANSAC outlier filtering,
and non-linear least-squares optimization to determine optimal camera pose
and lens parameters (Yaw, Pitch, Roll, FOV) automatically.
"""
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import cv2
from scipy.optimize import least_squares

from .rig_geometry import RigConfiguration, CameraPose
from .camera_model import CameraIntrinsics


@dataclass
class MatchPoint:
    """A pair of matching 2D pixel coordinates in Left and Right cameras."""
    u_left: float
    v_left: float
    u_right: float
    v_right: float
    weight: float = 1.0


@dataclass
class CalibrationResult:
    """Result of the automated stitch calibration."""
    success: bool
    num_matches: int
    initial_error_px: float
    optimized_error_px: float
    improvement_pct: float
    calibrated_rig: RigConfiguration
    details: Dict[str, Any] = field(default_factory=dict)
    message: str = ""


def extract_overlap_matches(
    frame_left: np.ndarray,
    frame_right: np.ndarray,
    max_features: int = 3000,
    ratio_thresh: float = 0.75,
    ransac_thresh_px: float = 4.0
) -> List[MatchPoint]:
    """
    Extracts high-quality correspondence points across the optical overlap seam
    of Left and Right video frames using SIFT and RANSAC geometric verification.
    """
    if frame_left is None or frame_right is None:
        return []

    h_l, w_l = frame_left.shape[:2]
    h_r, w_r = frame_right.shape[:2]

    # Convert to grayscale
    gray_l = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY) if len(frame_left.shape) == 3 else frame_left
    gray_r = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY) if len(frame_right.shape) == 3 else frame_right

    # Define ROI mask focusing on the overlap zone
    # Left camera: right 60% of frame (from x = 0.40 * w to w)
    mask_l = np.zeros((h_l, w_l), dtype=np.uint8)
    mask_l[:, int(w_l * 0.35):] = 255

    # Right camera: left 60% of frame (from x = 0 to 0.65 * w)
    mask_r = np.zeros((h_r, w_r), dtype=np.uint8)
    mask_r[:, :int(w_r * 0.65)] = 255

    # Try SIFT first, fallback to AKAZE or ORB if unavailable
    detector = None
    try:
        detector = cv2.SIFT_create(nfeatures=max_features, contrastThreshold=0.03, edgeThreshold=10)
    except Exception:
        try:
            detector = cv2.AKAZE_create()
        except Exception:
            detector = cv2.ORB_create(nfeatures=max_features)

    kp_l, des_l = detector.detectAndCompute(gray_l, mask=mask_l)
    kp_r, des_r = detector.detectAndCompute(gray_r, mask=mask_r)

    if des_l is None or des_r is None or len(kp_l) < 8 or len(kp_r) < 8:
        return []

    # Feature matching with FLANN or BFMatcher
    if des_l.dtype == np.float32:
        # SIFT descriptors
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        matcher = cv2.FlannBasedMatcher(index_params, search_params)
    else:
        # Binary descriptors (AKAZE/ORB)
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    try:
        raw_matches = matcher.knnMatch(des_l, des_r, k=2)
    except Exception:
        return []

    # Lowe's ratio test to filter out ambiguous grass/texture repetitions
    good_matches = []
    pts_l = []
    pts_r = []

    for m_tuple in raw_matches:
        if len(m_tuple) == 2:
            m, n = m_tuple
            if m.distance < ratio_thresh * n.distance:
                good_matches.append(m)
                pts_l.append(kp_l[m.queryIdx].pt)
                pts_r.append(kp_r[m.trainIdx].pt)

    if len(pts_l) < 8:
        return []

    pts_l_arr = np.float32(pts_l)
    pts_r_arr = np.float32(pts_r)

    # RANSAC fundamental matrix estimation to remove moving players/balls and spurious matches
    try:
        F, inliers_mask = cv2.findFundamentalMat(
            pts_l_arr, pts_r_arr,
            method=cv2.FM_RANSAC,
            ransacReprojThreshold=ransac_thresh_px,
            confidence=0.99
        )
    except Exception:
        inliers_mask = None

    result: List[MatchPoint] = []
    if inliers_mask is not None:
        inliers_mask = inliers_mask.ravel().astype(bool)
        for i, is_inlier in enumerate(inliers_mask):
            if is_inlier:
                u_l, v_l = pts_l_arr[i]
                u_r, v_r = pts_r_arr[i]
                result.append(MatchPoint(u_left=float(u_l), v_left=float(v_l),
                                         u_right=float(u_r), v_right=float(v_r)))
    else:
        for i in range(len(pts_l_arr)):
            u_l, v_l = pts_l_arr[i]
            u_r, v_r = pts_r_arr[i]
            result.append(MatchPoint(u_left=float(u_l), v_left=float(v_l),
                                     u_right=float(u_r), v_right=float(v_r)))

    return result


def unproject_pixel_to_cam_ray(u: np.ndarray, v: np.ndarray, intrinsics: CameraIntrinsics) -> np.ndarray:
    """
    Unprojects 2D image coordinates (u, v) into 3D normalized ray direction vectors in camera frame.
    Returns array of shape [N, 3].
    """
    if intrinsics.model_type == "fisheye":
        # Fisheye unprojection
        x_norm = (u - intrinsics.cx) / max(intrinsics.fx, 1e-5)
        y_norm = (v - intrinsics.cy) / max(intrinsics.fy, 1e-5)
        rho = np.sqrt(x_norm * x_norm + y_norm * y_norm)
        rho_safe = np.maximum(rho, 1e-7)

        # Approximate theta from polynomial distortion or simple equidistant
        theta = rho
        for _ in range(3): # Newton-Raphson refinement
            th2 = theta * theta
            f_val = theta * (1.0 + intrinsics.k1 * th2 + intrinsics.k2 * th2 * th2) - rho
            f_prime = 1.0 + 3.0 * intrinsics.k1 * th2 + 5.0 * intrinsics.k2 * th2 * th2
            theta = theta - f_val / np.maximum(f_prime, 1e-5)

        sin_th = np.sin(theta)
        cos_th = np.cos(theta)
        scale = sin_th / rho_safe
        xc = x_norm * scale
        yc = y_norm * scale
        zc = cos_th
    else:
        # Pinhole / Brown-Conrady rectilinear
        xc = (u - intrinsics.cx) / max(intrinsics.fx, 1e-5)
        yc = (v - intrinsics.cy) / max(intrinsics.fy, 1e-5)
        zc = np.ones_like(xc)

    norm = np.sqrt(xc * xc + yc * yc + zc * zc)
    norm = np.maximum(norm, 1e-7)
    return np.column_stack([xc / norm, yc / norm, zc / norm])


def project_cam_rays_to_pano(
    rays_cam: np.ndarray,
    camera_pose: CameraPose,
    r_global_level: np.ndarray,
    pano_hfov: float,
    out_w: int = 3840,
    out_h: int = 1080,
    vertical_crop_offset: float = 0.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Transforms rays from camera coordinate frame through pose and global leveling,
    then projects them onto the 32:9 cylindrical panorama canvas.
    Returns (pano_x, pano_y) arrays in pixel coordinates.
    """
    # 1. Transform from camera frame to leveled world frame:
    # In lut_generator: rays_cam = rays_level @ R_cam, so rays_level = rays_cam @ R_cam^T
    # And rays_level = rays_global @ R_global^T, so rays_global = rays_level @ R_global
    R_cam = camera_pose.rotation_matrix()
    rays_level = rays_cam @ R_cam.T
    rays_global = rays_level @ r_global_level

    Xg = rays_global[:, 0]
    Yg = rays_global[:, 1]
    Zg = rays_global[:, 2]

    # 2. Cylindrical Panorama Projection
    hfov_rad = np.radians(pano_hfov)
    lambda_angles = np.arctan2(Xg, np.maximum(Zg, 1e-7))

    f_pano = (out_w * 0.5) / np.tan(hfov_rad * 0.5)
    r_xz = np.sqrt(Xg * Xg + Zg * Zg)
    h_heights = Yg / np.maximum(r_xz, 1e-7)

    pano_x = (out_w * 0.5) + (lambda_angles / (hfov_rad * 0.5)) * (out_w * 0.5)
    y_center = (out_h * 0.5) + vertical_crop_offset * (out_h * 0.25)
    pano_y = y_center + h_heights * f_pano

    return pano_x, pano_y


def optimize_rig_calibration(
    matches: List[MatchPoint],
    initial_rig: RigConfiguration,
    out_w: int = 3840,
    out_h: int = 1080,
    optimize_fov: bool = True,
    lock_pitch_leveling: bool = True
) -> CalibrationResult:
    """
    Solves for the optimal 3D Rig parameters using non-linear least squares (Huber loss).
    Minimizes the pixel discrepancy of corresponding features in the 32:9 stitched panorama.
    """
    if len(matches) < 6:
        return CalibrationResult(
            success=False,
            num_matches=len(matches),
            initial_error_px=0.0,
            optimized_error_px=0.0,
            improvement_pct=0.0,
            calibrated_rig=initial_rig,
            message="Zu wenige Übereinstimmungen gefunden (mindestens 6 erforderlich)."
        )

    pts_l_u = np.array([m.u_left for m in matches], dtype=np.float64)
    pts_l_v = np.array([m.v_left for m in matches], dtype=np.float64)
    pts_r_u = np.array([m.u_right for m in matches], dtype=np.float64)
    pts_r_v = np.array([m.v_right for m in matches], dtype=np.float64)

    # Initial parameter vector:
    # p = [yaw_left, yaw_right, cam_pitch, roll_left, roll_right, fov_deg]
    p0 = np.array([
        initial_rig.left_pose.yaw,
        initial_rig.right_pose.yaw,
        initial_rig.left_pose.pitch,
        initial_rig.left_pose.roll,
        initial_rig.right_pose.roll,
        initial_rig.left_camera.hfov_deg
    ], dtype=np.float64)

    # Bounds for parameters
    bounds_lower = np.array([-75.0,  15.0, -40.0, -12.0, -12.0,  70.0])
    bounds_upper = np.array([-15.0,  75.0,  -2.0,  12.0,  12.0, 120.0])

    if not optimize_fov:
        # Clamp FOV bounds to initial value
        bounds_lower[5] = p0[5] - 1e-4
        bounds_upper[5] = p0[5] + 1e-4

    def compute_residuals(params: np.ndarray) -> np.ndarray:
        yaw_l, yaw_r, pitch_val, roll_l, roll_r, fov_val = params

        # Update intrinsics with current FOV candidate
        cam_l = CameraIntrinsics.from_dict(initial_rig.left_camera.to_dict())
        cam_r = CameraIntrinsics.from_dict(initial_rig.right_camera.to_dict())
        cam_l.set_fov(fov_val)
        cam_r.set_fov(fov_val)

        # Unproject points to camera rays
        rays_l = unproject_pixel_to_cam_ray(pts_l_u, pts_l_v, cam_l)
        rays_r = unproject_pixel_to_cam_ray(pts_r_u, pts_r_v, cam_r)

        # Poses
        pose_l = CameraPose(yaw=yaw_l, pitch=pitch_val, roll=roll_l, tx=initial_rig.left_pose.tx)
        pose_r = CameraPose(yaw=yaw_r, pitch=pitch_val, roll=roll_r, tx=initial_rig.right_pose.tx)

        # Global Leveling: If locked, counteract camera pitch directly
        g_pitch = -pitch_val if lock_pitch_leveling else initial_rig.global_pitch_correction
        r_pitch = np.radians(g_pitch)
        r_roll = np.radians(initial_rig.global_roll_correction)
        r_yaw = np.radians(initial_rig.global_yaw_center)

        Rx = np.array([[1, 0, 0], [0, np.cos(r_pitch), -np.sin(r_pitch)], [0, np.sin(r_pitch), np.cos(r_pitch)]], dtype=np.float64)
        Rz = np.array([[np.cos(r_roll), -np.sin(r_roll), 0], [np.sin(r_roll), np.cos(r_roll), 0], [0, 0, 1]], dtype=np.float64)
        Ry = np.array([[np.cos(r_yaw), 0, np.sin(r_yaw)], [0, 1, 0], [-np.sin(r_yaw), 0, np.cos(r_yaw)]], dtype=np.float64)
        R_global = Rz @ Rx @ Ry

        # Project Left and Right onto 32:9 panorama
        px_l, py_l = project_cam_rays_to_pano(
            rays_l, pose_l, R_global,
            initial_rig.pano_hfov, out_w, out_h, initial_rig.vertical_crop_offset
        )
        px_r, py_r = project_cam_rays_to_pano(
            rays_r, pose_r, R_global,
            initial_rig.pano_hfov, out_w, out_h, initial_rig.vertical_crop_offset
        )

        dx = px_l - px_r
        dy = py_l - py_r

        # Return concatenated 1D residual vector [dx_0, dy_0, dx_1, dy_1, ...]
        return np.column_stack([dx, dy]).ravel()

    # Calculate initial error
    res_initial = compute_residuals(p0)
    init_err_2d = res_initial.reshape(-1, 2)
    init_mean_err = float(np.mean(np.sqrt(init_err_2d[:, 0]**2 + init_err_2d[:, 1]**2)))

    # Run Least Squares with robust Huber loss function
    res = least_squares(
        compute_residuals,
        p0,
        bounds=(bounds_lower, bounds_upper),
        loss='huber',
        f_scale=2.0,
        max_nfev=250,
        xtol=1e-6,
        ftol=1e-6
    )

    opt_p = res.x
    res_opt = compute_residuals(opt_p)
    opt_err_2d = res_opt.reshape(-1, 2)
    opt_mean_err = float(np.mean(np.sqrt(opt_err_2d[:, 0]**2 + opt_err_2d[:, 1]**2)))

    improvement = 0.0
    if init_mean_err > 1e-4:
        improvement = max(0.0, float((init_mean_err - opt_mean_err) / init_mean_err * 100.0))

    # Construct optimized RigConfiguration
    opt_yaw_l, opt_yaw_r, opt_pitch, opt_roll_l, opt_roll_r, opt_fov = opt_p

    new_rig = RigConfiguration.from_dict(initial_rig.to_dict())
    new_rig.left_pose.yaw = float(round(opt_yaw_l, 2))
    new_rig.right_pose.yaw = float(round(opt_yaw_r, 2))
    new_rig.left_pose.pitch = float(round(opt_pitch, 2))
    new_rig.right_pose.pitch = float(round(opt_pitch, 2))
    new_rig.left_pose.roll = float(round(opt_roll_l, 2))
    new_rig.right_pose.roll = float(round(opt_roll_r, 2))
    if lock_pitch_leveling:
        new_rig.global_pitch_correction = float(round(-opt_pitch, 2))

    new_rig.left_camera.set_fov(float(round(opt_fov, 2)))
    new_rig.right_camera.set_fov(float(round(opt_fov, 2)))

    msg = (
        f"KI-Kalibrierung erfolgreich!\n\n"
        f"• Verwendete Nahtstellen-Punkte: {len(matches)}\n"
        f"• Mittlerer Nahtfehler: {init_mean_err:.1f} px ➔ {opt_mean_err:.2f} px "
        f"({improvement:.1f}% Verbesserung)\n\n"
        f"Berechnete Rig-Winkel:\n"
        f"• Spreizung: {abs(opt_yaw_l) + abs(opt_yaw_r):.2f}° "
        f"(Links {opt_yaw_l:.1f}°, Rechts +{opt_yaw_r:.1f}°)\n"
        f"• Neigung (Pitch): {opt_pitch:.1f}° (Ausgleich +{-opt_pitch:.1f}°)\n"
        f"• Roll-Feinjustage: L {opt_roll_l:+.1f}°, R {opt_roll_r:+.1f}°\n"
        f"• Objektiv HFOV: {opt_fov:.1f}°"
    )

    return CalibrationResult(
        success=True,
        num_matches=len(matches),
        initial_error_px=init_mean_err,
        optimized_error_px=opt_mean_err,
        improvement_pct=improvement,
        calibrated_rig=new_rig,
        details={
            "yaw_left": float(opt_yaw_l),
            "yaw_right": float(opt_yaw_r),
            "pitch": float(opt_pitch),
            "roll_left": float(opt_roll_l),
            "roll_right": float(opt_roll_r),
            "hfov": float(opt_fov),
            "spread_yaw": float(abs(opt_yaw_l) + abs(opt_yaw_r))
        },
        message=msg
    )


def calibrate_from_frames(
    frames_left: List[np.ndarray],
    frames_right: List[np.ndarray],
    initial_rig: RigConfiguration,
    out_w: int = 3840,
    out_h: int = 1080,
    optimize_fov: bool = True
) -> CalibrationResult:
    """
    Performs multi-frame calibration across a list of synchronized video frames.
    Aggregates robust feature matches from multiple time steps for supreme stability.
    """
    all_matches: List[MatchPoint] = []
    
    for fl, fr in zip(frames_left, frames_right):
        if fl is not None and fr is not None:
            m = extract_overlap_matches(fl, fr)
            all_matches.extend(m)

    if len(all_matches) < 6:
        return CalibrationResult(
            success=False,
            num_matches=len(all_matches),
            initial_error_px=0.0,
            optimized_error_px=0.0,
            improvement_pct=0.0,
            calibrated_rig=initial_rig,
            message="Keine ausreichenden Nahtstellen-Übereinstimmungen in den Video-Frames gefunden."
        )

    return optimize_rig_calibration(
        all_matches,
        initial_rig,
        out_w=out_w,
        out_h=out_h,
        optimize_fov=optimize_fov
    )

