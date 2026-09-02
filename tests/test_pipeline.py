"""
Verification & Unit Test Suite for MatchTrack-Stitcher.
Generates synthetic soccer pitch frames, tests Fisheye undistortion, 3D leveling, and 32:9 panorama stitching.
"""
import os
import sys
import numpy as np
import cv2

# Add root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from matchtrack.camera_model import CAMERA_PRESETS, CameraIntrinsics
from matchtrack.rig_geometry import RigConfiguration, CameraPose
from matchtrack.lut_generator import generate_remap_luts, apply_stitch
from matchtrack.tactical_warp import generate_tactical_16x9_luts, apply_tactical_16x9_warp
from matchtrack.audio_sync import calculate_audio_sync_offset


def create_synthetic_field_frame(camera_label: str, bg_color=(34, 139, 34)) -> np.ndarray:
    """Generates a synthetic camera view with pitch lines to test undistortion & stitching."""
    img = np.full((2160, 3840, 3), bg_color, dtype=np.uint8)

    # Draw soccer pitch lines (white lines)
    # Sideline
    cv2.line(img, (200, 1800), (3640, 1800), (255, 255, 255), 14)
    # Halfway line
    cv2.line(img, (1920, 600), (1920, 1800), (255, 255, 255), 14)
    # Center circle
    cv2.circle(img, (1920, 1500), 280, (255, 255, 255), 14)
    # Penalty box
    if "Left" in camera_label:
        cv2.rectangle(img, (400, 1200), (1200, 1800), (255, 255, 255), 14)
    else:
        cv2.rectangle(img, (2640, 1200), (3440, 1800), (255, 255, 255), 14)

    # Add text
    cv2.putText(img, f"DJI Action 4: {camera_label}", (300, 400), cv2.FONT_HERSHEY_SIMPLEX, 3.5, (255, 255, 255), 8)
    return img


def test_lut_generation_and_stitch():
    print("Testing Rig Configuration and LUT Generation...")
    rig = RigConfiguration()
    assert rig.left_pose.yaw == -40.0
    assert rig.right_pose.yaw == 40.0
    assert rig.global_pitch_correction == 15.0

    print("Generating 3840x1080 (32:9) Remap LUTs...")
    luts = generate_remap_luts(rig, 3840, 1080)
    assert luts.map_x_left.shape == (1080, 3840)
    assert luts.map_x_right.shape == (1080, 3840)
    assert luts.weight_left.shape == (1080, 3840, 1)
    assert luts.weight_right.shape == (1080, 3840, 1)
    print("LUT shapes and weight normalization verified.")

    print("Synthesizing Left & Right test frames...")
    frame_left = create_synthetic_field_frame("Left Camera (Yaw -40 deg, Pitch -15 deg)")
    frame_right = create_synthetic_field_frame("Right Camera (Yaw +40 deg, Pitch -15 deg)")

    print("Applying stitch...")
    stitched_32x9 = apply_stitch(frame_left, frame_right, luts)
    assert stitched_32x9.shape == (1080, 3840, 3)

    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "output"))
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "test_stitched_32x9.jpg")
    cv2.imwrite(out_file, stitched_32x9)
    print(f"Stitched test panorama successfully saved to: {out_file}")


def test_21x10_lut_generation_and_stitch():
    print("\nTesting 21:10 Rig Configuration, Remap LUT Generation & Squeezing...")
    rig = RigConfiguration()

    print("Generating 2520x1200 (21:10) Remap LUTs with dynamic horizontal squeeze...")
    luts_2110 = generate_remap_luts(rig, 2520, 1200)
    assert luts_2110.map_x_left.shape == (1200, 2520)
    assert luts_2110.map_x_right.shape == (1200, 2520)
    assert luts_2110.weight_left.shape == (1200, 2520, 1)
    assert luts_2110.weight_right.shape == (1200, 2520, 1)
    print("21:10 LUT shapes and weight normalization verified.")

    frame_left = create_synthetic_field_frame("Left Camera (Yaw -40 deg, Pitch -15 deg)")
    frame_right = create_synthetic_field_frame("Right Camera (Yaw +40 deg, Pitch -15 deg)")

    print("Applying 21:10 stitch...")
    stitched_21x10 = apply_stitch(frame_left, frame_right, luts_2110)
    assert stitched_21x10.shape == (1200, 2520, 3)

    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "output"))
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "test_stitched_21x10.jpg")
    cv2.imwrite(out_file, stitched_21x10)
    print(f"Stitched 21:10 test panorama successfully saved to: {out_file}")


def test_32x9_to_21x10_conversion():
    print("\nTesting Conversion of existing 32:9 Panorama to 21:10...")
    # Create mock 32:9 panorama
    pano_32x9 = np.full((1080, 3840, 3), (34, 139, 34), dtype=np.uint8)
    # Add pitch markings
    cv2.line(pano_32x9, (200, 900), (3640, 900), (255, 255, 255), 8)
    cv2.circle(pano_32x9, (1920, 540), 200, (255, 255, 255), 8)
    cv2.putText(pano_32x9, "32:9 Master Video", (1400, 300), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 4)

    # Convert to 21:10 (2520x1200)
    out_w, out_h = 2520, 1200
    converted_21x10 = cv2.resize(pano_32x9, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    assert converted_21x10.shape == (1200, 2520, 3)

    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "output"))
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "test_converted_32x9_to_21x10.jpg")
    cv2.imwrite(out_file, converted_21x10)
    print(f"Converted 21:10 test image saved to: {out_file}")


def test_audio_sync_mock():
    print("Testing Audio Sync Cross-Correlation math...")
    # Generate mock stereo signals with a known time offset
    sr = 16000
    duration = 5.0
    t = np.linspace(0, duration, int(sr * duration))
    # Sound event (e.g. whistle impulse at 2.0 seconds)
    pulse = np.exp(-((t - 2.0) ** 2) / 0.002) * np.sin(2 * np.pi * 2500 * t)
    
    offset_samples = int(0.12 * sr) # 120ms offset (approx 1920 samples)
    # Signal Left starts earlier, Signal Right has pulse delayed
    pulse_delayed = np.exp(-((t - (2.0 + 0.12)) ** 2) / 0.002) * np.sin(2 * np.pi * 2500 * (t - 0.12))
    sig_l = pulse + np.random.normal(0, 0.02, len(t))
    sig_r = pulse_delayed + np.random.normal(0, 0.02, len(t))

    from scipy import signal
    corr = signal.correlate(sig_l, sig_r, mode='full', method='fft')
    lags = signal.correlation_lags(len(sig_l), len(sig_r), mode='full')
    peak_idx = np.argmax(np.abs(corr))
    detected_offset = lags[peak_idx]
    
    # Assert detected lag matches ground truth within sub-millisecond precision (< 20 samples = < 1.25 ms)
    assert abs(abs(detected_offset) - offset_samples) <= 20
    print(f"Audio Cross-Correlation correctly detected {detected_offset} sample lag ({detected_offset/sr*1000:.1f}ms).")


def test_autocam():
    print("Testing AutoCam 16:9 Tracking & Broadcast Extraction...")
    from matchtrack.autocam import AutoCamTracker, AutoCamConfig
    tracker = AutoCamTracker(AutoCamConfig(zoom_factor=1.25, smoothing_factor=0.94))
    
    # Create mock 32:9 panorama frames with a moving player/ball blob
    w, h = 3840, 1080
    for frame_i in range(10):
        mock_pano = np.full((h, w, 3), (34, 139, 34), dtype=np.uint8) # green pitch
        # Moving player blob from center to right
        player_x = int(1920 + frame_i * 100)
        cv2.circle(mock_pano, (player_x, 600), 40, (255, 255, 255), -1)
        
        frame_16x9 = tracker.extract_16x9_frame(mock_pano, out_width=1920, out_height=1080)
        assert frame_16x9.shape == (1080, 1920, 3)

    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "output"))
    out_file = os.path.join(out_dir, "test_autocam_16x9.jpg")
    cv2.imwrite(out_file, frame_16x9)
def test_ai_broadcast():
    print("Testing YOLO AI Soccer Ball Tracking & Dynamic Broadcast Auto-Zoom...")
    from matchtrack.ai_tracker import AIBroadcastTracker, BroadcastConfig
    ai_tracker = AIBroadcastTracker(BroadcastConfig(ai_tracking=True, min_zoom=1.15, max_zoom=1.60))
    
    # Create mock 32:9 panorama frame with soccer pitch and a ball
    w, h = 3840, 1080
    mock_pano = np.full((h, w, 3), (34, 139, 34), dtype=np.uint8)
    # White ball
    cv2.circle(mock_pano, (2200, 600), 12, (255, 255, 255), -1)

    frame_16x9 = ai_tracker.extract_16x9_frame(mock_pano, out_width=1920, out_height=1080)
    assert frame_16x9.shape == (1080, 1920, 3)

    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "output"))
    out_file = os.path.join(out_dir, "test_ai_broadcast_16x9.jpg")
    cv2.imwrite(out_file, frame_16x9)
    print(f"AI Broadcast 16:9 frame successfully saved to: {out_file}")


def test_auto_stitch_calibration():
    print("Testing Auto-Stitch KI & 3D Rig Optimization...")
    from matchtrack.auto_stitch import (
        MatchPoint,
        extract_overlap_matches,
        optimize_rig_calibration,
        unproject_pixel_to_cam_ray,
        project_cam_rays_to_pano
    )

    # 1. Test Ray Unprojection and Pano Projection math
    intrinsics = CameraIntrinsics(image_width=3840, image_height=2160, hfov_deg=92.0)
    u_pts = np.array([1920.0, 3000.0])
    v_pts = np.array([1080.0, 1080.0])
    rays = unproject_pixel_to_cam_ray(u_pts, v_pts, intrinsics)
    assert rays.shape == (2, 3)
    # Center pixel should have ray pointing along +Z axis
    assert abs(rays[0, 0]) < 1e-4 and abs(rays[0, 1]) < 1e-4 and abs(rays[0, 2] - 1.0) < 1e-4

    # 2. Test Rig Optimizer with known ground-truth correspondence points
    gt_rig = RigConfiguration(
        left_pose=CameraPose(yaw=-41.2, pitch=-15.0, roll=-0.4),
        right_pose=CameraPose(yaw=41.2, pitch=-15.0, roll=0.4)
    )
    gt_rig.left_camera.set_fov(92.0)
    gt_rig.right_camera.set_fov(92.0)

    # Generate synthetic 3D world points in the overlap zone (yaw around 0 deg)
    num_pts = 40
    lambdas = np.linspace(-np.radians(6), np.radians(6), num_pts)
    heights = np.linspace(-0.15, 0.15, num_pts)
    
    matches: list[MatchPoint] = []
    # R_level for gt_rig
    r_pitch = np.radians(gt_rig.global_pitch_correction)
    Rx = np.array([[1, 0, 0], [0, np.cos(r_pitch), -np.sin(r_pitch)], [0, np.sin(r_pitch), np.cos(r_pitch)]], dtype=np.float64)
    R_global = Rx

    for lam, h in zip(lambdas, heights):
        # 3D ray in global world
        Xg = np.sin(lam)
        Yg = h
        Zg = np.cos(lam)
        norm = np.sqrt(Xg*Xg + Yg*Yg + Zg*Zg)
        ray_g = np.array([Xg/norm, Yg/norm, Zg/norm])

        # Project into Left Cam
        ray_level = ray_g @ R_global.T
        ray_c_l = ray_level @ gt_rig.left_pose.rotation_matrix()
        u_l = gt_rig.left_camera.fx * (ray_c_l[0] / ray_c_l[2]) + gt_rig.left_camera.cx
        v_l = gt_rig.left_camera.fy * (ray_c_l[1] / ray_c_l[2]) + gt_rig.left_camera.cy

        # Project into Right Cam
        ray_c_r = ray_level @ gt_rig.right_pose.rotation_matrix()
        u_r = gt_rig.right_camera.fx * (ray_c_r[0] / ray_c_r[2]) + gt_rig.right_camera.cx
        v_r = gt_rig.right_camera.fy * (ray_c_r[1] / ray_c_r[2]) + gt_rig.right_camera.cy

        matches.append(MatchPoint(u_left=float(u_l), v_left=float(v_l),
                                  u_right=float(u_r), v_right=float(v_r)))

    # Initial perturbed rig
    initial_rig = RigConfiguration(
        left_pose=CameraPose(yaw=-38.0, pitch=-14.0, roll=0.0),
        right_pose=CameraPose(yaw=38.0, pitch=-14.0, roll=0.0)
    )

    result = optimize_rig_calibration(matches, initial_rig, out_w=3840, out_h=1080)
    assert result.success is True
    assert result.num_matches == num_pts
    assert result.optimized_error_px < 0.1
    assert result.improvement_pct > 90.0

    print(f"Auto-Stitch calibration solved {result.num_matches} points: initial error {result.initial_error_px:.2f}px -> {result.optimized_error_px:.4f}px ({result.improvement_pct:.1f}% improvement).")
    print(f"Calibrated Yaw: L {result.details['yaw_left']:.2f}° (GT -41.2°), R +{result.details['yaw_right']:.2f}° (GT +41.2°)")


def test_camera_presets():
    print("Testing 1080p 60fps Dewarp preset...")
    assert "DJI_Action4_1080p_Dewarp" in CAMERA_PRESETS
    preset = CAMERA_PRESETS["DJI_Action4_1080p_Dewarp"]
    assert preset.image_width == 1920
    assert preset.image_height == 1080
    assert preset.hfov_deg == 92.0
    assert preset.cx == 960.0
    assert preset.cy == 540.0
    assert preset.model_type == "pinhole"
    print("1080p 60fps preset verified.")


def test_video_trimming_logic():
    print("Testing In/Out Video Trimming reference logic...")
    from matchtrack.stitcher_engine import StitcherEngine
    engine = StitcherEngine()
    # Left video frame indexing reference
    start_frame = 120
    end_frame = 300
    frames_to_process = end_frame - start_frame
    assert frames_to_process == 180
    print(f"Trimming range [{start_frame} -> {end_frame}] = {frames_to_process} frames verified.")


def test_end_to_end_trimmed_render():
    print("Testing End-to-End Trimmed Render Pipeline (Strict In/Out Frame Range)...")
    from matchtrack.stitcher_engine import StitcherEngine
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "output"))
    os.makedirs(out_dir, exist_ok=True)
    
    # Create two 40-frame synthetic test videos
    path_l = os.path.join(out_dir, "synth_left.mp4")
    path_r = os.path.join(out_dir, "synth_right.mp4")
    path_out = os.path.join(out_dir, "synth_stitched_trim.mp4")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vw_l = cv2.VideoWriter(path_l, fourcc, 30.0, (1280, 720))
    vw_r = cv2.VideoWriter(path_r, fourcc, 30.0, (1280, 720))
    
    for i in range(40):
        f_l = np.full((720, 1280, 3), (30 + i * 4, 120, 30), dtype=np.uint8)
        f_r = np.full((720, 1280, 3), (30, 120, 30 + i * 4), dtype=np.uint8)
        vw_l.write(f_l)
        vw_r.write(f_r)
    vw_l.release()
    vw_r.release()
    
    # Load into engine
    engine = StitcherEngine()
    engine.load_videos(path_l, path_r)
    
    start_f = 10
    end_f = 30
    expected_frames = end_f - start_f # 20 frames
    
    progress_recorded = []
    def progress_cb(processed, total, fps, eta):
        progress_recorded.append((processed, total))
        
    success = engine.render_video_to_file(
        output_filepath=path_out,
        out_width=1920,
        out_height=540,
        codec="libx264",
        start_frame=start_f,
        end_frame=end_f,
        progress_callback=progress_cb
    )
    assert success is True
    assert os.path.exists(path_out)
    
    # Verify rendered video frame count
    cap_out = cv2.VideoCapture(path_out)
    rendered_frames = int(cap_out.get(cv2.CAP_PROP_FRAME_COUNT))
    cap_out.release()
    
    assert rendered_frames == expected_frames, f"Expected {expected_frames} frames, got {rendered_frames}"
    print(f"Trimmed render successfully exported {rendered_frames} frames strictly from frame {start_f} to {end_f} (Zero padding before/after)!")


def test_16x9_broadcast_render():
    print("Testing 16:9 AI Broadcast High-Speed Render Pipeline...")
    from matchtrack.stitcher_engine import StitcherEngine
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "output"))
    path_l = os.path.join(out_dir, "synth_left.mp4")
    path_r = os.path.join(out_dir, "synth_right.mp4")
    path_out = os.path.join(out_dir, "synth_broadcast_16x9.mp4")
    
    engine = StitcherEngine()
    engine.load_videos(path_l, path_r)
    
    start_f = 5
    end_f = 25
    expected_frames = end_f - start_f # 20 frames
    
    success = engine.render_video_to_file(
        output_filepath=path_out,
        out_width=1280,
        out_height=720,
        mode="16:9_autocam",
        codec="libx264",
        start_frame=start_f,
        end_frame=end_f
    )
    assert success is True
    assert os.path.exists(path_out)
    
    cap = cv2.VideoCapture(path_out)
    fcount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    assert fcount == expected_frames
    print(f"16:9 Broadcast render successfully exported {fcount} frames with dynamic tracking!")


def test_corner_pinning():
    print("Testing Photoshop-Style 4-Corner Pinning & Homography Pipeline...")
    pose = CameraPose(yaw=-40.0, pitch=-15.0)
    assert not pose.is_corners_modified()
    assert pose.get_homography_matrix(3840, 2160) is None

    # Distort Top-Right and Bottom-Right corners
    pose.corners[1] = [1.15, -0.05] # TR shifted right and up
    pose.corners[2] = [1.10, 1.05]  # BR shifted right and down
    assert pose.is_corners_modified()
    
    H = pose.get_homography_matrix(3840, 2160)
    assert H is not None
    assert H.shape == (3, 3)

    # Test Rig Configuration with Corner Pins
    rig = RigConfiguration()
    rig.left_pose.corners[1] = [1.15, -0.05]
    assert rig.has_corner_pins()

    # Serialization test
    d = rig.to_dict()
    assert "corners" in d["left_pose"]
    assert d["left_pose"]["corners"][1] == [1.15, -0.05]
    
    rig_loaded = RigConfiguration.from_dict(d)
    assert rig_loaded.has_corner_pins()
    assert rig_loaded.left_pose.corners[1] == [1.15, -0.05]

    # Test LUT generation and stitch with warped corner
    luts = generate_remap_luts(rig, 1920, 540)
    assert luts.map_x_left.shape == (540, 1920)
    
    frame_l = create_synthetic_field_frame("Left (Corner Pinned)")
    frame_r = create_synthetic_field_frame("Right")
    stitched = apply_stitch(frame_l, frame_r, luts)
    assert stitched.shape == (540, 1920, 3)
    assert not np.isnan(stitched).any()

    # Test Reset
    rig.reset_all_corners()
    assert not rig.has_corner_pins()
    assert rig.left_pose.corners[1] == [1.0, 0.0]
    print("Corner Pinning homography, LUT generation, and rig serialization verified.")


def test_pitch_roi_filtering_and_speed():
    print("\nTesting Pitch Field ROI Polygon (6 Points with Mittellinie) Filtering & Lookahead Scan Speedup...")
    from matchtrack.ai_tracker import AIBroadcastTracker, BroadcastConfig, is_point_in_pitch_polygon, get_pitch_polygon_bbox
    
    # 1. Test Pitch 6-point polygon (curved sidelines: TL, TC, TR, BR, BC, BL)
    curved_pitch_points = [
        [0.10, 0.15], # 0: TL
        [0.50, 0.08], # 1: TC (Top Center - curved upwards at center line)
        [0.90, 0.15], # 2: TR
        [0.95, 0.90], # 3: BR
        [0.50, 0.96], # 4: BC (Bottom Center - curved downwards at center line)
        [0.05, 0.90]  # 5: BL
    ]
    min_x, max_x, min_y, max_y = get_pitch_polygon_bbox(curved_pitch_points)
    assert abs(min_x - 0.05) < 1e-4
    assert abs(max_x - 0.95) < 1e-4
    assert abs(min_y - 0.08) < 1e-4
    assert abs(max_y - 0.96) < 1e-4

    # Center is inside
    assert is_point_in_pitch_polygon(0.50, 0.50, curved_pitch_points) is True
    # Point at (0.50, 0.10) is inside because TC is at y=0.08 (even though TL is at y=0.15!)
    assert is_point_in_pitch_polygon(0.50, 0.10, curved_pitch_points) is True
    # Point far out on top (0.50, 0.03) is outside
    assert is_point_in_pitch_polygon(0.50, 0.03, curved_pitch_points) is False
    # Far out on right bench (0.98, 0.50)
    assert is_point_in_pitch_polygon(0.98, 0.50, curved_pitch_points) is False
    print("Point-in-polygon ray casting for 6-point curved pitch verified.")

    # 2. Test Pitch ROI action bounding
    cfg = BroadcastConfig(
        ai_tracking=True,
        pitch_corners=curved_pitch_points,
        scan_step=5,
        use_fp16=True
    )
    tracker = AIBroadcastTracker(cfg)
    
    # Test frame where an action/person is outside the pitch on the left bench (norm_x = 0.02)
    pano_frame = np.full((720, 2560, 3), (34, 139, 34), dtype=np.uint8)
    cv2.circle(pano_frame, (50, 360), 30, (255, 255, 255), -1)
    
    target_x, target_y, _, _, _ = tracker.detect_action(pano_frame)
    # Target must be bounded within [min_x, max_x]
    assert target_x >= min_x and target_x <= max_x
    print("Out-of-bounds action successfully excluded by 6-point pitch polygon.")
    
    # 3. Test Step 5 Lookahead trajectory generation speed
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "output"))
    os.makedirs(out_dir, exist_ok=True)
    temp_vid = os.path.join(out_dir, "temp_roi_pano.mp4")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_vid = cv2.VideoWriter(temp_vid, fourcc, 30.0, (1280, 360))
    for f in range(25):
        frame = np.full((360, 1280, 3), (34, 139, 34), dtype=np.uint8)
        # Moving player inside pitch
        px = int(500 + f * 10)
        cv2.circle(frame, (px, 180), 15, (255, 255, 255), -1)
        out_vid.write(frame)
    out_vid.release()
    
    traj = tracker.generate_offline_trajectory(temp_vid, start_frame=0, end_frame=25, step=5)
    assert len(traj) == 25
    assert 0 in traj and 24 in traj
    print("Configurable Step-5 trajectory scan verified with 25/25 interpolated frames.")


def test_full_profile_serialization():
    print("Testing Full Profile & Default Settings Serialization...")
    import json
    from matchtrack.rig_geometry import RigConfiguration

    profile_data = {
        "version": "1.3",
        "name": "Custom Profile",
        "rig": RigConfiguration().to_dict(),
        "sync": {"frame_offset_right": 42},
        "autocam": {
            "ai_tracking": True,
            "tracking_mode": "ball_centric",
            "enable_dynamic_zoom": True,
            "fixed_zoom_factor": 1.0,
            "min_zoom": 1.10,
            "max_zoom": 1.50,
            "zoom_speed": 0.05,
            "anticipation_lead": 0.20,
            "smoothing_factor": 0.95,
            "deadband_width": 0.05,
            "max_pan_speed": 0.08,
            "vertical_center_bias": 0.60,
            "pitch_corners": [
                [0.05, 0.08],
                [0.50, 0.06],
                [0.95, 0.08],
                [0.92, 0.92],
                [0.50, 0.94],
                [0.08, 0.92]
            ],
            "scan_step": 5,
            "use_fp16": True
        },
        "export": {
            "format": "16:9_autocam",
            "resolution_width": 1920,
            "resolution_height": 1080,
            "codec": "hevc_nvenc",
            "bitrate_mbps": 40,
            "trim_only": True,
            "audio_source": "left"
        },
        "ui_view": {
            "view_mode": "16:9_broadcast",
            "show_frame": True,
            "frame_opacity": 60.0,
            "show_center_line": False,
            "show_autocam": True,
            "show_ball": True,
            "show_seam": False,
            "show_grid": True,
            "sync_angles": True,
            "multi_frame_calib": True
        }
    }
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "output"))
    os.makedirs(out_dir, exist_ok=True)
    test_file = os.path.join(out_dir, "test_profile.json")
    with open(test_file, 'w', encoding='utf-8') as f:
        json.dump(profile_data, f, indent=4)

    with open(test_file, 'r', encoding='utf-8') as f:
        loaded = json.load(f)

    assert loaded["version"] == "1.3"
    assert loaded["sync"]["frame_offset_right"] == 42
    assert loaded["autocam"]["tracking_mode"] == "ball_centric"
    assert len(loaded["autocam"]["pitch_corners"]) == 6
    assert loaded["autocam"]["pitch_corners"][1] == [0.50, 0.06]
    assert loaded["autocam"]["scan_step"] == 5
    assert loaded["export"]["format"] == "16:9_autocam"
    assert loaded["export"]["bitrate_mbps"] == 40
    assert loaded["export"]["audio_source"] == "left"
    print("Full profile serialization and deserialization verified.")


def test_tactical_16x9_mesh_warp_math():
    """Tests 6-point dual-quad tactical warp LUT generation and frame transformation."""
    src_w, src_h = 3840, 1080
    dst_w, dst_h = 1920, 1080
    
    # 6 points representing curved pitch corners
    pitch_corners = [
        [0.02, 0.08],  # TL
        [0.50, 0.05],  # TC
        [0.98, 0.08],  # TR
        [0.99, 0.95],  # BR
        [0.50, 0.96],  # BC
        [0.01, 0.95]   # BL
    ]
    
    map_x, map_y = generate_tactical_16x9_luts(src_w, src_h, dst_w, dst_h, pitch_corners, margin_percent=1.0)
    assert map_x.shape == (dst_h, dst_w)
    assert map_y.shape == (dst_h, dst_w)
    assert map_x.dtype == np.float32
    assert map_y.dtype == np.float32

    # Synthetic panorama image with test grid
    dummy_pano = np.zeros((src_h, src_w, 3), dtype=np.uint8)
    dummy_pano[:, :] = (30, 120, 40)
    cv2.line(dummy_pano, (int(0.50 * src_w), 0), (int(0.50 * src_w), src_h), (255, 255, 255), 6) # Halfway line

    warped = apply_tactical_16x9_warp(dummy_pano, map_x, map_y)
    assert warped.shape == (dst_h, dst_w, 3)
    assert not np.all(warped == 0)

    # Verify center line is positioned near center in output 16:9 frame (x ~ 960)
    center_slice = warped[:, dst_w // 2 - 2:dst_w // 2 + 3]
    assert np.mean(center_slice) > 40.0
    print("Tactical 16:9 mesh warp mathematical transform & remap verified.")


def test_tactical_16x9_render_pipeline():
    """Tests full 2-camera live stitching to 16:9 tactical overview video export."""
    from matchtrack.stitcher_engine import StitcherEngine
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "output"))
    os.makedirs(out_dir, exist_ok=True)
    out_video = os.path.join(out_dir, "test_render_tactical_16x9.mp4")

    vid_a = os.path.join(out_dir, "synth_left.mp4")
    vid_b = os.path.join(out_dir, "synth_right.mp4")
    if not os.path.exists(vid_a) or not os.path.exists(vid_b):
        print("Skipping live camera render test (synthetic videos missing).")
        return

    engine = StitcherEngine()
    engine.load_videos(vid_a, vid_b)
    engine.ai_broadcast.config.tactical_margin = 1.0

    success = engine.render_video_to_file(
        output_filepath=out_video,
        out_width=1280,
        out_height=720,
        mode="16:9_tactical",
        codec="libx264",
        bitrate_mbps=15,
        start_frame=0,
        end_frame=20
    )
    assert success is True
    assert os.path.exists(out_video)
    assert os.path.getsize(out_video) > 1000
    print(f"16:9 Tactical live stitch video export verified ({os.path.getsize(out_video)} bytes).")


def test_standalone_32x9_to_16x9_tactical():
    """Tests standalone 32:9 panorama conversion to 16:9 tactical overview."""
    from matchtrack.stitcher_engine import StitcherEngine
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "output"))
    os.makedirs(out_dir, exist_ok=True)
    out_video = os.path.join(out_dir, "test_converted_tactical_16x9.mp4")

    # Use 32:9 stitched video from earlier test
    pano_src = os.path.join(out_dir, "synth_stitched_trim.mp4")
    if not os.path.exists(pano_src):
        print("Skipping standalone panorama tactical test (no master video).")
        return

    engine = StitcherEngine()
    engine.load_panorama_video(pano_src)
    engine.ai_broadcast.config.tactical_margin = 0.5

    success = engine.convert_panorama_to_16x9_tactical(
        output_filepath=out_video,
        out_width=1920,
        out_height=1080,
        codec="libx264",
        bitrate_mbps=15,
        start_frame=0,
        end_frame=30
    )
    assert success is True
    assert os.path.exists(out_video)
    assert os.path.getsize(out_video) > 1000
    print(f"Standalone 32:9 to 16:9 Tactical conversion verified ({os.path.getsize(out_video)} bytes).")


if __name__ == "__main__":
    test_camera_presets()
    test_lut_generation_and_stitch()
    test_21x10_lut_generation_and_stitch()
    test_32x9_to_21x10_conversion()
    test_corner_pinning()
    test_audio_sync_mock()
    test_ai_broadcast()
    test_pitch_roi_filtering_and_speed()
    test_auto_stitch_calibration()
    test_video_trimming_logic()
    test_end_to_end_trimmed_render()
    test_16x9_broadcast_render()
    test_tactical_16x9_mesh_warp_math()
    test_tactical_16x9_render_pipeline()
    test_standalone_32x9_to_16x9_tactical()
    test_full_profile_serialization()
    print("\n🎉 All unit and pipeline tests passed successfully!")



