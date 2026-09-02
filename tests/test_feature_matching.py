"""
End-to-End Test for SIFT Overlap Matching on Synthetic Textured Frames.
"""
import sys
import os
import numpy as np
import cv2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from matchtrack.auto_stitch import extract_overlap_matches, optimize_rig_calibration
from matchtrack.rig_geometry import RigConfiguration, CameraPose


def test_sift_overlap_matching():
    print("Testing End-to-End SIFT Overlap Matching on Synthetic Images...")
    w, h = 1920, 1080

    # Create synthetic soccer pitch scene
    scene = np.full((h, w * 2, 3), (34, 139, 34), dtype=np.uint8)

    # Draw complex texture: pitch lines, circle, sponsor ads, corner marks
    cv2.circle(scene, (w, 540), 200, (255, 255, 255), 10)
    cv2.line(scene, (w, 0), (w, h), (255, 255, 255), 12)
    cv2.line(scene, (0, 800), (w * 2, 800), (255, 255, 255), 12)

    # Add random textured features across the middle seam region
    np.random.seed(42)
    for _ in range(80):
        cx = np.random.randint(int(w * 0.7), int(w * 1.3))
        cy = np.random.randint(100, h - 100)
        radius = np.random.randint(4, 20)
        color = tuple(int(c) for c in np.random.randint(50, 255, 3))
        cv2.circle(scene, (cx, cy), radius, color, -1)

    # Left Camera views [0 ... w]
    # Right Camera views [w - overlap ... 2w - overlap]
    overlap_px = int(w * 0.40) # 40% overlap
    frame_left = scene[:, :w].copy()
    frame_right = scene[:, (w - overlap_px):(2 * w - overlap_px)].copy()

    # Extract matches
    matches = extract_overlap_matches(frame_left, frame_right, max_features=1500)
    print(f"Extracted {len(matches)} robust SIFT overlap matches!")
    assert len(matches) >= 20

    # Verify matching coordinate consistency
    # Left camera overlap starts at (w - overlap_px)
    # Right camera overlap starts at 0
    # True relationship: u_right = u_left - (w - overlap_px)
    diffs = [abs(m.u_right - (m.u_left - (w - overlap_px))) for m in matches]
    mean_diff = np.mean(diffs)
    print(f"Mean spatial alignment error across matches: {mean_diff:.2f} px")
    assert mean_diff < 3.0
    print("SIFT Feature extraction and RANSAC verification passed!")


if __name__ == "__main__":
    test_sift_overlap_matching()

