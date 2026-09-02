import os
import sys
import numpy as np
import cv2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from matchtrack.tactical_warp import generate_tactical_16x9_luts, apply_tactical_16x9_warp
from matchtrack.stitcher_engine import StitcherEngine

def run_tactical_tests():
    print("Testing 16:9 Tactical Warp math...")
    src_w, src_h = 3840, 1080
    dst_w, dst_h = 1920, 1080
    pitch_corners = [
        [0.02, 0.08],
        [0.50, 0.05],
        [0.98, 0.08],
        [0.99, 0.95],
        [0.50, 0.96],
        [0.01, 0.95]
    ]
    map_x, map_y = generate_tactical_16x9_luts(src_w, src_h, dst_w, dst_h, pitch_corners, margin_percent=1.0)
    assert map_x.shape == (dst_h, dst_w)
    assert map_y.shape == (dst_h, dst_w)
    assert map_x.dtype == np.float32
    assert map_y.dtype == np.float32

    dummy = np.zeros((src_h, src_w, 3), dtype=np.uint8)
    dummy[:, :] = (30, 140, 30)
    cv2.line(dummy, (int(0.50 * src_w), 0), (int(0.50 * src_w), src_h), (255, 255, 255), 6)
    warped = apply_tactical_16x9_warp(dummy, map_x, map_y)
    assert warped.shape == (dst_h, dst_w, 3)

    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "output"))
    os.makedirs(out_dir, exist_ok=True)
    out_img = os.path.join(out_dir, "test_tactical_16x9_rectified.jpg")
    cv2.imwrite(out_img, warped)
    print(f"Saved tactical warp test frame to: {out_img}")

    print("Testing preview mode '16:9_tactical' in StitcherEngine...")
    engine = StitcherEngine()
    engine.ai_broadcast.config.tactical_margin = 1.0
    engine.ai_broadcast.config.pitch_corners = pitch_corners

    # Create synthetic panorama video for instant testing
    synth_pano = os.path.join(out_dir, "synth_pano_test.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(synth_pano, fourcc, 30.0, (src_w, src_h))
    for _ in range(10):
        writer.write(dummy)
    writer.release()

    engine.load_panorama_video(synth_pano)
    p_frame, crop_box, ball_px, zoom = engine.render_preview_frame(0, 1920, 1080, view_mode="16:9_tactical")
    assert p_frame is not None
    assert p_frame.shape == (1080, 1920, 3)
    print("Preview render in 16:9_tactical mode verified.")

    # Convert panorama to 16:9 tactical
    tactical_out = os.path.join(out_dir, "test_tactical_output.mp4")
    success = engine.convert_panorama_to_16x9_tactical(
        output_filepath=tactical_out,
        out_width=1280,
        out_height=720,
        codec="libx264",
        bitrate_mbps=15,
        start_frame=0,
        end_frame=10
    )
    assert success is True
    assert os.path.exists(tactical_out)
    assert os.path.getsize(tactical_out) > 500
    print(f"16:9 Tactical batch export verified: {tactical_out} ({os.path.getsize(tactical_out)} bytes)")
    print("\nALL 16:9 TACTICAL TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    run_tactical_tests()

