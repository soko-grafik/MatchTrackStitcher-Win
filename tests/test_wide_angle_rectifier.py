"""
Unit Tests for WideAngleRectifier & Pitch Trapezoid Engine.
"""
import os
import sys
import time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from matchtrack.wide_angle_rectifier import WideAngleConfig, WideAngleRectifier


def test_config_dimensions_and_serialization():
    cfg = WideAngleConfig(target_format="16:9")
    assert cfg.get_target_dimensions() == (1920, 1080)
    
    cfg.target_format = "21:9"
    assert cfg.get_target_dimensions() == (2560, 1080)
    
    cfg.target_format = "32:9"
    assert cfg.get_target_dimensions() == (3840, 1080)
    
    cfg.target_format = "4:3"
    assert cfg.get_target_dimensions() == (1440, 1080)

    cfg.target_format = "custom"
    cfg.target_width = 1280
    cfg.target_height = 720
    assert cfg.get_target_dimensions() == (1280, 720)

    # Test serialization
    d = cfg.to_dict()
    assert d["target_format"] == "custom"
    assert len(d["pitch_corners"]) == 4

    restored = WideAngleConfig.from_dict(d)
    assert restored.target_width == 1280
    assert restored.target_height == 720
    assert restored.fisheye_strength == cfg.fisheye_strength


def test_remap_luts_generation_and_caching():
    cfg = WideAngleConfig(
        fisheye_strength=0.85,
        hfov_deg=142.0,
        geometry_mode="trapezoid",
        trapezoid_top_width_ratio=0.65,
        margin_percent=5.0
    )
    rectifier = WideAngleRectifier(cfg)
    
    map_x, map_y = rectifier.compute_remap_luts(src_w=1920, src_h=1080, dst_w=1280, dst_h=720)
    assert map_x.shape == (720, 1280)
    assert map_y.shape == (720, 1280)
    assert map_x.dtype == np.float32
    assert map_y.dtype == np.float32

    # Coordinates should be finite
    assert np.all(np.isfinite(map_x))
    assert np.all(np.isfinite(map_y))

    # Test cache hit
    map_x2, map_y2 = rectifier.compute_remap_luts(src_w=1920, src_h=1080, dst_w=1280, dst_h=720)
    assert map_x2 is map_x
    assert map_y2 is map_y

    # Change parameter -> cache should invalidate
    cfg.trapezoid_top_width_ratio = 0.50
    rectifier.update_config(cfg)
    map_x3, map_y3 = rectifier.compute_remap_luts(src_w=1920, src_h=1080, dst_w=1280, dst_h=720)
    assert map_x3 is not map_x


def test_rectifier_apply_execution():
    cfg = WideAngleConfig(target_format="16:9", fisheye_strength=0.80)
    rectifier = WideAngleRectifier(cfg)

    # Create dummy 1080p frame with crosshairs and border
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    frame[:, :] = (30, 40, 50)
    frame[540, :] = (255, 255, 255)
    frame[:, 960] = (255, 255, 255)

    out = rectifier.apply(frame, dst_w=1280, dst_h=720)
    assert out.shape == (720, 1280, 3)

    # Benchmark remap speed
    t0 = time.perf_counter()
    for _ in range(5):
        _ = rectifier.apply(frame, dst_w=1280, dst_h=720)
    avg_ms = ((time.perf_counter() - t0) / 5) * 1000.0
    print(f"Average Remap time: {avg_ms:.2f} ms")
    assert avg_ms < 60.0  # Must be fast enough for high FPS


def test_planar_rect_mode():
    cfg = WideAngleConfig(geometry_mode="planar_rect", target_format="16:9")
    rectifier = WideAngleRectifier(cfg)
    
    map_x, map_y = rectifier.compute_remap_luts(src_w=1920, src_h=1080, dst_w=1920, dst_h=1080)
    assert map_x.shape == (1080, 1920)
    assert np.all(np.isfinite(map_x))


if __name__ == "__main__":
    test_config_dimensions_and_serialization()
    print("test_config_dimensions_and_serialization: OK")
    test_remap_luts_generation_and_caching()
    print("test_remap_luts_generation_and_caching: OK")
    test_rectifier_apply_execution()
    print("test_rectifier_apply_execution: OK")
    test_planar_rect_mode()
    print("test_planar_rect_mode: OK")
    print("\nALL WIDE ANGLE TESTS PASSED SUCCESSFULLY!")

