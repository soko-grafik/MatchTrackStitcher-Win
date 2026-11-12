import unittest
import numpy as np
import json
from matchtrack.rig_geometry import RigConfiguration
from matchtrack.lut_generator import generate_remap_luts, apply_stitch
from matchtrack.color_matcher import ColorExposureMatcher


class TestCornerAndColorCalibration(unittest.TestCase):
    def test_16x9_profile_corner_coverage(self):
        """Verify 16:9 profiles cover the complete horizontal and vertical sensor area (no corner cut-offs)."""
        profiles = [
            'configs/dji_action4_1080p_16x9_dewarp_rocksteady.json',
            'configs/dji_action4_1080p_16x9_dewarp.json'
        ]
        for p in profiles:
            with open(p, 'r', encoding='utf-8') as f:
                data = json.load(f)
            rig = RigConfiguration.from_dict(data)
            luts = generate_remap_luts(rig, 2560, 720)
            
            # Left camera must span from 0.0 to 1919.0 horizontally
            self.assertEqual(luts.map_x_left.min(), 0.0, f"Profile {p} clipped left horizontal edge!")
            self.assertEqual(luts.map_x_left.max(), 1919.0, f"Profile {p} missing left camera seam coverage!")
            
            # Must cover the entire vertical height from near field to horizon
            self.assertEqual(luts.map_y_left.min(), 0.0, f"Profile {p} clipped top!")
            self.assertEqual(luts.map_y_left.max(), 1079.0, f"Profile {p} clipped bottom pitch corners!")

    def test_color_exposure_calibration(self):
        """Verify ColorExposureMatcher actively aligns color, exposure and white-balance."""
        with open('configs/dji_action4_1080p_16x9_dewarp_rocksteady.json', 'r', encoding='utf-8') as f:
            rig = RigConfiguration.from_dict(json.load(f))
        luts = generate_remap_luts(rig, 2560, 720)

        matcher = ColorExposureMatcher(enabled=True)

        # Left camera frame: brighter, cooler/greenish (120, 160, 90)
        fl = np.full((1080, 1920, 3), (120, 160, 90), dtype=np.uint8)
        # Right camera frame: darker, warmer/yellowish (80, 120, 100)
        fr = np.full((1080, 1920, 3), (80, 120, 100), dtype=np.uint8)

        res = matcher.calibrate(fl, fr, luts)
        self.assertTrue(res["success"])
        self.assertTrue(matcher.initialized)
        self.assertTrue(matcher.enabled)

        # Gains should bring both towards the average
        gl = np.array(res["gain_l"])
        gr = np.array(res["gain_r"])
        
        # Left was brighter, so gain_l should be < 1.0 on luminance
        self.assertLess(gl[1], 1.0)
        # Right was darker, so gain_r should be > 1.0 on luminance
        self.assertGreater(gr[1], 1.0)

        # Apply stitch with calibrated color matcher
        stitched = apply_stitch(fl, fr, luts, color_matcher=matcher)
        self.assertEqual(stitched.shape, (720, 2560, 3))


if __name__ == '__main__':
    unittest.main()
