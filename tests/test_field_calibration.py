import os
import sys
import unittest
import numpy as np
import cv2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from matchtrack.field_calibration import FieldMarking


class TestFieldCalibration(unittest.TestCase):
    def test_default_points(self):
        fm = FieldMarking()
        self.assertEqual(len(fm.points), 10)
        # Check ordering: P1 near left to P5 near right, P6 far right to P10 far left
        self.assertLess(fm.points[0][0], fm.points[4][0])
        self.assertGreater(fm.points[5][0], fm.points[9][0])

    def test_spline_curve(self):
        fm = FieldMarking()
        curve = fm.get_spline_curve([0, 1, 2, 3, 4], num_samples=30)
        self.assertEqual(len(curve), 30)
        # Verify monotonically increasing x
        for i in range(len(curve) - 1):
            self.assertLessEqual(curve[i][0], curve[i+1][0] + 0.01)

    def test_vertical_bounds(self):
        fm = FieldMarking()
        y_far, y_near = fm.get_vertical_bounds_at_x(0.5)
        self.assertLess(y_far, y_near)
        self.assertGreater(y_far, 0.3)
        self.assertLess(y_near, 0.9)

    def test_undo_and_reset(self):
        fm = FieldMarking()
        p0_orig = fm.points[0]
        fm.save_state()
        fm.set_point(0, 0.42, 0.88)
        self.assertAlmostEqual(fm.points[0][0], 0.42)
        
        # Undo
        self.assertTrue(fm.undo())
        self.assertEqual(fm.points[0], p0_orig)

    def test_serialization(self):
        fm = FieldMarking()
        fm.set_point(3, 0.77, 0.66)
        d = fm.to_dict()
        fm2 = FieldMarking.from_dict(d)
        self.assertEqual(len(fm2.points), 10)
        self.assertAlmostEqual(fm2.points[3][0], 0.77)
        self.assertAlmostEqual(fm2.points[3][1], 0.66)

    def test_auto_detect_synthetic(self):
        # Create a synthetic football pitch (green rectangle on black background)
        img = np.zeros((720, 1280, 3), dtype=np.uint8)
        # Green turf in center
        img[200:600, 100:1180] = [34, 139, 34]  # BGR Forest Green
        fm = FieldMarking()
        res = fm.auto_detect_from_image(img)
        self.assertTrue(res)
        self.assertEqual(len(fm.points), 10)


if __name__ == "__main__":
    unittest.main()
