import os
import sys
import unittest
import numpy as np
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt, QPoint, QRectF
from PySide6.QtGui import QImage

# Ensure offscreen Qt platform
os.environ["QT_QPA_PLATFORM"] = "offscreen"
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from matchtrack.gui.viewport import InteractivePanoramaViewport
from matchtrack.gui.main_window import MainWindow


class TestPhotoshopCropAndTransform(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(sys.argv)

    def test_viewport_crop_and_stretch(self):
        vp = InteractivePanoramaViewport()
        vp.resize(1280, 720)
        # Setup dummy test image
        img = QImage(3840, 1080, QImage.Format_RGB888)
        img.fill(Qt.green)
        vp.current_qimage = img
        vp.set_workflow_domain("stitching")

        # Initial state
        self.assertFalse(vp.crop_mode)
        self.assertEqual(vp.crop_rect_norm, [0.0, 0.0, 1.0, 1.0])
        self.assertEqual(vp.transform_stretch_x, 1.0)
        self.assertEqual(vp.transform_stretch_y, 1.0)

        # Toggle crop mode
        vp.set_crop_mode(True)
        self.assertTrue(vp.crop_mode)

        # Target rect with stretch
        tr1 = vp._get_target_rect()
        self.assertIsNotNone(tr1)
        w1 = tr1.width()

        # Apply anamorphic horizontal stretch (1.5x)
        vp.set_transform_stretch(1.5, 1.0)
        self.assertEqual(vp.transform_stretch_x, 1.5)
        tr2 = vp._get_target_rect()
        self.assertIsNotNone(tr2)

        # Test crop rect setting & clamping
        vp.set_crop_rect(0.1, 0.2, 0.6, 0.5)
        self.assertAlmostEqual(vp.crop_rect_norm[0], 0.1, places=3)
        self.assertAlmostEqual(vp.crop_rect_norm[1], 0.2, places=3)
        self.assertAlmostEqual(vp.crop_rect_norm[2], 0.6, places=3)
        self.assertAlmostEqual(vp.crop_rect_norm[3], 0.5, places=3)

        # Test pixel crop rect calculation
        crop_px = vp._get_crop_rect_px(tr2)
        self.assertAlmostEqual(crop_px.left(), tr2.left() + 0.1 * tr2.width(), delta=1.0)
        self.assertAlmostEqual(crop_px.top(), tr2.top() + 0.2 * tr2.height(), delta=1.0)
        self.assertAlmostEqual(crop_px.width(), 0.6 * tr2.width(), delta=1.0)
        self.assertAlmostEqual(crop_px.height(), 0.5 * tr2.height(), delta=1.0)

        # Test handle hit testing
        # Top-left corner handle
        pt_tl = QPoint(int(crop_px.left()), int(crop_px.top()))
        handle = vp._hit_test_crop_handle(pt_tl, tr2)
        self.assertEqual(handle, 'tl')

        # Center handle / move body
        pt_center = QPoint(int(crop_px.center().x()), int(crop_px.center().y()))
        handle_center = vp._hit_test_crop_handle(pt_center, tr2)
        self.assertEqual(handle_center, 'inside')

        # Bottom-right corner handle
        pt_br = QPoint(int(crop_px.right()), int(crop_px.bottom()))
        handle_br = vp._hit_test_crop_handle(pt_br, tr2)
        self.assertEqual(handle_br, 'br')

        # Test Aspect Ratio Lock
        vp.set_crop_aspect_ratio(16.0 / 9.0)
        self.assertAlmostEqual(vp.crop_aspect_ratio, 16.0 / 9.0, places=3)

    def test_main_window_crop_ui_integration(self):
        win = MainWindow()
        win.resize(1400, 900)

        # Verify crop widgets in Stitching tab
        self.assertTrue(hasattr(win, 'btn_crop_mode'))
        self.assertTrue(hasattr(win, 'combo_crop_ratio'))
        self.assertTrue(hasattr(win, 'spin_crop_x'))
        self.assertTrue(hasattr(win, 'spin_crop_y'))
        self.assertTrue(hasattr(win, 'spin_crop_w'))
        self.assertTrue(hasattr(win, 'spin_crop_h'))
        self.assertTrue(hasattr(win, 'ctrl_stretch_x'))
        self.assertTrue(hasattr(win, 'ctrl_stretch_y'))
        self.assertTrue(hasattr(win, 'btn_reset_crop'))

        # Toggle crop mode
        win.btn_crop_mode.setChecked(True)
        self.assertTrue(win.crop_mode)
        self.assertTrue(win.viewport.crop_mode)

        # Edit spinboxes and verify viewport receives update
        win.spin_crop_x.setValue(15.0)
        win.spin_crop_y.setValue(20.0)
        win.spin_crop_w.setValue(70.0)
        win.spin_crop_h.setValue(60.0)
        self.assertAlmostEqual(win.viewport.crop_rect_norm[0], 0.15, places=2)
        self.assertAlmostEqual(win.viewport.crop_rect_norm[1], 0.20, places=2)
        self.assertAlmostEqual(win.viewport.crop_rect_norm[2], 0.70, places=2)
        self.assertAlmostEqual(win.viewport.crop_rect_norm[3], 0.60, places=2)

        # Change stretch slider
        win.ctrl_stretch_x.setValue(1.25)
        self.assertAlmostEqual(win.transform_stretch_x, 1.25, places=2)
        self.assertAlmostEqual(win.viewport.transform_stretch_x, 1.25, places=2)

        # Test Reset
        win.reset_crop_and_transform()
        self.assertAlmostEqual(win.viewport.crop_rect_norm[0], 0.0, places=2)
        self.assertAlmostEqual(win.viewport.crop_rect_norm[1], 0.0, places=2)
        self.assertAlmostEqual(win.viewport.crop_rect_norm[2], 1.0, places=2)
        self.assertAlmostEqual(win.viewport.crop_rect_norm[3], 1.0, places=2)
        self.assertAlmostEqual(win.transform_stretch_x, 1.0, places=2)
        self.assertAlmostEqual(win.transform_stretch_y, 1.0, places=2)
        self.assertAlmostEqual(win.spin_crop_x.value(), 0.0, places=1)
        self.assertAlmostEqual(win.spin_crop_w.value(), 100.0, places=1)

        # Check export format has custom_crop option
        idx = win.combo_stitch_format.findData("custom_crop")
        self.assertGreaterEqual(idx, 0)
        win.combo_stitch_format.setCurrentIndex(idx)
        win.on_export_format_changed()
        self.assertGreater(win.combo_stitch_res.count(), 0)

        win.close()


if __name__ == "__main__":
    unittest.main()
