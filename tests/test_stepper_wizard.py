import os
import sys
import unittest
from PySide6.QtWidgets import QApplication

os.environ["QT_QPA_PLATFORM"] = "offscreen"
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from matchtrack.gui.main_window import MainWindow
from matchtrack.gui.tabs.stepper_wizard import StepperWizardWidget
from matchtrack.field_calibration import FieldMarking


class TestStepperWizard(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(sys.argv)
        cls.win = MainWindow()

    @classmethod
    def tearDownClass(cls):
        cls.win.close()

    def setUp(self):
        self.wizard = getattr(self.win, 'tab_stepper', None) or StepperWizardWidget(self.win)
        self.win.tab_stepper = self.wizard
        self.wizard.go_to_step(0)

    def test_wizard_step_navigation(self):
        self.assertEqual(self.wizard.current_step, 0)
        self.assertEqual(self.wizard.pages_stack.currentIndex(), 0)

        # Step 0 -> Step 1
        self.wizard.next_step()
        self.assertEqual(self.wizard.current_step, 1)
        self.assertEqual(self.wizard.pages_stack.currentIndex(), 1)

        # Step 1 -> Step 2
        self.wizard.next_step()
        self.assertEqual(self.wizard.current_step, 2)
        self.assertEqual(self.wizard.pages_stack.currentIndex(), 2)

        # Step 2 -> Step 3
        self.wizard.next_step()
        self.assertEqual(self.wizard.current_step, 3)

        # Step 3 -> Step 4
        self.wizard.next_step()
        self.assertEqual(self.wizard.current_step, 4)

        # Step 4 -> Step 3
        self.wizard.prev_step()
        self.assertEqual(self.wizard.current_step, 3)

        # Direct navigation
        self.wizard.go_to_step(0)
        self.assertEqual(self.wizard.current_step, 0)

    def test_step1_rig_and_sport_selection(self):
        self.wizard._select_rig("single_camera")
        self.assertEqual(self.wizard.rig_type, "single_camera")
        self.assertTrue(self.wizard.btn_rig_single.isChecked())
        self.assertFalse(self.wizard.btn_rig_two.isChecked())

        self.wizard._select_rig("two_cameras")
        self.assertEqual(self.wizard.rig_type, "two_cameras")
        self.assertTrue(self.wizard.btn_rig_two.isChecked())

        self.wizard._select_sport("football")
        self.assertEqual(self.wizard.sport, "football")

    def test_step4_framing_presets(self):
        self.wizard._select_framing_preset("tactical")
        self.assertEqual(self.wizard.framing_preset, "tactical")
        self.assertAlmostEqual(self.win.ctrl_min_zoom.value(), 1.05)
        self.assertAlmostEqual(self.win.ctrl_max_zoom.value(), 1.30)

        self.wizard._select_framing_preset("broadcast")
        self.assertEqual(self.wizard.framing_preset, "broadcast")
        self.assertAlmostEqual(self.win.ctrl_min_zoom.value(), 1.20)
        self.assertAlmostEqual(self.win.ctrl_max_zoom.value(), 2.00)

        self.wizard._select_framing_preset("fixed")
        self.assertEqual(self.wizard.framing_preset, "fixed")
        self.assertAlmostEqual(self.win.ctrl_min_zoom.value(), 1.20)
        self.assertAlmostEqual(self.win.ctrl_max_zoom.value(), 1.20)

    def test_step5_output_summary(self):
        self.wizard.update_summary()
        self.assertIn("1080", self.wizard.summary_fields["OUTPUT"].text())
        self.assertIn("Zwei Kameras", self.wizard.summary_fields["RIG - SPORT"].text())

    def test_ai_model_selection(self):
        # Combo must contain native and discovered models
        self.assertGreater(self.wizard.combo_wizard_model.count(), 3)
        
        # Test selecting Autocam Soccer model or native model
        idx_soccer = self.wizard.combo_wizard_model.findData("autocam_soccer_balldet_fp16")
        if idx_soccer >= 0:
            self.wizard.combo_wizard_model.setCurrentIndex(idx_soccer)
            self.wizard._on_wizard_model_changed(idx_soccer)
            self.assertEqual(self.win.combo_ai_engine.currentData(), "autocam_soccer_balldet_fp16")
            self.assertIn("Soccer", self.wizard.summary_fields["MODEL"].text())
        else:
            # Fallback test with native model
            idx_rtdetr = self.wizard.combo_wizard_model.findData("rtdetr")
            self.wizard.combo_wizard_model.setCurrentIndex(idx_rtdetr)
            self.wizard._on_wizard_model_changed(idx_rtdetr)
            self.assertEqual(self.win.combo_ai_engine.currentData(), "rtdetr")
            self.assertIn("RT-DETR", self.wizard.summary_fields["MODEL"].text())

    def test_step5_cloud_upload_controls(self):
        # Initial status
        self.assertFalse(self.wizard.chk_wizard_cloud_upload.isChecked())
        self.wizard.update_summary()
        self.assertEqual(self.wizard.summary_fields["CLOUD"].text(), "⚪ Deaktiviert")

        # Simulate authenticated state
        self.win.cloud_client.server_url = "https://cloud.matchtrack.de"
        self.win.cloud_client.access_token = "test_token"
        self.win.cloud_client.username = "CoachTester"
        self.win.cloud_client.user_info = {"username": "CoachTester", "role": "TRAINER"}
        
        self.wizard.update_cloud_ui()
        self.assertIn("CoachTester", self.wizard.lbl_wizard_cloud_status.text())
        self.assertEqual(self.wizard.btn_wizard_cloud_login.text(), "🔑 Konto")

        # Toggle cloud upload checkbox
        self.wizard.chk_wizard_cloud_upload.setChecked(True)
        self.wizard.update_summary()
        self.assertIn("Auto-Upload", self.wizard.summary_fields["CLOUD"].text())
        self.assertTrue(self.win.chk_cloud_auto_upload.isChecked())

        # Test cloud progress display
        self.wizard.on_cloud_upload_progress(5000, 10000, 12.5, 30.0, "Übertrage Teil 1/2")
        self.assertEqual(self.wizard.progress_bar.value(), 50)
        self.assertIn("12.5 MB/s", self.wizard.lbl_progress_status.text())
        self.assertFalse(self.wizard.progress_container.isHidden())
        self.assertTrue(self.wizard.btn_cancel_render.isEnabled())

        # Test cloud finished display
        self.wizard.on_cloud_upload_finished(True, "Erfolgreich", {"match_url": "https://cloud.matchtrack.de/m/123"})
        self.assertEqual(self.wizard.progress_bar.value(), 100)
        self.assertIn("hochgeladen", self.wizard.lbl_progress_status.text())
        self.assertFalse(self.wizard.btn_cancel_render.isEnabled())

    def test_step5_export_live_preview_and_telemetry(self):
        # Verify monitor widget presence
        self.assertTrue(hasattr(self.wizard, 'lbl_export_preview'))
        self.assertEqual(self.wizard.lbl_export_preview.width(), 480)
        self.assertEqual(self.wizard.lbl_export_preview.height(), 270)

        # Start export
        self.wizard.on_export_started()
        self.assertFalse(self.wizard.progress_container.isHidden())
        self.assertEqual(self.wizard.lbl_progress_pct.text(), "0.0%")
        self.assertIn("Processing", self.wizard.lbl_progress_badge.text())

        # Test stepping preview frame reception
        from PySide6.QtGui import QImage, QColor
        test_img = QImage(480, 270, QImage.Format_RGB888)
        test_img.fill(QColor(16, 185, 129))
        self.wizard.on_export_preview_frame(test_img)
        self.assertIsNotNone(self.wizard.lbl_export_preview.pixmap())
        self.assertFalse(self.wizard.lbl_export_preview.pixmap().isNull())

        # Test progress update and telemetry metrics (matching Autocam)
        self.wizard.on_export_progress(615, 10000, 88.8, 35.0, "16:9 Follow-Cam Rendering")
        self.assertEqual(self.wizard.lbl_progress_pct.text(), "6%")
        self.assertEqual(self.wizard.lbl_metric_fps.text(), "88.8")
        self.assertIn("615", self.wizard.lbl_metric_frames.text())
        self.assertIn("Echtzeit", self.wizard.lbl_metric_speed.text())

        # Finish export
        self.wizard.on_export_finished(True, "Fertig")
        self.assertEqual(self.wizard.lbl_progress_pct.text(), "100%")
        self.assertIn("Fertig", self.wizard.lbl_progress_badge.text())

    def test_video_path_display_and_trim_synchronization(self):
        # 1. Video path display
        self.wizard.update_left_video_path("C:/videos/game_left_1080p.mp4")
        self.assertIn("game_left_1080p.mp4", self.wizard.lbl_path_l.text())

        self.wizard.update_right_video_path("C:/videos/game_right_1080p.mp4")
        self.assertIn("game_right_1080p.mp4", self.wizard.lbl_path_r.text())

        self.wizard.update_panorama_video_path("C:/videos/panorama_32x9.mp4")
        self.assertIn("panorama_32x9.mp4", self.wizard.lbl_path_pano.text())

        # 2. Trim synchronization
        self.wizard.update_video_info(10000, 30.0)
        self.assertEqual(self.wizard.total_video_frames, 10000)

        # Test setting in/out from current player position
        self.win.current_frame_idx = 1500
        self.wizard._set_trim_start_from_current()
        self.assertEqual(self.wizard.spin_trim_start.value(), 1500)
        self.assertEqual(self.win.in_point_frame, 1500)

        self.win.current_frame_idx = 8500
        self.wizard._set_trim_end_from_current()
        self.assertEqual(self.wizard.spin_trim_end.value(), 8500)
        self.assertEqual(self.win.out_point_frame, 8500)

        # Test changing in main window updates wizard
        self.win.set_in_point(2000)
        self.assertEqual(self.wizard.spin_trim_start.value(), 2000)

        self.win.set_out_point(7000)
        self.assertEqual(self.wizard.spin_trim_end.value(), 7000)

    def test_dji_action4_1080p_16x9_preset(self):
        idx_1080p = self.wizard.combo_cam_preset.findData("DJI_Action4_1080p_16_9_Dewarp")
        self.assertGreaterEqual(idx_1080p, 0)
        self.wizard.combo_cam_preset.setCurrentIndex(idx_1080p)
        self.wizard._on_preset_selected(idx_1080p)

        self.assertEqual(self.win.rig.left_camera.image_width, 1920)
        self.assertEqual(self.win.rig.left_camera.image_height, 1080)
        self.assertAlmostEqual(self.win.rig.left_camera.hfov_deg, 92.0)
        self.assertAlmostEqual(self.win.ctrl_master_yaw.value(), 77.0)
        self.assertAlmostEqual(self.win.rig.pano_hfov, 154.0)
        self.assertAlmostEqual(self.win.rig.vertical_crop_offset, -0.16)

    def test_step2_framing_controls_sync(self):
        """Verifies Step 2 framing and centering controls update rig and sync properly."""
        self.wizard.go_to_step(1) # Step 2 (Source)

        # Test horizontal pan (Spielfeldmitte)
        self.wizard._on_step2_h_offset_changed(-4.5)
        self.assertAlmostEqual(self.win.rig.global_yaw_center, -4.5)
        self.assertAlmostEqual(self.win.ctrl_h_offset.value(), -4.5)

        # Test vertical shift (Untere Ecken / Höhe)
        self.wizard._on_step2_v_offset_changed(-0.18)
        self.assertAlmostEqual(self.win.rig.vertical_crop_offset, -0.18)
        self.assertAlmostEqual(self.win.ctrl_v_offset.value(), -0.18)

        # Test HFOV (Panorama-Breite)
        self.wizard._on_step2_hfov_changed(156.0)
        self.assertAlmostEqual(self.win.rig.pano_hfov, 156.0)
        self.assertAlmostEqual(self.win.ctrl_pano_hfov.value(), 156.0)

        # Test Curvature (Seitenlinien-Begradigung)
        self.wizard._on_step2_curvature_changed(0.06)
        self.assertAlmostEqual(self.win.rig.field_curvature_correction, 0.06)
        self.assertAlmostEqual(self.win.ctrl_curvature.value(), 0.06)

        # Test Click-to-center pitch
        self.win.on_pitch_center_picked(0.45) # 5% left of center
        self.assertAlmostEqual(self.wizard.ctrl_step2_h_offset.value(), self.win.rig.global_yaw_center)

    def test_viewport_paint_all_view_modes_including_8x3(self):
        """Verifies paintEvent does not crash on 8:3 tactical panorama or any other view mode."""
        from PySide6.QtGui import QImage, QColor, QPainter, QPixmap
        viewport = self.win.viewport

        test_img = QImage(2880, 1080, QImage.Format_RGB888)
        test_img.fill(QColor(30, 41, 59))
        viewport.current_qimage = test_img
        viewport.resize(960, 540)

        modes = [
            "8:3", "32:9", "grid", "cam1", "cam2",
            "wide_angle", "wide_angle_raw", "21:10", "16:9_tactical", "16:9", "custom"
        ]
        domains = ["stitching", "field_marking", "framing", "autocam", "livestream", "wide_angle"]

        pixmap = QPixmap(960, 540)
        for mode in modes:
            for domain in domains:
                viewport.view_mode = mode
                viewport.set_workflow_domain(domain)
                painter = QPainter(pixmap)
                try:
                    viewport._render_viewport(painter)
                finally:
                    painter.end()


if __name__ == "__main__":
    unittest.main()
