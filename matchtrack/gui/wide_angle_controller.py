"""
Wide-Angle Dewarping & 4-Point Pitch Trapezoid Controller Mixin for MainWindow.
Handles interactive corner marking, camera presets, geometry transformations,
and video opening.
"""

import os
import logging
from typing import Optional

from PySide6.QtWidgets import QFileDialog, QMessageBox

logger = logging.getLogger("MatchTrack")


class WideAngleControllerMixin:
    """Mixin containing Wide-Angle & 4-Point Trapezoid calibration methods for MainWindow."""

    def _on_wa_start_marking(self):
        """Starts 4-point sequential corner placement in the viewport."""
        self.set_view_mode_by_id("wide_angle_raw")
        self.viewport.start_wide_angle_marking()
        self.status_bar.showMessage("🎯 Klicken Sie nacheinander auf die 4 Eckfahnen: TL (Gelb) -> TR (Blau) -> BR (Rot) -> BL (Grün)")

    def _on_wa_reset_corners(self):
        """Resets the 4 pitch corners to standard default positions."""
        default_corners = [[0.15, 0.25], [0.85, 0.25], [0.95, 0.90], [0.05, 0.90]]
        self.wide_angle_config.corners = default_corners
        self.viewport.set_wide_angle_corners(default_corners)
        self.wide_angle_rectifier.config = self.wide_angle_config
        self.wide_angle_rectifier.update_maps()
        if hasattr(self, 'lbl_wa_calib_status'):
            self.lbl_wa_calib_status.setText("● 4/4 Ecken aktiv (Standard)")
            self.lbl_wa_calib_status.setStyleSheet("background-color: #064e3b; color: #34d399; font-weight: 700; font-size: 11px; padding: 4px 8px; border-radius: 6px;")
        self.refresh_preview()

    def _on_wa_edit_toggled(self, checked: bool):
        """Toggles direct mouse drag editing of corners in viewport."""
        self.viewport.set_wide_angle_edit_mode(checked)

    def _on_viewport_wa_corners_changed(self, corners: list):
        """Handles corner repositioning from viewport drag or click."""
        if hasattr(self, 'wide_angle_config'):
            self.wide_angle_config.corners = [[float(c[0]), float(c[1])] for c in corners]
            self.wide_angle_rectifier.config = self.wide_angle_config
            self.wide_angle_rectifier.update_maps()
            self.refresh_preview()
            if hasattr(self, 'lbl_wa_calib_status'):
                self.lbl_wa_calib_status.setText("● 4/4 Ecken aktiv")
                self.lbl_wa_calib_status.setStyleSheet("background-color: #064e3b; color: #34d399; font-weight: 700; font-size: 11px; padding: 4px 8px; border-radius: 6px;")

    def _on_viewport_wa_step_changed(self, step: int):
        """Updates UI status during sequential 4-point placement."""
        if not hasattr(self, 'lbl_wa_calib_status'):
            return
        step_names = [
            "1/4: Eckfahne OBEN LINKS (TL)",
            "2/4: Eckfahne OBEN RECHTS (TR)",
            "3/4: Eckfahne UNTEN RECHTS (BR)",
            "4/4: Eckfahne UNTEN LINKS (BL)"
        ]
        if 0 <= step < 4:
            self.lbl_wa_calib_status.setText(f"⏳ Schritt {step_names[step]}")
            self.lbl_wa_calib_status.setStyleSheet("background-color: #78350f; color: #fbbf24; font-weight: 700; font-size: 11px; padding: 4px 8px; border-radius: 6px;")
            if hasattr(self, 'combo_view_mode') and self.combo_view_mode.currentData() != "wide_angle_raw":
                self.set_view_mode_by_id("wide_angle_raw")
        else:
            self.lbl_wa_calib_status.setText("● 4/4 Ecken aktiv (Trapez kalibriert)")
            self.lbl_wa_calib_status.setStyleSheet("background-color: #064e3b; color: #34d399; font-weight: 700; font-size: 11px; padding: 4px 8px; border-radius: 6px;")
            if hasattr(self, 'combo_view_mode') and self.combo_view_mode.currentData() != "wide_angle":
                self.set_view_mode_by_id("wide_angle")
            self.refresh_preview()

    def _on_wa_preset_changed(self, index: int):
        """Applies lens and geometry presets."""
        preset_data = self.combo_wa_preset.currentData()
        if preset_data == "action4_155":
            self.wide_angle_config.hfov_deg = 138.0
            self.wide_angle_config.fisheye_k1 = 0.85
            self.wide_angle_config.zoom = 1.05
            self.wide_angle_config.top_width_ratio = 0.72
        elif preset_data == "action35_155":
            self.wide_angle_config.hfov_deg = 136.0
            self.wide_angle_config.fisheye_k1 = 0.82
            self.wide_angle_config.zoom = 1.05
            self.wide_angle_config.top_width_ratio = 0.72
        elif preset_data == "gopro_hyper":
            self.wide_angle_config.hfov_deg = 130.0
            self.wide_angle_config.fisheye_k1 = 0.78
            self.wide_angle_config.zoom = 1.08
            self.wide_angle_config.top_width_ratio = 0.70
        elif preset_data == "generic_fisheye":
            self.wide_angle_config.hfov_deg = 140.0
            self.wide_angle_config.fisheye_k1 = 0.90
            self.wide_angle_config.zoom = 1.00
            self.wide_angle_config.top_width_ratio = 0.72

        self._sync_wide_angle_ui_from_config()
        self.wide_angle_rectifier.config = self.wide_angle_config
        self.wide_angle_rectifier.update_maps()
        self.refresh_preview()

    def _on_wa_param_changed(self, *args):
        """Updates WideAngleConfig and remap LUTs when sliders change."""
        if not hasattr(self, 'ctrl_wa_strength'):
            return
        self.wide_angle_config.fisheye_k1 = float(self.ctrl_wa_strength.value())
        self.wide_angle_config.hfov_deg = float(self.ctrl_wa_hfov.value())
        self.wide_angle_config.zoom = float(self.ctrl_wa_zoom.value())
        self.wide_angle_config.tilt_deg = float(self.ctrl_wa_tilt.value())
        self.wide_angle_config.roll_deg = float(self.ctrl_wa_roll.value())
        self.wide_angle_config.v_offset = float(self.ctrl_wa_v_offset.value())
        self.wide_angle_config.top_width_ratio = float(self.ctrl_wa_top_ratio.value())
        self.wide_angle_config.pitch_margin = float(self.ctrl_wa_margin.value())

        self.wide_angle_rectifier.config = self.wide_angle_config
        self.wide_angle_rectifier.update_maps()
        self.refresh_preview()

    def _on_wa_geom_mode_toggled(self, checked: bool):
        """Toggles between trapezoid perspective and FIFA planar rectangle."""
        if self.radio_geom_planar.isChecked():
            self.wide_angle_config.trapezoid_mode = False
            if hasattr(self, 'ctrl_wa_top_ratio'):
                self.ctrl_wa_top_ratio.setEnabled(False)
        else:
            self.wide_angle_config.trapezoid_mode = True
            if hasattr(self, 'ctrl_wa_top_ratio'):
                self.ctrl_wa_top_ratio.setEnabled(True)
        self.wide_angle_rectifier.config = self.wide_angle_config
        self.wide_angle_rectifier.update_maps()
        self.refresh_preview()

    def _on_wa_target_format_changed(self, index: int):
        """Updates target aspect ratio and resolution."""
        data = self.combo_wa_target_format.currentData()
        if data == "custom":
            if hasattr(self, 'widget_wa_custom_res'):
                self.widget_wa_custom_res.setVisible(True)
            w = self.spin_wa_custom_w.value()
            h = self.spin_wa_custom_h.value()
            self.wide_angle_config.target_width = w
            self.wide_angle_config.target_height = h
            self.wide_angle_config.aspect_ratio = w / max(1, h)
        else:
            if hasattr(self, 'widget_wa_custom_res'):
                self.widget_wa_custom_res.setVisible(False)
            if isinstance(data, (tuple, list)) and len(data) == 2:
                w, h = data
                self.wide_angle_config.target_width = w
                self.wide_angle_config.target_height = h
                self.wide_angle_config.aspect_ratio = w / max(1, h)

        self.wide_angle_rectifier.config = self.wide_angle_config
        self.wide_angle_rectifier.update_maps()
        self.update_resolution_dropdown()
        self.refresh_preview()

    def _on_wa_custom_res_changed(self, val: int):
        """Handles custom resolution spinbox changes."""
        w = self.spin_wa_custom_w.value()
        h = self.spin_wa_custom_h.value()
        self.wide_angle_config.target_width = w
        self.wide_angle_config.target_height = h
        self.wide_angle_config.aspect_ratio = w / max(1, h)
        self.wide_angle_rectifier.config = self.wide_angle_config
        self.wide_angle_rectifier.update_maps()
        self.update_resolution_dropdown()
        self.refresh_preview()

    def _on_wa_export_clicked(self):
        """Switches to export tab pre-configured for Wide-Angle export."""
        self.show_workspace(3) # Export Tab
        idx = self.combo_exp_format.findData("wide_angle")
        if idx >= 0:
            self.combo_exp_format.setCurrentIndex(idx)

    def _sync_wide_angle_ui_from_config(self):
        """Synchronizes GUI sliders and selectors with self.wide_angle_config."""
        cfg = self.wide_angle_config
        widgets = [
            (getattr(self, 'ctrl_wa_strength', None), cfg.fisheye_k1),
            (getattr(self, 'ctrl_wa_hfov', None), cfg.hfov_deg),
            (getattr(self, 'ctrl_wa_zoom', None), cfg.zoom),
            (getattr(self, 'ctrl_wa_tilt', None), cfg.tilt_deg),
            (getattr(self, 'ctrl_wa_roll', None), cfg.roll_deg),
            (getattr(self, 'ctrl_wa_v_offset', None), cfg.v_offset),
            (getattr(self, 'ctrl_wa_top_ratio', None), cfg.top_width_ratio),
            (getattr(self, 'ctrl_wa_margin', None), cfg.pitch_margin),
        ]
        for ctrl, val in widgets:
            if ctrl is not None:
                ctrl.blockSignals(True)
                ctrl.setValue(val)
                ctrl.blockSignals(False)

        if hasattr(self, 'radio_geom_trapezoid') and hasattr(self, 'radio_geom_planar'):
            self.radio_geom_trapezoid.blockSignals(True)
            self.radio_geom_planar.blockSignals(True)
            self.radio_geom_trapezoid.setChecked(cfg.trapezoid_mode)
            self.radio_geom_planar.setChecked(not cfg.trapezoid_mode)
            self.radio_geom_trapezoid.blockSignals(False)
            self.radio_geom_planar.blockSignals(False)
            if hasattr(self, 'ctrl_wa_top_ratio'):
                self.ctrl_wa_top_ratio.setEnabled(cfg.trapezoid_mode)

    def open_wide_angle_video(self, file_path: Optional[str] = None):
        """Opens a single ultra-wide angle video file for dewarping and pitch calibration."""
        if not file_path:
            file_path, _ = QFileDialog.getOpenFileName(
                self, "155° Weitwinkel-Video öffnen", "",
                "Video Files (*.mp4 *.mov *.mkv *.avi *.ts);;All Files (*.*)"
            )
        if not file_path or not os.path.exists(file_path):
            return

        try:
            self.video_path_l = file_path
            self.video_path_r = None
            self.panorama_path = None
            self.engine.load_videos(file_path, None)
            self._after_videos_loaded()

            fn = os.path.basename(file_path)
            w = self.engine.video_left.width if self.engine.video_left else 0
            h = self.engine.video_left.height if self.engine.video_left else 0
            fps = self.engine.get_fps()
            if hasattr(self, 'lbl_wa_video_info'):
                self.lbl_wa_video_info.setText(f"📹 {fn} ({w}x{h} @ {fps:.1f} fps)")
                self.lbl_wa_video_info.setStyleSheet("color: #10b981; font-weight: 700; font-size: 11px;")
            if hasattr(self, 'lbl_video_status'):
                self.lbl_video_status.setText(f"📐 Weitwinkel: {fn}")
                self.lbl_video_status.setStyleSheet("color: #38bdf8; font-weight: 600;")

            self.set_view_mode_by_id("wide_angle")
            self.show_workspace(1)
            self.seek_frame(0)
            self.status_bar.showMessage(f"📐 Weitwinkel-Video geladen: {fn}")
        except Exception as e:
            QMessageBox.critical(self, "Fehler beim Laden", f"Video konnte nicht geöffnet werden: {e}")

    # =========================================================================
    # TAB 5: OBS-STYLE RTMP LIVE-STREAMING & CAMERA DIRECT CONNECT
    # =========================================================================


