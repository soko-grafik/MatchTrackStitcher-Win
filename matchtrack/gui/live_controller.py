"""
Live Stream and Camera Capture Controller Mixin for MatchTrack-Stitcher.
Handles UVC capture, DJI BLE camera pairing, multiview grid assembly,
and multi-channel RTMP streaming telemetry.
"""

import os
import cv2
import numpy as np
import logging
from typing import Dict, Any, Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QMessageBox

from ..rtmp_streamer import RTMPChannelConfig, StreamStats
from ..dji_ble import DjiCameraDevice
from ..live_capture import DualLiveCaptureManager, enumerate_video_devices
from .dialogs import DjiQRCodeDialog

logger = logging.getLogger("MatchTrack")


class LiveStreamMixin:
    """Mixin containing all live streaming, UVC capture, and DJI BLE methods for MainWindow."""

    def _on_stream_resolution_changed(self, idx: int):
        res = self.combo_stream_resolution.currentData() if hasattr(self, 'combo_stream_resolution') else "1080p"
        if hasattr(self, 'obs_settings_widget'):
            if res == "720p":
                self.obs_settings_widget.spn_bitrate.setValue(2500)
                self.obs_settings_widget.spn_max_bitrate.setValue(4000)
            elif res == "480p":
                self.obs_settings_widget.spn_bitrate.setValue(1000)
                self.obs_settings_widget.spn_max_bitrate.setValue(1800)
            elif res == "1080p":
                self.obs_settings_widget.spn_bitrate.setValue(3000)
                self.obs_settings_widget.spn_max_bitrate.setValue(5000)

    def _on_rtmp_preset_changed(self, idx: int):
        data = self.combo_rtmp_preset.currentData()
        if data != "custom":
            self.txt_rtmp_server_url.setText(data)

    def _on_grid_quadrant_clicked(self, quad: int):
        """Switches full-screen viewport preview when a quadrant in the 2x2 grid is clicked."""
        if quad == 0:
            self.set_view_mode_by_id("grid")
        elif quad == 1:
            mode = self.combo_live_proc_mode.currentData() if hasattr(self, 'combo_live_proc_mode') else "stitching_dual"
            if mode == "single_cam2":
                self.set_view_mode_by_id("cam2")
            elif mode in ("single_cam1", "dual_cams_nostitch"):
                self.set_view_mode_by_id("cam1")
            else:
                self.set_view_mode_by_id("16:9")
        elif quad == 2:
            mode = self.combo_live_proc_mode.currentData() if hasattr(self, 'combo_live_proc_mode') else "stitching_dual"
            if mode == "dual_cams_nostitch":
                self.set_view_mode_by_id("cam2")
            else:
                self.set_view_mode_by_id("8:3" if self.is_4_3_mode() else "32:9")
        elif quad == 3:
            self.set_view_mode_by_id("cam1")
        elif quad == 4:
            self.set_view_mode_by_id("cam2")

    def _on_live_proc_mode_changed(self):
        """Adapts UI, channels, and camera inputs based on stitching vs. single-stream mode."""
        mode = self.combo_live_proc_mode.currentData() if hasattr(self, 'combo_live_proc_mode') else "stitching_dual"
        if mode == "single_cam1":
            self.combo_uvc_left.setEnabled(True)
            self.combo_uvc_right.setEnabled(False)
            if hasattr(self, 'combo_feed_c1'):
                idx = self.combo_feed_c1.findData("cam1")
                if idx >= 0: self.combo_feed_c1.setCurrentIndex(idx)
            self.lbl_uvc_status.setText("💡 Einzelkamera-Modus: Nur Cam 1 aktiv. Kein Stitching = minimale CPU/GPU-Auslastung.")
        elif mode == "single_cam2":
            self.combo_uvc_left.setEnabled(False)
            self.combo_uvc_right.setEnabled(True)
            if hasattr(self, 'combo_feed_c1'):
                idx = self.combo_feed_c1.findData("cam2")
                if idx >= 0: self.combo_feed_c1.setCurrentIndex(idx)
            self.lbl_uvc_status.setText("💡 Einzelkamera-Modus: Nur Cam 2 aktiv. Kein Stitching = minimale CPU/GPU-Auslastung.")
        elif mode == "dual_cams_nostitch":
            self.combo_uvc_left.setEnabled(True)
            self.combo_uvc_right.setEnabled(True)
            if hasattr(self, 'combo_feed_c1'):
                idx1 = self.combo_feed_c1.findData("cam1")
                if idx1 >= 0: self.combo_feed_c1.setCurrentIndex(idx1)
            if hasattr(self, 'combo_feed_c2'):
                idx2 = self.combo_feed_c2.findData("cam2")
                if idx2 >= 0: self.combo_feed_c2.setCurrentIndex(idx2)
            self.lbl_uvc_status.setText("💡 Dual Direct-Stream: Cam 1 & Cam 2 werden parallel direkt gestreamt (ohne Stitching).")
        else:
            self.combo_uvc_left.setEnabled(True)
            self.combo_uvc_right.setEnabled(True)
            if hasattr(self, 'combo_feed_c1'):
                idx1 = self.combo_feed_c1.findData("broadcast_16x9")
                if idx1 >= 0: self.combo_feed_c1.setCurrentIndex(idx1)
            if hasattr(self, 'combo_feed_c2'):
                idx2 = self.combo_feed_c2.findData("panorama_8x3")
                if idx2 >= 0: self.combo_feed_c2.setCurrentIndex(idx2)
            self.lbl_uvc_status.setText("💡 Dual-Kamera Stitching aktiv: Echtzeit-Panorama & KI Ball-Follow Regie.")

        if getattr(self, 'view_mode', '') == "grid" and hasattr(self, 'live_capture_manager') and not self.live_capture_manager.is_running:
            standby = self._build_multiview_grid(None, None, None, None, mode=mode)
            if hasattr(self, 'viewport'):
                self.viewport.update_preview_image(standby)

    def _on_live_input_mode_changed(self, idx: int):
        mode = self.combo_live_input_mode.currentData()
        self.panel_uvc.setVisible(mode == "uvc")
        self.panel_ble.setVisible(mode == "ble")

    def _refresh_uvc_devices(self):
        self.combo_uvc_left.clear()
        self.combo_uvc_right.clear()
        devices = enumerate_video_devices(max_devices=6)
        if not devices:
            self.combo_uvc_left.addItem("Keine Webcams gefunden", -1)
            self.combo_uvc_right.addItem("--- (Keine / Einzelkamera-Modus) ---", -1)
            self.lbl_uvc_status.setText("Keine USB-Webcams oder Kameras in Windows erkannt.")
            return

        for dev in devices:
            self.combo_uvc_left.addItem(f"[{dev.index}] {dev.name}", dev.index)

        self.combo_uvc_right.addItem("--- (Keine / Einzelkamera-Modus) ---", -1)
        for dev in devices:
            self.combo_uvc_right.addItem(f"[{dev.index}] {dev.name}", dev.index)

        if len(devices) > 1:
            self.combo_uvc_right.setCurrentIndex(2)
            self.lbl_uvc_status.setText(f"{len(devices)} Webcams erkannt (Dual-Kamera Panorama-Stitching bereit).")
        else:
            self.combo_uvc_right.setCurrentIndex(0)
            self.lbl_uvc_status.setText(f"{len(devices)} Webcam erkannt (Einzelkamera Live-Broadcast bereit).")

    def _toggle_uvc_live_capture(self):
        if self.live_capture_manager.is_running:
            self.live_capture_manager.stop()
            self.live_capture_timer.stop()
            self.btn_toggle_uvc.setText("▶️ Live-Kameras aktivieren")
            self.btn_toggle_uvc.setStyleSheet("background-color: #0284c7; color: white; font-weight: bold; border-radius: 6px; padding: 5px 12px; font-size: 11px;")
            self.lbl_uvc_status.setText("Live-Kameras gestoppt.")
            if getattr(self, 'view_mode', '') == "grid":
                mode = self.combo_live_proc_mode.currentData() if hasattr(self, 'combo_live_proc_mode') else "stitching_dual"
                self.viewport.update_preview_image(self._build_multiview_grid(None, None, None, None, mode=mode))
        else:
            mode = self.combo_live_proc_mode.currentData() if hasattr(self, 'combo_live_proc_mode') else "stitching_dual"
            idx_l = self.combo_uvc_left.currentData()
            idx_r = self.combo_uvc_right.currentData()

            if mode == "single_cam2":
                if idx_r is None or idx_r < 0:
                    QMessageBox.warning(self, "Kamera wählen", "Bitte wähle eine gültige USB-Webcam für 'Cam Rechts' aus.")
                    return
                src_l = None
                src_r = idx_r
            elif mode == "single_cam1":
                if idx_l is None or idx_l < 0:
                    QMessageBox.warning(self, "Kamera wählen", "Bitte wähle eine gültige USB-Webcam für 'Cam Links' aus.")
                    return
                src_l = idx_l
                src_r = None
            else:
                if idx_l is None or idx_l < 0:
                    QMessageBox.warning(self, "Kamera wählen", "Bitte wähle mindestens eine gültige USB-Webcam für 'Cam Links' aus.")
                    return
                src_l = idx_l
                src_r = idx_r if (idx_r is not None and idx_r >= 0) else None

            fps_val = self.combo_stream_fps.currentData() or 30
            self.live_capture_manager.start(src_l, src_r, width=1920, height=1080, fps=fps_val)
            self.live_capture_timer.start()
            self.btn_toggle_uvc.setText("⏹ Kameras deaktivieren")
            self.btn_toggle_uvc.setStyleSheet("background-color: #b91c1c; color: white; font-weight: bold; border-radius: 6px; padding: 5px 12px; font-size: 11px;")

            if mode == "single_cam1":
                self.lbl_uvc_status.setText(f"Einzelkamera-Stream aktiv (Cam Links: [{src_l}], 1080p @ {fps_val} FPS - Ohne Stitching).")
            elif mode == "single_cam2":
                self.lbl_uvc_status.setText(f"Einzelkamera-Stream aktiv (Cam Rechts: [{src_r}], 1080p @ {fps_val} FPS - Ohne Stitching).")
            elif mode == "dual_cams_nostitch":
                self.lbl_uvc_status.setText(f"Dual Direct-Stream aktiv (Cams: [{src_l}] & [{src_r}] - Ohne Stitching).")
            else:
                if src_r is None:
                    self.lbl_uvc_status.setText(f"Einzelkamera-Erfassung aktiv (Quelle: [{src_l}], 1080p @ {fps_val} FPS).")
                else:
                    self.lbl_uvc_status.setText(f"Dual-Webcam-Erfassung aktiv (Quellen: [{src_l}] & [{src_r}], 1080p @ {fps_val} FPS).")

    def _build_multiview_grid(
        self,
        bcast_frame: Optional[np.ndarray],
        pano_frame: Optional[np.ndarray],
        frame_l: Optional[np.ndarray],
        frame_r: Optional[np.ndarray],
        mode: str = "stitching_dual"
    ) -> np.ndarray:
        """
        Renders a broadcast-grade 2x2 Multiview Grid (Quad-Split) in 1920x1080:
        Quadrant 1 (Top-Left):     Feed 1 - 16:9 KI-Regie (Ball-Follow) or Cam 1 Single
        Quadrant 2 (Top-Right):    Feed 2 - 8:3/32:9 Taktik-Panorama or Cam 2
        Quadrant 3 (Bottom-Left):  Feed 3 - Cam 1 Links (Roh-Feed)
        Quadrant 4 (Bottom-Right): Feed 4 - Cam 2 Rechts (Roh-Feed)
        """
        gw, gh = 958, 538
        grid_img = np.full((1080, 1920, 3), 16, dtype=np.uint8)

        # Dividing lines between 4 quadrants
        cv2.line(grid_img, (959, 0), (959, 1080), (38, 38, 54), 2)
        cv2.line(grid_img, (0, 539), (1920, 539), (38, 38, 54), 2)

        # Determine which feeds are actively streaming to RTMP
        is_live_any = hasattr(self, 'stream_manager') and self.stream_manager.is_any_live()
        c1_live = is_live_any and hasattr(self, 'chk_chan_1') and self.chk_chan_1.isChecked()
        c2_live = is_live_any and hasattr(self, 'chk_chan_2') and self.chk_chan_2.isChecked()
        c3_live = is_live_any and hasattr(self, 'chk_chan_3') and self.chk_chan_3.isChecked()
        c4_live = is_live_any and hasattr(self, 'chk_chan_4') and self.chk_chan_4.isChecked()

        feed_c1 = self.combo_feed_c1.currentData() if hasattr(self, 'combo_feed_c1') else "broadcast_16x9"
        feed_c2 = self.combo_feed_c2.currentData() if hasattr(self, 'combo_feed_c2') else "panorama_8x3"
        feed_c3 = self.combo_feed_c3.currentData() if hasattr(self, 'combo_feed_c3') else "cam1"
        feed_c4 = self.combo_feed_c4.currentData() if hasattr(self, 'combo_feed_c4') else "cam2"

        live_feeds = set()
        if c1_live and feed_c1: live_feeds.add(feed_c1)
        if c2_live and feed_c2: live_feeds.add(feed_c2)
        if c3_live and feed_c3: live_feeds.add(feed_c3)
        if c4_live and feed_c4: live_feeds.add(feed_c4)

        if mode == "single_cam1":
            q1_img = frame_l
            q1_title = "CAM 1 (LINKS / SINGLE STREAM)"
            q1_live = "cam1" in live_feeds or "broadcast_16x9" in live_feeds
            q1_sub = "HAUPT-STREAM AKTIV"

            q2_img = None
            q2_title = "TAKTIK-PANORAMA"
            q2_live = False
            q2_sub = "INAKTIV (Einzelkamera-Modus)"

            q3_img = frame_l
            q3_title = "CAM 1 LINKS (ROH-FEED)"
            q3_live = "cam1" in live_feeds
            q3_sub = "ROH-SIGNAL"

            q4_img = None
            q4_title = "CAM 2 RECHTS"
            q4_live = False
            q4_sub = "NICHT VERWENDET"

        elif mode == "single_cam2":
            q1_img = frame_r
            q1_title = "CAM 2 (RECHTS / SINGLE STREAM)"
            q1_live = "cam2" in live_feeds or "broadcast_16x9" in live_feeds
            q1_sub = "HAUPT-STREAM AKTIV"

            q2_img = None
            q2_title = "TAKTIK-PANORAMA"
            q2_live = False
            q2_sub = "INAKTIV (Einzelkamera-Modus)"

            q3_img = None
            q3_title = "CAM 1 LINKS"
            q3_live = False
            q3_sub = "NICHT VERWENDET"

            q4_img = frame_r
            q4_title = "CAM 2 RECHTS (ROH-FEED)"
            q4_live = "cam2" in live_feeds
            q4_sub = "ROH-SIGNAL"

        elif mode == "dual_cams_nostitch":
            q1_img = frame_l
            q1_title = "CAM 1 LINKS (DIRECT-STREAM)"
            q1_live = "cam1" in live_feeds or "broadcast_16x9" in live_feeds
            q1_sub = "DIREKT-STREAM"

            q2_img = frame_r
            q2_title = "CAM 2 RECHTS (DIRECT-STREAM)"
            q2_live = "cam2" in live_feeds or "panorama_8x3" in live_feeds
            q2_sub = "DIREKT-STREAM"

            q3_img = frame_l
            q3_title = "CAM 1 LINKS (ROH-FEED)"
            q3_live = "cam1" in live_feeds
            q3_sub = "ROH-SIGNAL"

            q4_img = frame_r
            q4_title = "CAM 2 RECHTS (ROH-FEED)"
            q4_live = "cam2" in live_feeds
            q4_sub = "ROH-SIGNAL"

        else: # stitching_dual
            q1_img = bcast_frame
            q1_title = "16:9 REGIE (BALL-FOLLOW KI)"
            q1_live = "broadcast_16x9" in live_feeds
            q1_sub = "KI-FOLLOW REGIE"

            q2_img = pano_frame
            q2_title = "TAKTIK-PANORAMA (8:3 / 32:9)"
            q2_live = "panorama_8x3" in live_feeds
            q2_sub = "STITCHED PANORAMA"

            q3_img = frame_l
            q3_title = "CAM 1 LINKS (ROH-FEED)"
            q3_live = "cam1" in live_feeds
            q3_sub = "ROH-SIGNAL"

            q4_img = frame_r
            q4_title = "CAM 2 RECHTS (ROH-FEED)"
            q4_live = "cam2" in live_feeds
            q4_sub = "ROH-SIGNAL"

        def render_tile(dest_slice, src_img, title, quad_num, is_live=False, standby_sub="STANDBY"):
            tile = np.full((gh, gw, 3), 18, dtype=np.uint8)
            if src_img is not None and src_img.size > 0:
                ih, iw = src_img.shape[:2]
                target_ar = gw / gh
                src_ar = iw / max(1, ih)

                if abs(src_ar - target_ar) < 0.05:
                    content = cv2.resize(src_img, (gw, gh), interpolation=cv2.INTER_LINEAR)
                elif src_ar > target_ar:
                    fit_h = max(10, int(gw / src_ar))
                    if fit_h > gh: fit_h = gh
                    resized = cv2.resize(src_img, (gw, fit_h), interpolation=cv2.INTER_LINEAR)
                    pad_top = (gh - fit_h) // 2
                    content = np.zeros((gh, gw, 3), dtype=np.uint8)
                    content[pad_top:pad_top+fit_h, 0:gw] = resized
                else:
                    fit_w = max(10, int(gh * src_ar))
                    if fit_w > gw: fit_w = gw
                    resized = cv2.resize(src_img, (fit_w, gh), interpolation=cv2.INTER_LINEAR)
                    pad_left = (gw - fit_w) // 2
                    content = np.zeros((gh, gw, 3), dtype=np.uint8)
                    content[0:gh, pad_left:pad_left+fit_w] = resized
                tile = content

                b_color = (0, 0, 235) if is_live else (65, 55, 45)
                b_thick = 2 if is_live else 1
                cv2.rectangle(tile, (0, 0), (gw-1, gh-1), b_color, b_thick)
            else:
                cv2.rectangle(tile, (0, 0), (gw-1, gh-1), (40, 35, 30), 1)
                # Big watermark number
                cv2.putText(tile, str(quad_num), (gw//2 - 35, gh//2 + 45), cv2.FONT_HERSHEY_DUPLEX, 3.6, (30, 28, 25), 4)
                # Title & Status text
                t_w = len(title) * 9
                cv2.putText(tile, title, (max(20, gw//2 - t_w//2), gh//2 - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (160, 160, 175), 2)
                s_w = len(standby_sub) * 7
                cv2.putText(tile, standby_sub, (max(20, gw//2 - s_w//2), gh//2 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 120), 1)

            # Top Header Bar Overlay
            cv2.rectangle(tile, (0, 0), (gw, 36), (14, 14, 18), -1)
            cv2.line(tile, (0, 36), (gw, 36), (35, 35, 48), 1)

            # Quadrant Number Badge
            badge_color = (37, 99, 235) if not is_live else (0, 0, 220)
            cv2.rectangle(tile, (8, 6), (34, 30), badge_color, -1)
            cv2.putText(tile, str(quad_num), (15, 25), cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 255), 2)

            # Title
            cv2.putText(tile, title, (42, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (230, 230, 240), 1, cv2.LINE_AA)

            # Indicator Badge
            if is_live:
                cv2.rectangle(tile, (gw - 85, 7), (gw - 10, 29), (0, 0, 200), -1)
                cv2.putText(tile, "LIVE", (gw - 72, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1, cv2.LINE_AA)
            elif src_img is not None and src_img.size > 0:
                cv2.rectangle(tile, (gw - 95, 7), (gw - 10, 29), (32, 32, 44), -1)
                cv2.putText(tile, "PREVIEW", (gw - 88, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (160, 160, 175), 1, cv2.LINE_AA)

            dest_slice[:] = tile

        # Render 4 quadrants
        render_tile(grid_img[0:gh, 0:gw], q1_img, q1_title, 1, is_live=q1_live, standby_sub=q1_sub)
        render_tile(grid_img[0:gh, 962:962+gw], q2_img, q2_title, 2, is_live=q2_live, standby_sub=q2_sub)
        render_tile(grid_img[542:542+gh, 0:gw], q3_img, q3_title, 3, is_live=q3_live, standby_sub=q3_sub)
        render_tile(grid_img[542:542+gh, 962:962+gw], q4_img, q4_title, 4, is_live=q4_live, standby_sub=q4_sub)

        return grid_img

    def _on_live_capture_tick(self):
        """Tick event capturing live frames from cameras, stitching/processing, and pushing to RTMP streams."""
        if not self.live_capture_manager.is_running:
            return

        frame_l, frame_r, is_synced = self.live_capture_manager.get_synced_frames()
        if frame_l is None and frame_r is None:
            return

        mode = self.combo_live_proc_mode.currentData() if hasattr(self, 'combo_live_proc_mode') else "stitching_dual"
        res_mode = self.combo_stream_resolution.currentData() if hasattr(self, 'combo_stream_resolution') else "1080p"
        is_4_3 = self.is_4_3_mode()

        if res_mode == "480p":
            pw, ph = (1280, 480) if is_4_3 else (1706, 480)
            bw, bh = 854, 480
        elif res_mode == "720p":
            pw, ph = (1920, 720) if is_4_3 else (2560, 720)
            bw, bh = 1280, 720
        else:
            pw, ph = (2560, 960) if is_4_3 else (3840, 1080)
            bw, bh = 1920, 1080

        pano_frame = None
        bcast_frame = None
        crop_box = None

        if mode == "stitching_dual" and frame_l is not None:
            try:
                pano_frame, bcast_frame, crop_box = self.engine.stitch_live_pair(
                    frame_l, frame_r,
                    pano_width=pw,
                    pano_height=ph,
                    broadcast_width=bw,
                    broadcast_height=bh
                )
            except Exception as e:
                logger.warning(f"Fehler beim Live-Stitching/Capture: {e}")
                return
        elif mode == "single_cam1" or (mode == "stitching_dual" and frame_r is None and frame_l is not None):
            bcast_frame = frame_l
            pano_frame = None
            crop_box = None
        elif mode == "single_cam2":
            bcast_frame = frame_r
            pano_frame = None
            crop_box = None
        elif mode == "dual_cams_nostitch":
            bcast_frame = frame_l
            pano_frame = frame_r
            crop_box = None

        # Push to active RTMP streaming channels via dynamic feed routing
        if self.stream_manager.is_any_live():
            feeds = {
                "broadcast_16x9": bcast_frame if bcast_frame is not None else frame_l,
                "panorama_8x3": pano_frame if pano_frame is not None else frame_r,
                "cam1": frame_l,
                "cam2": frame_r
            }
            self.stream_manager.push_feeds(feeds)

        # Update viewport preview
        vm = self.view_mode
        if vm == "grid":
            display_frame = self._build_multiview_grid(
                bcast_frame=bcast_frame,
                pano_frame=pano_frame,
                frame_l=frame_l,
                frame_r=frame_r,
                mode=mode
            )
            box = None
        elif vm == "cam1":
            display_frame = frame_l if frame_l is not None else bcast_frame
            box = None
        elif vm == "cam2":
            display_frame = frame_r if frame_r is not None else bcast_frame
            box = None
        elif vm.startswith("16:9") and bcast_frame is not None:
            display_frame = bcast_frame
            box = None
        else:
            display_frame = pano_frame if pano_frame is not None else (bcast_frame if bcast_frame is not None else frame_l)
            box = crop_box

        if display_frame is not None and hasattr(self, 'viewport') and self.viewport is not None:
            self.viewport.update_preview_image(display_frame, autocam_box=box)

    def _on_ble_scan_clicked(self):
        self.btn_ble_scan.setEnabled(False)
        self.lbl_ble_status.setText("Suche nach DJI Osmo Action Kameras via Bluetooth...")
        self.dji_ble_manager.scan_for_cameras(timeout_sec=5.0)
        QTimer.singleShot(6000, lambda: self.btn_ble_scan.setEnabled(True))

    def _on_dji_ble_device_found(self, dev: DjiCameraDevice):
        label = f"{dev.name} ({dev.address})"
        # Add to left/right combos if not already present
        if self.combo_dji_cam_left.findData(dev.address) < 0:
            self.combo_dji_cam_left.addItem(label, dev.address)
        if self.combo_dji_cam_right.findData(dev.address) < 0:
            self.combo_dji_cam_right.addItem(label, dev.address)
        self.lbl_ble_status.setText(f"Gefunden: {dev.name}")

    def _on_dji_ble_conn_changed(self, addr: str, connected: bool):
        status_txt = "Verbunden" if connected else "Getrennt"
        self.lbl_ble_status.setText(f"Kamera {addr}: {status_txt}")

    def _on_dji_ble_stream_started(self, addr: str, rtmp_url: str):
        self.lbl_ble_status.setText(f"🎉 Kamera {addr} streamt aktiv zu: {rtmp_url}")
        QMessageBox.information(self, "DJI Stream aktiv", f"Die DJI Action Kamera ({addr}) hat den RTMP-Stream erfolgreich gestartet:\n\n{rtmp_url}")

    def _on_dji_ble_error(self, err_msg: str):
        self.lbl_ble_status.setText(f"Fehler: {err_msg}")

    def _on_dji_connect_left_clicked(self):
        addr = self.combo_dji_cam_left.currentData()
        if addr:
            self.dji_ble_manager.connect_camera(addr)

    def _on_dji_connect_right_clicked(self):
        addr = self.combo_dji_cam_right.currentData()
        if addr:
            self.dji_ble_manager.connect_camera(addr)

    def _on_dji_start_rtmp_clicked(self):
        ssid = self.txt_dji_wifi_ssid.text().strip()
        pwd = self.txt_dji_wifi_pass.text().strip()
        server_url = self.txt_rtmp_server_url.text().strip().rstrip("/")
        if not ssid:
            QMessageBox.warning(self, "WLAN erforderlich", "Bitte gib den Namen deines WLAN-Netzwerks (SSID) ein.")
            return

        addr_l = self.combo_dji_cam_left.currentData()
        addr_r = self.combo_dji_cam_right.currentData()

        if not addr_l and not addr_r:
            QMessageBox.warning(self, "Kamera wählen", "Bitte wähle mindestens eine gefundene DJI-Kamera aus.")
            return

        bitrate = self.combo_stream_bitrate.currentData() or 6000

        if addr_l:
            url_l = f"{server_url}/{self.txt_stream_key_cam1.text().strip()}"
            self.dji_ble_manager.start_rtmp_streaming(addr_l, ssid, pwd, url_l, bitrate_kbps=bitrate)

        if addr_r:
            url_r = f"{server_url}/{self.txt_stream_key_cam2.text().strip()}"
            self.dji_ble_manager.start_rtmp_streaming(addr_r, ssid, pwd, url_r, bitrate_kbps=bitrate)

    def _on_show_dji_qr_dialog(self):
        ssid = self.txt_dji_wifi_ssid.text().strip()
        pwd = self.txt_dji_wifi_pass.text().strip()
        server_url = self.txt_rtmp_server_url.text().strip().rstrip("/")
        key = self.txt_stream_key_cam1.text().strip() or "cam1"
        full_url = f"{server_url}/{key}"

        qr_img = self.dji_ble_manager.generate_dji_stream_qr_image(ssid, pwd, full_url)
        dlg = DjiQRCodeDialog(qr_img, ssid, full_url, parent=self)
        dlg.exec()

    def _toggle_master_live_stream(self):
        """Starts or stops the OBS-Style multi-stream engine."""
        if self.stream_manager.is_any_live():
            self.stream_manager.stop_all()
            self.btn_master_stream_toggle.setText("🔴 LIVE-STREAM STARTEN (OBS-Engine)")
            self.btn_master_stream_toggle.setStyleSheet(
                "background-color: #dc2626; color: white; font-weight: 800; font-size: 13px; padding: 12px; border-radius: 10px;"
            )
            self.lbl_stream_tally.setText("⚪ BEREIT")
            self.lbl_stream_tally.setStyleSheet(
                "background-color: #27273a; color: #a1a1aa; font-weight: 900; font-size: 13px; padding: 6px 14px; border-radius: 8px;"
            )
            self.status_bar.showMessage("Stream beendet.")
        else:
            base_url = self.txt_rtmp_server_url.text().strip()
            if not base_url:
                QMessageBox.warning(self, "Server-URL fehlt", "Bitte gib eine gültige RTMP-Server-URL ein.")
                return

            # Full OBS Studio Output & Encoder settings
            obs_cfg = self.obs_settings_widget.get_settings() if hasattr(self, 'obs_settings_widget') else {}
            out_mode = obs_cfg.get("output_mode", "advanced")
            codec = obs_cfg.get("video_encoder", self.combo_stream_encoder.currentData() or "h264_nvenc")
            bitrate = int(obs_cfg.get("bitrate_kbps", 3000))
            max_bitrate = int(obs_cfg.get("max_bitrate_kbps", 5000))
            keyint_sec = float(obs_cfg.get("keyframe_interval_sec", 2.0))
            fps = int(obs_cfg.get("fps", self.combo_stream_fps.currentData() or 30))
            rc_mode = obs_cfg.get("rate_control", "vbr")
            preset = obs_cfg.get("preset", "p4")
            tuning = obs_cfg.get("tuning", "hq")
            multipass = obs_cfg.get("multipass", "qres")
            profile = obs_cfg.get("profile", "main")
            lookahead = bool(obs_cfg.get("lookahead", True))
            adaptive_quant = bool(obs_cfg.get("adaptive_quant", True))
            b_frames = int(obs_cfg.get("b_frames", 2))
            custom_opts = str(obs_cfg.get("custom_options", "")).strip()
            rescale_output = bool(obs_cfg.get("rescale_output", False))
            res_str = str(obs_cfg.get("rescale_resolution", "1920x1080"))
            rw, rh = (1920, 1080)
            if "x" in res_str:
                try:
                    parts = res_str.split("x")
                    rw, rh = int(parts[0].split()[0]), int(parts[1].split()[0])
                except Exception:
                    rw, rh = (1920, 1080)
            audio_track = int(obs_cfg.get("audio_track", 1))
            audio_encoder = obs_cfg.get("audio_encoder", "aac")

            res_mode = self.combo_stream_resolution.currentData() if hasattr(self, 'combo_stream_resolution') else "1080p"
            is_4_3 = self.is_4_3_mode()

            if res_mode == "480p":
                w_16x9, h_16x9 = 854, 480
                w_pano = 1280 if is_4_3 else 1706
                h_pano = 480
            elif res_mode == "720p":
                w_16x9, h_16x9 = 1280, 720
                w_pano = 1920 if is_4_3 else 2560
                h_pano = 720
            else:  # "1080p"
                w_16x9, h_16x9 = 1920, 1080
                w_pano = 2560 if is_4_3 else 3840
                h_pano = 960 if is_4_3 else 1080

            channels_data = [
                (1, getattr(self, 'chk_chan_1', self.chk_chan_bcast), getattr(self, 'combo_feed_c1', None), getattr(self, 'txt_stream_key_c1', self.txt_stream_key_bcast), "broadcast_16x9"),
                (2, getattr(self, 'chk_chan_2', self.chk_chan_pano), getattr(self, 'combo_feed_c2', None), getattr(self, 'txt_stream_key_c2', self.txt_stream_key_pano), "panorama_8x3"),
                (3, getattr(self, 'chk_chan_3', self.chk_chan_cam1), getattr(self, 'combo_feed_c3', None), getattr(self, 'txt_stream_key_c3', self.txt_stream_key_cam1), "cam1"),
                (4, getattr(self, 'chk_chan_4', self.chk_chan_cam2), getattr(self, 'combo_feed_c4', None), getattr(self, 'txt_stream_key_c4', self.txt_stream_key_cam2), "cam2"),
            ]

            active_count = 0
            for chan_num, chk, combo, txt, default_feed in channels_data:
                if chk and chk.isChecked():
                    feed = combo.currentData() if combo else default_feed
                    key = txt.text().strip() if txt else f"chan_{chan_num}"
                    cw, ch = (w_pano, h_pano) if feed == "panorama_8x3" else (w_16x9, h_16x9)

                    # In multi-stream, secondary feeds can slightly scale bitrate to preserve bandwidth
                    chan_bitrate = bitrate if (feed == "broadcast_16x9" or active_count == 0) else max(400, int(bitrate * 0.75))

                    cfg = RTMPChannelConfig(
                        channel_id=f"channel_{chan_num}",
                        name=f"Kanal {chan_num} ({feed})",
                        stream_url=base_url,
                        stream_key=key,
                        enabled=True,
                        width=cw,
                        height=ch,
                        fps=fps,
                        bitrate_kbps=chan_bitrate,
                        codec=codec,
                        keyframe_interval_sec=keyint_sec,
                        source_feed=feed,
                        output_mode=out_mode,
                        audio_track=audio_track,
                        audio_encoder=audio_encoder,
                        rescale_output=rescale_output,
                        rescale_width=rw,
                        rescale_height=rh,
                        rate_control=rc_mode,
                        max_bitrate_kbps=max_bitrate,
                        preset=preset,
                        tuning=tuning,
                        multipass=multipass,
                        profile=profile,
                        lookahead=lookahead,
                        adaptive_quant=adaptive_quant,
                        b_frames=b_frames,
                        custom_options=custom_opts
                    )
                    self.stream_manager.configure_channel(cfg)
                    active_count += 1

            if active_count == 0:
                QMessageBox.warning(self, "Keine Kanäle aktiv", "Bitte aktiviere mindestens einen Live-Stream Kanal (Kanal 1-4).")
                return

            self.stream_manager.start_all_enabled()

            # If UVC live capture is not running yet, auto-start it if cameras are selected
            if not self.live_capture_manager.is_running and self.combo_live_input_mode.currentData() == "uvc":
                self._toggle_uvc_live_capture()

            self.btn_master_stream_toggle.setText("⏹ LIVE-STREAM STOPPEN")
            self.btn_master_stream_toggle.setStyleSheet(
                "background-color: #059669; color: white; font-weight: 800; font-size: 13px; padding: 12px; border-radius: 10px;"
            )
            self.lbl_stream_tally.setText(f"🔴 LIVE ({res_mode.upper()})")
            self.lbl_stream_tally.setStyleSheet(
                "background-color: #dc2626; color: white; font-weight: 900; font-size: 13px; padding: 6px 14px; border-radius: 8px;"
            )
            self.status_bar.showMessage(f"OBS-Stream aktiv ({active_count} Kanäle @ {res_mode.upper()})")

    def _handle_stream_telemetry(self, stats_dict: dict):
        """Thread-safe slot updating GUI with current live stream telemetry."""
        if not stats_dict:
            return

        total_kbps = sum(s.current_bitrate_kbps for s in stats_dict.values())
        max_uptime = max((s.uptime_sec for s in stats_dict.values()), default=0.0)
        total_drops = sum(s.frames_dropped for s in stats_dict.values())
        avg_fps = max((s.current_fps for s in stats_dict.values()), default=0.0)

        m, s = divmod(int(max_uptime), 60)
        h, m = divmod(m, 60)
        uptime_str = f"{h:02d}:{m:02d}:{s:02d}"

        active_count = sum(1 for s in stats_dict.values() if s.is_live)
        if active_count > 0:
            self.lbl_stream_tally.setText(f"🔴 LIVE ({active_count})")
            self.lbl_stream_telemetry.setText(
                f"Bitrate: {total_kbps:.0f} kbps ({total_kbps/1024:.1f} MB/s) | FPS: {avg_fps:.1f} | Uptime: {uptime_str} | Drops: {total_drops}"
            )
        else:
            if self.stream_manager.is_any_live():
                self.lbl_stream_tally.setText("🟡 VERBINDE")
                self.lbl_stream_telemetry.setText("Verbinde mit RTMP-Server...")
            else:
                self.lbl_stream_tally.setText("⚪ BEREIT")
                self.lbl_stream_telemetry.setText("Bereit für Live-Stream.")



