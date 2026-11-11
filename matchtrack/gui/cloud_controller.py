"""
MatchTrack Online Cloud Integration Mixin for MainWindow.
Handles cloud authentication, match synchronization, background upload workers,
and telemetry.
"""

import os
import webbrowser
import logging
from typing import Optional, Dict, Any

from PySide6.QtWidgets import QMessageBox, QFileDialog, QDialog

from .dialogs import CloudLoginDialog
from .workers import CloudUploadWorker

logger = logging.getLogger("MatchTrack")


class CloudIntegrationMixin:
    """Mixin containing all MatchTrack Online Cloud-Upload methods for MainWindow."""

    def restore_cloud_session_on_startup(self):
        """Silently restores DPAPI-encrypted session if present."""
        try:
            profile = self.cloud_client.restore_saved_session()
            if profile:
                self._update_cloud_ui_connected(profile)
                self.refresh_cloud_teams_and_matches()
        except Exception as e:
            logger.debug(f"Cloud-Auto-Restore übersprungen: {e}")

    def _update_cloud_ui_connected(self, profile: Dict[str, Any]):
        username = profile.get("username") or self.cloud_client.username
        role = profile.get("role", "TRAINER")
        host = self.cloud_client.server_url.replace("https://", "").replace("http://", "")
        is_ak = bool(self.cloud_client.access_token and self.cloud_client.access_token.startswith("mto_ak_"))
        auth_tag = " [API-Key]" if is_ak else ""
        self.lbl_cloud_status.setText(f"🟢 {username} ({role}){auth_tag} @ {host}")
        self.lbl_cloud_status.setStyleSheet(
            "background-color: #0d2818; color: #34d399; border: 1px solid #059669; "
            "border-radius: 8px; padding: 5px 10px; font-size: 11px; font-weight: 700;"
        )
        self.btn_cloud_login.setText("🔑 Konto")
        self.btn_cloud_refresh.setEnabled(True)
        self.btn_cloud_logout.setVisible(True)
        self.saved_cloud_server_url = self.cloud_client.server_url
        if hasattr(self, 'tab_stepper'):
            self.tab_stepper.update_cloud_ui(profile)

    def _update_cloud_ui_disconnected(self):
        self.lbl_cloud_status.setText("🔴 Nicht verbunden")
        self.lbl_cloud_status.setStyleSheet(
            "background-color: #1a1520; color: #f87171; border: 1px solid #3f202b; "
            "border-radius: 8px; padding: 5px 10px; font-size: 11px; font-weight: 600;"
        )
        self.btn_cloud_login.setText("🔑 Verbinden")
        self.btn_cloud_refresh.setEnabled(False)
        self.btn_cloud_logout.setVisible(False)
        self.combo_cloud_team.clear()
        self.combo_cloud_team.addItem("— Bitte erst verbinden —", "")
        self.combo_cloud_existing_match.clear()
        if hasattr(self, 'tab_stepper'):
            self.tab_stepper.update_cloud_ui(None)

    def open_cloud_login_dialog(self):
        dlg = CloudLoginDialog(
            self.cloud_client,
            initial_url=self.cloud_client.server_url or self.saved_cloud_server_url,
            parent=self
        )
        if dlg.exec() == QDialog.Accepted and self.cloud_client.is_authenticated:
            self._update_cloud_ui_connected(self.cloud_client.user_info)
            self.refresh_cloud_teams_and_matches()
            self.status_bar.showMessage(f"☁️ Verbunden mit MatchTrack Online ({self.cloud_client.server_url})")

    def logout_cloud_session(self):
        self.cloud_client.logout()
        self._update_cloud_ui_disconnected()
        self.status_bar.showMessage("☁️ Von MatchTrack Online abgemeldet.")

    def refresh_cloud_teams_and_matches(self):
        if not self.cloud_client.is_authenticated:
            return
        try:
            teams = self.cloud_client.fetch_teams()
            prev_tid = self.combo_cloud_team.currentData() or self.saved_cloud_team_id
            self.combo_cloud_team.blockSignals(True)
            self.combo_cloud_team.clear()
            self.combo_cloud_team.addItem("— Keine Mannschaft / Allgemein —", "")
            restored_idx = 0
            for idx, t in enumerate(teams, start=1):
                t_id = t.get("id", "")
                t_name = t.get("name", "Mannschaft")
                age = t.get("age_group")
                label = f"{t_name} ({age})" if age else t_name
                self.combo_cloud_team.addItem(label, t_id)
                if prev_tid and t_id == prev_tid:
                    restored_idx = idx
            if restored_idx == 0 and len(teams) == 1:
                restored_idx = 1
            self.combo_cloud_team.setCurrentIndex(restored_idx)
            self.combo_cloud_team.blockSignals(False)

            matches = self.cloud_client.fetch_recent_matches(limit=30)
            self.combo_cloud_existing_match.blockSignals(True)
            self.combo_cloud_existing_match.clear()
            for m in matches:
                m_id = m.get("id", "")
                m_name = m.get("name", m_id)
                m_team = m.get("team_name") or ""
                v_type = m.get("video_type") or ""
                badge = f" [{v_type}]" if v_type else ""
                team_str = f" ({m_team})" if m_team else ""
                self.combo_cloud_existing_match.addItem(f"{m_name}{team_str}{badge}", m)
            self.combo_cloud_existing_match.blockSignals(False)
            if hasattr(self, 'tab_stepper'):
                self.tab_stepper.sync_cloud_teams()
        except Exception as e:
            logger.warning(f"Fehler beim Aktualisieren der Cloud-Daten: {e}")

    def on_cloud_target_mode_changed(self):
        is_existing = (self.combo_cloud_target_mode.currentData() == "existing")
        self.lbl_cloud_existing.setVisible(is_existing)
        self.combo_cloud_existing_match.setVisible(is_existing)
        if is_existing:
            self.on_cloud_existing_match_selected()

    def on_cloud_existing_match_selected(self):
        if self.combo_cloud_target_mode.currentData() != "existing":
            return
        m_data = self.combo_cloud_existing_match.currentData()
        if isinstance(m_data, dict):
            if m_data.get("name"):
                self.txt_cloud_match_name.setText(str(m_data["name"]))
            if m_data.get("team_id"):
                idx = self.combo_cloud_team.findData(m_data["team_id"])
                if idx >= 0:
                    self.combo_cloud_team.setCurrentIndex(idx)
            if m_data.get("category"):
                idx = self.combo_cloud_category.findData(m_data["category"])
                if idx >= 0:
                    self.combo_cloud_category.setCurrentIndex(idx)

    def _suggest_default_match_name(self, sample_filepath: Optional[str] = None) -> str:
        current = self.txt_cloud_match_name.text().strip()
        if current:
            return current
        if sample_filepath:
            folder_name = os.path.basename(os.path.dirname(os.path.abspath(sample_filepath)))
            base_name = os.path.splitext(os.path.basename(sample_filepath))[0]
            if folder_name and folder_name.lower() not in ("videos", "export", "output", "desktop", "downloads"):
                return folder_name
            return base_name
        return f"Spiel vom {self.date_cloud_recording.date().toString('dd.MM.yyyy')}"

    def pick_cloud_16x9_video(self):
        """Selects a finished 16:9 Broadcast MP4 video for uploading to MatchTrack Online."""
        fp, _ = QFileDialog.getOpenFileName(
            self,
            "Fertiges 16:9 Broadcast-Video auswählen",
            "",
            "MP4 Video (*.mp4 *.m4v *.mov)"
        )
        if fp:
            self.txt_cloud_video_16x9.setText(fp)
            if not self.txt_cloud_match_name.text().strip():
                self.txt_cloud_match_name.setText(self._suggest_default_match_name(fp))

    def pick_cloud_pano_video(self):
        """Selects a finished Panorama (32:9 / 8:3 / 21:10) MP4 video for uploading to MatchTrack Online."""
        fp, _ = QFileDialog.getOpenFileName(
            self,
            "Fertiges Panorama-Video (32:9 / 8:3 / 21:10) auswählen",
            "",
            "MP4 Video (*.mp4 *.m4v *.mov)"
        )
        if fp:
            self.txt_cloud_video_pano.setText(fp)
            if not self.txt_cloud_match_name.text().strip():
                self.txt_cloud_match_name.setText(self._suggest_default_match_name(fp))

    def start_manual_cloud_upload_dialog(self):
        """Uploads the selected Broadcast (16:9) and/or Panorama video(s), or prompts to pick them if empty."""
        if not self.cloud_client.is_authenticated:
            self.open_cloud_login_dialog()
            if not self.cloud_client.is_authenticated:
                return

        video_16x9 = self.txt_cloud_video_16x9.text().strip() if hasattr(self, 'txt_cloud_video_16x9') else ""
        video_32x9 = self.txt_cloud_video_pano.text().strip() if hasattr(self, 'txt_cloud_video_pano') else ""

        video_16x9 = video_16x9 if (video_16x9 and os.path.isfile(video_16x9)) else None
        video_32x9 = video_32x9 if (video_32x9 and os.path.isfile(video_32x9)) else None

        if not video_16x9 and not video_32x9:
            files, _ = QFileDialog.getOpenFileNames(
                self,
                "Fertige Videodatei(en) für MatchTrack Online auswählen (16:9 Broadcast und/oder Panorama)",
                "",
                "MP4 Video (*.mp4 *.m4v *.mov)"
            )
            if not files:
                return

            for fp in files:
                low = os.path.basename(fp).lower()
                is_pano_file = any(t in low for t in ("32x9", "8x3", "21x10", "pano", "master"))
                if not is_pano_file:
                    try:
                        cap = cv2.VideoCapture(fp)
                        if cap.isOpened():
                            vw = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                            vh = max(1.0, cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            if (vw / vh) >= 2.0:
                                is_pano_file = True
                        cap.release()
                    except Exception:
                        pass

                if is_pano_file and not video_32x9:
                    video_32x9 = fp
                elif not is_pano_file and not video_16x9:
                    video_16x9 = fp
                elif not video_32x9:
                    video_32x9 = fp
                elif not video_16x9:
                    video_16x9 = fp

            if video_16x9 and hasattr(self, 'txt_cloud_video_16x9'):
                self.txt_cloud_video_16x9.setText(video_16x9)
            if video_32x9 and hasattr(self, 'txt_cloud_video_pano'):
                self.txt_cloud_video_pano.setText(video_32x9)

        sample_fp = video_16x9 or video_32x9
        if sample_fp and not self.txt_cloud_match_name.text().strip():
            self.txt_cloud_match_name.setText(self._suggest_default_match_name(sample_fp))

        self._start_cloud_upload(video_16x9_path=video_16x9, video_32x9_path=video_32x9)

    def _start_cloud_upload(self, video_16x9_path: Optional[str] = None, video_32x9_path: Optional[str] = None):
        if self.cloud_upload_worker and self.cloud_upload_worker.isRunning():
            QMessageBox.information(self, "Cloud-Upload", "Ein Cloud-Upload läuft bereits im Hintergrund.")
            return

        if not self.cloud_client.is_authenticated:
            self.open_cloud_login_dialog()
            if not self.cloud_client.is_authenticated:
                return

        sample_file = video_16x9_path or video_32x9_path
        match_name = self._suggest_default_match_name(sample_file)
        self.txt_cloud_match_name.setText(match_name)

        team_id = self.combo_cloud_team.currentData() or None
        user_role = str(self.cloud_client.user_info.get("role", "")).upper()
        if user_role != "ADMIN" and not team_id:
            QMessageBox.warning(
                self,
                "Mannschaft erforderlich",
                "Bitte wähle im Dropdown 'Mannschaft' eine dir zugewiesene Mannschaft für den Upload aus."
            )
            return

        target_match_id = None
        if self.combo_cloud_target_mode.currentData() == "existing":
            m_data = self.combo_cloud_existing_match.currentData()
            if isinstance(m_data, dict) and m_data.get("id"):
                target_match_id = m_data["id"]

        res_data = self.combo_resolution.currentData()
        h_px = res_data[1] if isinstance(res_data, (tuple, list)) and len(res_data) == 2 else 1080
        video_quality = f"{h_px}p"

        rec_date_str = self.date_cloud_recording.date().toString("yyyy-MM-dd")
        is_pw = self.chk_cloud_password.isChecked() and bool(self.txt_cloud_password.text().strip())
        pw_val = self.txt_cloud_password.text().strip() if is_pw else None

        concurrency_val = self.combo_cloud_concurrency.currentData() if hasattr(self, 'combo_cloud_concurrency') else 3

        upload_kwargs = {
            "name": match_name,
            "team_id": team_id,
            "category": self.combo_cloud_category.currentData() or "Punktspiel",
            "recording_date": rec_date_str,
            "video_quality": video_quality,
            "match_id": target_match_id,
            "video_16x9_path": video_16x9_path,
            "video_32x9_path": video_32x9_path,
            "is_password_protected": is_pw,
            "password": pw_val,
            "skip_conversion": self.chk_cloud_fast_hls.isChecked(),
            "notify_team_members": self.chk_cloud_notify.isChecked(),
            "concurrency": concurrency_val or 3
        }

        self.btn_cloud_upload_now.setEnabled(False)
        self.btn_cloud_cancel.setEnabled(True)
        self.btn_cloud_open_browser.setEnabled(False)
        self.cloud_progress_bar.setValue(0)
        self.lbl_cloud_upload_stats.setText("☁️ Verbinde mit MatchTrack Online & starte Upload...")

        self.cloud_upload_worker = CloudUploadWorker(self.cloud_client, upload_kwargs)
        self.cloud_upload_worker.progress_signal.connect(self.on_cloud_upload_progress)
        self.cloud_upload_worker.finished_signal.connect(self.on_cloud_upload_finished)
        if hasattr(self, 'tab_stepper'):
            self.cloud_upload_worker.progress_signal.connect(self.tab_stepper.on_cloud_upload_progress)
            self.cloud_upload_worker.finished_signal.connect(self.tab_stepper.on_cloud_upload_finished)
        self.cloud_upload_worker.start()

    def on_cloud_upload_progress(self, sent_bytes: int, total_bytes: int, speed_mb_s: float, eta_sec: float, status_text: str):
        pct = int((sent_bytes / max(total_bytes, 1)) * 100)
        self.cloud_progress_bar.setValue(pct)
        if speed_mb_s > 0.01:
            m, s = divmod(int(eta_sec), 60)
            self.lbl_cloud_upload_stats.setText(f"{status_text} | {speed_mb_s:.1f} MB/s | Restzeit: {m:02d}:{s:02d}")
        else:
            self.lbl_cloud_upload_stats.setText(status_text)

    def on_cloud_upload_finished(self, success: bool, message: str, result: dict):
        self.btn_cloud_upload_now.setEnabled(True)
        self.btn_cloud_cancel.setEnabled(False)
        if success:
            self.cloud_progress_bar.setValue(100)
            match_url = result.get("match_url", "")
            self.last_uploaded_match_url = match_url
            self.btn_cloud_open_browser.setEnabled(bool(match_url))
            self.lbl_cloud_upload_stats.setText(f"✅ Upload abgeschlossen: {message}")
            self.status_bar.showMessage("☁️ Spiel erfolgreich auf MatchTrack Online veröffentlicht!")
            self.refresh_cloud_teams_and_matches()

            msg_box = QMessageBox(self)
            msg_box.setIcon(QMessageBox.Information)
            msg_box.setWindowTitle("☁️ MatchTrack Online Upload erfolgreich")
            msg_box.setText(
                f"🎉 Das Spiel wurde erfolgreich zu MatchTrack Online hochgeladen und angelegt!\n\n"
                f"Spiel: {result.get('match_name', self.txt_cloud_match_name.text())}\n"
                f"URL: {match_url}"
            )
            btn_open = msg_box.addButton("🌐 Jetzt im Browser öffnen", QMessageBox.AcceptRole)
            msg_box.addButton("Schließen", QMessageBox.RejectRole)
            msg_box.exec()
            if msg_box.clickedButton() == btn_open and match_url:
                QDesktopServices.openUrl(QUrl(match_url))
        else:
            self.lbl_cloud_upload_stats.setText(f"❌ Upload fehlgeschlagen: {message}")
            QMessageBox.warning(self, "Cloud-Upload fehlgeschlagen", f"Der Upload zu MatchTrack Online konnte nicht abgeschlossen werden:\n\n{message}")

    def cancel_cloud_upload(self):
        if self.cloud_upload_worker and self.cloud_upload_worker.isRunning():
            logger.info("Benutzer hat Abbruch des Cloud-Uploads angefordert.")
            self.cloud_upload_worker.cancel()
            self.lbl_cloud_upload_stats.setText("Upload-Abbruch wird ausgeführt...")

    def open_last_uploaded_match_in_browser(self):
        if self.last_uploaded_match_url:
            QDesktopServices.openUrl(QUrl(self.last_uploaded_match_url))

    # =========================================================================
    # TAB 2: 📐 WEITWINKEL-STUDIO & 4-PUNKT SPIELFELD-TRAPEZ
    # =========================================================================


