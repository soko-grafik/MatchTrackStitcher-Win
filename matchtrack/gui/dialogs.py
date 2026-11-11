"""
Modal dialogs for MatchTrack-Stitcher:
- DjiQRCodeDialog: Displays DJI Action RTMP QR code
- CloudLoginDialog: Secure connection dialog for MatchTrack Online
- OBSAdvancedOutputDialog: Dedicated window for OBS Studio streaming output settings
"""

import io
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QCheckBox, QWidget, QApplication
)

from ..cloud_uploader import MatchTrackOnlineClient
from .obs_output_settings import OBSOutputSettingsWidget

class DjiQRCodeDialog(QDialog):
    """Dialog displaying a DJI Action compatible QR-Code for instant camera live-stream configuration."""
    def __init__(self, qr_pil_image, wifi_ssid: str, rtmp_url: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("📱 DJI Live-Stream QR-Code")
        self.setMinimumWidth(380)
        self.setStyleSheet("""
            QDialog { background-color: #12121a; color: #f4f4f6; }
            QLabel { color: #e4e4e7; font-size: 12px; }
            QPushButton { background-color: #27273a; color: #f4f4f6; border: 1px solid #3f3f5a; border-radius: 8px; padding: 6px 14px; font-weight: 600; }
            QPushButton:hover { background-color: #3b3b55; }
        """)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(12)

        lbl_hdr = QLabel("📷 Halte die DJI Action vor diesen QR-Code:")
        lbl_hdr.setStyleSheet("font-weight: 700; font-size: 13px; color: #38bdf8;")
        lbl_hdr.setAlignment(Qt.AlignCenter)
        layout.addWidget(lbl_hdr)

        lbl_qr = QLabel()
        lbl_qr.setAlignment(Qt.AlignCenter)
        if qr_pil_image:
            import io
            buf = io.BytesIO()
            qr_pil_image.save(buf, format="PNG")
            pix = QPixmap()
            pix.loadFromData(buf.getvalue(), "PNG")
            lbl_qr.setPixmap(pix.scaled(280, 280, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            lbl_qr.setText("QR-Code konnte nicht generiert werden.")
        layout.addWidget(lbl_qr)

        info_box = QLabel(f"WLAN: <b>{wifi_ssid}</b><br>RTMP: <b>{rtmp_url}</b>")
        info_box.setStyleSheet("background-color: #1a1a26; border: 1px solid #28283a; border-radius: 6px; padding: 8px; font-size: 11px;")
        info_box.setAlignment(Qt.AlignCenter)
        info_box.setWordWrap(True)
        layout.addWidget(info_box)

        btn_close = QPushButton("Schließen")
        btn_close.clicked.connect(self.accept)
        layout.addWidget(btn_close)


class CloudLoginDialog(QDialog):
    """Secure login dialog for connecting MatchTrack-Stitcher to MatchTrack Online via API-Key or Username/Password."""
    def __init__(self, client: MatchTrackOnlineClient, initial_url: str = "", parent=None):
        super().__init__(parent)
        self.client = client
        self.setWindowTitle("☁️ Mit MatchTrack Online verbinden")
        self.setMinimumWidth(460)
        self.setStyleSheet("""
            QDialog { background-color: #12121a; color: #f4f4f6; }
            QLabel { color: #e4e4e7; font-size: 12px; }
            QLineEdit {
                background-color: #0b0b10;
                border: 1px solid #2e2e40;
                border-radius: 8px;
                padding: 8px 10px;
                color: #ffffff;
                font-size: 12px;
            }
            QLineEdit:focus { border-color: #3b82f6; }
        """)

        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(18, 18, 18, 18)

        lbl_header = QLabel("🔑 Verbindung mit MatchTrack Online")
        lbl_header.setStyleSheet("font-size: 15px; font-weight: bold; color: #60a5fa;")
        layout.addWidget(lbl_header)

        lbl_sub = QLabel(
            "Verbinde dich per API-Key (im Web-Admin unter 'Einstellungen → API-Keys & Stitcher' "
            "generierbar) oder mit Benutzername & Passwort. Die Zugangsdaten werden lokal "
            "über die Windows-DPAPI verschlüsselt gespeichert."
        )
        lbl_sub.setStyleSheet("color: #9ca3af; font-size: 11px;")
        lbl_sub.setWordWrap(True)
        layout.addWidget(lbl_sub)

        layout.addWidget(QLabel("Server-URL (HTTPS):"))
        self.txt_url = QLineEdit()
        self.txt_url.setPlaceholderText("https://analyse.dein-verein.de")
        self.txt_url.setText(initial_url or client.server_url or "")
        layout.addWidget(self.txt_url)

        layout.addWidget(QLabel("Authentifizierungs-Methode:"))
        mode_row = QHBoxLayout()
        self.btn_mode_apikey = QPushButton("🔑 API-Key (Empfohlen)")
        self.btn_mode_apikey.setCheckable(True)
        self.btn_mode_userpw = QPushButton("👤 Benutzername & Passwort")
        self.btn_mode_userpw.setCheckable(True)

        is_currently_apikey = not client.username or (client.access_token and client.access_token.startswith("mto_ak_"))
        self.btn_mode_apikey.setChecked(bool(is_currently_apikey))
        self.btn_mode_userpw.setChecked(not bool(is_currently_apikey))

        self.btn_mode_apikey.clicked.connect(lambda: self._switch_auth_mode("apikey"))
        self.btn_mode_userpw.clicked.connect(lambda: self._switch_auth_mode("userpw"))
        mode_row.addWidget(self.btn_mode_apikey)
        mode_row.addWidget(self.btn_mode_userpw)
        layout.addLayout(mode_row)

        # Container 1: API-Key Eingabe
        self.apikey_container = QWidget()
        ak_layout = QVBoxLayout(self.apikey_container)
        ak_layout.setContentsMargins(0, 4, 0, 0)
        ak_layout.setSpacing(6)
        ak_layout.addWidget(QLabel("API-Key (mto_ak_...):"))
        ak_row = QHBoxLayout()
        self.txt_apikey = QLineEdit()
        self.txt_apikey.setEchoMode(QLineEdit.Password)
        self.txt_apikey.setPlaceholderText("mto_ak_...")
        if client.access_token and client.access_token.startswith("mto_ak_"):
            self.txt_apikey.setText(client.access_token)
        btn_toggle_ak = QPushButton("👁")
        btn_toggle_ak.setCheckable(True)
        btn_toggle_ak.setFixedWidth(38)
        btn_toggle_ak.setToolTip("API-Key ein-/ausblenden")
        btn_toggle_ak.toggled.connect(
            lambda checked: self.txt_apikey.setEchoMode(QLineEdit.Normal if checked else QLineEdit.Password)
        )
        ak_row.addWidget(self.txt_apikey, 1)
        ak_row.addWidget(btn_toggle_ak)
        ak_layout.addLayout(ak_row)

        lbl_ak_hint = QLabel(
            "💡 Tipp: Einen neuen API-Key kannst du im Web-Adminbereich unter "
            "„System Einstellungen → API-Keys & Stitcher“ mit 1 Klick erzeugen."
        )
        lbl_ak_hint.setStyleSheet("color: #6ee7b7; font-size: 10px;")
        lbl_ak_hint.setWordWrap(True)
        ak_layout.addWidget(lbl_ak_hint)
        layout.addWidget(self.apikey_container)

        # Container 2: Benutzername & Passwort Eingabe
        self.userpw_container = QWidget()
        up_layout = QVBoxLayout(self.userpw_container)
        up_layout.setContentsMargins(0, 4, 0, 0)
        up_layout.setSpacing(6)
        up_layout.addWidget(QLabel("Benutzername:"))
        self.txt_user = QLineEdit()
        self.txt_user.setPlaceholderText("Benutzername eingeben...")
        self.txt_user.setText(client.username or "")
        up_layout.addWidget(self.txt_user)

        up_layout.addWidget(QLabel("Passwort:"))
        pw_row = QHBoxLayout()
        self.txt_pass = QLineEdit()
        self.txt_pass.setEchoMode(QLineEdit.Password)
        self.txt_pass.setPlaceholderText("Passwort eingeben...")
        btn_toggle_pw = QPushButton("👁")
        btn_toggle_pw.setCheckable(True)
        btn_toggle_pw.setFixedWidth(38)
        btn_toggle_pw.setToolTip("Passwort ein-/ausblenden")
        btn_toggle_pw.toggled.connect(
            lambda checked: self.txt_pass.setEchoMode(QLineEdit.Normal if checked else QLineEdit.Password)
        )
        pw_row.addWidget(self.txt_pass, 1)
        pw_row.addWidget(btn_toggle_pw)
        up_layout.addLayout(pw_row)
        layout.addWidget(self.userpw_container)

        self.chk_remember = QCheckBox("🔒 Zugangsdaten sicher über Windows-DPAPI auf diesem PC speichern")
        self.chk_remember.setChecked(True)
        self.chk_remember.setStyleSheet("color: #10b981; font-weight: 600; margin-top: 4px;")
        layout.addWidget(self.chk_remember)

        self.lbl_error = QLabel("")
        self.lbl_error.setStyleSheet("color: #ef4444; font-size: 11px; font-weight: 600;")
        self.lbl_error.setWordWrap(True)
        self.lbl_error.setVisible(False)
        layout.addWidget(self.lbl_error)

        btn_row = QHBoxLayout()
        btn_cancel = QPushButton("Abbrechen")
        btn_cancel.clicked.connect(self.reject)
        self.btn_login = QPushButton("☁️ Sicher verbinden")
        self.btn_login.setStyleSheet(
            "background-color: #2563eb; color: white; font-weight: bold; padding: 8px 16px; border-radius: 8px;"
        )
        self.btn_login.setDefault(True)
        self.btn_login.clicked.connect(self._do_login)
        btn_row.addStretch()
        btn_row.addWidget(btn_cancel)
        btn_row.addWidget(self.btn_login)
        layout.addLayout(btn_row)

        self._switch_auth_mode("apikey" if is_currently_apikey else "userpw")

    def _switch_auth_mode(self, mode: str):
        is_ak = (mode == "apikey")
        self.btn_mode_apikey.setChecked(is_ak)
        self.btn_mode_userpw.setChecked(not is_ak)
        active_style = (
            "background-color: #1e3a8a; color: #93c5fd; border: 1px solid #3b82f6; "
            "font-weight: bold; padding: 7px 10px; border-radius: 8px;"
        )
        inactive_style = (
            "background-color: #161622; color: #9ca3af; border: 1px solid #2e2e40; "
            "font-weight: normal; padding: 7px 10px; border-radius: 8px;"
        )
        self.btn_mode_apikey.setStyleSheet(active_style if is_ak else inactive_style)
        self.btn_mode_userpw.setStyleSheet(inactive_style if is_ak else active_style)
        self.apikey_container.setVisible(is_ak)
        self.userpw_container.setVisible(not is_ak)
        self.lbl_error.setVisible(False)
        self.adjustSize()

    def _do_login(self):
        self.lbl_error.setVisible(False)
        self.btn_login.setEnabled(False)
        self.btn_login.setText("Verbinde...")
        QApplication.processEvents()
        try:
            if self.btn_mode_apikey.isChecked():
                self.client.login_with_api_key(
                    server_url=self.txt_url.text(),
                    api_key=self.txt_apikey.text(),
                    remember=self.chk_remember.isChecked()
                )
            else:
                self.client.login(
                    server_url=self.txt_url.text(),
                    username=self.txt_user.text(),
                    password=self.txt_pass.text(),
                    remember=self.chk_remember.isChecked()
                )
            self.accept()
        except Exception as e:
            self.lbl_error.setText(f"❌ {e}")
            self.lbl_error.setVisible(True)
        finally:
            self.btn_login.setEnabled(True)
            self.btn_login.setText("☁️ Sicher verbinden")




class OBSAdvancedOutputDialog(QDialog):
    """
    Dedicated OBS Studio Output Settings Dialog window (matching OBS Studio 1:1).
    Provides extra breathing room, full form controls, and OK/Abbrechen/Übernehmen buttons.
    """
    def __init__(self, settings_widget: OBSOutputSettingsWidget, parent=None):
        super().__init__(parent)
        self.setWindowTitle("OBS Studio - Ausgabe-Einstellungen")
        self.resize(720, 600)
        self.setMinimumSize(640, 480)
        self.settings_widget = settings_widget

        # Snapshot current settings for cancel/revert
        self._initial_settings = settings_widget.get_settings()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        # Embedded copy of settings widget
        self.inner_widget = OBSOutputSettingsWidget(self)
        self.inner_widget.btn_popout.hide() # Hide popout button inside dialog
        self.inner_widget.set_settings(self._initial_settings)
        layout.addWidget(self.inner_widget, 1)

        # Bottom buttons: OK, Abbrechen, Übernehmen
        btn_box = QHBoxLayout()
        btn_box.addStretch()

        btn_ok = QPushButton("OK")
        btn_ok.setStyleSheet("background-color: #2563eb; color: white; font-weight: bold; padding: 6px 18px; border-radius: 6px; font-size: 11px;")
        btn_ok.clicked.connect(self._on_ok)
        btn_box.addWidget(btn_ok)

        btn_cancel = QPushButton("Abbrechen")
        btn_cancel.setStyleSheet("background-color: #1e2230; color: #94a3b8; padding: 6px 14px; border-radius: 6px; font-size: 11px;")
        btn_cancel.clicked.connect(self.reject)
        btn_box.addWidget(btn_cancel)

        btn_apply = QPushButton("Übernehmen")
        btn_apply.setStyleSheet("background-color: #1e293b; color: #38bdf8; font-weight: bold; padding: 6px 14px; border-radius: 6px; font-size: 11px;")
        btn_apply.clicked.connect(self._on_apply)
        btn_box.addWidget(btn_apply)

        layout.addLayout(btn_box)

    def _on_apply(self):
        new_cfg = self.inner_widget.get_settings()
        self.settings_widget.set_settings(new_cfg)

    def _on_ok(self):
        self._on_apply()
        self.accept()


