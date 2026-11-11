"""
Live-Streaming Studio Tab matching OBS Studio multi-channel RTMP architecture.
"""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QGroupBox,
    QComboBox, QPushButton, QCheckBox, QLineEdit, QFrame, QScrollArea
)
from ..obs_output_settings import OBSOutputSettingsWidget


def build_livestream_tab(win) -> QWidget:
    """
    Constructs Tab 4: OBS-Style RTMP Live-Streaming and DJI Action Direct Connect.
    """
    win.tab_livestream = QWidget()
    layout_tab_live = QVBoxLayout(win.tab_livestream)
    layout_tab_live.setContentsMargins(4, 4, 4, 4)

    scroll_live = QScrollArea()
    scroll_live.setWidgetResizable(True)
    widget_live = QWidget()
    live_layout = QVBoxLayout(widget_live)
    live_layout.setSpacing(10)
    live_layout.setContentsMargins(8, 8, 8, 8)

    # ---------------------------------------------------------------------
    # 1. Master Telemetry Header Card
    # ---------------------------------------------------------------------
    grp_tally = QGroupBox("🔴 LIVE-BROADCAST MONITOR")
    grp_tally.setStyleSheet("QGroupBox { font-weight: bold; color: #ef4444; border: 1px solid #3b2828; border-radius: 8px; margin-top: 6px; padding-top: 10px; background-color: #140d0d; }")
    tally_layout = QVBoxLayout(grp_tally)
    tally_layout.setSpacing(6)

    row_tally_hdr = QHBoxLayout()
    win.lbl_stream_tally = QLabel("⚪ BEREIT")
    win.lbl_stream_tally.setStyleSheet(
        "background-color: #27273a; color: #a1a1aa; font-weight: 900; font-size: 13px; padding: 6px 14px; border-radius: 8px;"
    )
    win.lbl_stream_tally.setAlignment(Qt.AlignCenter)

    win.lbl_stream_telemetry = QLabel("Bitrate: 0 kbps | FPS: 0.0 | Uptime: 00:00:00 | Drops: 0")
    win.lbl_stream_telemetry.setStyleSheet("color: #e4e4e7; font-size: 11px; font-weight: 600;")
    win.lbl_stream_telemetry.setWordWrap(True)

    row_tally_hdr.addWidget(win.lbl_stream_tally)
    row_tally_hdr.addWidget(win.lbl_stream_telemetry, 1)
    tally_layout.addLayout(row_tally_hdr)
    live_layout.addWidget(grp_tally)

    # ---------------------------------------------------------------------
    # 1b. Multiview & Preview Monitor Switcher
    # ---------------------------------------------------------------------
    grp_monitor = QGroupBox("📺 Multiview-Monitor & Vorschau-Auswahl")
    grp_monitor.setStyleSheet("QGroupBox { font-weight: bold; color: #38bdf8; border: 1px solid #282839; border-radius: 8px; margin-top: 4px; padding-top: 8px; }")
    layout_mon = QVBoxLayout(grp_monitor)
    layout_mon.setSpacing(6)

    row_mv_btns = QHBoxLayout()
    row_mv_btns.setSpacing(6)

    win.btn_mv_grid = QPushButton("🔲 4er-Grid (1, 2, 3, 4)")
    win.btn_mv_grid.setToolTip("Zeigt alle 4 Live-Streams gleichzeitig in einem 2x2 Regie-Grid (wie im Monitor gezeichnet)")
    win.btn_mv_grid.setStyleSheet("background-color: #2563eb; color: white; font-weight: 700; border-radius: 6px; padding: 6px 10px; font-size: 11px;")
    win.btn_mv_grid.clicked.connect(lambda: win.set_view_mode_by_id("grid"))

    win.btn_mv_bcast = QPushButton("🎯 16:9 Regie")
    win.btn_mv_bcast.setToolTip("Zeigt den virtuellen 16:9 TV-Broadcast mit Ball-Tracking im Vollbild")
    win.btn_mv_bcast.setStyleSheet("background-color: #1a1a26; border: 1px solid #2a2a3e; color: #cbd5e1; border-radius: 6px; padding: 6px 10px; font-size: 11px;")
    win.btn_mv_bcast.clicked.connect(lambda: win.set_view_mode_by_id("16:9"))

    win.btn_mv_pano = QPushButton("🏟️ Panorama")
    win.btn_mv_pano.setToolTip("Zeigt das vollständige gestitchte Stadion-Panorama")
    win.btn_mv_pano.setStyleSheet("background-color: #1a1a26; border: 1px solid #2a2a3e; color: #cbd5e1; border-radius: 6px; padding: 6px 10px; font-size: 11px;")
    win.btn_mv_pano.clicked.connect(lambda: win.set_view_mode_by_id("8:3" if win.is_4_3_mode() else "32:9"))

    win.btn_mv_cam1 = QPushButton("📹 Cam 1")
    win.btn_mv_cam1.setToolTip("Zeigt das Rohbild von Kamera Links")
    win.btn_mv_cam1.setStyleSheet("background-color: #1a1a26; border: 1px solid #2a2a3e; color: #cbd5e1; border-radius: 6px; padding: 6px 10px; font-size: 11px;")
    win.btn_mv_cam1.clicked.connect(lambda: win.set_view_mode_by_id("cam1"))

    win.btn_mv_cam2 = QPushButton("📹 Cam 2")
    win.btn_mv_cam2.setToolTip("Zeigt das Rohbild von Kamera Rechts")
    win.btn_mv_cam2.setStyleSheet("background-color: #1a1a26; border: 1px solid #2a2a3e; color: #cbd5e1; border-radius: 6px; padding: 6px 10px; font-size: 11px;")
    win.btn_mv_cam2.clicked.connect(lambda: win.set_view_mode_by_id("cam2"))

    row_mv_btns.addWidget(win.btn_mv_grid, 2)
    row_mv_btns.addWidget(win.btn_mv_bcast, 1)
    row_mv_btns.addWidget(win.btn_mv_pano, 1)
    row_mv_btns.addWidget(win.btn_mv_cam1, 1)
    row_mv_btns.addWidget(win.btn_mv_cam2, 1)
    layout_mon.addLayout(row_mv_btns)
    live_layout.addWidget(grp_monitor)

    # ---------------------------------------------------------------------
    # 2. Camera Input Source Group (USB Webcams / UVC vs. Wireless BLE)
    # ---------------------------------------------------------------------
    grp_input = QGroupBox("📹 Kamera-Eingangsquellen & Verarbeitungs-Modus")
    grp_input.setStyleSheet("QGroupBox { font-weight: bold; color: #38bdf8; border: 1px solid #282839; border-radius: 8px; margin-top: 6px; padding-top: 10px; }")
    input_layout = QVBoxLayout(grp_input)
    input_layout.setSpacing(8)

    # Live Processing Mode: Stitching vs. Single-Camera vs. Dual Single-Stream
    row_proc = QHBoxLayout()
    row_proc.addWidget(QLabel("Live-Modus:"))
    win.combo_live_proc_mode = QComboBox()
    win.combo_live_proc_mode.addItem("🧵 Dual-Kamera Panorama-Stitching & KI-Regie (Stitching + Ball-Follow 16:9)", "stitching_dual")
    win.combo_live_proc_mode.addItem("📹 Einzelkamera: Nur Cam 1 (Links - Direkt ohne Stitching / 0% Last)", "single_cam1")
    win.combo_live_proc_mode.addItem("📹 Einzelkamera: Nur Cam 2 (Rechts - Direkt ohne Stitching / 0% Last)", "single_cam2")
    win.combo_live_proc_mode.addItem("📹 Dual Direkt-Streams (Cam 1 & Cam 2 getrennt ohne Stitching)", "dual_cams_nostitch")
    win.combo_live_proc_mode.currentIndexChanged.connect(win._on_live_proc_mode_changed)
    win.combo_live_proc_mode.setStyleSheet("background-color: #111118; border: 1px solid #282839; border-radius: 6px; padding: 4px 8px; color: #f4f4f6; font-weight: 700;")
    row_proc.addWidget(win.combo_live_proc_mode, 1)
    input_layout.addLayout(row_proc)

    row_mode = QHBoxLayout()
    row_mode.addWidget(QLabel("Eingangs-Weg:"))
    win.combo_live_input_mode = QComboBox()
    win.combo_live_input_mode.addItem("🔌 USB-Kabel (Standard-Webcams, USB-C, DJI Action - 0ms Latenz)", "uvc")
    win.combo_live_input_mode.addItem("📡 Kabellose BLE-Fernsteuerung (Nur DJI Action 4/5 - Mimo Bypass)", "ble")
    win.combo_live_input_mode.currentIndexChanged.connect(win._on_live_input_mode_changed)
    win.combo_live_input_mode.setStyleSheet("background-color: #111118; border: 1px solid #282839; border-radius: 6px; padding: 4px 8px; color: #f4f4f6;")
    row_mode.addWidget(win.combo_live_input_mode, 1)
    input_layout.addLayout(row_mode)

    # Container A: USB Webcams & UVC Direct Controls
    win.panel_uvc = QFrame()
    win.panel_uvc.setStyleSheet("QFrame { background-color: #0b0b12; border: 1px solid #1e1e2d; border-radius: 6px; padding: 6px; }")
    uvc_layout = QVBoxLayout(win.panel_uvc)
    uvc_layout.setSpacing(6)

    row_uvc_sel = QGridLayout()
    row_uvc_sel.addWidget(QLabel("Cam Links / Einzelkamera:"), 0, 0)
    win.combo_uvc_left = QComboBox()
    win.combo_uvc_left.setStyleSheet("background-color: #12121a; border: 1px solid #27273a; border-radius: 6px; padding: 3px 6px; color: #f4f4f6;")
    row_uvc_sel.addWidget(win.combo_uvc_left, 0, 1)

    row_uvc_sel.addWidget(QLabel("Cam Rechts (für Dual-Stitching):"), 1, 0)
    win.combo_uvc_right = QComboBox()
    win.combo_uvc_right.setStyleSheet("background-color: #12121a; border: 1px solid #27273a; border-radius: 6px; padding: 3px 6px; color: #f4f4f6;")
    row_uvc_sel.addWidget(win.combo_uvc_right, 1, 1)
    uvc_layout.addLayout(row_uvc_sel)

    row_uvc_btns = QHBoxLayout()
    win.btn_refresh_uvc = QPushButton("🔄 Geräte suchen")
    win.btn_refresh_uvc.setStyleSheet("background-color: #1a1a26; border: 1px solid #28283a; border-radius: 6px; padding: 4px 8px; color: #94a3b8; font-size: 11px;")
    win.btn_refresh_uvc.clicked.connect(win._refresh_uvc_devices)

    win.btn_toggle_uvc = QPushButton("▶️ Live-Kameras aktivieren")
    win.btn_toggle_uvc.setStyleSheet("background-color: #0284c7; color: white; font-weight: bold; border-radius: 6px; padding: 5px 12px; font-size: 11px;")
    win.btn_toggle_uvc.clicked.connect(win._toggle_uvc_live_capture)

    row_uvc_btns.addWidget(win.btn_refresh_uvc)
    row_uvc_btns.addWidget(win.btn_toggle_uvc, 1)
    uvc_layout.addLayout(row_uvc_btns)

    win.lbl_uvc_status = QLabel("UVC-Kameras nicht gestartet.")
    win.lbl_uvc_status.setStyleSheet("color: #8e8e9c; font-size: 10px;")
    uvc_layout.addWidget(win.lbl_uvc_status)
    input_layout.addWidget(win.panel_uvc)

    # Container B: Wireless BLE Controls (DJI Mimo Bypass)
    win.panel_ble = QFrame()
    win.panel_ble.setVisible(False)
    win.panel_ble.setStyleSheet("QFrame { background-color: #0b0b12; border: 1px solid #1e1e2d; border-radius: 6px; padding: 6px; }")
    ble_layout = QVBoxLayout(win.panel_ble)
    ble_layout.setSpacing(6)

    row_wifi = QGridLayout()
    row_wifi.addWidget(QLabel("WLAN SSID:"), 0, 0)
    win.txt_dji_wifi_ssid = QLineEdit()
    win.txt_dji_wifi_ssid.setPlaceholderText("WLAN-Name am Platz...")
    win.txt_dji_wifi_ssid.setStyleSheet("background-color: #12121a; border: 1px solid #27273a; border-radius: 6px; padding: 3px 6px; color: #f4f4f6;")
    row_wifi.addWidget(win.txt_dji_wifi_ssid, 0, 1)

    row_wifi.addWidget(QLabel("WLAN Pass:"), 1, 0)
    win.txt_dji_wifi_pass = QLineEdit()
    win.txt_dji_wifi_pass.setEchoMode(QLineEdit.Password)
    win.txt_dji_wifi_pass.setPlaceholderText("WLAN-Passwort...")
    win.txt_dji_wifi_pass.setStyleSheet("background-color: #12121a; border: 1px solid #27273a; border-radius: 6px; padding: 3px 6px; color: #f4f4f6;")
    row_wifi.addWidget(win.txt_dji_wifi_pass, 1, 1)
    ble_layout.addLayout(row_wifi)

    row_ble_actions = QHBoxLayout()
    win.btn_ble_scan = QPushButton("📡 DJI Kameras suchen (BLE)")
    win.btn_ble_scan.setStyleSheet("background-color: #1e293b; border: 1px solid #3b82f6; color: #60a5fa; border-radius: 6px; padding: 5px 10px; font-weight: 600; font-size: 11px;")
    win.btn_ble_scan.clicked.connect(win._on_ble_scan_clicked)

    win.btn_dji_qr = QPushButton("📱 Live QR-Code")
    win.btn_dji_qr.setStyleSheet("background-color: #1a1a26; border: 1px solid #28283a; border-radius: 6px; padding: 5px 10px; color: #94a3b8; font-size: 11px;")
    win.btn_dji_qr.clicked.connect(win._on_show_dji_qr_dialog)

    row_ble_actions.addWidget(win.btn_ble_scan, 1)
    row_ble_actions.addWidget(win.btn_dji_qr)
    ble_layout.addLayout(row_ble_actions)

    row_ble_cams = QGridLayout()
    row_ble_cams.addWidget(QLabel("DJI Links:"), 0, 0)
    win.combo_dji_cam_left = QComboBox()
    win.combo_dji_cam_left.setStyleSheet("background-color: #12121a; border: 1px solid #27273a; border-radius: 6px; padding: 3px 6px; color: #f4f4f6;")
    win.btn_dji_conn_left = QPushButton("Koppeln")
    win.btn_dji_conn_left.setStyleSheet("background-color: #1a1a26; border: 1px solid #28283a; border-radius: 6px; padding: 3px 8px; font-size: 11px;")
    win.btn_dji_conn_left.clicked.connect(win._on_dji_connect_left_clicked)
    row_ble_cams.addWidget(win.combo_dji_cam_left, 0, 1)
    row_ble_cams.addWidget(win.btn_dji_conn_left, 0, 2)

    row_ble_cams.addWidget(QLabel("DJI Rechts:"), 1, 0)
    win.combo_dji_cam_right = QComboBox()
    win.combo_dji_cam_right.setStyleSheet("background-color: #12121a; border: 1px solid #27273a; border-radius: 6px; padding: 3px 6px; color: #f4f4f6;")
    win.btn_dji_conn_right = QPushButton("Koppeln")
    win.btn_dji_conn_right.setStyleSheet("background-color: #1a1a26; border: 1px solid #28283a; border-radius: 6px; padding: 3px 8px; font-size: 11px;")
    win.btn_dji_conn_right.clicked.connect(win._on_dji_connect_right_clicked)
    row_ble_cams.addWidget(win.combo_dji_cam_right, 1, 1)
    row_ble_cams.addWidget(win.btn_dji_conn_right, 1, 2)
    ble_layout.addLayout(row_ble_cams)

    win.btn_dji_start_rtmp = QPushButton("🚀 RTMP-Streams auf Kameras starten (Mimo Bypass)")
    win.btn_dji_start_rtmp.setStyleSheet("background-color: #059669; color: white; font-weight: bold; border-radius: 6px; padding: 6px 12px; font-size: 11px;")
    win.btn_dji_start_rtmp.clicked.connect(win._on_dji_start_rtmp_clicked)
    ble_layout.addWidget(win.btn_dji_start_rtmp)

    win.lbl_ble_status = QLabel("Bluetooth bereit.")
    win.lbl_ble_status.setStyleSheet("color: #8e8e9c; font-size: 10px;")
    ble_layout.addWidget(win.lbl_ble_status)
    input_layout.addWidget(win.panel_ble)

    live_layout.addWidget(grp_input)

    # ---------------------------------------------------------------------
    # 3. Multi-Destination RTMP Targets Group & Channel Mapping
    # ---------------------------------------------------------------------
    grp_targets = QGroupBox("🌐 Live-Stream Kanäle & Kamera-Zuordnung (Multi-Stream)")
    grp_targets.setStyleSheet("QGroupBox { font-weight: bold; color: #a78bfa; border: 1px solid #282839; border-radius: 8px; margin-top: 6px; padding-top: 10px; }")
    targets_layout = QVBoxLayout(grp_targets)
    targets_layout.setSpacing(6)

    row_preset = QHBoxLayout()
    row_preset.addWidget(QLabel("Server-Preset:"))
    win.combo_rtmp_preset = QComboBox()
    win.combo_rtmp_preset.addItem("MatchTrack Online (rtmp://matchtrack.de:1935/live)", "rtmp://matchtrack.de:1935/live")
    win.combo_rtmp_preset.addItem("YouTube Live (rtmp://a.rtmp.youtube.com/live2)", "rtmp://a.rtmp.youtube.com/live2")
    win.combo_rtmp_preset.addItem("Twitch (rtmp://live.twitch.tv/app)", "rtmp://live.twitch.tv/app")
    win.combo_rtmp_preset.addItem("Benutzerdefiniert (Custom RTMP)", "custom")
    win.combo_rtmp_preset.currentIndexChanged.connect(win._on_rtmp_preset_changed)
    win.combo_rtmp_preset.setStyleSheet("background-color: #111118; border: 1px solid #282839; border-radius: 6px; padding: 3px 6px; color: #f4f4f6;")
    row_preset.addWidget(win.combo_rtmp_preset, 1)
    targets_layout.addLayout(row_preset)

    row_server_url = QHBoxLayout()
    row_server_url.addWidget(QLabel("Server URL:"))
    win.txt_rtmp_server_url = QLineEdit("rtmp://matchtrack.de:1935/live")
    win.txt_rtmp_server_url.setStyleSheet("background-color: #111118; border: 1px solid #282839; border-radius: 6px; padding: 4px 8px; color: #f4f4f6; font-size: 11px;")
    row_server_url.addWidget(win.txt_rtmp_server_url, 1)
    targets_layout.addLayout(row_server_url)

    def make_feed_combo(default_idx: int):
        cb = QComboBox()
        cb.addItem("🎯 16:9 Regie (Ball-Follow KI)", "broadcast_16x9")
        cb.addItem("🏟️ 8:3 / 32:9 Taktik-Panorama", "panorama_8x3")
        cb.addItem("📹 Cam 1 (Links / Einzelkamera)", "cam1")
        cb.addItem("📹 Cam 2 (Rechts)", "cam2")
        cb.setCurrentIndex(default_idx)
        cb.setStyleSheet("background-color: #111118; border: 1px solid #282839; border-radius: 6px; padding: 3px 6px; color: #f4f4f6; font-size: 11px;")
        return cb

    # Channel 1
    row_c1 = QHBoxLayout()
    win.chk_chan_1 = QCheckBox("Kanal 1:")
    win.chk_chan_1.setChecked(True)
    win.chk_chan_1.setStyleSheet("color: #38bdf8; font-weight: 700; font-size: 11px;")
    win.combo_feed_c1 = make_feed_combo(0)  # broadcast_16x9
    win.txt_stream_key_c1 = QLineEdit("broadcast_1")
    win.txt_stream_key_c1.setPlaceholderText("Stream-Key...")
    win.txt_stream_key_c1.setStyleSheet("background-color: #111118; border: 1px solid #282839; border-radius: 6px; padding: 3px 6px; color: #f4f4f6; font-size: 11px;")
    row_c1.addWidget(win.chk_chan_1, 2)
    row_c1.addWidget(win.combo_feed_c1, 3)
    row_c1.addWidget(win.txt_stream_key_c1, 2)
    targets_layout.addLayout(row_c1)

    # Channel 2
    row_c2 = QHBoxLayout()
    win.chk_chan_2 = QCheckBox("Kanal 2:")
    win.chk_chan_2.setChecked(True)
    win.chk_chan_2.setStyleSheet("color: #a78bfa; font-weight: 700; font-size: 11px;")
    win.combo_feed_c2 = make_feed_combo(1)  # panorama_8x3
    win.txt_stream_key_c2 = QLineEdit("panorama_1")
    win.txt_stream_key_c2.setPlaceholderText("Stream-Key...")
    win.txt_stream_key_c2.setStyleSheet("background-color: #111118; border: 1px solid #282839; border-radius: 6px; padding: 3px 6px; color: #f4f4f6; font-size: 11px;")
    row_c2.addWidget(win.chk_chan_2, 2)
    row_c2.addWidget(win.combo_feed_c2, 3)
    row_c2.addWidget(win.txt_stream_key_c2, 2)
    targets_layout.addLayout(row_c2)

    # Channel 3
    row_c3 = QHBoxLayout()
    win.chk_chan_3 = QCheckBox("Kanal 3:")
    win.chk_chan_3.setChecked(False)
    win.chk_chan_3.setStyleSheet("color: #cbd5e1; font-weight: 600; font-size: 11px;")
    win.combo_feed_c3 = make_feed_combo(2)  # cam1
    win.txt_stream_key_c3 = QLineEdit("cam1")
    win.txt_stream_key_c3.setPlaceholderText("Stream-Key...")
    win.txt_stream_key_c3.setStyleSheet("background-color: #111118; border: 1px solid #282839; border-radius: 6px; padding: 3px 6px; color: #f4f4f6; font-size: 11px;")
    row_c3.addWidget(win.chk_chan_3, 2)
    row_c3.addWidget(win.combo_feed_c3, 3)
    row_c3.addWidget(win.txt_stream_key_c3, 2)
    targets_layout.addLayout(row_c3)

    # Channel 4
    row_c4 = QHBoxLayout()
    win.chk_chan_4 = QCheckBox("Kanal 4:")
    win.chk_chan_4.setChecked(False)
    win.chk_chan_4.setStyleSheet("color: #cbd5e1; font-weight: 600; font-size: 11px;")
    win.combo_feed_c4 = make_feed_combo(3)  # cam2
    win.txt_stream_key_c4 = QLineEdit("cam2")
    win.txt_stream_key_c4.setPlaceholderText("Stream-Key...")
    win.txt_stream_key_c4.setStyleSheet("background-color: #111118; border: 1px solid #282839; border-radius: 6px; padding: 3px 6px; color: #f4f4f6; font-size: 11px;")
    row_c4.addWidget(win.chk_chan_4, 2)
    row_c4.addWidget(win.combo_feed_c4, 3)
    row_c4.addWidget(win.txt_stream_key_c4, 2)
    targets_layout.addLayout(row_c4)

    # Backwards compatibility aliases
    win.chk_chan_bcast = win.chk_chan_1
    win.chk_chan_pano = win.chk_chan_2
    win.chk_chan_cam1 = win.chk_chan_3
    win.chk_chan_cam2 = win.chk_chan_4
    win.txt_stream_key_bcast = win.txt_stream_key_c1
    win.txt_stream_key_pano = win.txt_stream_key_c2
    win.txt_stream_key_cam1 = win.txt_stream_key_c3
    win.txt_stream_key_cam2 = win.txt_stream_key_c4

    live_layout.addWidget(grp_targets)

    # ---------------------------------------------------------------------
    # 4. OBS Studio Stream- & Kodierer-Einstellungen (1:1 OBS Engine)
    # ---------------------------------------------------------------------
    win.obs_settings_widget = OBSOutputSettingsWidget()
    live_layout.addWidget(win.obs_settings_widget)

    # Aliases for seamless integration with existing MainWindow methods
    win.combo_stream_encoder = win.obs_settings_widget.combo_video_encoder
    win.combo_stream_resolution = win.obs_settings_widget.combo_rescale_res
    win.combo_stream_fps = win.obs_settings_widget.combo_fps
    win.combo_stream_resolution.currentIndexChanged.connect(win._on_stream_resolution_changed)

    # Legacy proxy for bitrate queries
    class _BitrateProxy:
        def __init__(self, spn): self._spn = spn
        def currentData(self): return self._spn.value()
        def findData(self, val): return 0
        def setCurrentIndex(self, idx): pass
        def value(self): return self._spn.value()
        def setValue(self, v): self._spn.setValue(int(v))
    win.combo_stream_bitrate = _BitrateProxy(win.obs_settings_widget.spn_bitrate)

    # ---------------------------------------------------------------------
    # 5. Master Broadcast Action Buttons
    # ---------------------------------------------------------------------
    win.btn_master_stream_toggle = QPushButton("🔴 LIVE-STREAM STARTEN (OBS-Engine)")
    win.btn_master_stream_toggle.setStyleSheet(
        "background-color: #dc2626; color: white; font-weight: 800; font-size: 13px; padding: 12px; border-radius: 10px;"
    )
    win.btn_master_stream_toggle.clicked.connect(win._toggle_master_live_stream)
    live_layout.addWidget(win.btn_master_stream_toggle)

    live_layout.addStretch()
    scroll_live.setWidget(widget_live)
    layout_tab_live.addWidget(scroll_live)

    # Initial device enumeration
    win._refresh_uvc_devices()

    return win.tab_livestream
