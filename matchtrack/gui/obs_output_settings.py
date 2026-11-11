"""
OBS Studio Output Settings Panel and Widget matching OBS Studio 1:1.
"""

import os
from typing import Dict, Any
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QGroupBox, QComboBox,
    QSpinBox, QDoubleSpinBox, QLineEdit, QCheckBox, QRadioButton,
    QTabWidget, QPushButton, QFileDialog, QButtonGroup, QStackedWidget,
    QGridLayout
)
from .widgets import NoWheelComboBox, NoWheelSpinBox, NoWheelDoubleSpinBox

class OBSOutputSettingsWidget(QWidget):
    """
    1:1 OBS Studio Output Settings Panel.
    Provides identical layout, controls, and options to OBS Studio (Ausgabe -> Erweitert/Einfach).
    Includes Stream, Aufnahme, Audio, and Replaypuffer tabs.
    """
    settingsChanged = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(2, 4, 2, 4)
        main_layout.setSpacing(6)

        obs_style = """
            QGroupBox {
                font-weight: bold;
                color: #e2e8f0;
                border: 1px solid #282c3c;
                border-radius: 8px;
                margin-top: 6px;
                padding-top: 10px;
                background-color: #141622;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 4px;
            }
            QLabel {
                color: #cbd5e1;
                font-size: 11px;
            }
            QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {
                background-color: #0d0f17;
                border: 1px solid #282d3e;
                border-radius: 5px;
                padding: 3px 6px;
                color: #f1f5f9;
                font-size: 11px;
            }
            QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover, QLineEdit:hover {
                border-color: #3b82f6;
            }
            QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 18px;
                border-left: 1px solid #252939;
            }
            QCheckBox {
                color: #e2e8f0;
                font-size: 11px;
                spacing: 6px;
            }
            QRadioButton {
                color: #e2e8f0;
                font-size: 11px;
                spacing: 5px;
            }
            QTabWidget::pane {
                border: 1px solid #282c3c;
                background-color: #12141e;
                border-radius: 6px;
                padding: 6px;
            }
            QTabBar::tab {
                background-color: #1a1c28;
                color: #94a3b8;
                font-weight: 600;
                font-size: 11px;
                padding: 5px 12px;
                border-top-left-radius: 5px;
                border-top-right-radius: 5px;
                margin-right: 2px;
                border: 1px solid #242838;
                border-bottom: none;
            }
            QTabBar::tab:selected {
                background-color: #2563eb;
                color: #ffffff;
                border-color: #2563eb;
            }
            QTabBar::tab:hover:!selected {
                background-color: #232738;
                color: #cbd5e1;
            }
        """
        self.setStyleSheet(obs_style)

        # -------------------------------------------------------------
        # Header: Ausgabemodus [ Erweitert | Einfach ] + Popout Button
        # -------------------------------------------------------------
        header_box = QGroupBox("⚙️ OBS Studio Ausgabe")
        header_inner = QVBoxLayout(header_box)
        header_inner.setContentsMargins(6, 8, 6, 6)
        header_inner.setSpacing(6)

        mode_row = QHBoxLayout()
        mode_row.setSpacing(8)

        lbl_mode = QLabel("Ausgabemodus")
        lbl_mode.setStyleSheet("font-weight: 700; color: #f8fafc; font-size: 11px;")
        mode_row.addWidget(lbl_mode)

        self.combo_output_mode = NoWheelComboBox()
        self.combo_output_mode.addItem("Erweitert", "advanced")
        self.combo_output_mode.addItem("Einfach", "simple")
        self.combo_output_mode.setStyleSheet("background-color: #1a1e2c; font-weight: bold; min-width: 100px;")
        self.combo_output_mode.currentIndexChanged.connect(self._on_output_mode_changed)
        mode_row.addWidget(self.combo_output_mode, 1)

        self.btn_popout = QPushButton("🗗 OBS-Fenster")
        self.btn_popout.setToolTip("OBS Ausgabe-Einstellungen als eigenständiges Dialog-Fenster öffnen")
        self.btn_popout.setStyleSheet("""
            QPushButton {
                background-color: #1e293b;
                border: 1px solid #334155;
                color: #38bdf8;
                font-weight: bold;
                padding: 3px 8px;
                border-radius: 5px;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #2563eb;
                color: #ffffff;
                border-color: #3b82f6;
            }
        """)
        self.btn_popout.clicked.connect(self.open_popout_dialog)
        mode_row.addWidget(self.btn_popout)

        header_inner.addLayout(mode_row)

        # -------------------------------------------------------------
        # Stacked Widget: Page 0 = Erweitert (Tabs), Page 1 = Einfach
        # -------------------------------------------------------------
        self.stack = QStackedWidget()

        # =============================================================
        # PAGE 0: ADVANCED MODE (Erweitert) with 4 Tabs
        # =============================================================
        self.tab_widget = QTabWidget()

        # -------------------------------------------------------------
        # TAB 1: STREAM
        # -------------------------------------------------------------
        tab_stream = QWidget()
        stream_layout = QVBoxLayout(tab_stream)
        stream_layout.setContentsMargins(4, 4, 4, 4)
        stream_layout.setSpacing(8)

        # Streameinstellungen Header
        lbl_stream_head = QLabel("Streameinstellungen")
        lbl_stream_head.setStyleSheet("font-weight: 800; color: #f8fafc; font-size: 11px; border-bottom: 1px solid #282d3e; padding-bottom: 2px;")
        stream_layout.addWidget(lbl_stream_head)

        grid_stream = QGridLayout()
        grid_stream.setHorizontalSpacing(8)
        grid_stream.setVerticalSpacing(5)
        grid_stream.setColumnStretch(1, 1)

        # Row 0: Audiospur (Radio 1..6)
        lbl_at = QLabel("Audiospur")
        lbl_at.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        grid_stream.addWidget(lbl_at, 0, 0)

        audio_track_layout = QHBoxLayout()
        audio_track_layout.setSpacing(10)
        self.audio_group = QButtonGroup(self)
        self.radio_tracks = []
        for i in range(1, 7):
            rb = QRadioButton(str(i))
            if i == 1:
                rb.setChecked(True)
            self.audio_group.addButton(rb, i)
            self.radio_tracks.append(rb)
            audio_track_layout.addWidget(rb)
        audio_track_layout.addStretch()
        grid_stream.addLayout(audio_track_layout, 0, 1)

        # Row 1: Audiokodierer
        lbl_ae = QLabel("Audiokodierer")
        lbl_ae.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        grid_stream.addWidget(lbl_ae, 1, 0)
        self.combo_audio_encoder = NoWheelComboBox()
        self.combo_audio_encoder.addItem("FFmpeg AAC", "aac")
        grid_stream.addWidget(self.combo_audio_encoder, 1, 1)

        # Row 2: Videokodierer
        lbl_ve = QLabel("Videokodierer")
        lbl_ve.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        grid_stream.addWidget(lbl_ve, 2, 0)
        self.combo_video_encoder = NoWheelComboBox()
        self.combo_video_encoder.addItem("NVIDIA NVENC H.264", "h264_nvenc")
        self.combo_video_encoder.addItem("x264 (Software CPU)", "libx264")
        self.combo_video_encoder.addItem("Intel QuickSync H.264 (QSV)", "h264_qsv")
        self.combo_video_encoder.addItem("NVIDIA NVENC HEVC / H.265", "hevc_nvenc")
        self.combo_video_encoder.currentIndexChanged.connect(self._on_video_encoder_changed)
        grid_stream.addWidget(self.combo_video_encoder, 2, 1)

        # Row 3: Ausgabe umskalieren
        lbl_rescale = QLabel("Ausgabe umskalieren")
        lbl_rescale.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        grid_stream.addWidget(lbl_rescale, 3, 0)

        rescale_layout = QHBoxLayout()
        rescale_layout.setSpacing(6)
        self.chk_rescale = QCheckBox("Aktivieren")
        self.chk_rescale.setChecked(False)
        self.chk_rescale.toggled.connect(self._on_rescale_toggled)
        rescale_layout.addWidget(self.chk_rescale)

        self.combo_rescale_res = NoWheelComboBox()
        self.combo_rescale_res.addItem("1920x1080 (1080p)", "1080p")
        self.combo_rescale_res.addItem("1280x720 (720p)", "720p")
        self.combo_rescale_res.addItem("854x480 (480p)", "480p")
        self.combo_rescale_res.addItem("640x360 (360p)", "360p")
        self.combo_rescale_res.setEnabled(False)
        rescale_layout.addWidget(self.combo_rescale_res, 1)
        grid_stream.addLayout(rescale_layout, 3, 1)

        stream_layout.addLayout(grid_stream)

        # Kodierereinstellungen Header
        lbl_enc_head = QLabel("Kodierereinstellungen")
        lbl_enc_head.setStyleSheet("font-weight: 800; color: #f8fafc; font-size: 11px; border-bottom: 1px solid #282d3e; padding-bottom: 2px; margin-top: 4px;")
        stream_layout.addWidget(lbl_enc_head)

        grid_enc = QGridLayout()
        grid_enc.setHorizontalSpacing(8)
        grid_enc.setVerticalSpacing(5)
        grid_enc.setColumnStretch(1, 1)

        # 1. Qualitätsregulierungsmethode
        lbl_rc = QLabel("Qualitätsregulierungsmethode")
        lbl_rc.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        grid_enc.addWidget(lbl_rc, 0, 0)
        self.combo_rc = NoWheelComboBox()
        self.combo_rc.addItem("Variable Bitrate (VBR)", "vbr")
        self.combo_rc.addItem("Konstante Bitrate (CBR)", "cbr")
        self.combo_rc.addItem("Konstanter Quantisierungsparameter (CQP)", "cqp")
        self.combo_rc.addItem("Verlustfrei", "lossless")
        self.combo_rc.currentIndexChanged.connect(self._on_rc_changed)
        grid_enc.addWidget(self.combo_rc, 0, 1)

        # 2. Bitrate
        lbl_bitrate = QLabel("Bitrate")
        lbl_bitrate.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        grid_enc.addWidget(lbl_bitrate, 1, 0)
        self.spn_bitrate = NoWheelSpinBox()
        self.spn_bitrate.setRange(200, 50000)
        self.spn_bitrate.setSingleStep(250)
        self.spn_bitrate.setValue(3000)
        self.spn_bitrate.setSuffix(" Kbps")
        grid_enc.addWidget(self.spn_bitrate, 1, 1)

        # 3. Maximale Bitrate (0 = Pegelbegrenzung)
        self.lbl_max_bitrate = QLabel("Maximale Bitrate (0 = Pegelbegrenzung)")
        self.lbl_max_bitrate.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        grid_enc.addWidget(self.lbl_max_bitrate, 2, 0)
        self.spn_max_bitrate = NoWheelSpinBox()
        self.spn_max_bitrate.setRange(0, 50000)
        self.spn_max_bitrate.setSingleStep(250)
        self.spn_max_bitrate.setValue(5000)
        self.spn_max_bitrate.setSuffix(" Kbps")
        grid_enc.addWidget(self.spn_max_bitrate, 2, 1)

        # 4. Keyframeintervall (Sekunden, 0 = automatisch)
        lbl_keyint = QLabel("Keyframeintervall (Sekunden, 0 = automatisch)")
        lbl_keyint.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        grid_enc.addWidget(lbl_keyint, 3, 0)
        self.spn_keyint = NoWheelDoubleSpinBox()
        self.spn_keyint.setRange(0.0, 10.0)
        self.spn_keyint.setSingleStep(0.5)
        self.spn_keyint.setDecimals(1)
        self.spn_keyint.setValue(2.0)
        self.spn_keyint.setSuffix(" s")
        grid_enc.addWidget(self.spn_keyint, 3, 1)

        # 5. Voreinstellung
        lbl_preset = QLabel("Voreinstellung")
        lbl_preset.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        grid_enc.addWidget(lbl_preset, 4, 0)
        self.combo_preset = NoWheelComboBox()
        self.combo_preset.addItem("P1: Schnellste (Geringste Qualität)", "p1")
        self.combo_preset.addItem("P2: Schneller", "p2")
        self.combo_preset.addItem("P3: Schnell", "p3")
        self.combo_preset.addItem("P4: Mittel (Mittlere Qualität)", "p4")
        self.combo_preset.addItem("P5: Langsam (Gute Qualität)", "p5")
        self.combo_preset.addItem("P6: Langsamer (Bessere Qualität)", "p6")
        self.combo_preset.addItem("P7: Langsamste (Beste Qualität)", "p7")
        self.combo_preset.setCurrentIndex(3) # P4 default
        grid_enc.addWidget(self.combo_preset, 4, 1)

        # 6. Tuning
        self.lbl_tuning = QLabel("Tuning")
        self.lbl_tuning.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        grid_enc.addWidget(self.lbl_tuning, 5, 0)
        self.combo_tuning = NoWheelComboBox()
        self.combo_tuning.addItem("Hohe Qualität", "hq")
        self.combo_tuning.addItem("Niedrige Latenz", "ll")
        self.combo_tuning.addItem("Ultraniedrige Latenz", "ull")
        self.combo_tuning.addItem("Verlustfrei", "lossless")
        grid_enc.addWidget(self.combo_tuning, 5, 1)

        # 7. Mehrere Durchgänge
        self.lbl_multipass = QLabel("Mehrere Durchgänge")
        self.lbl_multipass.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        grid_enc.addWidget(self.lbl_multipass, 6, 0)
        self.combo_multipass = NoWheelComboBox()
        self.combo_multipass.addItem("Ein Durchgang", "disabled")
        self.combo_multipass.addItem("Zwei Durchgänge (Geviertelte Auflösung)", "qres")
        self.combo_multipass.addItem("Zwei Durchgänge (Volle Auflösung)", "fullres")
        self.combo_multipass.setCurrentIndex(1) # qres default
        grid_enc.addWidget(self.combo_multipass, 6, 1)

        # 8. Profil
        lbl_profile = QLabel("Profil")
        lbl_profile.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        grid_enc.addWidget(lbl_profile, 7, 0)
        self.combo_profile = NoWheelComboBox()
        self.combo_profile.addItem("main", "main")
        self.combo_profile.addItem("high", "high")
        self.combo_profile.addItem("baseline", "baseline")
        grid_enc.addWidget(self.combo_profile, 7, 1)

        # 9. Lookahead
        grid_enc.addWidget(QLabel(""), 8, 0)
        self.chk_lookahead = QCheckBox("Lookahead 🛈")
        self.chk_lookahead.setToolTip("Aktiviert dynamische Frame-Vorausschau (Lookahead-Puffer mit 32 Frames) für optimale Bitratenverteilung.")
        self.chk_lookahead.setChecked(True)
        grid_enc.addWidget(self.chk_lookahead, 8, 1)

        # 10. Adaptive Quantisierung
        grid_enc.addWidget(QLabel(""), 9, 0)
        self.chk_aq = QCheckBox("Adaptive Quantisierung 🛈")
        self.chk_aq.setToolTip("Aktiviert Spatial & Temporal Adaptive Quantization (Psycho-Visual Tuning) für schärfere Rasen- und Spielerkonturen.")
        self.chk_aq.setChecked(True)
        grid_enc.addWidget(self.chk_aq, 9, 1)

        # 11. B-Frames
        lbl_bf = QLabel("B-Frames")
        lbl_bf.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        grid_enc.addWidget(lbl_bf, 10, 0)
        self.spn_bframes = NoWheelSpinBox()
        self.spn_bframes.setRange(0, 16)
        self.spn_bframes.setValue(2)
        grid_enc.addWidget(self.spn_bframes, 10, 1)

        # 12. Benutzerdefinierte Kodiereroptionen
        lbl_co = QLabel("Benutzerdefinierte Kodiereroptionen 🛈")
        lbl_co.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        grid_enc.addWidget(lbl_co, 11, 0)
        self.txt_custom_opts = QLineEdit()
        self.txt_custom_opts.setPlaceholderText("z.B. no-scenecut=1 rc-lookahead=32")
        self.txt_custom_opts.setToolTip("Zusätzliche FFmpeg Parameter für den ausgewählten Kodierer.")
        grid_enc.addWidget(self.txt_custom_opts, 11, 1)

        # Framerate Row
        lbl_fps = QLabel("Framerate")
        lbl_fps.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        grid_enc.addWidget(lbl_fps, 12, 0)
        self.combo_fps = NoWheelComboBox()
        self.combo_fps.addItem("30 FPS (Standard für Live-Sport)", 30)
        self.combo_fps.addItem("60 FPS (Sehr flüssig)", 60)
        grid_enc.addWidget(self.combo_fps, 12, 1)

        stream_layout.addLayout(grid_enc)
        stream_layout.addStretch()

        self.tab_widget.addTab(tab_stream, "Stream")

        # -------------------------------------------------------------
        # TAB 2: AUFNAHME
        # -------------------------------------------------------------
        tab_rec = QWidget()
        rec_layout = QVBoxLayout(tab_rec)
        rec_layout.setContentsMargins(4, 4, 4, 4)
        rec_layout.setSpacing(8)

        lbl_rec_head = QLabel("Aufnahmeeinstellungen")
        lbl_rec_head.setStyleSheet("font-weight: 800; color: #f8fafc; font-size: 11px; border-bottom: 1px solid #282d3e; padding-bottom: 2px;")
        rec_layout.addWidget(lbl_rec_head)

        grid_rec = QGridLayout()
        grid_rec.setHorizontalSpacing(8)
        grid_rec.setVerticalSpacing(5)
        grid_rec.setColumnStretch(1, 1)

        grid_rec.addWidget(QLabel("Typ:"), 0, 0)
        self.combo_rec_type = NoWheelComboBox()
        self.combo_rec_type.addItem("Standard", "standard")
        grid_rec.addWidget(self.combo_rec_type, 0, 1)

        grid_rec.addWidget(QLabel("Aufnahmepfad:"), 1, 0)
        rec_path_layout = QHBoxLayout()
        self.txt_rec_path = QLineEdit("recordings")
        btn_rec_browse = QPushButton("Durchsuchen...")
        btn_rec_browse.setStyleSheet("background-color: #1e293b; color: #38bdf8; border: 1px solid #334155; padding: 2px 6px; border-radius: 4px;")
        btn_rec_browse.clicked.connect(self._browse_rec_path)
        rec_path_layout.addWidget(self.txt_rec_path, 1)
        rec_path_layout.addWidget(btn_rec_browse)
        grid_rec.addLayout(rec_path_layout, 1, 1)

        grid_rec.addWidget(QLabel("Aufnahmeformat:"), 2, 0)
        self.combo_rec_format = NoWheelComboBox()
        self.combo_rec_format.addItem("mp4 (MPEG-4)", "mp4")
        self.combo_rec_format.addItem("mkv (Matroska)", "mkv")
        self.combo_rec_format.addItem("mov (QuickTime)", "mov")
        self.combo_rec_format.addItem("ts (Transport Stream)", "ts")
        grid_rec.addWidget(self.combo_rec_format, 2, 1)

        grid_rec.addWidget(QLabel("Videokodierer:"), 3, 0)
        self.combo_rec_video_encoder = NoWheelComboBox()
        self.combo_rec_video_encoder.addItem("(Stream-Kodierer verwenden)", "same_as_stream")
        self.combo_rec_video_encoder.addItem("NVIDIA NVENC H.264", "h264_nvenc")
        self.combo_rec_video_encoder.addItem("x264 (Software CPU)", "libx264")
        grid_rec.addWidget(self.combo_rec_video_encoder, 3, 1)

        grid_rec.addWidget(QLabel("Audiokodierer:"), 4, 0)
        self.combo_rec_audio_encoder = NoWheelComboBox()
        self.combo_rec_audio_encoder.addItem("FFmpeg AAC", "aac")
        grid_rec.addWidget(self.combo_rec_audio_encoder, 4, 1)

        rec_layout.addLayout(grid_rec)
        rec_layout.addStretch()
        self.tab_widget.addTab(tab_rec, "Aufnahme")

        # -------------------------------------------------------------
        # TAB 3: AUDIO
        # -------------------------------------------------------------
        tab_audio = QWidget()
        audio_layout = QVBoxLayout(tab_audio)
        audio_layout.setContentsMargins(4, 4, 4, 4)
        audio_layout.setSpacing(8)

        lbl_aud_head = QLabel("Audiotrack-Bitraten")
        lbl_aud_head.setStyleSheet("font-weight: 800; color: #f8fafc; font-size: 11px; border-bottom: 1px solid #282d3e; padding-bottom: 2px;")
        audio_layout.addWidget(lbl_aud_head)

        grid_aud = QGridLayout()
        grid_aud.setHorizontalSpacing(8)
        grid_aud.setVerticalSpacing(5)
        grid_aud.setColumnStretch(1, 1)

        self.audio_bitrate_combos = []
        for i in range(1, 7):
            grid_aud.addWidget(QLabel(f"Audiospur {i} Bitrate:"), i - 1, 0)
            cb = NoWheelComboBox()
            cb.addItem("320 kbps (Höchste Studioqualität)", 320)
            cb.addItem("192 kbps (Hohe Qualität)", 192)
            cb.addItem("160 kbps (Standard)", 160)
            cb.addItem("128 kbps (Mittel)", 128)
            cb.addItem("64 kbps (Niedrig)", 64)
            cb.setCurrentIndex(2) # 160 kbps default
            grid_aud.addWidget(cb, i - 1, 1)
            self.audio_bitrate_combos.append(cb)

        audio_layout.addLayout(grid_aud)
        audio_layout.addStretch()
        self.tab_widget.addTab(tab_audio, "Audio")

        # -------------------------------------------------------------
        # TAB 4: REPLAYPUFFER
        # -------------------------------------------------------------
        tab_replay = QWidget()
        rep_layout = QVBoxLayout(tab_replay)
        rep_layout.setContentsMargins(4, 4, 4, 4)
        rep_layout.setSpacing(8)

        lbl_rep_head = QLabel("Replaypuffer-Einstellungen")
        lbl_rep_head.setStyleSheet("font-weight: 800; color: #f8fafc; font-size: 11px; border-bottom: 1px solid #282d3e; padding-bottom: 2px;")
        rep_layout.addWidget(lbl_rep_head)

        self.chk_replay = QCheckBox("Replaypuffer aktivieren")
        rep_layout.addWidget(self.chk_replay)

        grid_rep = QGridLayout()
        grid_rep.setHorizontalSpacing(8)
        grid_rep.setVerticalSpacing(5)
        grid_rep.setColumnStretch(1, 1)

        grid_rep.addWidget(QLabel("Maximale Replayzeit:"), 0, 0)
        self.spn_replay_sec = NoWheelSpinBox()
        self.spn_replay_sec.setRange(5, 300)
        self.spn_replay_sec.setValue(20)
        self.spn_replay_sec.setSuffix(" s")
        grid_rep.addWidget(self.spn_replay_sec, 0, 1)

        grid_rep.addWidget(QLabel("Maximaler Speicherverbrauch:"), 1, 0)
        self.spn_replay_mem = NoWheelSpinBox()
        self.spn_replay_mem.setRange(64, 8192)
        self.spn_replay_mem.setValue(512)
        self.spn_replay_mem.setSuffix(" MB")
        grid_rep.addWidget(self.spn_replay_mem, 1, 1)

        rep_layout.addLayout(grid_rep)
        rep_layout.addStretch()
        self.tab_widget.addTab(tab_replay, "Replaypuffer")

        self.stack.addWidget(self.tab_widget)

        # =============================================================
        # PAGE 1: SIMPLE MODE (Einfach)
        # =============================================================
        page_simple = QWidget()
        simple_layout = QVBoxLayout(page_simple)
        simple_layout.setContentsMargins(4, 4, 4, 4)
        simple_layout.setSpacing(8)

        lbl_simp_head = QLabel("Einfache Streaming-Ausgabe")
        lbl_simp_head.setStyleSheet("font-weight: 800; color: #f8fafc; font-size: 11px; border-bottom: 1px solid #282d3e; padding-bottom: 2px;")
        simple_layout.addWidget(lbl_simp_head)

        grid_simp = QGridLayout()
        grid_simp.setHorizontalSpacing(8)
        grid_simp.setVerticalSpacing(5)
        grid_simp.setColumnStretch(1, 1)

        grid_simp.addWidget(QLabel("Video-Bitrate:"), 0, 0)
        self.spn_simple_bitrate = NoWheelSpinBox()
        self.spn_simple_bitrate.setRange(200, 50000)
        self.spn_simple_bitrate.setSingleStep(250)
        self.spn_simple_bitrate.setValue(3000)
        self.spn_simple_bitrate.setSuffix(" Kbps")
        self.spn_simple_bitrate.valueChanged.connect(lambda v: self.spn_bitrate.setValue(v))
        grid_simp.addWidget(self.spn_simple_bitrate, 0, 1)

        grid_simp.addWidget(QLabel("Video-Kodierer:"), 1, 0)
        self.combo_simple_encoder = NoWheelComboBox()
        self.combo_simple_encoder.addItem("Hardware (NVENC H.264)", "h264_nvenc")
        self.combo_simple_encoder.addItem("Software (x264)", "libx264")
        self.combo_simple_encoder.addItem("Hardware (Intel QuickSync)", "h264_qsv")
        self.combo_simple_encoder.addItem("Hardware (NVENC HEVC)", "hevc_nvenc")
        self.combo_simple_encoder.currentIndexChanged.connect(self._on_simple_encoder_changed)
        grid_simp.addWidget(self.combo_simple_encoder, 1, 1)

        grid_simp.addWidget(QLabel("Audio-Bitrate:"), 2, 0)
        self.combo_simple_audio_bitrate = NoWheelComboBox()
        self.combo_simple_audio_bitrate.addItem("160 kbps", 160)
        self.combo_simple_audio_bitrate.addItem("128 kbps", 128)
        self.combo_simple_audio_bitrate.addItem("192 kbps", 192)
        grid_simp.addWidget(self.combo_simple_audio_bitrate, 2, 1)

        grid_simp.addWidget(QLabel("Kodierer-Voreinstellung:"), 3, 0)
        self.combo_simple_preset = NoWheelComboBox()
        self.combo_simple_preset.addItem("Schnell", "p2")
        self.combo_simple_preset.addItem("Sehr schnell", "p1")
        self.combo_simple_preset.addItem("Mittel / Ausgeglichen", "p4")
        self.combo_simple_preset.addItem("Hohe Qualität", "p6")
        self.combo_simple_preset.setCurrentIndex(2)
        grid_simp.addWidget(self.combo_simple_preset, 3, 1)

        simple_layout.addLayout(grid_simp)
        simple_layout.addStretch()

        self.stack.addWidget(page_simple)

        header_inner.addWidget(self.stack)
        main_layout.addWidget(header_box)

    def _on_output_mode_changed(self, idx: int):
        self.stack.setCurrentIndex(idx)
        self.settingsChanged.emit()

    def _on_video_encoder_changed(self, idx: int):
        codec = self.combo_video_encoder.currentData()
        is_nvenc = "nvenc" in codec.lower()
        self.lbl_tuning.setEnabled(is_nvenc)
        self.combo_tuning.setEnabled(is_nvenc)
        self.lbl_multipass.setEnabled(is_nvenc)
        self.combo_multipass.setEnabled(is_nvenc)
        self.chk_lookahead.setEnabled(is_nvenc)
        self.chk_aq.setEnabled(is_nvenc)
        self.settingsChanged.emit()

    def _on_simple_encoder_changed(self, idx: int):
        enc_data = self.combo_simple_encoder.currentData()
        enc_idx = self.combo_video_encoder.findData(enc_data)
        if enc_idx >= 0:
            self.combo_video_encoder.setCurrentIndex(enc_idx)

    def _on_rescale_toggled(self, checked: bool):
        self.combo_rescale_res.setEnabled(checked)
        self.settingsChanged.emit()

    def _on_rc_changed(self, idx: int):
        rc = self.combo_rc.currentData()
        is_vbr = (rc == "vbr")
        self.lbl_max_bitrate.setEnabled(is_vbr)
        self.spn_max_bitrate.setEnabled(is_vbr)
        self.settingsChanged.emit()

    def _browse_rec_path(self):
        path = QFileDialog.getExistingDirectory(self, "Aufnahmeverzeichnis auswählen", self.txt_rec_path.text())
        if path:
            self.txt_rec_path.setText(path)
            self.settingsChanged.emit()

    def open_popout_dialog(self):
        dialog = OBSAdvancedOutputDialog(self, parent=self.window())
        dialog.exec()

    def get_settings(self) -> dict:
        """Collects all OBS output settings into a clean dictionary."""
        return {
            "output_mode": self.combo_output_mode.currentData(),
            "audio_track": self.audio_group.checkedId() if self.audio_group.checkedId() > 0 else 1,
            "audio_encoder": self.combo_audio_encoder.currentData(),
            "video_encoder": self.combo_video_encoder.currentData(),
            "rescale_output": self.chk_rescale.isChecked(),
            "rescale_resolution": self.combo_rescale_res.currentText().split()[0],
            "rate_control": self.combo_rc.currentData(),
            "bitrate_kbps": self.spn_bitrate.value(),
            "max_bitrate_kbps": self.spn_max_bitrate.value(),
            "keyframe_interval_sec": self.spn_keyint.value(),
            "preset": self.combo_preset.currentData(),
            "tuning": self.combo_tuning.currentData(),
            "multipass": self.combo_multipass.currentData(),
            "profile": self.combo_profile.currentData(),
            "lookahead": self.chk_lookahead.isChecked(),
            "adaptive_quant": self.chk_aq.isChecked(),
            "b_frames": self.spn_bframes.value(),
            "custom_options": self.txt_custom_opts.text().strip(),
            "fps": self.combo_fps.currentData() or 30,
            "rec_type": self.combo_rec_type.currentData(),
            "rec_path": self.txt_rec_path.text().strip(),
            "rec_format": self.combo_rec_format.currentData(),
            "rec_video_encoder": self.combo_rec_video_encoder.currentData(),
            "rec_audio_encoder": self.combo_rec_audio_encoder.currentData(),
            "replay_enabled": self.chk_replay.isChecked(),
            "replay_sec": self.spn_replay_sec.value(),
            "replay_mem_mb": self.spn_replay_mem.value()
        }

    def set_settings(self, cfg: dict):
        """Populates all OBS output controls from a dictionary."""
        if not isinstance(cfg, dict):
            return

        if "output_mode" in cfg:
            idx = self.combo_output_mode.findData(cfg["output_mode"])
            if idx >= 0: self.combo_output_mode.setCurrentIndex(idx)

        if "audio_track" in cfg:
            track = int(cfg["audio_track"])
            btn = self.audio_group.button(track)
            if btn: btn.setChecked(True)

        if "video_encoder" in cfg:
            idx = self.combo_video_encoder.findData(cfg["video_encoder"])
            if idx >= 0: self.combo_video_encoder.setCurrentIndex(idx)

        if "rescale_output" in cfg:
            self.chk_rescale.setChecked(bool(cfg["rescale_output"]))

        if "rescale_resolution" in cfg:
            res_val = str(cfg["rescale_resolution"])
            for i in range(self.combo_rescale_res.count()):
                if res_val in self.combo_rescale_res.itemText(i):
                    self.combo_rescale_res.setCurrentIndex(i)
                    break

        if "rate_control" in cfg:
            idx = self.combo_rc.findData(cfg["rate_control"])
            if idx >= 0: self.combo_rc.setCurrentIndex(idx)

        if "bitrate_kbps" in cfg:
            self.spn_bitrate.setValue(int(cfg["bitrate_kbps"]))
            self.spn_simple_bitrate.setValue(int(cfg["bitrate_kbps"]))

        if "max_bitrate_kbps" in cfg:
            self.spn_max_bitrate.setValue(int(cfg["max_bitrate_kbps"]))

        if "keyframe_interval_sec" in cfg:
            self.spn_keyint.setValue(float(cfg["keyframe_interval_sec"]))

        if "preset" in cfg:
            idx = self.combo_preset.findData(cfg["preset"])
            if idx >= 0: self.combo_preset.setCurrentIndex(idx)

        if "tuning" in cfg:
            idx = self.combo_tuning.findData(cfg["tuning"])
            if idx >= 0: self.combo_tuning.setCurrentIndex(idx)

        if "multipass" in cfg:
            idx = self.combo_multipass.findData(cfg["multipass"])
            if idx >= 0: self.combo_multipass.setCurrentIndex(idx)

        if "profile" in cfg:
            idx = self.combo_profile.findData(cfg["profile"])
            if idx >= 0: self.combo_profile.setCurrentIndex(idx)

        if "lookahead" in cfg:
            self.chk_lookahead.setChecked(bool(cfg["lookahead"]))

        if "adaptive_quant" in cfg:
            self.chk_aq.setChecked(bool(cfg["adaptive_quant"]))

        if "b_frames" in cfg:
            self.spn_bframes.setValue(int(cfg["b_frames"]))

        if "custom_options" in cfg:
            self.txt_custom_opts.setText(str(cfg["custom_options"]))

        if "fps" in cfg:
            idx = self.combo_fps.findData(int(cfg["fps"]))
            if idx >= 0: self.combo_fps.setCurrentIndex(idx)

        if "rec_path" in cfg:
            self.txt_rec_path.setText(str(cfg["rec_path"]))

        if "rec_format" in cfg:
            idx = self.combo_rec_format.findData(cfg["rec_format"])
            if idx >= 0: self.combo_rec_format.setCurrentIndex(idx)

        if "replay_enabled" in cfg:
            self.chk_replay.setChecked(bool(cfg["replay_enabled"]))

        if "replay_sec" in cfg:
            self.spn_replay_sec.setValue(int(cfg["replay_sec"]))

        if "replay_mem_mb" in cfg:
            self.spn_replay_mem.setValue(int(cfg["replay_mem_mb"]))


