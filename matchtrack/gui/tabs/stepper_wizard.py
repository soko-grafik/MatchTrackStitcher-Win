"""
Once Sport Autocam 5-Step Stepper Wizard for MatchTrack-Stitcher.
Implements the clean linear 5-step guided workflow:
  1 Setup ➔ 2 Source ➔ 3 Field marking ➔ 4 Framing ➔ 5 Output
Equivalent to Once Sport Autocam v4.1.3.
"""

import os
import time
import cv2
import numpy as np
import logging
from typing import Optional, Dict, Any, Tuple

from PySide6.QtCore import Qt, Signal, QRectF, QPointF, QSize, QTimer
from PySide6.QtGui import QImage, QPixmap, QColor, QFont, QPen, QPainter, QIcon
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QPushButton,
    QStackedWidget, QFrame, QFileDialog, QComboBox, QSlider, QCheckBox,
    QLineEdit, QScrollArea, QProgressBar, QMessageBox, QGroupBox, QSpinBox
)

from ..widgets import NoWheelSpinBox, NoWheelDoubleSpinBox, LabeledSliderSpinBox
from ...field_calibration import FieldMarking

def create_codec_combobox() -> QComboBox:
    cb = QComboBox()
    cb.addItem("⚡ Auto (Hardware NVENC H.264 falls verf\u00fcgbar)", "h264_nvenc")
    cb.addItem("NVIDIA NVENC H.264 (8-Bit yuv420p)", "h264_nvenc")
    cb.addItem("Intel QuickSync H.264 (h264_qsv)", "h264_qsv")
    cb.addItem("AMD AMF H.264 (h264_amf)", "h264_amf")
    cb.addItem("CPU x264 Software (libx264 - 8-Bit yuv420p)", "libx264")
    return cb

logger = logging.getLogger("MatchTrack")


class StepperWizardWidget(QWidget):
    """
    Main 5-Step Wizard Container replicating the Once Sport Autocam interface.
    """
    step_changed = Signal(int) # Emits current step index (0..4)

    def __init__(self, main_win, parent=None):
        super().__init__(parent)
        self.win = main_win
        self.current_step = 0

        # State Model
        self.rig_type: str = "two_cameras"  # "single_camera" or "two_cameras"
        self.sport: str = "football"        # "football", "basketball", "handball", "futsal", "other"
        self.framing_preset: str = "broadcast" # "tactical", "broadcast", "custom", "fixed"
        self.club_logo_path: Optional[str] = None

        # Timeline Trim State
        self.total_video_frames: int = 0
        self.video_fps: float = 30.0
        self.trim_start_frame: int = 0
        self.trim_end_frame: int = 0

        self._init_ui()
        QTimer.singleShot(700, self.update_cloud_ui)

    def _init_ui(self):
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(12, 12, 12, 12)
        root_layout.setSpacing(12)

        # -------------------------------------------------------------
        # TOP STEPPER HEADER
        # -------------------------------------------------------------
        self.header_card = QFrame()
        self.header_card.setStyleSheet("QFrame { background: #0c0c14; border: 1px solid #1e1e2d; border-radius: 12px; padding: 6px 12px; }")
        header_layout = QHBoxLayout(self.header_card)
        header_layout.setContentsMargins(4, 4, 4, 4)
        header_layout.setSpacing(8)

        self.step_buttons = []
        step_titles = [
            ("1", "Setup"),
            ("2", "Source"),
            ("3", "Field marking"),
            ("4", "Framing"),
            ("5", "Output")
        ]

        for idx, (num, name) in enumerate(step_titles):
            btn = QPushButton(f"  {num}  {name}  ")
            btn.setCheckable(True)
            btn.setStyleSheet("""
                QPushButton {
                    background-color: transparent;
                    color: #94a3b8;
                    font-weight: 700;
                    font-size: 12px;
                    padding: 8px 16px;
                    border-radius: 8px;
                    border: 1px solid transparent;
                }
                QPushButton:hover {
                    color: #ffffff;
                    background-color: #1a1a28;
                }
                QPushButton:checked {
                    background-color: #10b981;
                    color: #ffffff;
                    border: 1px solid #059669;
                }
            """)
            btn.clicked.connect(lambda _, i=idx: self.go_to_step(i))
            self.step_buttons.append(btn)
            header_layout.addWidget(btn)

        header_layout.addStretch()

        root_layout.addWidget(self.header_card)

        # -------------------------------------------------------------
        # MIDDLE STACKED WIDGET (Pages 0..4)
        # -------------------------------------------------------------
        self.pages_stack = QStackedWidget()
        self.pages_stack.addWidget(self._build_step1_setup())
        self.pages_stack.addWidget(self._build_step2_source())
        self.pages_stack.addWidget(self._build_step3_field())
        self.pages_stack.addWidget(self._build_step4_framing())
        self.pages_stack.addWidget(self._build_step5_output())

        root_layout.addWidget(self.pages_stack, 1)

        # -------------------------------------------------------------
        # BOTTOM NAVIGATION ACTION BAR
        # -------------------------------------------------------------
        self.bottom_bar = QFrame()
        self.bottom_bar.setStyleSheet("QFrame { background: #0c0c14; border: 1px solid #1e1e2d; border-radius: 12px; padding: 6px 12px; }")
        bottom_layout = QHBoxLayout(self.bottom_bar)
        bottom_layout.setContentsMargins(8, 6, 8, 6)

        self.btn_nav_back = QPushButton("❮ Zurück")
        self.btn_nav_back.setStyleSheet("padding: 8px 20px; font-weight: 700; font-size: 12px; background-color: #1e1e2d; color: #cbd5e1; border-radius: 8px; border: 1px solid #28283a;")
        self.btn_nav_back.clicked.connect(self.prev_step)
        bottom_layout.addWidget(self.btn_nav_back)

        bottom_layout.addStretch()

        self.btn_nav_next = QPushButton("Weiter: Source ❯")
        self.btn_nav_next.setStyleSheet("padding: 9px 26px; font-weight: 800; font-size: 13px; background-color: #10b981; color: #ffffff; border-radius: 8px; border: none;")
        self.btn_nav_next.clicked.connect(self.next_step)
        bottom_layout.addWidget(self.btn_nav_next)

        root_layout.addWidget(self.bottom_bar)

        self.update_stepper_ui()

    # =========================================================================
    # STEP 1: SETUP
    # =========================================================================
    def _build_step1_setup(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(20)

        lbl_title = QLabel("Aufnahme & Sportart konfigurieren (Set up the recording)")
        lbl_title.setStyleSheet("font-size: 20px; font-weight: 800; color: #f8fafc;")
        layout.addWidget(lbl_title)

        lbl_desc = QLabel("Wähle aus, wie das Spiel aufgenommen wurde und welche Sportart analysiert werden soll.")
        lbl_desc.setStyleSheet("color: #94a3b8; font-size: 13px;")
        layout.addWidget(lbl_desc)

        # 1. Camera Rig Section
        lbl_rig_hdr = QLabel("KAMERA-SETUP (CAMERA RIG)")
        lbl_rig_hdr.setStyleSheet("color: #64748b; font-size: 11px; font-weight: 800; letter-spacing: 1px; margin-top: 10px;")
        layout.addWidget(lbl_rig_hdr)

        row_rig = QHBoxLayout()
        row_rig.setSpacing(16)

        self.btn_rig_single = QPushButton("📹 Einzelkamera / Fertiges Panorama\nEine Weitwinkel-Kamera oder fertiges Panoramavideo")
        self.btn_rig_single.setCheckable(True)
        self.btn_rig_single.setStyleSheet(self._card_button_style())
        self.btn_rig_single.clicked.connect(lambda: self._select_rig("single_camera"))

        self.btn_rig_two = QPushButton("🧵 Zwei Kameras - Stitching\nLinke und rechte Kamera nahtlos gestitcht (Dual DJI Action 4)")
        self.btn_rig_two.setCheckable(True)
        self.btn_rig_two.setChecked(True)
        self.btn_rig_two.setStyleSheet(self._card_button_style(active=True))
        self.btn_rig_two.clicked.connect(lambda: self._select_rig("two_cameras"))

        row_rig.addWidget(self.btn_rig_single, 1)
        row_rig.addWidget(self.btn_rig_two, 1)
        layout.addLayout(row_rig)

        # 2. Sport Section
        lbl_sport_hdr = QLabel("SPORTART (SPORT)")
        lbl_sport_hdr.setStyleSheet("color: #64748b; font-size: 11px; font-weight: 800; letter-spacing: 1px; margin-top: 15px;")
        layout.addWidget(lbl_sport_hdr)

        row_sports = QHBoxLayout()
        row_sports.setSpacing(12)

        self.sport_btns = {}
        sports = [
            ("football", "⚽ Fußball", "Ballgeführtes Panning mit distanzbasiertem Zoom"),
            ("basketball", "🏀 Basketball", "Halbfeld-Tracking, konstanter Zoom"),
            ("handball", "🤾 Handball", "Schnelles End-to-End Spiel"),
            ("futsal", "🥅 Futsal", "Kleinfeld-Hallenfußball"),
            ("other", "🎯 Sonstige", "Allgemeines Ball-Tracking")
        ]

        for s_key, s_title, s_sub in sports:
            btn = QPushButton(f"{s_title}\n{s_sub}")
            btn.setCheckable(True)
            active = (s_key == "football")
            btn.setChecked(active)
            btn.setStyleSheet(self._card_button_style(active=active))
            btn.clicked.connect(lambda _, k=s_key: self._select_sport(k))
            self.sport_btns[s_key] = btn
            row_sports.addWidget(btn, 1)

        layout.addLayout(row_sports)
        layout.addStretch()

        return page

    def _select_rig(self, rig_type: str):
        self.rig_type = rig_type
        self.btn_rig_single.setChecked(rig_type == "single_camera")
        self.btn_rig_two.setChecked(rig_type == "two_cameras")
        self.btn_rig_single.setStyleSheet(self._card_button_style(active=(rig_type == "single_camera")))
        self.btn_rig_two.setStyleSheet(self._card_button_style(active=(rig_type == "two_cameras")))
        self._update_source_page_visibility()

    def _select_sport(self, sport: str):
        self.sport = sport
        for k, btn in self.sport_btns.items():
            is_act = (k == sport)
            btn.setChecked(is_act)
            btn.setStyleSheet(self._card_button_style(active=is_act))

    def _card_button_style(self, active: bool = False) -> str:
        if active:
            return """
                QPushButton {
                    background-color: #0f2e22;
                    border: 2px solid #10b981;
                    border-radius: 12px;
                    color: #ffffff;
                    font-weight: 700;
                    font-size: 12px;
                    padding: 16px;
                    text-align: left;
                }
            """
        return """
            QPushButton {
                background-color: #0c0c14;
                border: 1px solid #1e1e2d;
                border-radius: 12px;
                color: #94a3b8;
                font-weight: 600;
                font-size: 12px;
                padding: 16px;
                text-align: left;
            }
            QPushButton:hover {
                border-color: #3b82f6;
                color: #ffffff;
                background-color: #141420;
            }
        """

    # =========================================================================
    # STEP 2: SOURCE
    # =========================================================================
    def _build_step2_source(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)

        lbl_title = QLabel("Videoquellen & Trim-Bereich (Where is the footage coming from?)")
        lbl_title.setStyleSheet("font-size: 20px; font-weight: 800; color: #f8fafc;")
        layout.addWidget(lbl_title)

        # Container for Two-Cam mode
        self.box_two_cam = QWidget()
        layout_two = QVBoxLayout(self.box_two_cam)
        layout_two.setContentsMargins(0, 0, 0, 0)
        layout_two.setSpacing(12)

        row_cams = QHBoxLayout()
        row_cams.setSpacing(16)

        # Left Cam Card
        card_l = QFrame()
        card_l.setStyleSheet("QFrame { background: #0c0c14; border: 1px solid #1e1e2d; border-radius: 10px; padding: 12px; }")
        l_layout = QVBoxLayout(card_l)
        l_layout.addWidget(QLabel("📷 LINKES VIDEO (LEFT CAMERA)"))
        self.lbl_path_l = QLabel("Kein Video gewählt")
        self.lbl_path_l.setStyleSheet("color: #94a3b8; font-size: 11px;")
        l_layout.addWidget(self.lbl_path_l)
        btn_open_l = QPushButton("📁 Linke Videodatei wählen...")
        btn_open_l.setStyleSheet("padding: 8px; font-weight: 700; background: #1e1e2d; color: white; border-radius: 6px;")
        btn_open_l.clicked.connect(self.win.open_left_video)
        l_layout.addWidget(btn_open_l)
        row_cams.addWidget(card_l, 1)

        # Right Cam Card
        card_r = QFrame()
        card_r.setStyleSheet("QFrame { background: #0c0c14; border: 1px solid #1e1e2d; border-radius: 10px; padding: 12px; }")
        r_layout = QVBoxLayout(card_r)
        r_layout.addWidget(QLabel("📷 RECHTES VIDEO (RIGHT CAMERA)"))
        self.lbl_path_r = QLabel("Kein Video gewählt")
        self.lbl_path_r.setStyleSheet("color: #94a3b8; font-size: 11px;")
        r_layout.addWidget(self.lbl_path_r)
        btn_open_r = QPushButton("📁 Rechte Videodatei wählen...")
        btn_open_r.setStyleSheet("padding: 8px; font-weight: 700; background: #1e1e2d; color: white; border-radius: 6px;")
        btn_open_r.clicked.connect(self.win.open_right_video)
        r_layout.addWidget(btn_open_r)
        row_cams.addWidget(card_r, 1)

        layout_two.addLayout(row_cams)

        # Calibration Preset Picker Row
        row_calib = QHBoxLayout()
        row_calib.addWidget(QLabel("Kamera-Kalibrierung (Calibration):"))
        self.combo_cam_preset = QComboBox()
        self.combo_cam_preset.addItem("⭐ DJI Osmo Action 4 - 4:3 RockSteady (Empfohlen)", "DJI_Action4_1080p_4_3_Dewarp_RockSteady")
        self.combo_cam_preset.addItem("⭐ DJI Osmo Action 4 - 2.7K 16:9 Dewarp", "DJI_Action4_2_7K_16_9")
        self.combo_cam_preset.addItem("⭐ DJI Osmo Action 4 - 4K 16:9 Dewarp", "DJI_Action4_4K_16_9")
        self.combo_cam_preset.addItem("📷 GoPro Hero 11 / 12 / 13 Wide", "GoPro13_Wide")
        self.combo_cam_preset.addItem("📱 Apple iPhone 0.5x UltraWide", "iPhone_UltraWide")
        self.combo_cam_preset.currentIndexChanged.connect(self._on_preset_selected)
        row_calib.addWidget(self.combo_cam_preset, 1)

        self.btn_prep_stitch = QPushButton("🧵 Stitching vorbereiten (Prepare stitching)")
        self.btn_prep_stitch.setStyleSheet("padding: 8px 16px; font-weight: 800; background: #2563eb; color: white; border-radius: 6px;")
        self.btn_prep_stitch.clicked.connect(self._prepare_stitching)
        row_calib.addWidget(self.btn_prep_stitch)

        layout_two.addLayout(row_calib)
        layout.addWidget(self.box_two_cam)

        # Container for Single-Cam / Pre-stitched Panorama mode
        self.box_single_cam = QWidget()
        layout_single = QVBoxLayout(self.box_single_cam)
        layout_single.setContentsMargins(0, 0, 0, 0)
        layout_single.setSpacing(12)

        card_pano = QFrame()
        card_pano.setStyleSheet("QFrame { background: #0c0c14; border: 1px solid #1e1e2d; border-radius: 10px; padding: 16px; }")
        p_layout = QVBoxLayout(card_pano)
        p_layout.addWidget(QLabel("🎬 FERTIGES PANORAMA-VIDEO (PANORAMIC FOOTAGE)"))
        self.lbl_path_pano = QLabel("Kein Panoramavideo geladen")
        self.lbl_path_pano.setStyleSheet("color: #94a3b8; font-size: 12px;")
        p_layout.addWidget(self.lbl_path_pano)

        btn_open_pano = QPushButton("📁 Fertiges Panoramavideo wählen (32:9 / 8:3)...")
        btn_open_pano.setStyleSheet("padding: 10px; font-weight: 800; background: #10b981; color: white; border-radius: 8px;")
        btn_open_pano.clicked.connect(self.win.open_panorama_video)
        p_layout.addWidget(btn_open_pano)

        layout_single.addWidget(card_pano)
        layout.addWidget(self.box_single_cam)

        # 3. Trim Section (Timeline Range)
        grp_trim = QFrame()
        grp_trim.setStyleSheet("QFrame { background: #0c0c14; border: 1px solid #1e1e2d; border-radius: 10px; padding: 12px; }")
        t_layout = QVBoxLayout(grp_trim)
        t_layout.setSpacing(10)
        
        lbl_trim_hdr = QLabel("✂️ SCHNITTBEREICH (TRIM RANGE)")
        lbl_trim_hdr.setStyleSheet("color: #64748b; font-size: 11px; font-weight: 800; letter-spacing: 1px;")
        t_layout.addWidget(lbl_trim_hdr)

        row_trim_ctrls = QHBoxLayout()
        row_trim_ctrls.setSpacing(12)

        lbl_s = QLabel("Start-Frame:")
        lbl_s.setStyleSheet("color: #cbd5e1; font-weight: 700; font-size: 11px;")
        self.spin_trim_start = QSpinBox()
        self.spin_trim_start.setRange(0, 999999)
        self.spin_trim_start.setValue(0)
        self.spin_trim_start.setFixedWidth(100)
        self.spin_trim_start.setStyleSheet("background: #1e1e2d; color: #10b981; font-weight: 700; padding: 6px; border-radius: 6px;")
        self.spin_trim_start.valueChanged.connect(self._on_trim_changed)
        row_trim_ctrls.addWidget(lbl_s)
        row_trim_ctrls.addWidget(self.spin_trim_start)

        lbl_e = QLabel("End-Frame:")
        lbl_e.setStyleSheet("color: #cbd5e1; font-weight: 700; font-size: 11px;")
        self.spin_trim_end = QSpinBox()
        self.spin_trim_end.setRange(0, 999999)
        self.spin_trim_end.setValue(0)
        self.spin_trim_end.setFixedWidth(100)
        self.spin_trim_end.setStyleSheet("background: #1e1e2d; color: #ef4444; font-weight: 700; padding: 6px; border-radius: 6px;")
        self.spin_trim_end.valueChanged.connect(self._on_trim_changed)
        row_trim_ctrls.addWidget(lbl_e)
        row_trim_ctrls.addWidget(self.spin_trim_end)

        btn_full_range = QPushButton("🔄 Gesamtes Video")
        btn_full_range.setStyleSheet("background: #1e1e2d; color: #94a3b8; font-weight: 700; padding: 6px 12px; border-radius: 6px; border: 1px solid #28283a;")
        btn_full_range.clicked.connect(self._reset_trim_to_full)
        row_trim_ctrls.addWidget(btn_full_range)
        row_trim_ctrls.addStretch()

        t_layout.addLayout(row_trim_ctrls)

        row_trim_info = QHBoxLayout()
        self.lbl_trim_start = QLabel("START: 00:00:00 (#0)")
        self.lbl_trim_start.setStyleSheet("color: #10b981; font-weight: 700; font-family: monospace; font-size: 11px;")
        self.lbl_trim_end = QLabel("ENDE: 00:00:00 (#0)")
        self.lbl_trim_end.setStyleSheet("color: #ef4444; font-weight: 700; font-family: monospace; font-size: 11px;")
        self.lbl_trim_len = QLabel("Länge: 00:00:00 (0 Frames)")
        self.lbl_trim_len.setStyleSheet("color: #38bdf8; font-weight: 700; font-size: 11px;")

        row_trim_info.addWidget(self.lbl_trim_start)
        row_trim_info.addWidget(self.lbl_trim_end)
        row_trim_info.addStretch()
        row_trim_info.addWidget(self.lbl_trim_len)
        t_layout.addLayout(row_trim_info)

        layout.addWidget(grp_trim)
        layout.addStretch()

        self._update_source_page_visibility()
        return page

    def format_tc(self, frame_idx: int) -> str:
        fps = max(1.0, getattr(self, 'video_fps', 30.0))
        sec = max(0, frame_idx) / fps
        m, s = divmod(int(sec), 60)
        h, m = divmod(m, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    def update_video_info(self, total_frames: int, fps: float):
        self.total_video_frames = max(1, total_frames)
        self.video_fps = max(1.0, fps)
        self.trim_start_frame = 0
        self.trim_end_frame = self.total_video_frames

        if hasattr(self, 'spin_trim_start') and hasattr(self, 'spin_trim_end'):
            self.spin_trim_start.blockSignals(True)
            self.spin_trim_end.blockSignals(True)
            self.spin_trim_start.setRange(0, self.total_video_frames)
            self.spin_trim_end.setRange(0, self.total_video_frames)
            self.spin_trim_start.setValue(0)
            self.spin_trim_end.setValue(self.total_video_frames)
            self.spin_trim_start.blockSignals(False)
            self.spin_trim_end.blockSignals(False)

        self._refresh_trim_labels()
        self.update_summary()

    def _on_trim_changed(self):
        s_val = self.spin_trim_start.value()
        e_val = self.spin_trim_end.value()
        if s_val > e_val:
            s_val = e_val
            self.spin_trim_start.blockSignals(True)
            self.spin_trim_start.setValue(s_val)
            self.spin_trim_start.blockSignals(False)

        self.trim_start_frame = s_val
        self.trim_end_frame = e_val
        self._refresh_trim_labels()
        self.update_summary()

    def _reset_trim_to_full(self):
        if hasattr(self, 'spin_trim_start') and hasattr(self, 'spin_trim_end'):
            self.spin_trim_start.setValue(0)
            self.spin_trim_end.setValue(self.total_video_frames)

    def _refresh_trim_labels(self):
        if not hasattr(self, 'lbl_trim_start'):
            return
        diff = max(0, self.trim_end_frame - self.trim_start_frame)
        self.lbl_trim_start.setText(f"START: {self.format_tc(self.trim_start_frame)} (#{self.trim_start_frame})")
        self.lbl_trim_end.setText(f"ENDE: {self.format_tc(self.trim_end_frame)} (#{self.trim_end_frame})")
        self.lbl_trim_len.setText(f"Länge: {self.format_tc(diff)} ({diff:,} Frames)")

    def _update_source_page_visibility(self):
        is_two = (self.rig_type == "two_cameras")
        self.box_two_cam.setVisible(is_two)
        self.box_single_cam.setVisible(not is_two)

    def _on_preset_selected(self, idx: int):
        val = self.combo_cam_preset.currentData()
        if val and hasattr(self.win, 'apply_action4_4x3_rocksteady_preset'):
            self.win.apply_action4_4x3_rocksteady_preset()

    def _prepare_stitching(self):
        self.win.run_auto_stitch_calibration()

    # =========================================================================
    # STEP 3: FIELD MARKING (10-Point Sideline Model)
    # =========================================================================
    def _build_step3_field(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(14)

        lbl_title = QLabel("Spielfeld markieren (Mark the playing field)")
        lbl_title.setStyleSheet("font-size: 20px; font-weight: 800; color: #f8fafc;")
        layout.addWidget(lbl_title)

        lbl_desc = QLabel(
            "Platziere zehn Punkte – fünf auf jeder Seitenlinie. Beginne an der nahen linken Ecke und gehe gegen den "
            "Uhrzeigersinn: Entlang der nahen Seitenlinie (ein Punkt an jedem Viertel) zur rechten Ecke, hinauf zur "
            "fernen rechten Ecke und zurück entlang der Gegengeraden zur fernen linken Ecke."
        )
        lbl_desc.setWordWrap(True)
        lbl_desc.setStyleSheet("color: #94a3b8; font-size: 13px;")
        layout.addWidget(lbl_desc)

        # Action Bar (Auto mark, Undo, Reset)
        row_actions = QHBoxLayout()
        row_actions.setSpacing(12)

        self.btn_auto_mark = QPushButton("🎯 Auto-Mark (Linien automatisch erkennen)")
        self.btn_auto_mark.setStyleSheet("padding: 10px 18px; font-weight: 800; font-size: 13px; background: #10b981; color: white; border-radius: 8px;")
        self.btn_auto_mark.clicked.connect(self._auto_mark_field)
        row_actions.addWidget(self.btn_auto_mark)

        self.btn_undo_mark = QPushButton("↺ Rückgängig (Undo)")
        self.btn_undo_mark.setStyleSheet("padding: 10px 16px; font-weight: 700; background: #1e1e2d; color: #cbd5e1; border-radius: 8px; border: 1px solid #28283a;")
        self.btn_undo_mark.clicked.connect(self._undo_field_mark)
        row_actions.addWidget(self.btn_undo_mark)

        self.btn_reset_mark = QPushButton("Punkte zurücksetzen")
        self.btn_reset_mark.setStyleSheet("padding: 10px 16px; font-weight: 700; background: #1e1e2d; color: #cbd5e1; border-radius: 8px; border: 1px solid #28283a;")
        self.btn_reset_mark.clicked.connect(self._reset_field_mark)
        row_actions.addWidget(self.btn_reset_mark)

        row_actions.addStretch()

        # 10/10 Points Badge Indicator
        self.lbl_points_badge = QLabel("🟢 10 / 10 Punkte gesetzt – Bereit!")
        self.lbl_points_badge.setStyleSheet("color: #10b981; font-weight: 800; font-size: 12px; padding: 6px 14px; background: rgba(16, 185, 129, 0.15); border: 1px solid #10b981; border-radius: 20px;")
        row_actions.addWidget(self.lbl_points_badge)

        layout.addLayout(row_actions)

        # Instructions / Point Table Preview
        info_card = QFrame()
        info_card.setStyleSheet("QFrame { background: #0c0c14; border: 1px solid #1e1e2d; border-radius: 10px; padding: 14px; }")
        ic_layout = QVBoxLayout(info_card)
        ic_layout.setSpacing(6)

        ic_layout.addWidget(QLabel("💡 TIPPS ZUR PUNKTE-PLATZIERUNG:"))
        t1 = QLabel("• Punkte 1 bis 5: Nahe Seitenauslinie (1: Eckfahne links, 2: 1/4 Feld, 3: Mittellinie, 4: 3/4 Feld, 5: Eckfahne rechts)")
        t1.setStyleSheet("color: #cbd5e1; font-size: 12px;")
        t2 = QLabel("• Punkte 6 bis 10: Ferne Gegenauslinie (6: Eckfahne rechts, 7: 3/4, 8: Mittellinie, 9: 1/4, 10: Eckfahne links)")
        t2.setStyleSheet("color: #cbd5e1; font-size: 12px;")
        t3 = QLabel("• Du kannst jeden Punkt im Panorama-Fenster mit der Maus anfassen und punktgenau auf die Kreidelinie ziehen.")
        t3.setStyleSheet("color: #10b981; font-size: 12px; font-weight: 600;")
        ic_layout.addWidget(t1)
        ic_layout.addWidget(t2)
        ic_layout.addWidget(t3)

        layout.addWidget(info_card)
        layout.addStretch()

        return page

    def _auto_mark_field(self):
        curr_bgr = self.win.viewport.current_frame_bgr if hasattr(self.win, 'viewport') else None
        if curr_bgr is not None and hasattr(self.win.viewport, 'field_marking'):
            ok = self.win.viewport.field_marking.auto_detect_from_image(curr_bgr)
            if ok:
                self.win.viewport.update()
                self.win.status_bar.showMessage("🎯 Spielfeld 10-Punkte Linien automatisch erkannt!")
            else:
                self.win.status_bar.showMessage("Auto-Mark: Konnte Linien nicht sicher trennen, Standard-Punkte aktiv.")
        else:
            self.win.status_bar.showMessage("Bitte zuerst ein Video laden!")

    def _undo_field_mark(self):
        if hasattr(self.win, 'viewport') and hasattr(self.win.viewport, 'field_marking'):
            if self.win.viewport.field_marking.undo():
                self.win.viewport.update()

    def _reset_field_mark(self):
        if hasattr(self.win, 'viewport') and hasattr(self.win.viewport, 'field_marking'):
            self.win.viewport.field_marking.reset_to_defaults()
            self.win.viewport.update()

    # =========================================================================
    # STEP 4: FRAMING (Dual Live Preview Monitors & 2-Box Zoom Control)
    # =========================================================================
    def _build_step4_framing(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)

        lbl_title = QLabel("Bildausschnitt & Zoom-Grenzen festlegen (Frame the shot)")
        lbl_title.setStyleSheet("font-size: 20px; font-weight: 800; color: #f8fafc;")
        layout.addWidget(lbl_title)

        lbl_desc = QLabel(
            "Die virtuelle KI-Kamera rendert eine 16:9 Ausgabe zwischen diesen beiden Bildausschnitten. "
            "Der Zoom vergrößert das Bild stufenlos, je weiter der Ball von der Kamera entfernt ist."
        )
        lbl_desc.setStyleSheet("color: #94a3b8; font-size: 13px;")
        layout.addWidget(lbl_desc)

        # Framing Preset Selection Row
        row_presets = QHBoxLayout()
        row_presets.setSpacing(12)

        self.framing_btns = {}
        f_presets = [
            ("tactical", "🏟️ Tactical", "Bleibt weit für volle Spielformation"),
            ("broadcast", "🎥 Broadcast", "Folgt dynamisch dem Spiel wie im TV (Standard)"),
            ("custom", "⚙️ Custom", "Individuelle Glättungs- & Zoom-Parameter"),
            ("fixed", "📌 Fixed view", "Fester Bildausschnitt ohne Kameraschwenk")
        ]

        for p_key, p_title, p_sub in f_presets:
            btn = QPushButton(f"{p_title}\n{p_sub}")
            btn.setCheckable(True)
            active = (p_key == "broadcast")
            btn.setChecked(active)
            btn.setStyleSheet(self._card_button_style(active=active))
            btn.clicked.connect(lambda _, k=p_key: self._select_framing_preset(k))
            self.framing_btns[p_key] = btn
            row_presets.addWidget(btn, 1)

        layout.addLayout(row_presets)

        # Dual Live-Preview Cards Row (Lowest zoom vs Highest zoom)
        lbl_monitors_hdr = QLabel("ECHTZEIT-VORSCHAU DER BEIDEN ZOOM-GRENZEN (LIVE PREVIEW MONITORS):")
        lbl_monitors_hdr.setStyleSheet("color: #64748b; font-size: 11px; font-weight: 800; letter-spacing: 1px; margin-top: 10px;")
        layout.addWidget(lbl_monitors_hdr)

        row_monitors = QHBoxLayout()
        row_monitors.setSpacing(16)

        # Card 1: Lowest Zoom (Widest 1.2x)
        card_low = QFrame()
        card_low.setStyleSheet("QFrame { background: #0c0c14; border: 2px solid #10b981; border-radius: 10px; padding: 8px; }")
        cl_layout = QVBoxLayout(card_low)
        cl_layout.setContentsMargins(6, 6, 6, 6)
        cl_layout.setSpacing(6)

        hdr_low = QHBoxLayout()
        self.lbl_low_title = QLabel("🟢 Weitester Zoom (Widest - 1.2x)")
        self.lbl_low_title.setStyleSheet("color: #10b981; font-weight: 800; font-size: 12px;")
        hdr_low.addWidget(self.lbl_low_title)
        hdr_low.addStretch()
        cl_layout.addLayout(hdr_low)

        self.preview_monitor_low = QLabel()
        self.preview_monitor_low.setFixedSize(384, 216)
        self.preview_monitor_low.setStyleSheet("background: #000; border-radius: 6px;")
        self.preview_monitor_low.setAlignment(Qt.AlignCenter)
        self.preview_monitor_low.setText("Live Min-Zoom Vorschau")
        cl_layout.addWidget(self.preview_monitor_low)

        lbl_low_hint = QLabel("Ziehe den grünen Rahmen im Panorama oder verändere die Ecke.")
        lbl_low_hint.setStyleSheet("color: #94a3b8; font-size: 11px;")
        cl_layout.addWidget(lbl_low_hint)

        row_monitors.addWidget(card_low, 1)

        # Card 2: Highest Zoom (Tightest 2.0x)
        card_high = QFrame()
        card_high.setStyleSheet("QFrame { background: #0c0c14; border: 2px dashed #ffffff; border-radius: 10px; padding: 8px; }")
        ch_layout = QVBoxLayout(card_high)
        ch_layout.setContentsMargins(6, 6, 6, 6)
        ch_layout.setSpacing(6)

        hdr_high = QHBoxLayout()
        self.lbl_high_title = QLabel("⚪ Engster Zoom (Tightest - 2.0x)")
        self.lbl_high_title.setStyleSheet("color: #ffffff; font-weight: 800; font-size: 12px;")
        hdr_high.addWidget(self.lbl_high_title)
        hdr_high.addStretch()
        ch_layout.addLayout(hdr_high)

        self.preview_monitor_high = QLabel()
        self.preview_monitor_high.setFixedSize(384, 216)
        self.preview_monitor_high.setStyleSheet("background: #000; border-radius: 6px;")
        self.preview_monitor_high.setAlignment(Qt.AlignCenter)
        self.preview_monitor_high.setText("Live Max-Zoom Vorschau")
        ch_layout.addWidget(self.preview_monitor_high)

        lbl_high_hint = QLabel("Ziehe den weißen Rahmen im Panorama oder skaliere den Zoom.")
        lbl_high_hint.setStyleSheet("color: #94a3b8; font-size: 11px;")
        ch_layout.addWidget(lbl_high_hint)

        row_monitors.addWidget(card_high, 1)

        layout.addLayout(row_monitors)

        # AI Tracking Model Selection Row
        grp_model = QFrame()
        grp_model.setStyleSheet("QFrame { background: #0c0c14; border: 1px solid #1e1e2d; border-radius: 10px; padding: 12px 16px; margin-top: 4px; }")
        l_model = QHBoxLayout(grp_model)
        l_model.setSpacing(14)

        lbl_m_title = QLabel("🤖 KI-Tracking & Ball-Modell:")
        lbl_m_title.setStyleSheet("color: #e2e8f0; font-weight: 800; font-size: 12px;")
        l_model.addWidget(lbl_m_title)

        from ...paths import get_available_ai_models
        self.combo_wizard_model = QComboBox()
        self.combo_wizard_model.setStyleSheet("""
            QComboBox {
                background: #141420;
                color: #f8fafc;
                font-weight: 700;
                font-size: 12px;
                padding: 6px 12px;
                border: 1px solid #28283c;
                border-radius: 6px;
            }
            QComboBox::drop-down { border: none; }
            QComboBox QAbstractItemView {
                background-color: #12121c;
                color: #f8fafc;
                selection-background-color: #10b981;
                border: 1px solid #28283c;
            }
        """)
        for m in get_available_ai_models():
            self.combo_wizard_model.addItem(m["name"], m["id"])
        self.combo_wizard_model.addItem("📁 Eigenes Modell (.onnx / .pt) wählen...", "custom_browse")
        self.combo_wizard_model.currentIndexChanged.connect(self._on_wizard_model_changed)
        l_model.addWidget(self.combo_wizard_model, 1)

        layout.addWidget(grp_model)
        layout.addStretch()

        return page

    def _on_wizard_model_changed(self, idx: int):
        data = self.combo_wizard_model.currentData()
        if data == "custom_browse":
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Eigenes KI-Modell wählen (.onnx oder .pt)", "",
                "KI-Modelle (*.onnx *.pt);;ONNX Modelle (*.onnx);;PyTorch Modelle (*.pt);;Alle Dateien (*.*)"
            )
            if file_path:
                fname = os.path.basename(file_path)
                custom_id = f"custom_{fname}"
                insert_idx = max(0, self.combo_wizard_model.count() - 1)
                self.combo_wizard_model.insertItem(insert_idx, f"📁 {fname} (Benutzerdefiniert)", custom_id)
                self.combo_wizard_model.blockSignals(True)
                self.combo_wizard_model.setCurrentIndex(insert_idx)
                self.combo_wizard_model.blockSignals(False)
                data = custom_id
                if hasattr(self.win.engine.ai_broadcast, 'config'):
                    self.win.engine.ai_broadcast.config.custom_model_path = file_path
            else:
                idx0 = self.combo_wizard_model.findData("yolo11")
                if idx0 >= 0:
                    self.combo_wizard_model.blockSignals(True)
                    self.combo_wizard_model.setCurrentIndex(idx0)
                    self.combo_wizard_model.blockSignals(False)
                return

        if hasattr(self.win, 'combo_ai_engine'):
            m_idx = self.win.combo_ai_engine.findData(data)
            if m_idx >= 0 and self.win.combo_ai_engine.currentIndex() != m_idx:
                self.win.combo_ai_engine.blockSignals(True)
                self.win.combo_ai_engine.setCurrentIndex(m_idx)
                self.win.combo_ai_engine.blockSignals(False)
        if hasattr(self.win.engine, 'ai_broadcast'):
            self.win.engine.ai_broadcast.set_backend(data)
        self.update_summary()

    def _select_framing_preset(self, preset: str):
        self.framing_preset = preset
        for k, btn in self.framing_btns.items():
            is_act = (k == preset)
            btn.setChecked(is_act)
            btn.setStyleSheet(self._card_button_style(active=is_act))

        # Adjust zoom levels in engine
        if hasattr(self.win, 'ctrl_min_zoom') and hasattr(self.win, 'ctrl_max_zoom'):
            if preset == "tactical":
                self.win.ctrl_min_zoom.setValue(1.05)
                self.win.ctrl_max_zoom.setValue(1.30)
            elif preset == "broadcast":
                self.win.ctrl_min_zoom.setValue(1.20)
                self.win.ctrl_max_zoom.setValue(2.00)
            elif preset == "fixed":
                self.win.ctrl_min_zoom.setValue(1.20)
                self.win.ctrl_max_zoom.setValue(1.20)
        self.update_framing_previews()

    def update_framing_previews(self):
        """Renders live side-by-side cropped preview monitors at lowest and highest zoom."""
        if not hasattr(self.win, 'viewport') or self.win.viewport.current_frame_bgr is None:
            return
        frame = self.win.viewport.current_frame_bgr
        fh, fw = frame.shape[:2]

        min_z = float(self.win.ctrl_min_zoom.value()) if hasattr(self.win, 'ctrl_min_zoom') else 1.2
        max_z = float(self.win.ctrl_max_zoom.value()) if hasattr(self.win, 'ctrl_max_zoom') else 2.0

        self.lbl_low_title.setText(f"🟢 Weitester Zoom (Widest - {min_z:.1f}x)")
        self.lbl_high_title.setText(f"⚪ Engster Zoom (Tightest - {max_z:.1f}x)")

        # Crop 16:9 around center for preview
        aspect = 16.0 / 9.0
        
        # Low zoom crop
        ch_low = int(fh / max(0.5, min_z))
        cw_low = int(ch_low * aspect)
        cx = fw // 2
        cy = fh // 2
        x1_l = max(0, cx - cw_low // 2)
        y1_l = max(0, cy - ch_low // 2)
        crop_low = frame[y1_l:min(fh, y1_l + ch_low), x1_l:min(fw, x1_l + cw_low)]
        
        if crop_low.size > 0:
            thumb_l = cv2.resize(crop_low, (384, 216))
            rgb_l = cv2.cvtColor(thumb_l, cv2.COLOR_BGR2RGB)
            qimg_l = QImage(rgb_l.data, 384, 216, 384 * 3, QImage.Format_RGB888)
            self.preview_monitor_low.setPixmap(QPixmap.fromImage(qimg_l))

        # High zoom crop
        ch_high = int(fh / max(0.5, max_z))
        cw_high = int(ch_high * aspect)
        x1_h = max(0, cx - cw_high // 2)
        y1_h = max(0, cy - ch_high // 2)
        crop_high = frame[y1_h:min(fh, y1_h + ch_high), x1_h:min(fw, x1_h + cw_high)]
        
        if crop_high.size > 0:
            thumb_h = cv2.resize(crop_high, (384, 216))
            rgb_h = cv2.cvtColor(thumb_h, cv2.COLOR_BGR2RGB)
            qimg_h = QImage(rgb_h.data, 384, 216, 384 * 3, QImage.Format_RGB888)
            self.preview_monitor_high.setPixmap(QPixmap.fromImage(qimg_h))

    # =========================================================================
    # STEP 5: OUTPUT (Summary Dashboard & 1-Click Export)
    # =========================================================================
    def _build_step5_output(self) -> QWidget:
        page = QWidget()
        layout = QHBoxLayout(page)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(24)

        # Left Column: Export & Encoder Settings
        col_left = QVBoxLayout()
        col_left.setSpacing(14)

        lbl_title = QLabel("Ausgabe & Export (Output)")
        lbl_title.setStyleSheet("font-size: 20px; font-weight: 800; color: #f8fafc;")
        col_left.addWidget(lbl_title)

        # Output Path Row
        lbl_dest = QLabel("SPEICHERORT:")
        lbl_dest.setStyleSheet("color: #64748b; font-size: 11px; font-weight: 800;")
        col_left.addWidget(lbl_dest)

        row_path = QHBoxLayout()
        self.txt_out_path = QLineEdit()
        self.txt_out_path.setReadOnly(True)
        self.txt_out_path.setStyleSheet("background: #0c0c14; border: 1px solid #1e1e2d; border-radius: 6px; padding: 8px; color: #f8fafc;")
        row_path.addWidget(self.txt_out_path, 1)

        btn_browse_out = QPushButton("Speichern unter...")
        btn_browse_out.setStyleSheet("padding: 8px 14px; font-weight: 700; background: #1e1e2d; color: white; border-radius: 6px;")
        btn_browse_out.clicked.connect(self._browse_output_path)
        row_path.addWidget(btn_browse_out)
        col_left.addLayout(row_path)

        # Resolution & Bitrate Grid
        grid_out = QGridLayout()
        grid_out.setSpacing(10)

        grid_out.addWidget(QLabel("Auflösung:"), 0, 0)
        self.combo_wizard_res = QComboBox()
        self.combo_wizard_res.addItem("1920x1080 (Full HD 1080p - Standard)", (1920, 1080))
        self.combo_wizard_res.addItem("3840x2160 (4K UHD 2160p - Hohe Details)", (3840, 2160))
        self.combo_wizard_res.addItem("2560x1440 (2K QHD 1440p)", (2560, 1440))
        self.combo_wizard_res.addItem("1280x720 (HD 720p - Schnell)", (1280, 720))
        grid_out.addWidget(self.combo_wizard_res, 0, 1)

        grid_out.addWidget(QLabel("Bitrate:"), 1, 0)
        self.ctrl_wizard_bitrate = LabeledSliderSpinBox("", 4, 100, 16, 1, " Mbps")
        grid_out.addWidget(self.ctrl_wizard_bitrate, 1, 1)

        grid_out.addWidget(QLabel("Hardware-Encoder:"), 2, 0)
        self.combo_wizard_codec = create_codec_combobox()
        grid_out.addWidget(self.combo_wizard_codec, 2, 1)

        col_left.addLayout(grid_out)

        # Logo Overlay Row (Optional)
        grp_logo = QFrame()
        grp_logo.setStyleSheet("QFrame { background: #0c0c14; border: 1px solid #1e1e2d; border-radius: 8px; padding: 10px; }")
        l_layout = QHBoxLayout(grp_logo)
        l_layout.setSpacing(10)

        self.lbl_logo_info = QLabel("Vereinswappen / Logo-Overlay (Optional): Kein Logo gewählt")
        self.lbl_logo_info.setStyleSheet("color: #94a3b8; font-size: 11px;")
        l_layout.addWidget(self.lbl_logo_info, 1)

        btn_logo = QPushButton("📁 Wappen wählen...")
        btn_logo.setStyleSheet("padding: 6px 12px; font-weight: 700; background: #1e1e2d; color: white; border-radius: 6px;")
        btn_logo.clicked.connect(self._browse_logo)
        l_layout.addWidget(btn_logo)

        col_left.addWidget(grp_logo)

        # Audio Mute
        self.chk_mute_audio = QCheckBox("Audio-Spur stummschalten (Mute audio)")
        self.chk_mute_audio.setStyleSheet("color: #cbd5e1; font-weight: 600;")
        col_left.addWidget(self.chk_mute_audio)

        # Lookahead Option (1-Pass vs 2-Pass)
        grp_mode = QFrame()
        grp_mode.setStyleSheet("QFrame { background: #0c0c14; border: 1px solid #1e1e2d; border-radius: 8px; padding: 12px; }")
        gm_layout = QVBoxLayout(grp_mode)
        gm_layout.setSpacing(6)

        self.chk_wizard_lookahead = QCheckBox("🎬 Filmische 2-Pass KI-Glättung (Lookahead)")
        self.chk_wizard_lookahead.setChecked(False)
        self.chk_wizard_lookahead.setStyleSheet("color: #f8fafc; font-weight: 700; font-size: 13px;")
        self.chk_wizard_lookahead.stateChanged.connect(lambda: self.update_summary())
        gm_layout.addWidget(self.chk_wizard_lookahead)

        lbl_mode_desc = QLabel("• Aus (1-Pass): Startet sofort in Echtzeit (~60-100 FPS), ideal für schnelle Ergebnisse.\n• Ein (2-Pass): Scannt vorab den Spielverlauf für filmische Schwenks ohne Ruckeln.")
        lbl_mode_desc.setStyleSheet("color: #94a3b8; font-size: 11px; line-height: 1.4;")
        gm_layout.addWidget(lbl_mode_desc)
        col_left.addWidget(grp_mode)

        # MatchTrack Online Cloud Integration (Auto-Upload Option)
        grp_cloud = QFrame()
        grp_cloud.setStyleSheet("QFrame { background: #0c0c14; border: 1px solid #1e1e2d; border-radius: 8px; padding: 12px; }")
        gc_layout = QVBoxLayout(grp_cloud)
        gc_layout.setSpacing(8)

        # Header Row: Checkbox + Login/Konto Button
        row_c_hdr = QHBoxLayout()
        self.chk_wizard_cloud_upload = QCheckBox("☁️ Nach Fertigstellung in MatchTrack Online Cloud hochladen")
        self.chk_wizard_cloud_upload.setChecked(False)
        self.chk_wizard_cloud_upload.setStyleSheet("color: #f8fafc; font-weight: 700; font-size: 13px;")
        self.chk_wizard_cloud_upload.stateChanged.connect(self._on_cloud_upload_toggled)
        row_c_hdr.addWidget(self.chk_wizard_cloud_upload, 1)

        self.btn_wizard_cloud_login = QPushButton("🔑 Verbinden")
        self.btn_wizard_cloud_login.setStyleSheet("background: #1e293b; color: #60a5fa; font-weight: 700; padding: 4px 10px; border-radius: 6px; border: 1px solid #3b82f6;")
        self.btn_wizard_cloud_login.clicked.connect(self._on_cloud_login_clicked)
        row_c_hdr.addWidget(self.btn_wizard_cloud_login)
        gc_layout.addLayout(row_c_hdr)

        # Details Row: Status badge + Team selection
        self.row_cloud_details = QHBoxLayout()
        self.row_cloud_details.setSpacing(10)

        self.lbl_wizard_cloud_status = QLabel("⚪ Nicht angemeldet")
        self.lbl_wizard_cloud_status.setStyleSheet("color: #94a3b8; font-size: 11px; font-weight: 600;")
        self.row_cloud_details.addWidget(self.lbl_wizard_cloud_status, 1)

        lbl_team_title = QLabel("Mannschaft:")
        lbl_team_title.setStyleSheet("color: #94a3b8; font-size: 11px; font-weight: 700;")
        self.row_cloud_details.addWidget(lbl_team_title)

        self.combo_wizard_cloud_team = QComboBox()
        self.combo_wizard_cloud_team.setStyleSheet("""
            QComboBox {
                background: #141420;
                color: #f8fafc;
                font-weight: 700;
                font-size: 11px;
                padding: 4px 8px;
                border: 1px solid #28283c;
                border-radius: 6px;
                min-width: 150px;
            }
            QComboBox::drop-down { border: none; }
            QComboBox QAbstractItemView {
                background-color: #12121c;
                color: #f8fafc;
                selection-background-color: #10b981;
            }
        """)
        self.combo_wizard_cloud_team.currentIndexChanged.connect(self._on_wizard_cloud_team_changed)
        self.row_cloud_details.addWidget(self.combo_wizard_cloud_team)

        gc_layout.addLayout(self.row_cloud_details)
        col_left.addWidget(grp_cloud)

        col_left.addStretch()

        # Big Action Button
        self.btn_start_render = QPushButton("🚀 Jetzt Video erstellen (Start processing)")
        self.btn_start_render.setStyleSheet("""
            QPushButton {
                background-color: #10b981;
                color: #ffffff;
                font-weight: 800;
                font-size: 15px;
                padding: 16px;
                border-radius: 10px;
                border: none;
            }
            QPushButton:hover {
                background-color: #059669;
            }
            QPushButton:disabled {
                background-color: #1e293b;
                color: #64748b;
            }
        """)
        self.btn_start_render.clicked.connect(self._start_export)
        col_left.addWidget(self.btn_start_render)

        # Live Progress & Stepping Preview Box in Step 5 (Matching Once Sport Autocam)
        self.progress_container = QFrame()
        self.progress_container.setObjectName("progress_container")
        self.progress_container.setStyleSheet("""
            QFrame#progress_container {
                background: #09090f;
                border: 1px solid #1e1e2d;
                border-radius: 12px;
                padding: 14px;
            }
        """)
        p_layout = QVBoxLayout(self.progress_container)
        p_layout.setSpacing(10)

        # 1. Top Header: Filename & Status Badge
        hdr_row = QHBoxLayout()
        self.lbl_progress_file = QLabel("🎬 MatchTrack_Autocam_Broadcast.mp4")
        self.lbl_progress_file.setStyleSheet("color: #f8fafc; font-weight: 800; font-size: 13px;")
        hdr_row.addWidget(self.lbl_progress_file, 1)

        self.lbl_progress_badge = QLabel("● Bereit")
        self.lbl_progress_badge.setStyleSheet("color: #94a3b8; font-weight: 700; font-size: 11px; padding: 3px 8px; background: #141420; border: 1px solid #28283a; border-radius: 8px;")
        hdr_row.addWidget(self.lbl_progress_badge)
        p_layout.addLayout(hdr_row)

        # 2. Percentage & ETA Row
        row_pct_eta = QHBoxLayout()
        self.lbl_progress_pct = QLabel("0.0%")
        self.lbl_progress_pct.setStyleSheet("color: #ffffff; font-weight: 900; font-size: 26px;")
        row_pct_eta.addWidget(self.lbl_progress_pct)

        self.lbl_progress_stage = QLabel("● Initialisierung")
        self.lbl_progress_stage.setStyleSheet("color: #10b981; font-weight: 700; font-size: 11px; margin-left: 8px;")
        row_pct_eta.addWidget(self.lbl_progress_stage)

        row_pct_eta.addStretch()

        self.lbl_progress_time = QLabel("ETA --:--  •  Dauer 00:00")
        self.lbl_progress_time.setStyleSheet("color: #94a3b8; font-weight: 700; font-size: 11px; font-family: monospace;")
        row_pct_eta.addWidget(self.lbl_progress_time)
        p_layout.addLayout(row_pct_eta)

        # 3. Progress Bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFixedHeight(12)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                background: #141420;
                border-radius: 6px;
                border: 1px solid #222233;
                text-align: center;
                color: transparent;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #059669, stop:1 #10b981);
                border-radius: 5px;
            }
        """)
        p_layout.addWidget(self.progress_bar)

        # 4. 16:9 Stepping Follow-Cam Preview Monitor Card
        card_preview = QFrame()
        card_preview.setStyleSheet("background: #000000; border: 1px solid #222233; border-radius: 10px;")
        cp_layout = QVBoxLayout(card_preview)
        cp_layout.setContentsMargins(4, 4, 4, 4)
        cp_layout.setAlignment(Qt.AlignCenter)

        self.lbl_export_preview = QLabel()
        self.lbl_export_preview.setFixedSize(480, 270)
        self.lbl_export_preview.setAlignment(Qt.AlignCenter)
        self.lbl_export_preview.setStyleSheet("background: #050508; border-radius: 8px; color: #64748b; font-size: 12px; font-weight: 700;")
        self.lbl_export_preview.setText("🎥 Live Follow-Cam Vorschau\n(Startet mit dem Rendering)")
        cp_layout.addWidget(self.lbl_export_preview)
        p_layout.addWidget(card_preview, alignment=Qt.AlignCenter)

        # 5. Telemetry Metrics Cards Row (Matching Autocam: Speed, FPS, Frames, Output Size)
        row_metrics = QHBoxLayout()
        row_metrics.setSpacing(8)

        def make_metric_card(title: str, default_val: str):
            box = QFrame()
            box.setStyleSheet("background: #10101a; border: 1px solid #1e1e2d; border-radius: 8px; padding: 4px 8px;")
            l = QVBoxLayout(box)
            l.setContentsMargins(4, 4, 4, 4)
            l.setSpacing(2)
            lbl_t = QLabel(title)
            lbl_t.setStyleSheet("color: #64748b; font-size: 10px; font-weight: 800; letter-spacing: 0.5px;")
            lbl_v = QLabel(default_val)
            lbl_v.setStyleSheet("color: #f8fafc; font-size: 12px; font-weight: 800;")
            lbl_v.setAlignment(Qt.AlignCenter)
            l.addWidget(lbl_t, alignment=Qt.AlignCenter)
            l.addWidget(lbl_v, alignment=Qt.AlignCenter)
            return box, lbl_v

        card_spd, self.lbl_metric_speed = make_metric_card("SPEED", "-")
        card_fps, self.lbl_metric_fps = make_metric_card("FPS", "-")
        card_frm, self.lbl_metric_frames = make_metric_card("FRAMES", "0")
        card_siz, self.lbl_metric_size = make_metric_card("OUTPUT SIZE", "-")

        row_metrics.addWidget(card_spd, 1)
        row_metrics.addWidget(card_fps, 1)
        row_metrics.addWidget(card_frm, 1)
        row_metrics.addWidget(card_siz, 1)
        p_layout.addLayout(row_metrics)

        # 6. Bottom Status & Stop / Abbrechen Row
        row_p_btns = QHBoxLayout()
        self.lbl_progress_status = QLabel("Bereit zum Exportieren.")
        self.lbl_progress_status.setStyleSheet("color: #94a3b8; font-weight: 600; font-size: 11px;")
        row_p_btns.addWidget(self.lbl_progress_status, 1)

        self.btn_cancel_render = QPushButton("⏹ Stop / Abbrechen")
        self.btn_cancel_render.setEnabled(False)
        self.btn_cancel_render.setStyleSheet("""
            QPushButton {
                background: #1f1215;
                color: #ef4444;
                font-weight: 800;
                font-size: 12px;
                padding: 6px 16px;
                border-radius: 8px;
                border: 1px solid #ef4444;
            }
            QPushButton:hover {
                background: #ef4444;
                color: #ffffff;
            }
            QPushButton:disabled {
                background: #141420;
                color: #64748b;
                border: 1px solid #28283a;
            }
        """)
        self.btn_cancel_render.clicked.connect(self._on_cancel_clicked)
        row_p_btns.addWidget(self.btn_cancel_render)
        p_layout.addLayout(row_p_btns)

        self.progress_container.setVisible(False)
        col_left.addWidget(self.progress_container)

        layout.addLayout(col_left, 1)

        # Right Column: Autocam Live Summary Card
        self.summary_card = QFrame()
        self.summary_card.setFixedWidth(340)
        self.summary_card.setStyleSheet("QFrame { background: #0c0c14; border: 1px solid #1e1e2d; border-radius: 12px; padding: 16px; }")
        s_layout = QVBoxLayout(self.summary_card)
        s_layout.setSpacing(12)

        lbl_sum_hdr = QLabel("ZUSAMMENFASSUNG (SUMMARY)")
        lbl_sum_hdr.setStyleSheet("color: #64748b; font-size: 11px; font-weight: 800; letter-spacing: 1px;")
        s_layout.addWidget(lbl_sum_hdr)

        self.summary_fields = {}
        items = [
            ("RIG - SPORT", "Zwei Kameras - Fußball"),
            ("SOURCE", "Keine Quelle gewählt"),
            ("FIELD", "10/10 Punkte markiert"),
            ("FRAMING", "1,2x - 2,0x (Tilt-korrigiert)"),
            ("MODEL", "MatchTrack YOLO11n"),
            ("OUTPUT", "1920x1080 - 16 Mb/s (H.264)"),
            ("CLOUD", "Deaktiviert"),
            ("EST. SIZE", "~1.8 GB"),
            ("EST. RAM", "~1.5 GB")
        ]

        for key, val in items:
            row = QHBoxLayout()
            lbl_k = QLabel(key)
            lbl_k.setStyleSheet("color: #64748b; font-size: 11px; font-weight: 700;")
            lbl_v = QLabel(val)
            lbl_v.setStyleSheet("color: #f8fafc; font-size: 11px; font-weight: 700; text-align: right;")
            lbl_v.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.summary_fields[key] = lbl_v
            row.addWidget(lbl_k)
            row.addWidget(lbl_v)
            s_layout.addLayout(row)

        s_layout.addStretch()

        btn_save_preset = QPushButton("💾 Als Voreinstellung sichern...")
        btn_save_preset.setStyleSheet("padding: 8px; font-weight: 700; background: #1e1e2d; color: #cbd5e1; border-radius: 6px; border: 1px solid #28283a;")
        btn_save_preset.clicked.connect(self.win.save_as_default_settings)
        s_layout.addWidget(btn_save_preset)

        layout.addWidget(self.summary_card)

        return page

    def _browse_output_path(self):
        f, _ = QFileDialog.getSaveFileName(self, "Export-Videodatei speichern", "MatchTrack_Autocam_Broadcast.mp4", "MP4 Videos (*.mp4)")
        if f:
            self.txt_out_path.setText(f)
            self.update_summary()

    def _browse_logo(self):
        f, _ = QFileDialog.getOpenFileName(self, "Vereinswappen wählen", "", "Bilder (*.png *.jpg *.jpeg)")
        if f:
            self.club_logo_path = f
            self.lbl_logo_info.setText(f"Wappen: {os.path.basename(f)}")

    def update_summary(self):
        """Updates summary statistics on the right card."""
        is_two = (self.rig_type == "two_cameras")
        rig_str = f"{'Zwei Kameras' if is_two else 'Einzelkamera'} - {self.sport.capitalize()}"
        self.summary_fields["RIG - SPORT"].setText(rig_str)

        src_str = "Dual DJI Action 4" if is_two else "Panorama Master"
        self.summary_fields["SOURCE"].setText(src_str)

        min_z = float(self.win.ctrl_min_zoom.value()) if hasattr(self.win, 'ctrl_min_zoom') else 1.2
        max_z = float(self.win.ctrl_max_zoom.value()) if hasattr(self.win, 'ctrl_max_zoom') else 2.0
        self.summary_fields["FRAMING"].setText(f"{min_z:.1f}x - {max_z:.1f}x (Broadcast)")

        if hasattr(self, 'combo_wizard_model') and "MODEL" in self.summary_fields:
            m_txt = self.combo_wizard_model.currentText()
            # Abbreviate for compact summary card
            if "YOLO11" in m_txt:
                disp_m = "YOLO11n (PyTorch)"
            elif "RT-DETR" in m_txt:
                disp_m = "RT-DETR (Attention)"
            elif "Pose" in m_txt:
                disp_m = "YOLO-Pose (Skelett)"
            elif "Soccer Ball (FP16" in m_txt:
                disp_m = "Autocam Soccer FP16"
            elif "Soccer Ball (INT8" in m_txt:
                disp_m = "Autocam Soccer INT8"
            elif "Basketball Ball (FP16" in m_txt:
                disp_m = "Autocam B-ball FP16"
            elif "Basketball Ball (INT8" in m_txt:
                disp_m = "Autocam B-ball INT8"
            elif "Keypoints" in m_txt:
                disp_m = "Autocam Keypoints"
            else:
                disp_m = m_txt.split("(")[0].strip()[:20]
            self.summary_fields["MODEL"].setText(disp_m)

        res_w, res_h = self.combo_wizard_res.currentData() or (1920, 1080)
        bitrate = int(self.ctrl_wizard_bitrate.value())
        use_lookahead = self.chk_wizard_lookahead.isChecked() if hasattr(self, 'chk_wizard_lookahead') else False
        mode_txt = "2-Pass (Lookahead)" if use_lookahead else "1-Pass (Direkt)"
        self.summary_fields["OUTPUT"].setText(f"{res_w}x{res_h} - {bitrate}M ({mode_txt})")

        # Cloud upload status in summary
        if hasattr(self, 'chk_wizard_cloud_upload') and "CLOUD" in self.summary_fields:
            if self.chk_wizard_cloud_upload.isChecked():
                team_txt = self.combo_wizard_cloud_team.currentText()
                if team_txt and not team_txt.startswith("—"):
                    self.summary_fields["CLOUD"].setText(f"🟢 {team_txt[:16]}")
                else:
                    self.summary_fields["CLOUD"].setText("🟢 Auto-Upload")
            else:
                self.summary_fields["CLOUD"].setText("⚪ Deaktiviert")

        # Estimate size: bitrate * duration
        fps = getattr(self, 'video_fps', 30.0)
        diff_f = max(0, getattr(self, 'trim_end_frame', 0) - getattr(self, 'trim_start_frame', 0))
        if diff_f > 0:
            total_sec = diff_f / max(1.0, fps)
        elif self.win.engine.video_left:
            total_sec = self.win.engine.video_left.total_frames / max(1.0, self.win.engine.video_left.fps)
        elif self.win.engine.video_panorama:
            total_sec = self.win.engine.video_panorama.total_frames / max(1.0, self.win.engine.video_panorama.fps)
        else:
            total_sec = 60.0
        est_mb = (bitrate * total_sec) / 8.0
        if est_mb > 1024:
            self.summary_fields["EST. SIZE"].setText(f"~{est_mb / 1024.0:.1f} GB")
        else:
            self.summary_fields["EST. SIZE"].setText(f"~{est_mb:.0f} MB")

    def _start_export(self):
        path = self.txt_out_path.text()
        if not path:
            self._browse_output_path()
            path = self.txt_out_path.text()
            if not path:
                return

        # Trigger MainWindow export worker with H.264 yuv420p
        res_data = self.combo_wizard_res.currentData() or (1920, 1080)
        w, h = res_data
        bitrate = int(self.ctrl_wizard_bitrate.value())
        codec = self.combo_wizard_codec.currentData() or "h264_nvenc"
        use_lookahead = self.chk_wizard_lookahead.isChecked() if hasattr(self, 'chk_wizard_lookahead') else False

        # Apply settings to main window controls and start render
        if hasattr(self.win, 'combo_codec'):
            idx = self.win.combo_codec.findData(codec)
            if idx >= 0: self.win.combo_codec.setCurrentIndex(idx)
        if hasattr(self.win, 'ctrl_bitrate'):
            self.win.ctrl_bitrate.setValue(bitrate)

        # Synchronize Cloud Auto-Upload settings
        if hasattr(self, 'chk_wizard_cloud_upload'):
            auto_cloud = self.chk_wizard_cloud_upload.isChecked()
            if hasattr(self.win, 'chk_cloud_auto_upload'):
                self.win.chk_cloud_auto_upload.setChecked(auto_cloud)
            if hasattr(self.win, 'txt_cloud_video_16x9'):
                self.win.txt_cloud_video_16x9.setText(path)
            if hasattr(self, 'combo_wizard_cloud_team') and hasattr(self.win, 'combo_cloud_team'):
                t_data = self.combo_wizard_cloud_team.currentData()
                if t_data:
                    t_idx = self.win.combo_cloud_team.findData(t_data)
                    if t_idx >= 0:
                        self.win.combo_cloud_team.setCurrentIndex(t_idx)

        start_f = getattr(self, 'trim_start_frame', 0)
        end_f = getattr(self, 'trim_end_frame', None)
        if end_f == 0 and self.total_video_frames > 0:
            end_f = self.total_video_frames

        self.on_export_started()

        # Trigger rendering in MainWindow
        self.win.start_export(
            custom_out_path=path, 
            custom_w=w, 
            custom_h=h, 
            mode="16:9_autocam",
            custom_start_frame=start_f,
            custom_end_frame=end_f,
            use_lookahead=use_lookahead
        )

    def on_export_started(self):
        self._export_start_time = time.time()
        self.btn_start_render.setEnabled(False)
        self.btn_start_render.setText("⏳ Export läuft...")
        self.btn_cancel_render.setEnabled(True)
        self.progress_bar.setValue(0)
        if hasattr(self, 'lbl_progress_pct'):
            self.lbl_progress_pct.setText("0.0%")
        if hasattr(self, 'lbl_progress_badge'):
            self.lbl_progress_badge.setText("● Processing")
            self.lbl_progress_badge.setStyleSheet("color: #10b981; font-weight: 700; font-size: 11px; padding: 3px 8px; background: rgba(16, 185, 129, 0.15); border: 1px solid #10b981; border-radius: 8px;")
        if hasattr(self, 'lbl_progress_file'):
            fn = os.path.basename(self.txt_out_path.text()) if self.txt_out_path.text() else "MatchTrack_Autocam_Broadcast.mp4"
            self.lbl_progress_file.setText(f"🎬 {fn}")
        if hasattr(self, 'lbl_progress_stage'):
            self.lbl_progress_stage.setText("● Initialisierung")
        self.lbl_progress_status.setText("Export wird vorbereitet...")
        self.progress_container.setVisible(True)

    def on_export_progress(self, processed: int, total: int, fps: float, eta: float, stage_text: str = ""):
        pct = int((processed / max(total, 1)) * 100)
        self.progress_bar.setValue(pct)
        if hasattr(self, 'lbl_progress_pct'):
            self.lbl_progress_pct.setText(f"{pct}%")
        
        m, s = divmod(int(eta), 60)
        dur_sec = max(0, time.time() - getattr(self, '_export_start_time', time.time()))
        dm, ds = divmod(int(dur_sec), 60)
        if hasattr(self, 'lbl_progress_time'):
            self.lbl_progress_time.setText(f"ETA {m:02d}:{s:02d}  •  Dauer {dm:02d}:{ds:02d}")

        if stage_text and hasattr(self, 'lbl_progress_stage'):
            self.lbl_progress_stage.setText(f"● {stage_text}")

        # Update Telemetry Metrics Cards
        nominal_fps = getattr(self, 'video_fps', 30.0) or 30.0
        speed_x = fps / max(1.0, nominal_fps)
        if hasattr(self, 'lbl_metric_speed'):
            self.lbl_metric_speed.setText(f"{speed_x:.2f}x Echtzeit")
        if hasattr(self, 'lbl_metric_fps'):
            self.lbl_metric_fps.setText(f"{fps:.1f}")
        if hasattr(self, 'lbl_metric_frames'):
            self.lbl_metric_frames.setText(f"{processed:,} / {total:,}")

        if hasattr(self, 'lbl_metric_size'):
            bitrate = int(self.ctrl_wizard_bitrate.value()) if hasattr(self, 'ctrl_wizard_bitrate') else 16
            processed_sec = processed / max(1.0, nominal_fps)
            est_mb = (bitrate * processed_sec) / 8.0
            if est_mb > 1024:
                self.lbl_metric_size.setText(f"{est_mb / 1024.0:.1f} GB")
            else:
                self.lbl_metric_size.setText(f"{est_mb:.1f} MB")

        prefix = f"[{stage_text}] " if stage_text else ""
        self.lbl_progress_status.setText(f"{prefix}{processed:,}/{total:,} Frames ({pct}%) | {fps:.1f} FPS | Restzeit: {m:02d}:{s:02d}")

    def on_export_preview_frame(self, qimg: QImage):
        """Displays the stepping follow-cam preview frame in the monitor."""
        if not hasattr(self, 'lbl_export_preview') or qimg is None:
            return
        pix = QPixmap.fromImage(qimg)
        scaled = pix.scaled(
            self.lbl_export_preview.size(), 
            Qt.KeepAspectRatio, 
            Qt.SmoothTransformation
        )
        self.lbl_export_preview.setPixmap(scaled)

    def on_export_finished(self, success: bool, message: str):
        self.btn_start_render.setEnabled(True)
        self.btn_start_render.setText("🚀 Jetzt Video erstellen (Start processing)")
        self.btn_cancel_render.setEnabled(False)
        if success:
            self.progress_bar.setValue(100)
            if hasattr(self, 'lbl_progress_pct'):
                self.lbl_progress_pct.setText("100%")
            if hasattr(self, 'lbl_progress_badge'):
                self.lbl_progress_badge.setText("● Fertig")
                self.lbl_progress_badge.setStyleSheet("color: #10b981; font-weight: 700; font-size: 11px; padding: 3px 8px; background: rgba(16, 185, 129, 0.15); border: 1px solid #10b981; border-radius: 8px;")
            if hasattr(self, 'chk_wizard_cloud_upload') and self.chk_wizard_cloud_upload.isChecked():
                self.lbl_progress_status.setText("☁️ Video fertig! Starte automatischen Upload zu MatchTrack Online...")
                self.btn_cancel_render.setEnabled(True)
                self.btn_cancel_render.setText("✖ Upload abbrechen")
            else:
                self.lbl_progress_status.setText("🎉 Video-Export erfolgreich abgeschlossen!")
        else:
            if hasattr(self, 'lbl_progress_badge'):
                self.lbl_progress_badge.setText("● Fehlgeschlagen")
                self.lbl_progress_badge.setStyleSheet("color: #ef4444; font-weight: 700; font-size: 11px; padding: 3px 8px; background: rgba(239, 68, 68, 0.15); border: 1px solid #ef4444; border-radius: 8px;")
            self.lbl_progress_status.setText("Export abgebrochen oder fehlgeschlagen.")

    def on_export_cancelled(self):
        self.btn_start_render.setEnabled(True)
        self.btn_start_render.setText("🚀 Jetzt Video erstellen (Start processing)")
        self.btn_cancel_render.setEnabled(False)
        if hasattr(self, 'lbl_progress_badge'):
            self.lbl_progress_badge.setText("● Abgebrochen")
            self.lbl_progress_badge.setStyleSheet("color: #f59e0b; font-weight: 700; font-size: 11px; padding: 3px 8px; background: rgba(245, 158, 11, 0.15); border: 1px solid #f59e0b; border-radius: 8px;")
        self.lbl_progress_status.setText("Export abgebrochen.")

    def _on_cancel_clicked(self):
        if hasattr(self.win, 'cloud_upload_worker') and self.win.cloud_upload_worker and self.win.cloud_upload_worker.isRunning():
            self.win.cancel_cloud_upload()
            self.lbl_progress_status.setText("Cloud-Upload wird abgebrochen...")
        else:
            self.win.cancel_video_export()

    def _on_cloud_login_clicked(self):
        self.win.open_cloud_login_dialog()
        self.update_cloud_ui()

    def _on_cloud_upload_toggled(self, state: int):
        checked = bool(state)
        if checked and not self.win.cloud_client.is_authenticated:
            self.win.open_cloud_login_dialog()
            if not self.win.cloud_client.is_authenticated:
                self.chk_wizard_cloud_upload.blockSignals(True)
                self.chk_wizard_cloud_upload.setChecked(False)
                self.chk_wizard_cloud_upload.blockSignals(False)
                return
        if hasattr(self.win, 'chk_cloud_auto_upload'):
            self.win.chk_cloud_auto_upload.setChecked(self.chk_wizard_cloud_upload.isChecked())
        self.update_summary()

    def _on_wizard_cloud_team_changed(self, idx: int):
        data = self.combo_wizard_cloud_team.currentData()
        if hasattr(self.win, 'combo_cloud_team') and data is not None:
            t_idx = self.win.combo_cloud_team.findData(data)
            if t_idx >= 0 and self.win.combo_cloud_team.currentIndex() != t_idx:
                self.win.combo_cloud_team.blockSignals(True)
                self.win.combo_cloud_team.setCurrentIndex(t_idx)
                self.win.combo_cloud_team.blockSignals(False)
        self.update_summary()

    def update_cloud_ui(self, profile: Optional[Dict[str, Any]] = None):
        if not hasattr(self, 'lbl_wizard_cloud_status'):
            return
        if self.win.cloud_client.is_authenticated:
            info = profile or self.win.cloud_client.user_info or {}
            username = info.get("username") or self.win.cloud_client.username or "Angemeldet"
            role = info.get("role", "TRAINER")
            host = self.win.cloud_client.server_url.replace("https://", "").replace("http://", "")
            self.lbl_wizard_cloud_status.setText(f"🟢 {username} ({role}) @ {host}")
            self.lbl_wizard_cloud_status.setStyleSheet("color: #34d399; font-size: 11px; font-weight: 700;")
            self.btn_wizard_cloud_login.setText("🔑 Konto")
        else:
            self.lbl_wizard_cloud_status.setText("⚪ Nicht angemeldet")
            self.lbl_wizard_cloud_status.setStyleSheet("color: #94a3b8; font-size: 11px; font-weight: 600;")
            self.btn_wizard_cloud_login.setText("🔑 Verbinden")
        self.sync_cloud_teams()
        self.update_summary()

    def sync_cloud_teams(self):
        if not hasattr(self, 'combo_wizard_cloud_team') or not hasattr(self.win, 'combo_cloud_team'):
            return
        self.combo_wizard_cloud_team.blockSignals(True)
        self.combo_wizard_cloud_team.clear()
        for i in range(self.win.combo_cloud_team.count()):
            txt = self.win.combo_cloud_team.itemText(i)
            dat = self.win.combo_cloud_team.itemData(i)
            self.combo_wizard_cloud_team.addItem(txt, dat)
        self.combo_wizard_cloud_team.setCurrentIndex(self.win.combo_cloud_team.currentIndex())
        self.combo_wizard_cloud_team.blockSignals(False)

    def on_cloud_upload_progress(self, sent_bytes: int, total_bytes: int, speed_mb_s: float, eta_sec: float, status_text: str):
        pct = int((sent_bytes / max(total_bytes, 1)) * 100)
        self.progress_bar.setValue(pct)
        self.progress_container.setVisible(True)
        if speed_mb_s > 0.01:
            m, s = divmod(int(eta_sec), 60)
            self.lbl_progress_status.setText(f"☁️ Cloud-Upload: {pct}% ({speed_mb_s:.1f} MB/s) | Restzeit: {m:02d}:{s:02d}")
        else:
            self.lbl_progress_status.setText(f"☁️ Cloud-Upload: {status_text}")
        self.btn_cancel_render.setEnabled(True)
        self.btn_cancel_render.setText("✖ Upload abbrechen")

    def on_cloud_upload_finished(self, success: bool, message: str, result: dict):
        self.btn_cancel_render.setEnabled(False)
        self.btn_cancel_render.setText("✖ Export abbrechen")
        self.btn_start_render.setEnabled(True)
        if success:
            self.progress_bar.setValue(100)
            self.lbl_progress_status.setText("✅ Video erfolgreich in MatchTrack Online Cloud hochgeladen!")
        else:
            self.lbl_progress_status.setText(f"❌ Cloud-Upload fehlgeschlagen: {message}")

    # =========================================================================
    # STEPPER NAVIGATION
    # =========================================================================
    def go_to_step(self, step_idx: int):
        if 0 <= step_idx < 5:
            self.current_step = step_idx
            self.pages_stack.setCurrentIndex(step_idx)
            self.update_stepper_ui()
            self.step_changed.emit(step_idx)

            # Switch Viewport domain dynamically
            if hasattr(self.win, 'viewport'):
                if step_idx == 2: # Field marking
                    self.win.viewport.set_workflow_domain("field_marking")
                elif step_idx == 3: # Framing
                    self.win.viewport.set_workflow_domain("framing")
                    self.update_framing_previews()
                elif step_idx == 1: # Source
                    self.win.viewport.set_workflow_domain("stitching")
                elif step_idx == 4: # Output
                    self.win.viewport.set_workflow_domain("autocam")
                    self.update_summary()
                    if hasattr(self, 'lbl_export_preview') and hasattr(self, 'preview_monitor_high'):
                        pm = self.preview_monitor_high.pixmap()
                        if pm and not pm.isNull():
                            self.lbl_export_preview.setPixmap(pm.scaled(
                                self.lbl_export_preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
                            ))

    def next_step(self):
        if self.current_step < 4:
            self.go_to_step(self.current_step + 1)
        elif self.current_step == 4:
            self._start_export()

    def prev_step(self):
        if self.current_step > 0:
            self.go_to_step(self.current_step - 1)

    def update_stepper_ui(self):
        for idx, btn in enumerate(self.step_buttons):
            btn.setChecked(idx == self.current_step)

        # Update Bottom Button texts
        next_names = ["Source ❯", "Field marking ❯", "Framing ❯", "Output ❯", "Verarbeitung starten 🚀"]
        self.btn_nav_next.setText(f"Weiter: {next_names[self.current_step]}")
        self.btn_nav_back.setEnabled(self.current_step > 0)
