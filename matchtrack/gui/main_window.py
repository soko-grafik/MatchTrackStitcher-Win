"""
MatchTrack-Stitcher Modern Desktop UI (PySide6 / Qt6).
Ultra-Responsive Real-Time Slider Engine (Zero Lag / Debounced Draft Mode),
Dedicated Soccer Ball Tracking (Ball Reticle & Trajectory Follow), and 32:9/16:9 Broadcast Pipelines.
"""
import os
import sys
import math
from typing import Optional, Tuple, List, Dict, Any
import cv2
import numpy as np

from PySide6.QtCore import Qt, QTimer, QThread, Signal, Slot, QPoint, QPointF, QRectF, QObject, QEvent, QDate, QUrl
from PySide6.QtGui import QImage, QPixmap, QIcon, QFont, QColor, QPainter, QPen, QWheelEvent, QMouseEvent, QPolygonF, QDesktopServices, QShortcut, QKeySequence, QPainterPath
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QSlider, QSpinBox, QDoubleSpinBox, QComboBox, QAbstractSpinBox,
    QFileDialog, QProgressBar, QGroupBox, QTabWidget, QSplitter,
    QCheckBox, QMessageBox, QFrame, QScrollArea, QStatusBar, QToolButton,
    QPlainTextEdit, QMenu, QWidgetAction, QLineEdit, QDateEdit, QDialog,
    QRadioButton, QStackedWidget, QButtonGroup, QFormLayout
)


import json
import threading
from datetime import datetime
from ..camera_model import CAMERA_PRESETS, CameraIntrinsics
from ..rig_geometry import RigConfiguration, CameraPose
from ..lut_generator import generate_remap_luts, RemapLUTs, apply_stitch
from ..stitcher_engine import StitcherEngine
from ..audio_sync import calculate_audio_sync_offset
from ..auto_stitch import (
    extract_overlap_matches,
    optimize_rig_calibration,
    calibrate_from_frames,
    CalibrationResult
)
from ..autocam import AutoCamConfig
from ..tactical_warp import generate_tactical_16x9_luts, generate_tactical_16x9_canvas_luts, apply_tactical_16x9_warp
from ..radar_minimap import SoccerRadarRenderer
from ..ai_detector import SKELETON_CONNECTIONS
from ..logger import get_logger, get_gui_handler, GuiLogRecord, setup_logging, install_crash_handlers
from ..paths import get_log_file_path, get_default_settings_path, get_resource_path
from ..cloud_uploader import MatchTrackOnlineClient
from ..rtmp_streamer import MultiStreamManager, RTMPChannelConfig, StreamStats
from ..live_capture import DualLiveCaptureManager, enumerate_video_devices
from ..dji_ble import DjiBLEManager, DjiCameraDevice
from .tactics_dialog import TacticsDashboardDialog
from ..wide_angle_rectifier import WideAngleConfig, WideAngleRectifier

logger = get_logger("gui")



# -----------------------------------------------------------------------------
# Modular GUI Sub-Components
# -----------------------------------------------------------------------------
from .widgets import (
    NoWheelSlider, NoWheelDoubleSpinBox, NoWheelSpinBox, NoWheelComboBox,
    ScrollAreaWheelFilter, LabeledSliderSpinBox, LogEmitter, LogViewerWidget
)
from .workers import RenderWorker, CloudUploadWorker
from .dialogs import DjiQRCodeDialog, CloudLoginDialog, OBSAdvancedOutputDialog
from .obs_output_settings import OBSOutputSettingsWidget
from .viewport import InteractivePanoramaViewport
from .live_controller import LiveStreamMixin
from .cloud_controller import CloudIntegrationMixin
from .wide_angle_controller import WideAngleControllerMixin
from .tabs import (
    build_landing_page,
    build_stitching_tab,
    build_wide_angle_tab,
    build_autocam_tab,
    build_livestream_tab,
    build_cloud_tab,
    create_export_upload_subpage,
    create_codec_combobox,
    create_audio_combobox
)


class MainWindow(
    QMainWindow,
    LiveStreamMixin,
    CloudIntegrationMixin,
    WideAngleControllerMixin
):
    stream_telemetry_signal = Signal(dict)
    ble_device_found_signal = Signal(object)
    ble_conn_changed_signal = Signal(str, bool)
    ble_stream_started_signal = Signal(str, str)
    ble_error_signal = Signal(str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("MatchTrack-Stitcher | 32:9 Panorama & 16:9 Ball-Follow Broadcast (DJI Action 4)")
        self.resize(1550, 940)
        self.setMinimumSize(1100, 720)

        # Initialize rig, engine and autocam config
        self.rig = RigConfiguration()
        self.engine = StitcherEngine(self.rig)
        self.render_worker: Optional[RenderWorker] = None
        self.cloud_client = MatchTrackOnlineClient()
        self.cloud_upload_worker: Optional[CloudUploadWorker] = None
        self.last_uploaded_match_url: str = ""
        self.saved_cloud_server_url: str = ""
        self.saved_cloud_team_id: str = ""

        # Live-Streaming and Hardware Capture Managers
        self._view_mode: str = "32:9"
        self.stream_manager = MultiStreamManager()
        self.live_capture_manager = DualLiveCaptureManager()
        self.dji_ble_manager = DjiBLEManager()

        # Wide-Angle Rectifier & Pitch Trapezoid Engine (DJI Action 4)
        self.wide_angle_config = WideAngleConfig()
        self.wide_angle_rectifier = WideAngleRectifier(self.wide_angle_config)

        # Connect thread-safe background signals to GUI slots
        self.stream_telemetry_signal.connect(self._handle_stream_telemetry)
        self.ble_device_found_signal.connect(self._on_dji_ble_device_found)
        self.ble_conn_changed_signal.connect(self._on_dji_ble_conn_changed)
        self.ble_stream_started_signal.connect(self._on_dji_ble_stream_started)
        self.ble_error_signal.connect(self._on_dji_ble_error)

        self.stream_manager.telemetry_callback = lambda stats: self.stream_telemetry_signal.emit(stats)
        self.dji_ble_manager.on_device_found = lambda dev: self.ble_device_found_signal.emit(dev)
        self.dji_ble_manager.on_connection_change = lambda addr, c: self.ble_conn_changed_signal.emit(addr, c)
        self.dji_ble_manager.on_stream_started = lambda addr, u: self.ble_stream_started_signal.emit(addr, u)
        self.dji_ble_manager.on_error = lambda err: self.ble_error_signal.emit(err)

        # Live streaming capture tick timer (~30 FPS)
        self.live_capture_timer = QTimer(self)
        self.live_capture_timer.setInterval(33)
        self.live_capture_timer.timeout.connect(self._on_live_capture_tick)

        # Playback timer
        self.playback_timer = QTimer(self)
        self.playback_timer.setInterval(33) # ~30 fps
        self.playback_timer.timeout.connect(self.on_play_step)
        
        # Debounce timer for high-res preview updates during fast slider dragging
        self.hq_preview_timer = QTimer(self)
        self.hq_preview_timer.setSingleShot(True)
        self.hq_preview_timer.setInterval(45) # 45ms debounce
        self.hq_preview_timer.timeout.connect(self.render_high_res_preview)
        
        self.is_playing = False
        self.current_frame_idx = 0
        self.sync_angles = True
        self.in_point_frame = 0
        self.out_point_frame = 0
        self.crop_mode: bool = False
        self.transform_stretch_x: float = 1.0
        self.transform_stretch_y: float = 1.0

        # Set Window Icon
        icon_path = os.path.join(os.path.dirname(__file__), "..", "assets", "icon.ico")
        if not os.path.exists(icon_path):
            icon_path = os.path.join(os.path.dirname(__file__), "..", "assets", "icon.png")
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))

        self.init_ui()

        # Connect Viewport Wide-Angle Signals
        self.viewport.wide_angle_corners_changed.connect(self._on_viewport_wa_corners_changed)
        self.viewport.wide_angle_step_changed.connect(self._on_viewport_wa_step_changed)
        if hasattr(self.viewport, 'crop_rect_changed'):
            self.viewport.crop_rect_changed.connect(self.on_crop_rect_from_viewport)

        self.apply_dark_theme()
        self.load_default_startup_settings()
        self.show_landing_page()
        QTimer.singleShot(400, self.restore_cloud_session_on_startup)

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        root_layout = QVBoxLayout(central_widget)
        root_layout.setContentsMargins(8, 8, 8, 8)
        root_layout.setSpacing(6)

        # 1. TOP APP HEADER BAR (Samsung One UI Style)
        top_bar = QFrame()
        top_bar.setObjectName("appHeader")
        top_bar.setStyleSheet("""
            #appHeader {
                background-color: #12121a;
                border: 1px solid #20202e;
                border-radius: 14px;
                padding: 6px 12px;
            }
        """)
        top_layout = QHBoxLayout(top_bar)
        top_layout.setContentsMargins(10, 6, 10, 6)
        top_layout.setSpacing(10)

        # Brand / Logo
        logo_icon_path = os.path.join(os.path.dirname(__file__), "..", "assets", "icon_small.png")
        if os.path.exists(logo_icon_path):
            lbl_logo = QLabel()
            pix = QPixmap(logo_icon_path).scaled(28, 28, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            lbl_logo.setPixmap(pix)
            top_layout.addWidget(lbl_logo)

        lbl_brand = QLabel("MatchTrack Stitcher")
        lbl_brand.setStyleSheet("font-weight: 800; font-size: 15px; color: #f4f4f6; letter-spacing: 0.5px;")
        top_layout.addWidget(lbl_brand)

        # Video Status Indicator Pill
        self.lbl_video_status = QLabel("⚪ Keine Videos geladen")
        self.lbl_video_status.setStyleSheet("background: #181824; color: #94a3b8; border: 1px solid #28283a; border-radius: 12px; padding: 4px 12px; font-size: 11px; font-weight: 600;")
        top_layout.addWidget(self.lbl_video_status)

        top_layout.addStretch()

        # Section Navigation Hub Pills (Centered in top bar)
        nav_frame = QFrame()
        nav_frame.setStyleSheet("QFrame { background: #0c0c14; border: 1px solid #1e1e2d; border-radius: 10px; padding: 2px 4px; }")
        nav_layout = QHBoxLayout(nav_frame)
        nav_layout.setContentsMargins(2, 2, 2, 2)
        nav_layout.setSpacing(4)

        self.btn_nav_wizard = QPushButton("⚡ Autocam Assistent")
        self.btn_nav_home = QPushButton("🏠 Zentrale")
        self.btn_nav_stitch = QPushButton("🧵 Stitching")
        self.btn_nav_wide = QPushButton("📐 Weitwinkel")
        self.btn_nav_export = QPushButton("🚀 Export")
        self.btn_nav_live = QPushButton("🔴 Live-Stream")
        self.btn_nav_cloud = QPushButton("☁️ Cloud-Upload")

        self.btn_nav_wizard.clicked.connect(lambda: self.show_workspace(0))
        self.btn_nav_home.clicked.connect(self.show_landing_page)
        self.btn_nav_stitch.clicked.connect(lambda: self.show_workspace(1))
        self.btn_nav_wide.clicked.connect(lambda: self.show_workspace(2))
        self.btn_nav_export.clicked.connect(lambda: self.show_workspace(3))
        self.btn_nav_live.clicked.connect(lambda: self.show_workspace(4))
        self.btn_nav_cloud.clicked.connect(lambda: self.show_workspace(5))

        for btn in [self.btn_nav_wizard, self.btn_nav_home, self.btn_nav_stitch, self.btn_nav_wide, self.btn_nav_export, self.btn_nav_live, self.btn_nav_cloud]:
            btn.setStyleSheet("background-color: transparent; border: none; border-radius: 6px; font-weight: 700; font-size: 11px; padding: 5px 12px; color: #94a3b8;")
            nav_layout.addWidget(btn)

        top_layout.addWidget(nav_frame)
        top_layout.addStretch()

        # Hide native menubar and use clean global shortcuts
        self.menuBar().clear()
        self.menuBar().setVisible(False)
        self.menuBar().hide()
        QShortcut(QKeySequence("Ctrl+H"), self, self.show_landing_page)
        QShortcut(QKeySequence("Ctrl+1"), self, lambda: self.show_workspace(0))
        QShortcut(QKeySequence("Ctrl+2"), self, lambda: self.show_workspace(1))
        QShortcut(QKeySequence("Ctrl+3"), self, lambda: self.show_workspace(2))
        QShortcut(QKeySequence("Ctrl+4"), self, lambda: self.show_workspace(3))
        QShortcut(QKeySequence("Ctrl+5"), self, lambda: self.show_workspace(4))
        QShortcut(QKeySequence("Ctrl+6"), self, lambda: self.show_workspace(5))

        # Profile / Settings Management Buttons
        self.btn_load_prof_tb = QPushButton("📁 Profil")
        self.btn_load_prof_tb.setStyleSheet("background-color: #1c1c28; border: 1px solid #2a2a3e; font-size: 11px; padding: 6px 12px; border-radius: 8px; color: #cbd5e1; font-weight: 600;")
        self.btn_load_prof_tb.setToolTip("Gesamtes Profil aus Datei laden")
        self.btn_load_prof_tb.clicked.connect(self.load_full_profile_from)

        self.btn_save_prof_tb = QPushButton("💾 Speichern")
        self.btn_save_prof_tb.setStyleSheet("background-color: #1c1c28; border: 1px solid #2a2a3e; font-size: 11px; padding: 6px 12px; border-radius: 8px; color: #cbd5e1; font-weight: 600;")
        self.btn_save_prof_tb.setToolTip("Gesamtes Profil in Datei speichern")
        self.btn_save_prof_tb.clicked.connect(self.save_full_profile_as)

        self.btn_save_default_tb = QPushButton("⭐ Standard")
        self.btn_save_default_tb.setStyleSheet("background-color: #2563eb; border: none; font-weight: bold; font-size: 11px; padding: 6px 12px; border-radius: 8px; color: white;")
        self.btn_save_default_tb.setToolTip("Aktuelle Einstellungen als Start-Standard speichern")
        self.btn_save_default_tb.clicked.connect(self.save_as_default_settings)

        top_layout.addWidget(self.btn_load_prof_tb)
        top_layout.addWidget(self.btn_save_prof_tb)
        top_layout.addWidget(self.btn_save_default_tb)

        root_layout.addWidget(top_bar)

        # Main Resizable Splitter (Left: Large Viewport & Timeline | Right: Controls)
        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)

        # =========================================================================
        # LEFT PANE: Large Panorama Viewport + Timeline
        # =========================================================================
        left_pane = QWidget()
        left_layout = QVBoxLayout(left_pane)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(6)

        # Viewport Header Toolbar (Clean Samsung One UI Style)
        vp_tool_bar = QHBoxLayout()
        vp_tool_bar.setContentsMargins(4, 2, 4, 4)
        vp_tool_bar.setSpacing(8)

        lbl_vp_title = QLabel("LIVE VORSCHAU")
        lbl_vp_title.setStyleSheet("font-weight: 800; color: #f4f4f6; font-size: 11px; letter-spacing: 0.8px;")
        vp_tool_bar.addWidget(lbl_vp_title)

        # View Mode Dropdown
        self.combo_view_mode = QComboBox()
        self.combo_view_mode.addItem("🔲 4-Kanal Multiview Grid (1, 2, 3, 4)", "grid")
        self.combo_view_mode.addItem("🌟 32:9 Taktik-Panorama (16:9 Profil)", "32:9")
        self.combo_view_mode.addItem("🌟 8:3 Taktik-Panorama (4:3 Profil)", "8:3")
        self.combo_view_mode.addItem("📐 21:10 Gestauchtes Panorama (Gesamt)", "21:10")
        self.combo_view_mode.addItem("📐 16:9 Taktik-Warp (Spielfeld entzerrt)", "16:9_tactical")
        self.combo_view_mode.addItem("📐 Weitwinkel Begradigt (DJI Action 4 Trapez)", "wide_angle")
        self.combo_view_mode.addItem("📷 Weitwinkel Original (4-Punkt Ecken)", "wide_angle_raw")
        self.combo_view_mode.addItem("🎥 16:9 Auto-Broadcast (Follow-Cam)", "16:9")
        self.combo_view_mode.addItem("📹 Cam 1 Links (Live-Feed)", "cam1")
        self.combo_view_mode.addItem("📹 Cam 2 Rechts (Live-Feed)", "cam2")
        self.combo_view_mode.setStyleSheet("font-weight: 600; background-color: #181824; padding: 5px 10px; border: 1px solid #2a2a3e; border-radius: 8px;")
        self.combo_view_mode.currentIndexChanged.connect(self.on_view_mode_changed)
        vp_tool_bar.addWidget(self.combo_view_mode)

        # Clean "Einblendungen" Dropdown ToolButton with QMenu
        self.btn_overlays_menu = QToolButton()
        self.btn_overlays_menu.setText("👁️ Einblendungen ▾")
        self.btn_overlays_menu.setPopupMode(QToolButton.InstantPopup)
        self.btn_overlays_menu.setStyleSheet("""
            QToolButton { background-color: #181824; border: 1px solid #2a2a3e; border-radius: 8px; padding: 5px 12px; font-weight: 600; color: #cbd5e1; }
            QToolButton:hover { background-color: #242436; border-color: #3b82f6; color: #ffffff; }
        """)
        menu_overlays = QMenu(self.btn_overlays_menu)
        menu_overlays.setStyleSheet("background-color: #151520; color: #f4f4f6; border: 1px solid #2a2a3e; border-radius: 8px; padding: 6px;")

        self.chk_show_frame = QCheckBox("🔲 32:9 / 16:9 Rahmen")
        self.chk_show_frame.setChecked(True)
        self.chk_show_frame.toggled.connect(self.toggle_frame_overlay)
        act_frame = QWidgetAction(self)
        act_frame.setDefaultWidget(self.chk_show_frame)
        menu_overlays.addAction(act_frame)

        self.chk_show_center_line = QCheckBox("🎯 Spielfeld-Mitte & Hilfslinien")
        self.chk_show_center_line.setChecked(True)
        self.chk_show_center_line.toggled.connect(self.toggle_center_line)
        act_center = QWidgetAction(self)
        act_center.setDefaultWidget(self.chk_show_center_line)
        menu_overlays.addAction(act_center)

        self.chk_show_autocam = QCheckBox("🎥 16:9 Kamera-Kasten")
        self.chk_show_autocam.setChecked(True)
        self.chk_show_autocam.toggled.connect(self.toggle_autocam_box)
        act_autocam = QWidgetAction(self)
        act_autocam.setDefaultWidget(self.chk_show_autocam)
        menu_overlays.addAction(act_autocam)

        self.chk_show_ball = QCheckBox("⚽ Ball-Fokus")
        self.chk_show_ball.setChecked(True)
        self.chk_show_ball.toggled.connect(self.toggle_ball_reticle)
        act_ball = QWidgetAction(self)
        act_ball.setDefaultWidget(self.chk_show_ball)
        menu_overlays.addAction(act_ball)

        self.chk_show_seam = QCheckBox("🪡 Nahtstelle (Mitte)")
        self.chk_show_seam.setChecked(True)
        self.chk_show_seam.toggled.connect(self.toggle_seam_overlay)
        act_seam = QWidgetAction(self)
        act_seam.setDefaultWidget(self.chk_show_seam)
        menu_overlays.addAction(act_seam)

        self.chk_show_grid = QCheckBox("📏 Horizontlinien")
        self.chk_show_grid.setChecked(True)
        self.chk_show_grid.toggled.connect(self.toggle_grid_overlay)
        act_grid = QWidgetAction(self)
        act_grid.setDefaultWidget(self.chk_show_grid)
        menu_overlays.addAction(act_grid)

        self.chk_show_radar_menu = QCheckBox("🗺️ 2D Taktik-Radar")
        self.chk_show_radar_menu.setChecked(True)
        self.chk_show_radar_menu.toggled.connect(self.toggle_radar_overlay)
        act_radar = QWidgetAction(self)
        act_radar.setDefaultWidget(self.chk_show_radar_menu)
        menu_overlays.addAction(act_radar)

        self.chk_show_offside_menu = QCheckBox("🚩 Virtuelle Abseitslinie (VAR)")
        self.chk_show_offside_menu.setChecked(True)
        self.chk_show_offside_menu.toggled.connect(self.toggle_offside_line)
        act_offside = QWidgetAction(self)
        act_offside.setDefaultWidget(self.chk_show_offside_menu)
        menu_overlays.addAction(act_offside)

        self.chk_show_skeletons_menu = QCheckBox("🦴 Spieler-Skelette & Pose")
        self.chk_show_skeletons_menu.setChecked(False)
        self.chk_show_skeletons_menu.toggled.connect(self.toggle_skeletons)
        act_skeletons = QWidgetAction(self)
        act_skeletons.setDefaultWidget(self.chk_show_skeletons_menu)
        menu_overlays.addAction(act_skeletons)

        self.chk_show_speed_menu = QCheckBox("⚡ Live-Geschwindigkeit (km/h)")
        self.chk_show_speed_menu.setChecked(False)
        self.chk_show_speed_menu.toggled.connect(self.toggle_speed_badges)
        act_speed = QWidgetAction(self)
        act_speed.setDefaultWidget(self.chk_show_speed_menu)
        menu_overlays.addAction(act_speed)

        self.chk_sideline_menu = QCheckBox("🚶 Trainer-/Auslinien-Filter")
        self.chk_sideline_menu.setChecked(True)
        self.chk_sideline_menu.toggled.connect(self.toggle_sideline_filter)
        act_sideline = QWidgetAction(self)
        act_sideline.setDefaultWidget(self.chk_sideline_menu)
        menu_overlays.addAction(act_sideline)

        self.btn_overlays_menu.setMenu(menu_overlays)
        vp_tool_bar.addWidget(self.btn_overlays_menu)

        vp_tool_bar.addStretch()

        # Placeholders for legacy references
        self.btn_pitch_roi_tb = None
        self.btn_toggle_tactical_view = None
        self.btn_corner_pins_tb = None
        self.btn_pick_center_tb = None

        # Clean Zoom Controls (One UI Pills)
        btn_zoom_fit = QPushButton("🔍 Fit")
        btn_zoom_fit.setToolTip("Passt die Vorschau optimal in das Fenster ein")
        btn_zoom_fit.setStyleSheet("background-color: #181824; border: 1px solid #2a2a3e; border-radius: 8px; padding: 5px 10px; font-weight: 600; font-size: 11px; color: #cbd5e1;")
        btn_zoom_fit.clicked.connect(lambda: self.viewport.reset_zoom())
        vp_tool_bar.addWidget(btn_zoom_fit)

        btn_zoom_seam = QPushButton("🔎 200% Zoom")
        btn_zoom_seam.setToolTip("Vergrößert auf 200% zur genauen Nahtstellen-Ausrichtung")
        btn_zoom_seam.setStyleSheet("background-color: #181824; border: 1px solid #2a2a3e; border-radius: 8px; padding: 5px 10px; font-weight: 600; font-size: 11px; color: #cbd5e1;")
        btn_zoom_seam.clicked.connect(self.zoom_to_seam)
        vp_tool_bar.addWidget(btn_zoom_seam)

        left_layout.addLayout(vp_tool_bar)

        # High-Resolution Interactive Viewport
        self.viewport = InteractivePanoramaViewport()
        self.viewport.set_rig(self.rig)
        self.viewport.framing_dragged.connect(self.shift_framing)
        self.viewport.center_picked.connect(self.on_pitch_center_picked)
        self.viewport.corner_dragged.connect(self.on_viewport_corner_dragged)
        self.viewport.corner_drag_finished.connect(self.on_viewport_corner_drag_finished)
        self.viewport.pitch_corners_changed.connect(self.on_viewport_pitch_corners_changed)
        self.viewport.grid_quadrant_clicked.connect(self._on_grid_quadrant_clicked)
        left_layout.addWidget(self.viewport, 1)

        # Timeline & Playback Controls Frame (One UI Style)
        self.timeline_frame = QFrame()
        self.timeline_frame.setObjectName("timelineFrame")
        self.timeline_frame.setStyleSheet("""
            #timelineFrame {
                background-color: #12121a;
                border: 1px solid #20202e;
                border-radius: 14px;
                padding: 8px 12px;
            }
        """)
        timeline_layout = QVBoxLayout(self.timeline_frame)
        timeline_layout.setContentsMargins(6, 6, 6, 6)
        timeline_layout.setSpacing(6)

        # Row 1: Scrubber slider with timecodes
        slider_row = QHBoxLayout()
        slider_row.setSpacing(10)
        self.lbl_current_time = QLabel("00:00:00.00 (F 0)")
        self.lbl_current_time.setStyleSheet("font-weight: 700; color: #60a5fa; font-family: 'Consolas', monospace; font-size: 13px; min-width: 140px;")
        
        self.slider_timeline = QSlider(Qt.Horizontal)
        self.slider_timeline.setRange(0, 100)
        self.slider_timeline.setStyleSheet("""
            QSlider::groove:horizontal { height: 6px; background: #222232; border-radius: 3px; }
            QSlider::sub-page:horizontal { background: #2563eb; border-radius: 3px; }
            QSlider::handle:horizontal { background: #ffffff; border: 2px solid #2563eb; width: 16px; margin: -5px 0; border-radius: 8px; }
            QSlider::handle:horizontal:hover { background: #93c5fd; }
        """)
        self.slider_timeline.valueChanged.connect(self.on_slider_seek)

        self.lbl_total_time = QLabel("00:00:00")
        self.lbl_total_time.setStyleSheet("color: #71717a; font-family: 'Consolas', monospace; font-size: 12px; min-width: 65px;")

        slider_row.addWidget(self.lbl_current_time)
        slider_row.addWidget(self.slider_timeline, 1)
        slider_row.addWidget(self.lbl_total_time)
        timeline_layout.addLayout(slider_row)

        # Row 2: Transport Controls (Left/Center) + Trim Controls (Right)
        ctrl_row = QHBoxLayout()
        ctrl_row.setSpacing(6)

        # Transport
        self.btn_step_back = QPushButton("⏮ -1F")
        self.btn_step_back.setStyleSheet("background-color: #181824; border: 1px solid #28283a; padding: 5px 10px; font-weight: 600; border-radius: 8px; color: #cbd5e1;")
        self.btn_step_back.clicked.connect(lambda: self.step_frame(-1))

        self.btn_play_pause = QPushButton("▶ Abspielen")
        self.btn_play_pause.setStyleSheet("font-weight: 700; min-width: 110px; background-color: #2563eb; border: none; border-radius: 8px; color: white; padding: 6px 14px;")
        self.btn_play_pause.clicked.connect(self.toggle_playback)

        self.btn_step_fwd = QPushButton("+1F ⏭")
        self.btn_step_fwd.setStyleSheet("background-color: #181824; border: 1px solid #28283a; padding: 5px 10px; font-weight: 600; border-radius: 8px; color: #cbd5e1;")
        self.btn_step_fwd.clicked.connect(lambda: self.step_frame(1))

        ctrl_row.addWidget(self.btn_step_back)
        ctrl_row.addWidget(self.btn_play_pause)
        ctrl_row.addWidget(self.btn_step_fwd)

        ctrl_row.addSpacing(14)

        # Trim Controls
        self.btn_set_in = QPushButton("🚩 In (I)")
        self.btn_set_in.setStyleSheet("background-color: #181824; border: 1px solid #28283a; font-weight: 600; padding: 5px 10px; color: #34d399; border-radius: 8px;")
        self.btn_set_in.setToolTip("Startpunkt für Video-Export auf aktuellen Frame setzen (Taste 'I')")
        self.btn_set_in.clicked.connect(lambda: self.set_in_point())

        self.btn_jump_in = QPushButton("⏮ In")
        self.btn_jump_in.setStyleSheet("background-color: #181824; border: 1px solid #28283a; padding: 5px 8px; color: #a1a1aa; border-radius: 8px;")
        self.btn_jump_in.setToolTip("Springe zum Startpunkt (Taste 'Pos1' / 'Home')")
        self.btn_jump_in.clicked.connect(self.jump_to_in_point)

        self.btn_set_out = QPushButton("🏁 Out (O)")
        self.btn_set_out.setStyleSheet("background-color: #181824; border: 1px solid #28283a; font-weight: 600; padding: 5px 10px; color: #f87171; border-radius: 8px;")
        self.btn_set_out.setToolTip("Endpunkt für Video-Export auf aktuellen Frame setzen (Taste 'O')")
        self.btn_set_out.clicked.connect(lambda: self.set_out_point())

        self.btn_jump_out = QPushButton("Out ⏭")
        self.btn_jump_out.setStyleSheet("background-color: #181824; border: 1px solid #28283a; padding: 5px 8px; color: #a1a1aa; border-radius: 8px;")
        self.btn_jump_out.setToolTip("Springe zum Endpunkt (Taste 'Ende')")
        self.btn_jump_out.clicked.connect(self.jump_to_out_point)

        self.btn_reset_trim = QToolButton()
        self.btn_reset_trim.setText("↺")
        self.btn_reset_trim.setToolTip("Schnittbereich auf gesamtes Video zurücksetzen")
        self.btn_reset_trim.setStyleSheet("background-color: #181824; border: 1px solid #28283a; border-radius: 8px; color: #a1a1aa; font-weight: bold; padding: 5px 8px;")
        self.btn_reset_trim.clicked.connect(self.reset_in_out_points)

        self.lbl_trim_info = QLabel("✂️ Schnitt: 00:00:00 ➔ 00:00:00 (Gesamtes Video)")
        self.lbl_trim_info.setStyleSheet("background-color: rgba(37, 99, 235, 0.12); border: 1px solid #2563eb; border-radius: 8px; padding: 5px 12px; color: #60a5fa; font-weight: 600; font-size: 11px;")

        ctrl_row.addWidget(self.btn_set_in)
        ctrl_row.addWidget(self.btn_jump_in)
        ctrl_row.addWidget(self.btn_set_out)
        ctrl_row.addWidget(self.btn_jump_out)
        ctrl_row.addWidget(self.btn_reset_trim)
        ctrl_row.addWidget(self.lbl_trim_info, 1)

        timeline_layout.addLayout(ctrl_row)
        left_layout.addWidget(self.timeline_frame)

        splitter.addWidget(left_pane)

        # =========================================================================
        # RIGHT PANE: Clean 4-Step Control Panel (Accordion Style)
        # =========================================================================
        self.right_pane = QTabWidget()
        self.right_pane.setMinimumWidth(400)
        self.right_pane.setDocumentMode(True)
        self.right_pane.setStyleSheet("QTabWidget::pane { border: none; }")
        self.right_pane.tabBar().hide()
        self.right_pane.currentChanged.connect(self._on_tab_changed)

        # -------------------------------------------------------------------------
        # TAB 0: ⚡ AUTOCAM 5-STEP ASSISTENT (Once Sport Autocam Style)
        # -------------------------------------------------------------------------
        self._init_stepper_wizard()
        self.right_pane.addTab(self.tab_stepper, "⚡ Autocam")

        # Dock Stepper's Live Process & Stepping Preview Container into the LEFT Pane
        if hasattr(self, 'tab_stepper') and hasattr(self.tab_stepper, 'progress_container'):
            left_layout.addWidget(self.tab_stepper.progress_container)
            self.tab_stepper.progress_container.setVisible(False)

        # -------------------------------------------------------------------------
        # TAB 1: 🧵 STITCHING (Quellformat, Videos, Sync & Naht-/Farbkalibrierung)
        # -------------------------------------------------------------------------
        self._init_stitching_tab()
        self.right_pane.addTab(self.tab_stitching, "🧵 1. Stitching")

        # -------------------------------------------------------------------------
        # TAB 2: 📐 Weitwinkel-Studio & 4-Punkt Spielfeld-Trapez
        # -------------------------------------------------------------------------
        self._init_wide_angle_tab()
        self.right_pane.addTab(self.tab_wide_angle, "📐 2. Weitwinkel")

        # TAB 3: 🎯 KI-Broadcast & Follow-Cam (Legacy-Modus - durch Autocam 5-Schritte ersetzt)
        self._init_autocam_tab()
        # Hinweis: self.tab_autocam wird nicht mehr zu right_pane hinzugefügt, da durch Autocam 5-Schritte Assistent ersetzt.

        # -------------------------------------------------------------------------
        # TAB 3: 🚀 Export (Samsung One UI Style)
        # -------------------------------------------------------------------------
        self.tab_export = QWidget()
        layout_tab_exp = QVBoxLayout(self.tab_export)
        layout_tab_exp.setContentsMargins(4, 4, 4, 4)
        scroll_exp = QScrollArea()
        scroll_exp.setWidgetResizable(True)
        widget_exp = QWidget()
        export_layout = QVBoxLayout(widget_exp)
        export_layout.setSpacing(12)

        grp_trim = QGroupBox("✂️ Export-Schnittbereich")
        layout_trim = QVBoxLayout(grp_trim)
        layout_trim.setSpacing(8)

        self.chk_export_trim_only = QCheckBox("Nur ausgewählten In/Out-Schnittbereich exportieren")
        self.chk_export_trim_only.setChecked(True)
        self.chk_export_trim_only.setStyleSheet("font-weight: 600; color: #10b981;")
        self.chk_export_trim_only.toggled.connect(self.on_export_trim_toggled)
        layout_trim.addWidget(self.chk_export_trim_only)

        row_in = QHBoxLayout()
        lbl_in = QLabel("Start (In):")
        lbl_in.setStyleSheet("font-weight: 600; min-width: 60px; color: #f4f4f6;")
        self.spin_export_in = NoWheelSpinBox()
        self.spin_export_in.setRange(0, 10000000)
        self.spin_export_in.setValue(0)
        self.spin_export_in.setSuffix(" F")
        self.spin_export_in.valueChanged.connect(self.on_spin_export_in_changed)
        self.lbl_export_in_time = QLabel("00:00:00.00")
        self.lbl_export_in_time.setStyleSheet("color: #60a5fa; font-weight: bold; min-width: 80px;")
        self.btn_export_set_in_curr = QPushButton("📍 Playhead")
        self.btn_export_set_in_curr.setStyleSheet("background-color: #1a1a26; border: 1px solid #28283a; border-radius: 6px; padding: 4px 8px; font-size: 11px;")
        self.btn_export_set_in_curr.clicked.connect(lambda: self.set_in_point())
        row_in.addWidget(lbl_in)
        row_in.addWidget(self.spin_export_in)
        row_in.addWidget(self.lbl_export_in_time)
        row_in.addWidget(self.btn_export_set_in_curr)
        layout_trim.addLayout(row_in)

        row_out = QHBoxLayout()
        lbl_out = QLabel("Ende (Out):")
        lbl_out.setStyleSheet("font-weight: 600; min-width: 60px; color: #f4f4f6;")
        self.spin_export_out = NoWheelSpinBox()
        self.spin_export_out.setRange(0, 10000000)
        self.spin_export_out.setValue(0)
        self.spin_export_out.setSuffix(" F")
        self.spin_export_out.valueChanged.connect(self.on_spin_export_out_changed)
        self.lbl_export_out_time = QLabel("00:00:00.00")
        self.lbl_export_out_time.setStyleSheet("color: #60a5fa; font-weight: bold; min-width: 80px;")
        self.btn_export_set_out_curr = QPushButton("📍 Playhead")
        self.btn_export_set_out_curr.setStyleSheet("background-color: #1a1a26; border: 1px solid #28283a; border-radius: 6px; padding: 4px 8px; font-size: 11px;")
        self.btn_export_set_out_curr.clicked.connect(lambda: self.set_out_point())
        row_out.addWidget(lbl_out)
        row_out.addWidget(self.spin_export_out)
        row_out.addWidget(self.lbl_export_out_time)
        row_out.addWidget(self.btn_export_set_out_curr)
        layout_trim.addLayout(row_out)

        row_dur = QHBoxLayout()
        self.lbl_export_duration_info = QLabel("Render-Dauer: 00:00:00 (0 Frames)")
        self.lbl_export_duration_info.setStyleSheet("color: #10b981; font-weight: 600; font-size: 11px;")
        btn_reset_export_range = QPushButton("↺ Alles")
        btn_reset_export_range.setStyleSheet("background-color: #1a1a26; border: 1px solid #28283a; border-radius: 6px; padding: 4px 8px; font-size: 11px;")
        btn_reset_export_range.clicked.connect(self.reset_in_out_points)
        row_dur.addWidget(self.lbl_export_duration_info, 1)
        row_dur.addWidget(btn_reset_export_range)
        layout_trim.addLayout(row_dur)

        export_layout.addWidget(grp_trim)

        grp_exp_settings = QGroupBox("⚙️ Render- & Export-Einstellungen")
        exp_form = QVBoxLayout(grp_exp_settings)
        exp_form.setSpacing(8)

        exp_form.addWidget(QLabel("Video-Format:"))
        self.combo_exp_format = QComboBox()
        self.combo_exp_format.addItem("🌟 32:9 Panorama (Taktik-Ansicht / Ganzes Feld)", "32:9")
        self.combo_exp_format.addItem("📐 21:10 Gestaucht (Taktik-Panorama / Ganzes Feld gestaucht)", "21:10")
        self.combo_exp_format.addItem("📐 16:9 Taktik-Warp (Entzerrtes Spielfeld ohne Schwenk)", "16:9_tactical")
        self.combo_exp_format.addItem("🎥 16:9 AutoCam (TV-Broadcast mit automatischer Kameraführung)", "16:9_autocam")
        self.combo_exp_format.addItem("📐 Weitwinkel Begradigt (DJI Action 4 Trapez)", "wide_angle")
        self.combo_exp_format.addItem("🔄 Beides exportieren", "both")
        self.combo_exp_format.currentIndexChanged.connect(self.on_export_format_changed)
        exp_form.addWidget(self.combo_exp_format)

        exp_form.addWidget(QLabel("Ausgabe-Auflösung:"))
        self.combo_resolution = QComboBox()
        self.update_resolution_dropdown()
        exp_form.addWidget(self.combo_resolution)

        exp_form.addWidget(QLabel("Hardware Video-Encoder (NVIDIA RTX / Intel QuickSync):"))
        self.combo_codec = QComboBox()
        self.combo_codec.addItem("🚀 NVIDIA RTX 3070 Laptop (Asus TUF Dash F15 - Ampere NVENC Ultra Fast)", "hevc_nvenc")
        self.combo_codec.addItem("🌟 NVIDIA RTX 3070 Laptop (Asus TUF Dash F15 - Hohe Qualität p6)", "hevc_nvenc_hq")
        self.combo_codec.addItem("⚡ NVIDIA RTX 2070 / Desktop (Turing NVENC Ultra Fast)", "hevc_nvenc")
        self.combo_codec.addItem("🌟 NVIDIA RTX 2070 / Desktop (Turing NVENC Hohe Qualität p6)", "hevc_nvenc_hq")
        self.combo_codec.addItem("⚡ NVIDIA NVENC H.264 (RTX 3070 / 2070 High-Speed)", "h264_nvenc")
        self.combo_codec.addItem("🔋 Intel Core i7-11370H QuickSync HEVC (Akku-schonend / Iris Xe HW)", "hevc_qsv")
        self.combo_codec.addItem("💻 Software CPU x264 (Multi-Core)", "libx264")
        exp_form.addWidget(self.combo_codec)

        self.ctrl_bitrate = LabeledSliderSpinBox("Export-Bitrate:", 10.0, 150.0, 50.0, 1.0, "Mbps", decimals=0)
        exp_form.addWidget(self.ctrl_bitrate)

        exp_form.addWidget(QLabel("Audio-Quelle:"))
        self.combo_audio_source = QComboBox()
        self.combo_audio_source.addItem("🎤 Kamera Links (Audioquelle Video A)", "left")
        self.combo_audio_source.addItem("🎤 Kamera Rechts (Audioquelle Video B, synchronisiert)", "right")
        self.combo_audio_source.addItem("🎚️ Beide Kameras mischen (Audio A + B)", "mix")
        self.combo_audio_source.addItem("🔇 Stumm (Kein Audio)", "none")
        exp_form.addWidget(self.combo_audio_source)

        self.chk_lookahead = QCheckBox("🎬 Filmreife Lookahead-Glättung (2-Pass antizipatorische Schwenks)")
        self.chk_lookahead.setChecked(True)
        self.chk_lookahead.setStyleSheet("font-weight: 600; color: #f59e0b;")
        exp_form.addWidget(self.chk_lookahead)

        self.chk_exp_burn_radar = QCheckBox("🗺️ 2D Taktik-Radar in Video einbrennen")
        self.chk_exp_burn_radar.setChecked(False)
        self.chk_exp_burn_radar.setStyleSheet("font-weight: 600; color: #38bdf8;")
        self.chk_exp_burn_radar.setToolTip("Rendert das 2D Taktik-Radar (Mini-Map) mit Live-Spielern und Ball direkt in das exportierte MP4-Video.")
        self.chk_exp_burn_radar.toggled.connect(self.on_burn_radar_toggled)
        exp_form.addWidget(self.chk_exp_burn_radar)

        export_layout.addWidget(grp_exp_settings)

        # Batch Export Action
        grp_render = QGroupBox("🚀 GPU Video-Export")
        render_box = QVBoxLayout(grp_render)
        render_box.setSpacing(8)

        self.btn_start_render = QPushButton("🚀 Video jetzt Rendern & Exportieren")
        self.btn_start_render.setStyleSheet("background-color: #2563eb; font-weight: bold; padding: 12px; font-size: 14px; color: white; border-radius: 10px;")
        self.btn_start_render.clicked.connect(self.start_video_export)
        render_box.addWidget(self.btn_start_render)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        render_box.addWidget(self.progress_bar)

        self.lbl_render_stats = QLabel("Bereit zum Exportieren.")
        self.lbl_render_stats.setStyleSheet("color: #8e8e9c; font-size: 11px;")
        render_box.addWidget(self.lbl_render_stats)

        self.btn_cancel_render = QPushButton("Abbrechen")
        self.btn_cancel_render.setEnabled(False)
        self.btn_cancel_render.setStyleSheet("background-color: #1a1a26; border: 1px solid #28283a; border-radius: 8px; padding: 6px; font-size: 11px; color: #94a3b8;")
        self.btn_cancel_render.clicked.connect(self.cancel_video_export)
        render_box.addWidget(self.btn_cancel_render)

        export_layout.addWidget(grp_render)

        export_layout.addStretch()

        scroll_exp.setWidget(widget_exp)
        layout_tab_exp.addWidget(scroll_exp)

        self.right_pane.addTab(self.tab_export, "🚀 4. Export")

        # -------------------------------------------------------------------------
        # TAB 5: OBS-STYLE RTMP LIVE-STREAMING & CAMERA DIRECT CONNECT
        # -------------------------------------------------------------------------
        self._init_livestream_tab()
        self.right_pane.addTab(self.tab_livestream, "🔴 5. Live-Stream")

        # -------------------------------------------------------------------------
        # TAB 6: MATCHTRACK ONLINE CLOUD-UPLOAD
        # -------------------------------------------------------------------------
        self._init_cloud_tab()
        self.right_pane.addTab(self.tab_cloud, "☁️ 6. Cloud-Upload")

        splitter.addWidget(self.right_pane)
        splitter.setSizes([1200, 480])
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)

        # Install wheel filter on right pane to prevent slider value changes on scrolling
        self._install_scroll_wheel_filter(self.right_pane)

        # Bottom Log Viewer Dock
        self.log_viewer = QPlainTextEdit()
        self.log_viewer.setReadOnly(True)
        self.log_viewer.setMaximumBlockCount(1500)
        self.log_viewer.setMaximumHeight(85)
        self.log_viewer.setStyleSheet("""
            QPlainTextEdit {
                background-color: #0b0b10;
                color: #8e8e9c;
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 11px;
                border: 1px solid #20202e;
                border-radius: 10px;
                padding: 4px 8px;
            }
        """)

        # Workspace Container (Houses Splitter and Log Viewer)
        self.workspace_widget = QWidget()
        workspace_layout = QVBoxLayout(self.workspace_widget)
        workspace_layout.setContentsMargins(0, 0, 0, 0)
        workspace_layout.setSpacing(6)
        workspace_layout.addWidget(splitter, 1)
        workspace_layout.addWidget(self.log_viewer)

        # Initialize Landing Page (Start-Zentrale)
        self._init_landing_page()

        # Stacked Widget housing Landing Page (Index 0) and Workspace (Index 1)
        self.main_stack = QStackedWidget()
        self.main_stack.addWidget(self.landing_widget)
        self.main_stack.addWidget(self.workspace_widget)

        root_layout.addWidget(self.main_stack, 1)

        # Bottom status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Bereit. Bitte Videos laden oder Profil öffnen.")

        # Setup GUI logging sink via thread-safe Qt Signal/Slot
        self.log_emitter = LogEmitter(self)
        self.log_emitter.log_signal.connect(self._safe_append_log, Qt.QueuedConnection)
        self.gui_handler = get_gui_handler()
        self._log_sub_cb = lambda rec: self.log_emitter.log_signal.emit(rec)
        self.gui_handler.subscribe(self._log_sub_cb)

    @Slot(object)
    def _safe_append_log(self, record: GuiLogRecord):
        if hasattr(self, 'log_viewer') and self.log_viewer is not None:
            self.log_viewer.appendPlainText(record.formatted())

    def append_log_message(self, record: GuiLogRecord):
        """Thread-safe forwarder for legacy callers."""
        if hasattr(self, 'log_emitter'):
            self.log_emitter.log_signal.emit(record)
        else:
            self._safe_append_log(record)

    def _install_scroll_wheel_filter(self, container: QWidget):
        """
        Installs ScrollAreaWheelFilter on all interactive controls inside container
        so that mouse-wheel scrolling scrolls the parent QScrollArea without
        altering slider, spinbox, or combobox values.
        """
        if not hasattr(self, "_wheel_filter"):
            self._wheel_filter = ScrollAreaWheelFilter(self)
        
        if isinstance(container, (QSlider, QAbstractSpinBox, QComboBox)):
            container.installEventFilter(self._wheel_filter)
        for child in container.findChildren(QWidget):
            if isinstance(child, (QSlider, QAbstractSpinBox, QComboBox)):
                child.installEventFilter(self._wheel_filter)

    def apply_dark_theme(self):
        self.setStyleSheet("""
            QMainWindow { background-color: #0b0b10; }
            QWidget { color: #f4f4f6; font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, 'Roboto', sans-serif; font-size: 12px; }
            
            /* Samsung One UI Cards */
            QGroupBox {
                font-weight: 700;
                font-size: 13px;
                border: 1px solid #232332;
                border-radius: 14px;
                margin-top: 14px;
                padding: 18px 12px 12px 12px;
                background-color: #15151f;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 14px;
                top: 1px;
                padding: 0 8px;
                background-color: #15151f;
                color: #f4f4f6;
                font-size: 12px;
                font-weight: 700;
                border-radius: 4px;
            }
            
            /* Samsung One UI Buttons */
            QPushButton {
                background-color: #1e1e2b;
                border: 1px solid #2e2e40;
                border-radius: 10px;
                padding: 7px 14px;
                font-weight: 600;
                font-size: 12px;
                color: #f4f4f6;
            }
            QPushButton:hover {
                background-color: #28283a;
                border-color: #404058;
                color: #ffffff;
            }
            QPushButton:pressed {
                background-color: #14141e;
                border-color: #2563eb;
            }
            QPushButton:checked {
                background-color: #2563eb;
                border-color: #3b82f6;
                color: #ffffff;
                font-weight: bold;
            }
            QPushButton:disabled {
                background-color: #13131c;
                color: #484858;
                border-color: #1e1e2a;
            }
            
            /* Samsung One UI Sliders */
            QSlider::groove:horizontal { 
                height: 6px; 
                background: #252536; 
                border-radius: 3px; 
            }
            QSlider::sub-page:horizontal { 
                background: #2563eb; 
                border-radius: 3px; 
            }
            QSlider::handle:horizontal { 
                background: #ffffff; 
                border: 2px solid #2563eb; 
                width: 16px; 
                height: 16px; 
                margin: -5px 0; 
                border-radius: 8px; 
            }
            QSlider::handle:horizontal:hover { 
                background: #93c5fd; 
                border-color: #3b82f6; 
            }
            
            /* Samsung One UI Inputs */
            QComboBox, QSpinBox, QDoubleSpinBox {
                background-color: #111118;
                border: 1px solid #282839;
                border-radius: 8px;
                padding: 5px 8px;
                color: #f4f4f6;
                font-weight: 500;
            }
            QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover {
                border-color: #2563eb;
            }
            QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {
                border-color: #3b82f6;
                background-color: #13131c;
            }
            QComboBox::drop-down {
                border: none;
                width: 22px;
            }
            
            /* Samsung One UI Segmented Control TabWidget */
            QTabWidget {
                background: transparent;
            }
            QTabWidget::pane {
                border: none;
                background: transparent;
                padding-top: 4px;
            }
            QTabBar {
                background-color: #12121a;
                border: 1px solid #222230;
                border-radius: 12px;
                padding: 3px;
                qproperty-expanding: true;
            }
            QTabBar::tab {
                background: transparent;
                color: #8e8e9c;
                border-radius: 9px;
                padding: 8px 10px;
                font-weight: 600;
                font-size: 12px;
                border: none;
                margin: 1px;
            }
            QTabBar::tab:selected {
                background-color: #2563eb;
                color: #ffffff;
                font-weight: bold;
            }
            QTabBar::tab:hover:!selected {
                background-color: #1a1a26;
                color: #ffffff;
            }
            
            /* Samsung One UI CheckBoxes */
            QCheckBox {
                spacing: 8px;
                font-weight: 500;
                color: #e4e4e7;
                font-size: 12px;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
                border-radius: 5px;
                border: 1px solid #36364a;
                background-color: #121218;
            }
            QCheckBox::indicator:hover {
                border-color: #2563eb;
            }
            QCheckBox::indicator:checked {
                background-color: #2563eb;
                border-color: #3b82f6;
            }
            
            /* Samsung One UI Progress Bar */
            QProgressBar {
                border: 1px solid #262638;
                border-radius: 8px;
                text-align: center;
                background: #121218;
                color: #ffffff;
                font-weight: 700;
                font-size: 11px;
                height: 22px;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #1d4ed8, stop:1 #2563eb);
                border-radius: 7px;
            }
            
            QSplitter::handle {
                background-color: #181822;
                width: 5px;
            }
            QSplitter::handle:hover {
                background-color: #2563eb;
            }
            
            QScrollArea {
                border: none;
                background: transparent;
            }
            
            QScrollBar:vertical {
                background: #0f0f15;
                width: 7px;
                border-radius: 3px;
                margin: 2px 0 2px 0;
            }
            QScrollBar::handle:vertical {
                background: #2a2a3b;
                border-radius: 3px;
                min-height: 24px;
            }
            QScrollBar::handle:vertical:hover {
                background: #3f3f58;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
            
            QStatusBar {
                background-color: #101016;
                color: #8e8e9c;
                border-top: 1px solid #1e1e2b;
                font-size: 11px;
            }
        """)

    def has_active_video(self) -> bool:
        return (self.engine.video_panorama is not None) or (self.engine.video_left is not None)

    def is_4_3_mode(self) -> bool:
        """Returns True if the current camera profile or loaded video is 4:3 (yielding an 8:3 panorama)."""
        if hasattr(self, 'engine') and hasattr(self.engine, 'is_4_3_mode'):
            return self.engine.is_4_3_mode()
        if hasattr(self, 'rig') and self.rig and hasattr(self.rig, 'left_camera') and self.rig.left_camera:
            w = self.rig.left_camera.image_width
            h = self.rig.left_camera.image_height
            if h > 0 and (w / h) < 1.55:
                return True
        return False

    def get_panorama_aspect(self) -> float:
        """Returns nominal panorama aspect ratio (8.0/3.0 for 4:3 inputs, 32.0/9.0 for 16:9 inputs)."""
        return (8.0 / 3.0) if self.is_4_3_mode() else (32.0 / 9.0)

    def _update_format_combo_labels(self):
        """Updates UI dropdown labels and live preview mode to accurately reflect 8:3 (for 4:3 inputs) or 32:9 (for 16:9 inputs)."""
        is_4_3 = self.is_4_3_mode()
        if hasattr(self, 'viewport'):
            self.viewport.is_4_3 = is_4_3
        if hasattr(self, 'combo_view_mode'):
            cur_mode = self.combo_view_mode.currentData()
            if is_4_3 and cur_mode == "32:9":
                idx_8_3 = self.combo_view_mode.findData("8:3")
                if idx_8_3 >= 0:
                    self.combo_view_mode.blockSignals(True)
                    self.combo_view_mode.setCurrentIndex(idx_8_3)
                    self.combo_view_mode.blockSignals(False)
                    if hasattr(self, 'viewport'):
                        self.viewport.view_mode = "8:3"
            elif not is_4_3 and cur_mode == "8:3":
                idx_32_9 = self.combo_view_mode.findData("32:9")
                if idx_32_9 >= 0:
                    self.combo_view_mode.blockSignals(True)
                    self.combo_view_mode.setCurrentIndex(idx_32_9)
                    self.combo_view_mode.blockSignals(False)
                    if hasattr(self, 'viewport'):
                        self.viewport.view_mode = "32:9"
        if hasattr(self, 'chk_show_frame'):
            self.chk_show_frame.setText("🔲 8:3 / 16:9 Rahmen" if is_4_3 else "🔲 32:9 / 16:9 Rahmen")
        if hasattr(self, 'btn_view_pano') and self.btn_view_pano is not None:
            self.btn_view_pano.setText("🌟 8:3 Panorama" if is_4_3 else "🌟 32:9 Panorama")
        if hasattr(self, 'combo_exp_format'):
            idx_pano = self.combo_exp_format.findData("32:9")
            if idx_pano >= 0:
                txt = "🌟 8:3 Panorama (Taktik-Ansicht / Ganzes Feld)" if is_4_3 else "🌟 32:9 Panorama (Taktik-Ansicht / Ganzes Feld)"
                self.combo_exp_format.setItemText(idx_pano, txt)
            idx_both = self.combo_exp_format.findData("both")
            if idx_both >= 0:
                txt_b = "🔄 Beides exportieren (8:3 Master + 16:9 Broadcast)" if is_4_3 else "🔄 Beides exportieren (32:9 Master + 16:9 Broadcast)"
                self.combo_exp_format.setItemText(idx_both, txt_b)

    # Video loading & seeking
    def open_left_video(self, file_path_override: Optional[str] = None):
        file_path = file_path_override or QFileDialog.getOpenFileName(self, "Linkes Video öffnen (DJI Action 4)", "", "Video Files (*.mp4 *.mov *.mkv *.avi)")[0]
        if file_path:
            self.video_path_l = file_path
            self.lbl_file_left.setText(os.path.basename(file_path))
            self.lbl_file_left.setStyleSheet("color: #60a5fa; font-weight: 600;")
            if hasattr(self, 'badge_left'):
                self.badge_left.setText("🟢 Bereit")
                self.badge_left.setStyleSheet("background: #064e3b; color: #34d399; border-radius: 6px; padding: 2px 8px; font-size: 11px; font-weight: 600;")
            if hasattr(self, 'tab_stepper') and hasattr(self.tab_stepper, 'update_left_video_path'):
                self.tab_stepper.update_left_video_path(file_path)
            self._try_init_engine()

    def open_right_video(self, file_path_override: Optional[str] = None):
        file_path = file_path_override or QFileDialog.getOpenFileName(self, "Rechtes Video öffnen (DJI Action 4)", "", "Video Files (*.mp4 *.mov *.mkv *.avi)")[0]
        if file_path:
            self.video_path_r = file_path
            self.lbl_file_right.setText(os.path.basename(file_path))
            self.lbl_file_right.setStyleSheet("color: #60a5fa; font-weight: 600;")
            if hasattr(self, 'badge_right'):
                self.badge_right.setText("🟢 Bereit")
                self.badge_right.setStyleSheet("background: #064e3b; color: #34d399; border-radius: 6px; padding: 2px 8px; font-size: 11px; font-weight: 600;")
            if hasattr(self, 'tab_stepper') and hasattr(self.tab_stepper, 'update_right_video_path'):
                self.tab_stepper.update_right_video_path(file_path)
            self._try_init_engine()

    def open_panorama_video(self, file_path_override: Optional[str] = None):
        file_path = file_path_override or QFileDialog.getOpenFileName(self, "32:9 Panorama-Mastervideo öffnen", "", "Video Files (*.mp4 *.mov *.mkv *.avi)")[0]
        if file_path:
            self.panorama_path = file_path
            self.lbl_file_left.setText(f"🎬 {os.path.basename(file_path)}")
            self.lbl_file_left.setStyleSheet("color: #34d399; font-weight: 600;")
            self.lbl_file_right.setText("(1-Video Modus)")
            self.lbl_file_right.setStyleSheet("color: #8e8e9c; font-style: italic;")
            if hasattr(self, 'badge_left'):
                self.badge_left.setText("🎬 Panorama")
                self.badge_left.setStyleSheet("background: #064e3b; color: #34d399; border-radius: 6px; padding: 2px 8px; font-size: 11px; font-weight: 600;")
            if hasattr(self, 'badge_right'):
                self.badge_right.setText("—")
                self.badge_right.setStyleSheet("background: #181824; color: #8e8e9c; border-radius: 6px; padding: 2px 8px; font-size: 11px;")
            if hasattr(self, 'tab_stepper') and hasattr(self.tab_stepper, 'update_panorama_video_path'):
                self.tab_stepper.update_panorama_video_path(file_path)
            try:
                self.engine.load_panorama_video(file_path)
                self.update_ui_for_mode(is_panorama=True)
                self._after_videos_loaded()
            except Exception as e:
                QMessageBox.critical(self, "Fehler beim Laden", str(e))

    def _try_init_engine(self):
        if hasattr(self, 'video_path_l') and hasattr(self, 'video_path_r'):
            try:
                self.engine.load_videos(self.video_path_l, self.video_path_r)
                self.update_ui_for_mode(is_panorama=False)
                self._after_videos_loaded()
            except Exception as e:
                QMessageBox.critical(self, "Fehler beim Laden", str(e))

    def _after_videos_loaded(self):
        total_frames = self.engine.get_max_duration_frames()
        self.slider_timeline.setRange(0, max(0, total_frames - 1))
        self.slider_timeline.setValue(0)
        
        self.in_point_frame = 0
        self.out_point_frame = max(0, total_frames - 1)
        self.spin_export_in.setRange(0, max(0, total_frames - 1))
        self.spin_export_out.setRange(0, max(0, total_frames - 1))

        fps = self.engine.get_fps()
        dur_sec = total_frames / max(fps, 1.0)
        m, s = divmod(int(dur_sec), 60)
        h, m = divmod(m, 60)
        self.lbl_total_time.setText(f"{h:02d}:{m:02d}:{s:02d}")
        
        self.update_in_out_display()
        if hasattr(self, 'tab_stepper'):
            if hasattr(self.tab_stepper, 'update_video_info'):
                self.tab_stepper.update_video_info(total_frames, fps)
            if hasattr(self, 'video_path_l') and self.video_path_l:
                self.tab_stepper.update_left_video_path(self.video_path_l)
            if hasattr(self, 'video_path_r') and self.video_path_r:
                self.tab_stepper.update_right_video_path(self.video_path_r)
            if hasattr(self, 'panorama_path') and self.panorama_path:
                self.tab_stepper.update_panorama_video_path(self.panorama_path)
            if hasattr(self.tab_stepper, 'set_trim_range'):
                self.tab_stepper.set_trim_range(self.in_point_frame, self.out_point_frame)
        if self.engine.is_panorama_mode():
            if hasattr(self, 'lbl_video_status'):
                status_txt = "🎬 8:3 Panorama geladen" if self.is_4_3_mode() else "🎬 32:9 Panorama geladen"
                self.lbl_video_status.setText(status_txt)
                self.lbl_video_status.setStyleSheet("background: #064e3b; color: #34d399; border: 1px solid #059669; border-radius: 12px; padding: 3px 10px; font-size: 11px; font-weight: bold;")
            self.status_bar.showMessage(f"🎬 Panorama geladen: {self.engine.video_panorama.width}x{self.engine.video_panorama.height} bei {fps:.2f} FPS. Bereit für 16:9 Broadcast.")
        else:
            if hasattr(self, 'lbl_video_status'):
                status_txt = "🟢 2 Kameras (4:3) geladen" if self.is_4_3_mode() else "🟢 2 Kameras geladen"
                self.lbl_video_status.setText(status_txt)
                self.lbl_video_status.setStyleSheet("background: #064e3b; color: #34d399; border: 1px solid #059669; border-radius: 12px; padding: 3px 10px; font-size: 11px; font-weight: bold;")
            self.status_bar.showMessage(f"Videos geladen: {self.engine.video_left.width}x{self.engine.video_left.height} bei {fps:.2f} FPS. Bereit.")
        self._update_format_combo_labels()
        self.update_resolution_dropdown()
        self.refresh_preview()

    def update_ui_for_mode(self, is_panorama: bool):
        if is_panorama:
            self.setWindowTitle("MatchTrack-Stitcher | 🎬 32:9 Panorama-Modus ➔ 16:9 Follow-Cam Broadcast")
            self.btn_auto_sync.setEnabled(False)
            self.btn_auto_stitch.setEnabled(False)
            if getattr(self, 'btn_corner_pins_tb', None) is not None:
                self.btn_corner_pins_tb.setEnabled(False)
            if hasattr(self, 'btn_corner_pins_side') and self.btn_corner_pins_side is not None:
                self.btn_corner_pins_side.setEnabled(False)
            self.chk_show_seam.setEnabled(False)
            self.chk_show_grid.setEnabled(False)

            self.combo_exp_format.blockSignals(True)
            self.combo_exp_format.clear()
            self.combo_exp_format.addItem("📐 21:10 Gestaucht (Taktik-Panorama / 32:9 zu 21:10)", "21:10")
            self.combo_exp_format.addItem("🎥 16:9 Broadcast Follow-Cam (aus 32:9 Video)", "16:9_autocam")
            self.combo_exp_format.blockSignals(False)
            self.combo_audio_source.setEnabled(False)
            self.update_resolution_dropdown()

            if hasattr(self, 'tab_autocam') and hasattr(self, 'right_pane'):
                self.right_pane.setCurrentWidget(self.tab_autocam)
        else:
            self.setWindowTitle("MatchTrack-Stitcher | 32:9 & 21:10 Panorama & 16:9 Ball-Follow Broadcast (DJI Action 4)")
            self.btn_auto_sync.setEnabled(True)
            self.btn_auto_stitch.setEnabled(True)
            if getattr(self, 'btn_corner_pins_tb', None) is not None:
                self.btn_corner_pins_tb.setEnabled(True)
            if hasattr(self, 'btn_corner_pins_side') and self.btn_corner_pins_side is not None:
                self.btn_corner_pins_side.setEnabled(True)
            self.chk_show_seam.setEnabled(True)
            self.chk_show_grid.setEnabled(True)

            self.combo_exp_format.blockSignals(True)
            self.combo_exp_format.clear()
            pano_label = "🌟 8:3 Panorama Master (Taktik-Ansicht / Ganzes Feld)" if self.is_4_3_mode() else "🌟 32:9 Panorama Master (Taktik-Ansicht / Ganzes Feld)"
            both_label = "🔄 Beides exportieren (8:3 Master + 16:9 Broadcast)" if self.is_4_3_mode() else "🔄 Beides exportieren (32:9 Master + 16:9 Broadcast)"
            self.combo_exp_format.addItem(pano_label, "32:9")
            self.combo_exp_format.addItem("📐 21:10 Gestaucht (Taktik-Panorama / Ganzes Feld gestaucht)", "21:10")
            self.combo_exp_format.addItem("🎥 16:9 Broadcast Follow-Cam (2-Stufig)", "16:9_autocam")
            self.combo_exp_format.addItem(both_label, "both")
            self.combo_exp_format.blockSignals(False)
            self.combo_audio_source.setEnabled(True)
            self.update_resolution_dropdown()

    def run_audio_sync(self):
        if not hasattr(self, 'video_path_l') or not hasattr(self, 'video_path_r'):
            QMessageBox.warning(self, "Hinweis", "Bitte zuerst beide Videos öffnen!")
            return

        self.status_bar.showMessage("Analysiere Audio-Wellenformen mit FFT Cross-Correlation...")
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            offset_frames, delay_sec, conf = calculate_audio_sync_offset(
                self.video_path_l, 
                self.video_path_r,
                fps=self.engine.video_left.fps
            )
            self.spin_offset.setValue(offset_frames)
            self.status_bar.showMessage(f"Audio-Sync abgeschlossen! Offset: {offset_frames} Frames ({delay_sec*1000:.1f} ms) | Konfidenz: {conf*100:.1f}%")
            QMessageBox.information(
                self, 
                "Audio-Synchronisation", 
                f"Synchronisation erfolgreich berechnet!\n\n"
                f"• Versatz: {offset_frames} Frames ({delay_sec:.3f} Sekunden)\n"
                f"• Signal-Konfidenz: {conf*100:.1f}%\n\n"
                f"Der Offset wurde automatisch auf die rechte Kamera angewendet."
            )
        except Exception as e:
            QMessageBox.warning(self, "Audio Sync Fehler", f"Konnte Audio nicht synchronisieren: {e}")
        finally:
            QApplication.restoreOverrideCursor()

    def run_auto_stitch_calibration(self):
        if not hasattr(self, 'video_path_l') or not hasattr(self, 'video_path_r'):
            QMessageBox.warning(self, "Hinweis", "Bitte zuerst beide Videos öffnen!")
            return

        if not self.engine.video_left or not self.engine.video_right:
            QMessageBox.warning(self, "Hinweis", "Videos sind noch nicht vollständig geladen.")
            return

        use_multi_frame = getattr(self, 'chk_multi_frame_calib', None) and self.chk_multi_frame_calib.isChecked()
        
        self.status_bar.showMessage("Analysiere optische Nahtstellen-Überlappung mit SIFT KI-Feature-Matching...")
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            if use_multi_frame:
                total_f = self.engine.get_max_duration_frames()
                # Sample 5 timestamps across video duration
                sample_indices = [
                    int(total_f * 0.10),
                    int(total_f * 0.30),
                    int(total_f * 0.50),
                    int(total_f * 0.70),
                    int(total_f * 0.90)
                ]
                frames_l = []
                frames_r = []
                for f_idx in sample_indices:
                    idx_l = max(0, min(f_idx, self.engine.video_left.total_frames - 1))
                    idx_r = max(0, min(f_idx - self.engine.frame_offset_right, self.engine.video_right.total_frames - 1))
                    fl = self.engine.video_left.get_frame(idx_l)
                    fr = self.engine.video_right.get_frame(idx_r)
                    if fl is not None and fr is not None:
                        frames_l.append(fl)
                        frames_r.append(fr)

                result = calibrate_from_frames(frames_l, frames_r, self.rig, out_w=3840, out_h=1080)
            else:
                idx_l = self.current_frame_idx
                idx_r = self.current_frame_idx - self.engine.frame_offset_right
                fl = self.engine.video_left.get_frame(idx_l)
                fr = self.engine.video_right.get_frame(idx_r)
                if fl is None or fr is None:
                    QMessageBox.warning(self, "Fehler", "Konnte aktuelle Frames nicht aus den Videos lesen.")
                    return
                matches = extract_overlap_matches(fl, fr)
                result = optimize_rig_calibration(matches, self.rig, out_w=3840, out_h=1080)

            if not result.success:
                self.status_bar.showMessage("KI-Kalibrierung nicht möglich: " + result.message)
                QMessageBox.warning(self, "Kalibrierung fehlgeschlagen", result.message)
                return

            # Apply optimized parameters to UI controls
            d = result.details
            self.ctrl_cam_fov.setValue(d["hfov"])
            
            # Symmetrical or asymmetrical yaw
            if abs(abs(d["yaw_left"]) - abs(d["yaw_right"])) < 0.6:
                self.chk_sync_yaw.setChecked(True)
                self.ctrl_master_yaw.setValue(d["spread_yaw"])
            else:
                self.chk_sync_yaw.setChecked(False)
                self.ctrl_left_yaw.setValue(d["yaw_left"])
                self.ctrl_right_yaw.setValue(d["yaw_right"])

            self.ctrl_cam_pitch.setValue(d["pitch"])
            self.ctrl_global_pitch.setValue(-d["pitch"])
            self.ctrl_left_roll.setValue(d["roll_left"])
            self.ctrl_right_roll.setValue(d["roll_right"])

            if hasattr(self, 'ctrl_bottom_taper_x') and "bottom_taper_x" in d:
                self.ctrl_bottom_taper_x.setValue(d["bottom_taper_x"])
            if hasattr(self, 'ctrl_bottom_taper_y') and "bottom_taper_y" in d:
                self.ctrl_bottom_taper_y.setValue(d["bottom_taper_y"])

            # Ensure optimal field coverage (protect bottom pitch corners & full sidelines)
            cam = getattr(self.rig, 'left_camera', None)
            cam_w = getattr(cam, 'image_width', 1920)
            cam_h = getattr(cam, 'image_height', 1080)
            if cam_w / max(1, cam_h) >= 1.55:
                if hasattr(self, 'ctrl_squeeze') and self.ctrl_squeeze.value() > 0.60:
                    self.ctrl_squeeze.setValue(0.52)
                    self.rig.horizontal_squeeze = 0.52
                if hasattr(self, 'ctrl_v_offset') and self.ctrl_v_offset.value() > -0.05:
                    self.ctrl_v_offset.setValue(-0.12)
                    self.rig.vertical_crop_offset = -0.12
                if hasattr(self, 'ctrl_pano_hfov') and self.ctrl_pano_hfov.value() < 145.0:
                    self.ctrl_pano_hfov.setValue(146.0)
                    self.rig.pano_hfov = 146.0

            # Automatic Color & Exposure Matching across left and right cameras
            self.rig.auto_color_match = True
            self.rig.color_match_strength = 1.0
            if hasattr(self, 'chk_auto_color_match'):
                self.chk_auto_color_match.setChecked(True)

            self.engine.invalidate_luts()
            luts_updated = self.engine.update_luts(2560, 720)

            # Calibrate color matcher with current frames
            calib_fl = frames_l[len(frames_l)//2] if (use_multi_frame and frames_l) else (fl if 'fl' in locals() else None)
            calib_fr = frames_r[len(frames_r)//2] if (use_multi_frame and frames_r) else (fr if 'fr' in locals() else None)
            color_info = ""
            if calib_fl is not None and calib_fr is not None:
                color_res = self.engine.color_matcher.calibrate(calib_fl, calib_fr, luts_updated)
                if color_res.get("success"):
                    gl = color_res.get("gain_l", [1.0, 1.0, 1.0])
                    gr = color_res.get("gain_r", [1.0, 1.0, 1.0])
                    color_info = f"\n\nFarb- & Belichtungsabgleich:\n• Kameras L ↔ R automatisch harmonisiert\n• Gains: L=[{gl[0]:.2f}, {gl[1]:.2f}, {gl[2]:.2f}], R=[{gr[0]:.2f}, {gr[1]:.2f}, {gr[2]:.2f}]"

            if self.rig.smart_seam_enabled:
                curr_l = self.engine.video_left.get_frame(self.current_frame_idx) if self.engine.video_left else None
                curr_r = self.engine.video_right.get_frame(max(0, self.current_frame_idx - self.engine.frame_offset_right)) if self.engine.video_right else None
                if curr_l is not None and curr_r is not None:
                    self.engine.luts.update_smart_seam(curr_l, curr_r, self.rig)
            self.render_high_res_preview()

            self.status_bar.showMessage(
                f"KI-Kalibrierung fertig! {result.num_matches} Matches | "
                f"Nahtfehler: {result.initial_error_px:.1f}px ➔ {result.optimized_error_px:.2f}px ({result.improvement_pct:.1f}% verbessert) | Farbabgleich aktiv"
            )
            QMessageBox.information(self, "KI-Nahtstellen-Kalibrierung", result.message + color_info)

        except Exception as e:
            self.status_bar.showMessage(f"Fehler bei KI-Kalibrierung: {e}")
            QMessageBox.critical(self, "Fehler bei KI-Kalibrierung", str(e))
        finally:
            QApplication.restoreOverrideCursor()

    def on_offset_slider_changed(self, val):
        if self.spin_offset.value() != val:
            self.spin_offset.setValue(val)

    def on_offset_spin_changed(self, val):
        if self.slider_offset.value() != val:
            self.slider_offset.blockSignals(True)
            self.slider_offset.setValue(val)
            self.slider_offset.blockSignals(False)
        self.engine.frame_offset_right = val
        self.refresh_preview()

    def adjust_offset_by(self, delta: int):
        """Fine-adjust the right camera audio/video frame offset by delta (+/- 1F, +/- 5F)."""
        self.spin_offset.setValue(self.spin_offset.value() + delta)

    def format_timecode(self, frame_idx: int) -> str:
        fps = self.engine.get_fps() if self.engine else 30.0
        sec = frame_idx / max(fps, 1.0)
        m, s = divmod(int(sec), 60)
        h, m = divmod(m, 60)
        millis = int((sec - int(sec)) * 100)
        return f"{h:02d}:{m:02d}:{s:02d}.{millis:02d}"

    def set_in_point(self, frame_idx: Optional[int] = None):
        if frame_idx is None:
            frame_idx = self.current_frame_idx
        if self.has_active_video():
            max_f = max(0, self.engine.get_max_duration_frames() - 1)
            frame_idx = max(0, min(max_f, frame_idx))
        if frame_idx > self.out_point_frame:
            self.out_point_frame = frame_idx
        self.in_point_frame = frame_idx
        self.update_in_out_display()
        self.status_bar.showMessage(f"🚩 Startpunkt (In) auf Frame {frame_idx} ({self.format_timecode(frame_idx)}) gesetzt.")

    def set_out_point(self, frame_idx: Optional[int] = None):
        if frame_idx is None:
            frame_idx = self.current_frame_idx
        if self.has_active_video():
            max_f = max(0, self.engine.get_max_duration_frames() - 1)
            frame_idx = max(0, min(max_f, frame_idx))
        if frame_idx < self.in_point_frame:
            self.in_point_frame = frame_idx
        self.out_point_frame = frame_idx
        self.update_in_out_display()
        self.status_bar.showMessage(f"🏁 Endpunkt (Out) auf Frame {frame_idx} ({self.format_timecode(frame_idx)}) gesetzt.")

    def jump_to_in_point(self):
        self.slider_timeline.setValue(self.in_point_frame)

    def jump_to_out_point(self):
        self.slider_timeline.setValue(self.out_point_frame)

    def reset_in_out_points(self):
        if not self.has_active_video():
            self.in_point_frame = 0
            self.out_point_frame = 0
        else:
            self.in_point_frame = 0
            self.out_point_frame = max(0, self.engine.get_max_duration_frames() - 1)
        self.update_in_out_display()
        self.status_bar.showMessage("Schnittbereich auf gesamtes Video zurückgesetzt.")

    def update_in_out_display(self):
        # Also sync stepper wizard if present
        if hasattr(self, 'tab_stepper') and hasattr(self.tab_stepper, 'set_trim_range'):
            self.tab_stepper.set_trim_range(self.in_point_frame, self.out_point_frame)

        if not self.has_active_video():
            self.lbl_trim_info.setText("✂️ Schnittbereich: Kein Video geladen")
            return

        total_avail = self.engine.get_max_duration_frames()
        in_f = max(0, min(self.in_point_frame, total_avail - 1))
        out_f = max(in_f, min(self.out_point_frame, total_avail - 1))
        
        in_tc = self.format_timecode(in_f)
        out_tc = self.format_timecode(out_f)
        
        diff_frames = (out_f - in_f + 1)
        fps = self.engine.get_fps()
        dur_sec = diff_frames / max(fps, 1.0)
        m, s = divmod(int(dur_sec), 60)
        h, m = divmod(m, 60)
        dur_str = f"{h:02d}:{m:02d}:{s:02d}"

        self.lbl_trim_info.setText(f"✂️ In: {in_tc} (F {in_f}) ➔ Out: {out_tc} (F {out_f}) | Schnitt-Dauer: {dur_str} ({diff_frames:,} F)")

        # Update export tab widgets without triggering recursion
        self.spin_export_in.blockSignals(True)
        self.spin_export_out.blockSignals(True)
        self.spin_export_in.setValue(in_f)
        self.spin_export_out.setValue(out_f)
        self.spin_export_in.blockSignals(False)
        self.spin_export_out.blockSignals(False)

        self.lbl_export_in_time.setText(in_tc)
        self.lbl_export_out_time.setText(out_tc)

        # Also sync stepper wizard if present
        if hasattr(self, 'tab_stepper') and hasattr(self.tab_stepper, 'set_trim_range'):
            self.tab_stepper.set_trim_range(in_f, out_f)
        
        if self.chk_export_trim_only.isChecked():
            self.lbl_export_duration_info.setText(f"Ausgewählte Export-Dauer: {dur_str} ({diff_frames:,} Frames)")
        else:
            tot_dur = total_avail / max(fps, 1.0)
            tm, ts = divmod(int(tot_dur), 60)
            th, tm = divmod(tm, 60)
            self.lbl_export_duration_info.setText(f"Gesamte Video-Dauer: {th:02d}:{tm:02d}:{ts:02d} ({total_avail:,} Frames)")

    def on_export_trim_toggled(self, checked: bool):
        self.spin_export_in.setEnabled(checked)
        self.spin_export_out.setEnabled(checked)
        self.btn_export_set_in_curr.setEnabled(checked)
        self.btn_export_set_out_curr.setEnabled(checked)
        self.update_in_out_display()

    def on_spin_export_in_changed(self, val: int):
        self.set_in_point(val)

    def on_spin_export_out_changed(self, val: int):
        self.set_out_point(val)

    def keyPressEvent(self, event):
        focused = QApplication.focusWidget()
        if isinstance(focused, (QSpinBox, QDoubleSpinBox)):
            super().keyPressEvent(event)
            return
        
        if event.key() == Qt.Key_I:
            self.set_in_point()
            event.accept()
        elif event.key() == Qt.Key_O:
            self.set_out_point()
            event.accept()
        elif event.key() == Qt.Key_Space:
            self.toggle_playback()
            event.accept()
        elif event.key() == Qt.Key_Left:
            self.step_frame(-1)
            event.accept()
        elif event.key() == Qt.Key_Right:
            self.step_frame(1)
            event.accept()
        elif event.key() == Qt.Key_Home:
            self.jump_to_in_point()
            event.accept()
        elif event.key() == Qt.Key_End:
            self.jump_to_out_point()
            event.accept()
        else:
            super().keyPressEvent(event)

    def on_slider_seek(self, val):
        self.current_frame_idx = val
        if self.has_active_video():
            tc = self.format_timecode(val)
            self.lbl_current_time.setText(f"{tc} (F {val})")
        self.refresh_preview()

    def step_frame(self, step):
        new_val = max(0, self.current_frame_idx + step)
        self.slider_timeline.setValue(new_val)

    def seek_frame(self, frame_idx: int):
        """Seeks the video playhead to the given frame index."""
        new_val = max(0, int(frame_idx))
        if hasattr(self, 'slider_timeline'):
            self.slider_timeline.setValue(new_val)
        else:
            self.on_slider_seek(new_val)

    def toggle_playback(self):
        if self.is_playing:
            self.playback_timer.stop()
            self.btn_play_pause.setText("▶ Abspielen")
            self.btn_play_pause.setStyleSheet("font-weight: bold; min-width: 100px; background-color: #27ae60;")
            self.is_playing = False
        else:
            self.playback_timer.start()
            self.btn_play_pause.setText("⏸ Pause")
            self.btn_play_pause.setStyleSheet("font-weight: bold; min-width: 100px; background-color: #e67e22;")
            self.is_playing = True

    def on_play_step(self):
        max_f = self.slider_timeline.maximum()
        if self.current_frame_idx >= max_f:
            self.toggle_playback()
            return
        self.slider_timeline.setValue(self.current_frame_idx + 1)

    def toggle_seam_overlay(self, checked):
        self.viewport.show_seam = checked
        self.viewport.update()

    def toggle_grid_overlay(self, checked):
        self.viewport.show_grid = checked
        self.viewport.update()

    def toggle_autocam_box(self, checked):
        self.viewport.show_autocam_box = checked
        self.viewport.update()

    def toggle_ball_reticle(self, checked):
        self.viewport.show_ball_tracking = checked
        self.viewport.update()

    def toggle_radar_overlay(self, checked: bool):
        self.viewport.set_radar_enabled(checked)
        if hasattr(self, 'chk_show_radar'):
            self.chk_show_radar.blockSignals(True)
            self.chk_show_radar.setChecked(checked)
            self.chk_show_radar.blockSignals(False)
        cfg = getattr(self.engine.ai_broadcast, 'config', None)
        if cfg:
            cfg.enable_radar = checked

    def on_radar_param_changed(self):
        show_r = self.chk_show_radar.isChecked() if hasattr(self, 'chk_show_radar') else True
        pos = self.combo_radar_pos.currentData() if hasattr(self, 'combo_radar_pos') else "top_right"
        w = int(self.ctrl_radar_width.value()) if hasattr(self, 'ctrl_radar_width') else 320
        op = float(self.ctrl_radar_opacity.value()) if hasattr(self, 'ctrl_radar_opacity') else 0.85

        self.viewport.set_radar_enabled(show_r)
        self.viewport.set_radar_position(pos)
        self.viewport.set_radar_width(w)
        self.viewport.set_radar_opacity(op)

        if hasattr(self, 'chk_show_radar_menu'):
            self.chk_show_radar_menu.blockSignals(True)
            self.chk_show_radar_menu.setChecked(show_r)
            self.chk_show_radar_menu.blockSignals(False)

        cfg = getattr(self.engine.ai_broadcast, 'config', None)
        if cfg:
            cfg.enable_radar = show_r
            cfg.radar_position = pos
            cfg.radar_width = w
            cfg.radar_opacity = op

    def toggle_offside_line(self, checked: bool):
        if hasattr(self.viewport, 'set_offside_line_enabled'):
            self.viewport.set_offside_line_enabled(checked)
        if hasattr(self, 'chk_show_offside'):
            self.chk_show_offside.blockSignals(True)
            self.chk_show_offside.setChecked(checked)
            self.chk_show_offside.blockSignals(False)
        if hasattr(self, 'chk_show_offside_menu'):
            self.chk_show_offside_menu.blockSignals(True)
            self.chk_show_offside_menu.setChecked(checked)
            self.chk_show_offside_menu.blockSignals(False)
        cfg = getattr(self.engine.ai_broadcast, 'config', None)
        if cfg:
            cfg.enable_offside_line = checked
        self.viewport.update()

    def toggle_skeletons(self, checked: bool):
        if hasattr(self.viewport, 'set_skeletons_enabled'):
            self.viewport.set_skeletons_enabled(checked)
        if hasattr(self, 'chk_show_skeletons'):
            self.chk_show_skeletons.blockSignals(True)
            self.chk_show_skeletons.setChecked(checked)
            self.chk_show_skeletons.blockSignals(False)
        if hasattr(self, 'chk_show_skeletons_menu'):
            self.chk_show_skeletons_menu.blockSignals(True)
            self.chk_show_skeletons_menu.setChecked(checked)
            self.chk_show_skeletons_menu.blockSignals(False)
        cfg = getattr(self.engine.ai_broadcast, 'config', None)
        if cfg:
            cfg.enable_skeletons = checked
        self.viewport.update()

    def toggle_speed_badges(self, checked: bool):
        if hasattr(self.viewport, 'set_speed_badges_enabled'):
            self.viewport.set_speed_badges_enabled(checked)
        if hasattr(self, 'chk_show_speed'):
            self.chk_show_speed.blockSignals(True)
            self.chk_show_speed.setChecked(checked)
            self.chk_show_speed.blockSignals(False)
        if hasattr(self, 'chk_show_speed_menu'):
            self.chk_show_speed_menu.blockSignals(True)
            self.chk_show_speed_menu.setChecked(checked)
            self.chk_show_speed_menu.blockSignals(False)
        cfg = getattr(self.engine.ai_broadcast, 'config', None)
        if cfg:
            cfg.enable_speed_badges = checked
        self.viewport.update()

    def toggle_sideline_filter(self, checked: bool):
        if hasattr(self, 'chk_sideline_filter'):
            self.chk_sideline_filter.blockSignals(True)
            self.chk_sideline_filter.setChecked(checked)
            self.chk_sideline_filter.blockSignals(False)
        if hasattr(self, 'chk_sideline_menu'):
            self.chk_sideline_menu.blockSignals(True)
            self.chk_sideline_menu.setChecked(checked)
            self.chk_sideline_menu.blockSignals(False)
        if hasattr(self, 'ctrl_pitch_margin'):
            self.ctrl_pitch_margin.setVisible(checked)
        cfg = getattr(self.engine.ai_broadcast, 'config', None)
        if cfg:
            cfg.enable_sideline_filter = checked
        self.refresh_preview()

    def open_tactics_dashboard(self):
        """Opens the full interactive tactics and scouting dashboard."""
        ai_tracker = getattr(self.engine, 'ai_broadcast', None)
        if not ai_tracker or not hasattr(ai_tracker, 'get_tactics_summary'):
            QMessageBox.warning(self, "Taktik-Dashboard", "Keine KI-Tracking-Daten verfügbar.")
            return

        summary = ai_tracker.get_tactics_summary()
        dlg = TacticsDashboardDialog(summary, parent=self)
        dlg.exec()

    def on_burn_radar_toggled(self, checked: bool):
        if hasattr(self, 'chk_burn_radar'):
            self.chk_burn_radar.blockSignals(True)
            self.chk_burn_radar.setChecked(checked)
            self.chk_burn_radar.blockSignals(False)
        if hasattr(self, 'chk_exp_burn_radar'):
            self.chk_exp_burn_radar.blockSignals(True)
            self.chk_exp_burn_radar.setChecked(checked)
            self.chk_exp_burn_radar.blockSignals(False)

        cfg = getattr(self.engine.ai_broadcast, 'config', None)
        if cfg:
            cfg.burn_radar_in_export = checked

    def on_follow_ball_toggled(self, checked):
        self.engine.autocam.config.follow_ball = checked
        self.refresh_preview()

    def show_landing_page(self):
        """Switches to the Workflow Hub / Start-Zentrale Landingpage."""
        if hasattr(self, 'main_stack'):
            self.main_stack.setCurrentIndex(0)
        self._update_nav_highlight("home")
        if hasattr(self, 'status_bar'):
            self.status_bar.clearMessage()

    def set_stitching_subpage(self, index: int):
        """Switches between Media/Sync, Calibration and Export inside the combined Stitching tab."""
        if hasattr(self, 'stitch_sub_stack'):
            self.stitch_sub_stack.setCurrentIndex(index)
        btns = [getattr(self, 'btn_stitch_sub_media', None),
                getattr(self, 'btn_stitch_sub_calib', None),
                getattr(self, 'btn_stitch_sub_export', None)]
        for i, b in enumerate(btns):
            if b:
                if i == index:
                    b.setStyleSheet("background-color: #2563eb; color: #ffffff; font-weight: 700; font-size: 11px; padding: 6px 12px; border-radius: 6px; border: none;")
                else:
                    b.setStyleSheet("background-color: transparent; color: #94a3b8; font-weight: 600; font-size: 11px; padding: 6px 12px; border-radius: 6px; border: none;")

    def set_wide_angle_subpage(self, index: int):
        """Switches between Begradigung/Kalibrierung and Export inside the Weitwinkel tab."""
        if hasattr(self, 'wa_sub_stack'):
            self.wa_sub_stack.setCurrentIndex(index)
        btns = [getattr(self, 'btn_wa_sub_calib', None),
                getattr(self, 'btn_wa_sub_export', None)]
        for i, b in enumerate(btns):
            if b:
                if i == index:
                    b.setStyleSheet("background-color: #2563eb; color: #ffffff; font-weight: 700; font-size: 11px; padding: 6px 12px; border-radius: 6px; border: none;")
                else:
                    b.setStyleSheet("background-color: transparent; color: #94a3b8; font-weight: 600; font-size: 11px; padding: 6px 12px; border-radius: 6px; border: none;")

    def set_autocam_subpage(self, index: int):
        """Switches between Regie/KI-Tracking and Export inside the KI-Broadcast tab."""
        if hasattr(self, 'ai_sub_stack'):
            self.ai_sub_stack.setCurrentIndex(index)
        btns = [getattr(self, 'btn_ai_sub_main', None),
                getattr(self, 'btn_ai_sub_export', None)]
        for i, b in enumerate(btns):
            if b:
                if i == index:
                    b.setStyleSheet("background-color: #2563eb; color: #ffffff; font-weight: 700; font-size: 11px; padding: 6px 12px; border-radius: 6px; border: none;")
                else:
                    b.setStyleSheet("background-color: transparent; color: #94a3b8; font-weight: 600; font-size: 11px; padding: 6px 12px; border-radius: 6px; border: none;")

    def show_workspace(self, tab_identifier: Any = 0):
        """Switches to the interactive workspace and activates designated workflow tab."""
        tab_map = {
            "wizard": 0, 0: 0,
            "stitch": 1, 1: 1,
            "wide": 2, 2: 2,
            "ai": 0,      # Legacy 'ai' tab now maps directly to Autocam 5-Schritte Assistent (Tab 0)
            "export": 3, 3: 3,
            "live": 4, 4: 4,
            "cloud": 5, 5: 5,
        }
        tab_idx = tab_map.get(tab_identifier, 0)
        if hasattr(self, 'main_stack'):
            self.main_stack.setCurrentIndex(1)
        if hasattr(self, 'right_pane') and 0 <= tab_idx < self.right_pane.count():
            self.right_pane.setCurrentIndex(tab_idx)
        self._on_tab_changed(tab_idx)

    def switch_to_step(self, index: int):
        """Legacy helper to switch tab and display workspace."""
        self.show_workspace(index)

    def _on_tab_changed(self, index: int):
        """Handles sidebar tab switching and updates top step navigation pills."""
        tab_key_map = {
            0: "wizard",
            1: "stitch",
            2: "wide",
            3: "export",
            4: "live",
            5: "cloud",
        }
        self._update_nav_highlight(tab_key_map.get(index, "wizard"))

        if index == 0:  # Autocam Assistent
            if hasattr(self, 'tab_stepper'):
                self.tab_stepper.go_to_step(self.tab_stepper.current_step)
        elif index == 2:  # Weitwinkel tab
            if hasattr(self, 'viewport'):
                self.viewport.set_wide_angle_edit_mode(True)
            if hasattr(self, 'combo_view_mode'):
                cur_vm = self.combo_view_mode.currentData()
                if cur_vm not in ("wide_angle", "wide_angle_raw"):
                    self.set_view_mode_by_id("wide_angle")
                self.refresh_preview()
        else:
            if hasattr(self, 'combo_view_mode'):
                cur_vm = self.combo_view_mode.currentData()
                if cur_vm not in ("wide_angle", "wide_angle_raw") and hasattr(self, 'viewport'):
                    self.viewport.set_wide_angle_edit_mode(False)

        if index == 4:  # Live-Stream tab (now at index 4)
            if hasattr(self, 'combo_view_mode') and self.combo_view_mode.findData("grid") >= 0:
                self.set_view_mode_by_id("grid")
            if hasattr(self, 'live_capture_manager') and not self.live_capture_manager.is_running:
                mode = self.combo_live_proc_mode.currentData() if hasattr(self, 'combo_live_proc_mode') else "stitching_dual"
                standby_grid = self._build_multiview_grid(None, None, None, None, mode=mode)
                if hasattr(self, 'viewport'):
                    self.viewport.update_preview_image(standby_grid)

    def _update_nav_highlight(self, active_key: str):
        """Updates highlight styling for top navigation pills."""
        pills = {
            "wizard": getattr(self, "btn_nav_wizard", None),
            "home": getattr(self, "btn_nav_home", None),
            "stitch": getattr(self, "btn_nav_stitch", None),
            "wide": getattr(self, "btn_nav_wide", None),
            "export": getattr(self, "btn_nav_export", None),
            "live": getattr(self, "btn_nav_live", None),
            "cloud": getattr(self, "btn_nav_cloud", None),
        }
        for key, btn in pills.items():
            if btn is None:
                continue
            if key == active_key:
                if key == "wizard":
                    btn.setStyleSheet(
                        "background-color: #10b981; color: #ffffff; border: none; border-radius: 6px; "
                        "font-weight: 800; font-size: 11px; padding: 5px 12px;"
                    )
                else:
                    btn.setStyleSheet(
                        "background-color: #2563eb; border: none; border-radius: 6px; "
                        "font-weight: 700; font-size: 11px; padding: 5px 12px; color: #ffffff;"
                    )
            else:
                btn.setStyleSheet(
                    "background-color: transparent; border: none; border-radius: 6px; "
                    "font-weight: 700; font-size: 11px; padding: 5px 12px; color: #94a3b8;"
                )

    def _launch_workflow_stitching(self, fmt: str = "4:3"):
        """Launches stitching workflow with preselected 4:3 or 16:9 profile."""
        if fmt == "4:3":
            if hasattr(self, 'radio_src_4_3'):
                self.radio_src_4_3.setChecked(True)
            self._on_source_format_radio_changed(True)
        else:
            if hasattr(self, 'radio_src_16_9'):
                self.radio_src_16_9.setChecked(True)
            self._on_source_format_radio_changed(True)
        self.show_workspace(1)

    def _on_source_format_radio_changed(self, checked: bool = True):
        """Applies presets according to selected camera source aspect ratio."""
        if not checked:
            return
        if hasattr(self, 'radio_src_4_3') and self.radio_src_4_3.isChecked():
            if hasattr(self, 'lbl_src_format_desc'):
                self.lbl_src_format_desc.setText("💡 Ausgabeprofil: 8:3 Taktik-Panorama (2560x960) | DJI RockSteady Dewarp aktiv.")
                self.lbl_src_format_desc.setStyleSheet("color: #38bdf8; font-size: 11px; font-weight: 600;")
            self._load_rig_json_silently("dji_action4_1080p_4x3_dewarp_rocksteady.json")
            self.set_view_mode_by_id("8:3")
            if hasattr(self, 'combo_exp_format'):
                idx = self.combo_exp_format.findData("8:3")
                if idx >= 0:
                    self.combo_exp_format.setCurrentIndex(idx)
            if hasattr(self, 'status_bar'):
                self.status_bar.showMessage("Profil vorgewählt: 4:3 DJI Action 4 (8:3 Taktik-Panorama)")
        elif hasattr(self, 'radio_src_16_9') and self.radio_src_16_9.isChecked():
            if hasattr(self, 'lbl_src_format_desc'):
                self.lbl_src_format_desc.setText("💡 Ausgabeprofil: 32:9 Ultra-Panorama (3840x1080) | Standard 16:9 Rigs.")
                self.lbl_src_format_desc.setStyleSheet("color: #34d399; font-size: 11px; font-weight: 600;")
            self._load_rig_json_silently("default_rig_action4_80deg.json")
            self.set_view_mode_by_id("32:9")
            if hasattr(self, 'combo_exp_format'):
                idx = self.combo_exp_format.findData("32:9")
                if idx >= 0:
                    self.combo_exp_format.setCurrentIndex(idx)
            if hasattr(self, 'status_bar'):
                self.status_bar.showMessage("Profil vorgewählt: 16:9 Standard-Cams (32:9 Ultra-Panorama)")

    def _load_rig_json_silently(self, file_path: str) -> bool:
        """Loads a rig profile JSON silently without popup dialogs."""
        resolved_path = file_path
        if not os.path.exists(resolved_path):
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            alt_path = os.path.join(base_dir, file_path)
            if os.path.exists(alt_path):
                resolved_path = alt_path
            else:
                logger.warning(f"Rig profile file not found: {file_path}")
                return False
        try:
            with open(resolved_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.apply_settings_dict(data)
            logger.info(f"Rig profile silently loaded from {resolved_path}")
            return True
        except Exception as e:
            logger.error(f"Error loading rig JSON {resolved_path}: {e}")
            return False

    @property
    def view_mode(self) -> str:
        """Returns the current active viewport presentation mode (e.g. '32:9', '8:3', '16:9', '16:9_tactical')."""
        if hasattr(self, 'combo_view_mode') and self.combo_view_mode is not None:
            data = self.combo_view_mode.currentData()
            if data:
                return str(data)
        if hasattr(self, 'viewport') and hasattr(self.viewport, 'view_mode') and self.viewport.view_mode:
            return str(self.viewport.view_mode)
        return getattr(self, '_view_mode', "32:9")

    @view_mode.setter
    def view_mode(self, mode: str):
        self._view_mode = mode
        self.set_view_mode_by_id(mode)

    def set_view_mode_by_id(self, mode_id: str):
        """Switches the viewport presentation mode by identifier (e.g. '32:9', '16:9_tactical', '16:9')."""
        if not hasattr(self, 'combo_view_mode'):
            return
        idx = self.combo_view_mode.findData(mode_id)
        if idx >= 0:
            if self.combo_view_mode.currentIndex() != idx:
                self.combo_view_mode.setCurrentIndex(idx)
            else:
                self.on_view_mode_changed()

    def toggle_tactical_view(self, checked: bool):
        """Quick 1-click toggle between 32:9/8:3 panorama overview and rectified 16:9 tactical field view."""
        target_mode = "16:9_tactical" if checked else ("8:3" if self.is_4_3_mode() else "32:9")
        self.set_view_mode_by_id(target_mode)

    def on_view_mode_changed(self):
        mode = self.combo_view_mode.currentData()
        self.viewport.view_mode = mode
        if hasattr(self, 'btn_toggle_tactical_view') and self.btn_toggle_tactical_view is not None:
            self.btn_toggle_tactical_view.blockSignals(True)
            self.btn_toggle_tactical_view.setChecked(mode == "16:9_tactical")
            self.btn_toggle_tactical_view.blockSignals(False)
        if mode == "16:9_tactical":
            self.status_bar.showMessage("📐 16:9 Taktik-Warp aktiv: Ziehen Sie die 4 Ecken & 2 Mittellinien-Punkte direkt im Bild, um das Video live zu verzerren!")
        elif mode in ("wide_angle", "wide_angle_raw"):
            self.viewport.set_wide_angle_edit_mode(True)
            self.status_bar.showMessage("📐 Weitwinkel-Entzerrung aktiv: Setzen oder ziehen Sie die 4 Spielfeldecken direkt im Bild!")
        self.refresh_preview()

    def zoom_to_seam(self):
        self.viewport.zoom_level = 2.2
        self.viewport.pan_offset = QPoint(0, 0)
        self.viewport.update()

    def toggle_sync_angles(self, checked):
        self.sync_angles = checked

    def on_master_yaw_changed(self, total_yaw: float):
        half = total_yaw * 0.5
        self.ctrl_left_yaw.blockSignals(True)
        self.ctrl_right_yaw.blockSignals(True)
        self.ctrl_left_yaw.setValue(-half)
        self.ctrl_right_yaw.setValue(half)
        self.ctrl_left_yaw.blockSignals(False)
        self.ctrl_right_yaw.blockSignals(False)

        self.rig.left_pose.yaw = -half
        self.rig.right_pose.yaw = half
        self.engine.invalidate_luts()
        self.trigger_smooth_preview()

    def on_left_yaw_changed(self, val: float):
        if self.sync_angles:
            self.ctrl_right_yaw.blockSignals(True)
            self.ctrl_right_yaw.setValue(abs(val))
            self.ctrl_right_yaw.blockSignals(False)
            self.rig.right_pose.yaw = abs(val)
            self.ctrl_master_yaw.blockSignals(True)
            self.ctrl_master_yaw.setValue(abs(val) * 2.0)
            self.ctrl_master_yaw.blockSignals(False)
        
        self.rig.left_pose.yaw = val
        self.engine.invalidate_luts()
        self.trigger_smooth_preview()

    def on_right_yaw_changed(self, val: float):
        if self.sync_angles:
            self.ctrl_left_yaw.blockSignals(True)
            self.ctrl_left_yaw.setValue(-abs(val))
            self.ctrl_left_yaw.blockSignals(False)
            self.rig.left_pose.yaw = -abs(val)
            self.ctrl_master_yaw.blockSignals(True)
            self.ctrl_master_yaw.setValue(abs(val) * 2.0)
            self.ctrl_master_yaw.blockSignals(False)
            
        self.rig.right_pose.yaw = val
        self.engine.invalidate_luts()
        self.trigger_smooth_preview()

    def on_cam_fov_changed(self, val: float):
        self.rig.left_camera.set_fov(val)
        self.rig.right_camera.set_fov(val)
        self.engine.invalidate_luts()
        self.trigger_smooth_preview()

    def on_preset_changed(self, val: Any = None):
        key = None
        if hasattr(self, 'combo_preset'):
            data = self.combo_preset.currentData()
            if data and data in CAMERA_PRESETS:
                key = data
        if not key and isinstance(val, str):
            if val in CAMERA_PRESETS:
                key = val
            elif hasattr(self, 'combo_preset'):
                txt = self.combo_preset.currentText()
                for k, p in CAMERA_PRESETS.items():
                    if k in txt or p.name in txt:
                        key = k
                        break

        if key and key in CAMERA_PRESETS:
            preset = CAMERA_PRESETS[key]
            self.rig.left_camera = CameraIntrinsics.from_dict(preset.to_dict())
            self.rig.right_camera = CameraIntrinsics.from_dict(preset.to_dict())
            if hasattr(self, 'ctrl_cam_fov'):
                self.ctrl_cam_fov.blockSignals(True)
                self.ctrl_cam_fov.setValue(preset.hfov_deg)
                self.ctrl_cam_fov.blockSignals(False)
            if hasattr(self, 'combo_preset'):
                idx = self.combo_preset.findData(key)
                if idx >= 0 and self.combo_preset.currentIndex() != idx:
                    self.combo_preset.blockSignals(True)
                    self.combo_preset.setCurrentIndex(idx)
                    self.combo_preset.blockSignals(False)
            self.engine.invalidate_luts()
            self._update_format_combo_labels()
            self.update_resolution_dropdown()
            self.refresh_preview()
            if hasattr(self, 'status_bar'):
                self.status_bar.showMessage(f"📷 Profil aktiviert: {preset.name} ({preset.image_width}x{preset.image_height}, HFOV {preset.hfov_deg}°)")

    def on_curvature_changed(self, val: float):
        """Adjusts the field line curvature compensation to straighten ground lines."""
        self.rig.field_curvature_correction = float(val)
        self.engine.invalidate_luts()
        self.trigger_smooth_preview()

    def on_projection_changed(self, idx: int):
        """Switches panorama projection between cylindrical, pannini, and rectilinear."""
        if hasattr(self, 'combo_projection'):
            proj = self.combo_projection.currentData()
            if proj:
                self.rig.projection_type = proj
                self.engine.invalidate_luts()
                self.trigger_smooth_preview()

    def on_auto_color_match_toggled(self, checked: bool):
        """Toggles automated color and white-balance matching between left and right cameras."""
        self.rig.auto_color_match = checked
        if hasattr(self, 'engine'):
            self.engine.color_matcher.enabled = checked
            if not checked:
                self.engine.color_matcher.reset()
        self.trigger_smooth_preview()

    def on_color_match_strength_changed(self, val: float):
        """Adjusts the intensity of the automatic color matching (0% = none, 100% = full)."""
        strength = float(val) / 100.0
        self.rig.color_match_strength = strength
        if hasattr(self, 'engine'):
            self.engine.color_matcher.set_strength(strength)
        self.trigger_smooth_preview()

    def on_rig_param_changed(self):
        base_pitch = self.ctrl_cam_pitch.value()
        pitch_diff = self.ctrl_pitch_diff.value() if hasattr(self, 'ctrl_pitch_diff') else 0.0
        self.rig.left_pose.pitch = base_pitch
        self.rig.right_pose.pitch = base_pitch + pitch_diff
        self.rig.left_pose.roll = self.ctrl_left_roll.value()
        self.rig.right_pose.roll = self.ctrl_right_roll.value()
        if hasattr(self, 'ctrl_seam_pos'):
            self.rig.seam_angle_offset = self.ctrl_seam_pos.value()
        if hasattr(self, 'ctrl_bottom_taper_x'):
            self.rig.bottom_seam_taper_x = self.ctrl_bottom_taper_x.value()
        if hasattr(self, 'ctrl_bottom_taper_y'):
            self.rig.bottom_seam_taper_y = self.ctrl_bottom_taper_y.value()
        if hasattr(self, 'chk_smart_seam'):
            self.rig.smart_seam_enabled = self.chk_smart_seam.isChecked()

        self.rig.global_pitch_correction = self.ctrl_global_pitch.value()
        self.rig.global_yaw_center = self.ctrl_h_offset.value()
        self.rig.pano_hfov = self.ctrl_pano_hfov.value()
        self.rig.vertical_crop_offset = self.ctrl_v_offset.value()
        self.rig.horizontal_squeeze = self.ctrl_squeeze.value()
        self.rig.blend_width_deg = self.ctrl_blend_width.value()
        if hasattr(self, 'chk_auto_color_match'):
            self.rig.auto_color_match = self.chk_auto_color_match.isChecked()
            self.engine.color_matcher.enabled = self.rig.auto_color_match
        if hasattr(self, 'ctrl_color_match_strength'):
            self.rig.color_match_strength = self.ctrl_color_match_strength.value() / 100.0
            self.engine.color_matcher.set_strength(self.rig.color_match_strength)
        if hasattr(self, 'ctrl_curvature'):
            self.rig.field_curvature_correction = self.ctrl_curvature.value()
        if hasattr(self, 'combo_projection'):
            proj = self.combo_projection.currentData()
            if proj:
                self.rig.projection_type = proj

        self.engine.invalidate_luts()
        self.trigger_smooth_preview()

    def on_smart_seam_toggled(self, checked: bool):
        self.rig.smart_seam_enabled = checked
        self.engine.invalidate_luts()
        if checked and hasattr(self, 'current_frame_idx') and self.engine.video_left and self.engine.video_right:
            idx_l = self.current_frame_idx
            idx_r = self.current_frame_idx - self.engine.frame_offset_right
            fl = self.engine.video_left.get_frame(idx_l)
            fr = self.engine.video_right.get_frame(idx_r)
            if fl is not None and fr is not None:
                pw, ph = (1920, 720) if self.is_4_3_mode() else (2560, 720)
                luts = self.engine.update_luts(pw, ph)
                luts.update_smart_seam(fl, fr, self.rig)
                self.viewport.luts_ref = luts
        self.update_preview(self.current_frame_idx if hasattr(self, 'current_frame_idx') else 0)

    def on_pick_center_toggled(self, checked: bool):
        """Activates interactive click-to-center mode on the panorama preview."""
        if hasattr(self, 'btn_pick_center_tb') and self.btn_pick_center_tb is not None and self.btn_pick_center_tb.isChecked() != checked:
            self.btn_pick_center_tb.blockSignals(True)
            self.btn_pick_center_tb.setChecked(checked)
            self.btn_pick_center_tb.blockSignals(False)

        if hasattr(self, 'btn_pick_center_calib') and self.btn_pick_center_calib.isChecked() != checked:
            self.btn_pick_center_calib.blockSignals(True)
            self.btn_pick_center_calib.setChecked(checked)
            self.btn_pick_center_calib.blockSignals(False)

        self.viewport.set_pick_center_mode(checked)
        if checked:
            self.status_bar.showMessage("📍 Klicken Sie in der Vorschau auf die Mittellinie / den Anstoßpunkt zum automatischen Zentrieren...")
        else:
            self.status_bar.showMessage("Bereit")

    def on_pitch_center_picked(self, norm_x: float):
        """Calculates exact yaw angle from clicked pixel and centers the soccer pitch symmetrically."""
        # norm_x is in [0.0 ... 1.0] where 0.5 is current center
        angle_offset = (norm_x - 0.5) * self.rig.pano_hfov
        new_yaw = float(np.clip(self.rig.global_yaw_center + angle_offset, -60.0, 60.0))
        
        self.ctrl_h_offset.setValue(new_yaw)
        self.on_pick_center_toggled(False)
        self.status_bar.showMessage(f"🎯 Spielfeld-Mitte ausgerichtet (Versatz: {new_yaw:+.1f}°) – Beide Tore sind jetzt im Bild!")

    def toggle_center_line(self, checked: bool):
        self.viewport.show_center_line = checked
        self.viewport.update()

    def pan_offset_by(self, delta_deg: float):
        new_yaw = float(np.clip(self.ctrl_h_offset.value() + delta_deg, -60.0, 60.0))
        self.ctrl_h_offset.setValue(new_yaw)

    def shift_framing(self, dh_deg: float, dv_shift: float):
        """Nudges the panorama framing horizontally (yaw pan) and vertically (shift)."""
        new_h = float(np.clip(self.ctrl_h_offset.value() + dh_deg, -60.0, 60.0))
        new_v = float(np.clip(self.ctrl_v_offset.value() + dv_shift, -2.0, 2.0))
        self.ctrl_h_offset.setValue(new_h)
        self.ctrl_v_offset.setValue(new_v)

    def reset_framing(self):
        """Resets framing to default center."""
        self.ctrl_h_offset.setValue(0.0)
        self.ctrl_v_offset.setValue(0.12)


    def apply_full_pitch_preset(self):
        """Applies full pitch 145° HFOV preset with near-field bottom corner focus and anamorphic squeeze."""
        self.ctrl_pano_hfov.blockSignals(True)
        self.ctrl_h_offset.blockSignals(True)
        self.ctrl_v_offset.blockSignals(True)
        self.ctrl_squeeze.blockSignals(True)
        self.ctrl_safety_margin.blockSignals(True)
        
        self.ctrl_pano_hfov.setValue(145.0)
        self.ctrl_h_offset.setValue(0.0)
        self.ctrl_v_offset.setValue(0.12)
        self.ctrl_squeeze.setValue(1.15)
        self.ctrl_safety_margin.setValue(0.5)
        self.chk_auto_crop.setChecked(True)
        
        self.ctrl_pano_hfov.blockSignals(False)
        self.ctrl_h_offset.blockSignals(False)
        self.ctrl_v_offset.blockSignals(False)
        self.ctrl_squeeze.blockSignals(False)
        self.ctrl_safety_margin.blockSignals(False)
        
        self.rig.pano_hfov = 145.0
        self.rig.global_yaw_center = 0.0
        self.rig.vertical_crop_offset = 0.12
        self.rig.horizontal_squeeze = 1.15
        self.rig.lir_safety_margin = 0.005
        self.rig.auto_crop_lir = True
        
        self.engine.invalidate_luts()
        self._update_format_combo_labels()
        self.update_resolution_dropdown()
        self.refresh_preview()
        self.status_bar.showMessage("🏟️ Preset aktiviert: 145° Weitwinkel-Panorama + Breitenstauchung für volle Spielfeld- & Nah-Ecken-Abdeckung!")

    def apply_action4_4x3_rocksteady_preset(self):
        """Applies complete stitching preset for DJI Action 4 1080p 4:3 Dewarp (RockSteady ON, 30fps)."""
        preset_key = "DJI_Action4_1080p_4_3_Dewarp_RockSteady"
        if preset_key not in CAMERA_PRESETS:
            return
        preset = CAMERA_PRESETS[preset_key]
        self.rig.left_camera = CameraIntrinsics.from_dict(preset.to_dict())
        self.rig.right_camera = CameraIntrinsics.from_dict(preset.to_dict())

        # Rig poses optimized for 82.0° FOV cameras (~9° central overlap)
        self.rig.left_pose.yaw = -36.5
        self.rig.left_pose.pitch = -2.0
        self.rig.left_pose.roll = 0.0
        self.rig.left_pose.reset_corners()

        self.rig.right_pose.yaw = 36.5
        self.rig.right_pose.pitch = -2.0
        self.rig.right_pose.roll = 0.0
        self.rig.right_pose.reset_corners()

        self.rig.global_pitch_correction = 2.0
        self.rig.global_roll_correction = 0.0
        self.rig.global_yaw_center = 0.0
        self.rig.pano_hfov = 142.0
        self.rig.projection_type = "cylindrical"
        self.rig.field_curvature_correction = 0.0
        self.rig.blend_width_deg = 4.0
        self.rig.seam_angle_offset = 0.0
        self.rig.bottom_seam_taper_x = 0.0
        self.rig.bottom_seam_taper_y = 0.0
        self.rig.smart_seam_enabled = False
        self.rig.auto_color_match = True
        self.rig.color_match_strength = 1.0
        self.rig.vertical_crop_offset = 0.0
        self.rig.horizontal_squeeze = 0.85
        self.rig.auto_crop_lir = True
        self.rig.lir_safety_margin = 0.005

        # Update UI Controls safely
        for ctrl in [
            self.ctrl_left_yaw, self.ctrl_right_yaw, self.ctrl_master_yaw,
            self.ctrl_cam_pitch, self.ctrl_left_roll, self.ctrl_right_roll,
            self.ctrl_global_pitch, self.ctrl_h_offset, self.ctrl_pano_hfov,
            self.ctrl_v_offset, self.ctrl_squeeze, self.ctrl_blend_width,
            self.ctrl_cam_fov, self.ctrl_safety_margin, self.chk_auto_crop
        ]:
            ctrl.blockSignals(True)

        self.ctrl_left_yaw.setValue(-36.5)
        self.ctrl_right_yaw.setValue(36.5)
        self.ctrl_master_yaw.setValue(73.0)
        self.ctrl_cam_pitch.setValue(-2.0)
        self.ctrl_left_roll.setValue(0.0)
        self.ctrl_right_roll.setValue(0.0)
        self.ctrl_global_pitch.setValue(2.0)
        self.ctrl_h_offset.setValue(0.0)
        self.ctrl_pano_hfov.setValue(142.0)
        self.ctrl_v_offset.setValue(0.0)
        self.ctrl_squeeze.setValue(0.85)
        self.ctrl_blend_width.setValue(4.0)
        self.ctrl_cam_fov.setValue(82.0)
        self.ctrl_safety_margin.setValue(0.5)
        self.chk_auto_crop.setChecked(True)
        if hasattr(self, 'ctrl_pitch_diff'):
            self.ctrl_pitch_diff.setValue(0.0)
        if hasattr(self, 'ctrl_seam_pos'):
            self.ctrl_seam_pos.setValue(0.0)
        if hasattr(self, 'ctrl_bottom_taper_x'):
            self.ctrl_bottom_taper_x.setValue(0.0)
        if hasattr(self, 'ctrl_bottom_taper_y'):
            self.ctrl_bottom_taper_y.setValue(0.0)
        if hasattr(self, 'chk_smart_seam'):
            self.chk_smart_seam.setChecked(False)

        for ctrl in [
            self.ctrl_left_yaw, self.ctrl_right_yaw, self.ctrl_master_yaw,
            self.ctrl_cam_pitch, self.ctrl_left_roll, self.ctrl_right_roll,
            self.ctrl_global_pitch, self.ctrl_h_offset, self.ctrl_pano_hfov,
            self.ctrl_v_offset, self.ctrl_squeeze, self.ctrl_blend_width,
            self.ctrl_cam_fov, self.ctrl_safety_margin, self.chk_auto_crop
        ]:
            ctrl.blockSignals(False)

        if hasattr(self, 'combo_preset'):
            idx = self.combo_preset.findData(preset_key)
            if idx >= 0:
                self.combo_preset.blockSignals(True)
                self.combo_preset.setCurrentIndex(idx)
                self.combo_preset.blockSignals(False)

        if hasattr(self, 'combo_projection'):
            self.combo_projection.blockSignals(True)
            idx = self.combo_projection.findData("cylindrical")
            if idx >= 0:
                self.combo_projection.setCurrentIndex(idx)
            self.combo_projection.blockSignals(False)

        self.update_corner_spinboxes_from_rig()
        self.engine.invalidate_luts()
        self._update_format_combo_labels()
        self.update_resolution_dropdown()
        self.refresh_preview()
        self.status_bar.showMessage("📹 Preset aktiviert: DJI Action 4 1080p 4:3 Dewarp (RockSteady ON, 30fps) - 8:3 Live-Vorschau & Rig optimiert!")

    def apply_action4_1080p_16x9_dewarp_preset(self):
        """Applies complete stitching preset for DJI Action 4 1080p (16:9) Dewarp Standard (60fps)."""
        preset_key = "DJI_Action4_1080p_16_9_Dewarp"
        if preset_key not in CAMERA_PRESETS:
            preset_key = "DJI_Action4_1080p_Dewarp"
        if preset_key not in CAMERA_PRESETS:
            return
        preset = CAMERA_PRESETS[preset_key]
        self.rig.left_camera = CameraIntrinsics.from_dict(preset.to_dict())
        self.rig.right_camera = CameraIntrinsics.from_dict(preset.to_dict())

        # Dual-Rig poses optimized for 16:9 92.0° FOV cameras (~10° central overlap at 77° spread)
        self.rig.left_pose.yaw = -38.5
        self.rig.left_pose.pitch = -2.0
        self.rig.left_pose.roll = 0.0
        self.rig.left_pose.reset_corners()

        self.rig.right_pose.yaw = 38.5
        self.rig.right_pose.pitch = -2.0
        self.rig.right_pose.roll = 0.0
        self.rig.right_pose.reset_corners()

        self.rig.global_pitch_correction = 2.0
        self.rig.global_roll_correction = 0.0
        self.rig.global_yaw_center = 0.0
        self.rig.pano_hfov = 154.0
        self.rig.projection_type = "cylindrical"
        self.rig.field_curvature_correction = 0.0
        self.rig.blend_width_deg = 3.5
        self.rig.seam_angle_offset = 0.0
        self.rig.bottom_seam_taper_x = 0.0
        self.rig.bottom_seam_taper_y = 0.0
        self.rig.smart_seam_enabled = False
        self.rig.auto_color_match = True
        self.rig.color_match_strength = 1.0
        self.rig.vertical_crop_offset = -0.12
        self.rig.horizontal_squeeze = 0.52
        self.rig.auto_crop_lir = True
        self.rig.lir_safety_margin = 0.005

        for ctrl in [
            self.ctrl_left_yaw, self.ctrl_right_yaw, self.ctrl_master_yaw,
            self.ctrl_cam_pitch, self.ctrl_left_roll, self.ctrl_right_roll,
            self.ctrl_global_pitch, self.ctrl_h_offset, self.ctrl_pano_hfov,
            self.ctrl_v_offset, self.ctrl_squeeze, self.ctrl_blend_width,
            self.ctrl_cam_fov, self.ctrl_safety_margin, self.chk_auto_crop
        ]:
            ctrl.blockSignals(True)

        self.ctrl_left_yaw.setValue(-38.5)
        self.ctrl_right_yaw.setValue(38.5)
        self.ctrl_master_yaw.setValue(77.0)
        self.ctrl_cam_pitch.setValue(-2.0)
        self.ctrl_left_roll.setValue(0.0)
        self.ctrl_right_roll.setValue(0.0)
        self.ctrl_global_pitch.setValue(2.0)
        self.ctrl_h_offset.setValue(0.0)
        self.ctrl_pano_hfov.setValue(154.0)
        self.ctrl_v_offset.setValue(-0.12)
        self.ctrl_squeeze.setValue(0.52)
        self.ctrl_blend_width.setValue(3.5)
        self.ctrl_cam_fov.setValue(92.0)
        self.ctrl_safety_margin.setValue(0.5)
        self.chk_auto_crop.setChecked(True)

        for ctrl in [
            self.ctrl_left_yaw, self.ctrl_right_yaw, self.ctrl_master_yaw,
            self.ctrl_cam_pitch, self.ctrl_left_roll, self.ctrl_right_roll,
            self.ctrl_global_pitch, self.ctrl_h_offset, self.ctrl_pano_hfov,
            self.ctrl_v_offset, self.ctrl_squeeze, self.ctrl_blend_width,
            self.ctrl_cam_fov, self.ctrl_safety_margin, self.chk_auto_crop
        ]:
            ctrl.blockSignals(False)

        if hasattr(self, 'combo_preset'):
            idx = self.combo_preset.findData(preset_key)
            if idx >= 0:
                self.combo_preset.blockSignals(True)
                self.combo_preset.setCurrentIndex(idx)
                self.combo_preset.blockSignals(False)

        self.update_corner_spinboxes_from_rig()
        self.engine.invalidate_luts()
        self._update_format_combo_labels()
        self.update_resolution_dropdown()
        self.refresh_preview()
        self.status_bar.showMessage("📹 Preset aktiviert: DJI Action 4 1080p 16:9 Dewarp Standard (60fps)")

    def apply_action4_1080p_16x9_rocksteady_preset(self):
        """Applies complete stitching preset for DJI Action 4 1080p (16:9) Dewarp RockSteady."""
        preset_key = "DJI_Action4_1080p_Dewarp_RockSteady"
        if preset_key not in CAMERA_PRESETS:
            return
        preset = CAMERA_PRESETS[preset_key]
        self.rig.left_camera = CameraIntrinsics.from_dict(preset.to_dict())
        self.rig.right_camera = CameraIntrinsics.from_dict(preset.to_dict())

        self.rig.left_pose.yaw = -36.5
        self.rig.left_pose.pitch = -2.0
        self.rig.left_pose.roll = 0.0
        self.rig.left_pose.reset_corners()

        self.rig.right_pose.yaw = 36.5
        self.rig.right_pose.pitch = -2.0
        self.rig.right_pose.roll = 0.0
        self.rig.right_pose.reset_corners()

        self.rig.global_pitch_correction = 2.0
        self.rig.global_roll_correction = 0.0
        self.rig.global_yaw_center = 0.0
        self.rig.pano_hfov = 146.0
        self.rig.projection_type = "cylindrical"
        self.rig.field_curvature_correction = 0.0
        self.rig.blend_width_deg = 3.5
        self.rig.seam_angle_offset = 0.0
        self.rig.bottom_seam_taper_x = 0.0
        self.rig.bottom_seam_taper_y = 0.0
        self.rig.smart_seam_enabled = False
        self.rig.auto_color_match = True
        self.rig.color_match_strength = 1.0
        self.rig.vertical_crop_offset = -0.12
        self.rig.horizontal_squeeze = 0.52
        self.rig.auto_crop_lir = True
        self.rig.lir_safety_margin = 0.005

        for ctrl in [
            self.ctrl_left_yaw, self.ctrl_right_yaw, self.ctrl_master_yaw,
            self.ctrl_cam_pitch, self.ctrl_left_roll, self.ctrl_right_roll,
            self.ctrl_global_pitch, self.ctrl_h_offset, self.ctrl_pano_hfov,
            self.ctrl_v_offset, self.ctrl_squeeze, self.ctrl_blend_width,
            self.ctrl_cam_fov, self.ctrl_safety_margin, self.chk_auto_crop
        ]:
            ctrl.blockSignals(True)

        self.ctrl_left_yaw.setValue(-36.5)
        self.ctrl_right_yaw.setValue(36.5)
        self.ctrl_master_yaw.setValue(73.0)
        self.ctrl_cam_pitch.setValue(-2.0)
        self.ctrl_left_roll.setValue(0.0)
        self.ctrl_right_roll.setValue(0.0)
        self.ctrl_global_pitch.setValue(2.0)
        self.ctrl_h_offset.setValue(0.0)
        self.ctrl_pano_hfov.setValue(146.0)
        self.ctrl_v_offset.setValue(-0.12)
        self.ctrl_squeeze.setValue(0.52)
        self.ctrl_blend_width.setValue(3.5)
        self.ctrl_cam_fov.setValue(82.0)
        self.ctrl_safety_margin.setValue(0.5)
        self.chk_auto_crop.setChecked(True)

        for ctrl in [
            self.ctrl_left_yaw, self.ctrl_right_yaw, self.ctrl_master_yaw,
            self.ctrl_cam_pitch, self.ctrl_left_roll, self.ctrl_right_roll,
            self.ctrl_global_pitch, self.ctrl_h_offset, self.ctrl_pano_hfov,
            self.ctrl_v_offset, self.ctrl_squeeze, self.ctrl_blend_width,
            self.ctrl_cam_fov, self.ctrl_safety_margin, self.chk_auto_crop
        ]:
            ctrl.blockSignals(False)

        if hasattr(self, 'combo_preset'):
            idx = self.combo_preset.findData(preset_key)
            if idx >= 0:
                self.combo_preset.blockSignals(True)
                self.combo_preset.setCurrentIndex(idx)
                self.combo_preset.blockSignals(False)

        self.update_corner_spinboxes_from_rig()
        self.engine.invalidate_luts()
        self._update_format_combo_labels()
        self.update_resolution_dropdown()
        self.refresh_preview()
        self.status_bar.showMessage("📹 Preset aktiviert: DJI Action 4 1080p 16:9 RockSteady")

    def apply_rig_preset_by_key(self, preset_key: str):
        """Applies rig preset dynamically by preset key string."""
        if preset_key in ("DJI_Action4_1080p_16_9_Dewarp", "DJI_Action4_1080p_Dewarp"):
            self.apply_action4_1080p_16x9_dewarp_preset()
        elif preset_key == "DJI_Action4_1080p_Dewarp_RockSteady":
            self.apply_action4_1080p_16x9_rocksteady_preset()
        elif preset_key == "DJI_Action4_1080p_4_3_Dewarp_RockSteady":
            self.apply_action4_4x3_rocksteady_preset()
        elif preset_key in CAMERA_PRESETS:
            preset = CAMERA_PRESETS[preset_key]
            self.rig.left_camera = CameraIntrinsics.from_dict(preset.to_dict())
            self.rig.right_camera = CameraIntrinsics.from_dict(preset.to_dict())
            self.ctrl_cam_fov.setValue(preset.hfov_deg)
            self.engine.invalidate_luts()
            self.refresh_preview()
            self.status_bar.showMessage(f"Kamera-Preset aktiviert: {preset.name}")

    def toggle_auto_crop(self, checked: bool):
        self.rig.auto_crop_lir = checked
        self.engine.invalidate_luts()
        self._update_format_combo_labels()
        self.update_resolution_dropdown()
        self.refresh_preview()

    def on_safety_margin_changed(self, val: float):
        self.rig.lir_safety_margin = val * 0.01
        self.engine.invalidate_luts()
        self.trigger_smooth_preview()

    def on_ai_yolo_toggled(self, checked: bool):
        self.engine.ai_broadcast.config.ai_tracking = checked
        self.refresh_preview()


    def on_dynamic_zoom_toggled(self, checked: bool):
        self.engine.ai_broadcast.config.enable_dynamic_zoom = checked
        if hasattr(self, 'ctrl_min_zoom'):
            self.ctrl_min_zoom.setVisible(checked)
        if hasattr(self, 'ctrl_max_zoom'):
            self.ctrl_max_zoom.setVisible(checked)
        if hasattr(self, 'ctrl_zoom_speed'):
            self.ctrl_zoom_speed.setVisible(checked)
        if hasattr(self, 'ctrl_fixed_zoom'):
            self.ctrl_fixed_zoom.setVisible(not checked)
        self.on_autocam_param_changed()

    def on_autocam_param_changed(self):
        if hasattr(self, 'combo_ai_model'):
            self.engine.ai_broadcast.config.tracking_mode = self.combo_ai_model.currentData()
        if hasattr(self, 'ctrl_ball_weight'):
            self.engine.ai_broadcast.config.ball_weight = self.ctrl_ball_weight.value()
        if hasattr(self, 'ctrl_players_weight'):
            self.engine.ai_broadcast.config.players_weight = self.ctrl_players_weight.value()
        if hasattr(self, 'chk_dynamic_zoom'):
            self.engine.ai_broadcast.config.enable_dynamic_zoom = self.chk_dynamic_zoom.isChecked()
        if hasattr(self, 'ctrl_fixed_zoom'):
            self.engine.ai_broadcast.config.fixed_zoom_factor = self.ctrl_fixed_zoom.value()
        if hasattr(self, 'ctrl_min_zoom'):
            self.engine.ai_broadcast.config.min_zoom = self.ctrl_min_zoom.value()
        if hasattr(self, 'ctrl_max_zoom'):
            self.engine.ai_broadcast.config.max_zoom = self.ctrl_max_zoom.value()
        if hasattr(self, 'ctrl_zoom_speed'):
            self.engine.ai_broadcast.config.zoom_speed = self.ctrl_zoom_speed.value()
        if hasattr(self, 'ctrl_ac_lead'):
            self.engine.ai_broadcast.config.anticipation_lead = self.ctrl_ac_lead.value()
        if hasattr(self, 'ctrl_ac_smooth'):
            self.engine.ai_broadcast.config.smoothing_factor = self.ctrl_ac_smooth.value()
        if hasattr(self, 'ctrl_ac_deadband'):
            self.engine.ai_broadcast.config.deadband_width = self.ctrl_ac_deadband.value()
        if hasattr(self, 'ctrl_ac_speed'):
            self.engine.ai_broadcast.config.max_pan_speed = self.ctrl_ac_speed.value()
        if hasattr(self, 'ctrl_ac_vpos'):
            self.engine.ai_broadcast.config.vertical_center_bias = self.ctrl_ac_vpos.value()
        if hasattr(self, 'chk_zone_damping'):
            self.engine.ai_broadcast.config.enable_zone_damping = self.chk_zone_damping.isChecked()
            if hasattr(self, 'ctrl_zone_boost'):
                self.ctrl_zone_boost.setVisible(self.chk_zone_damping.isChecked())
        if hasattr(self, 'ctrl_zone_boost'):
            self.engine.ai_broadcast.config.zone_boost_factor = self.ctrl_zone_boost.value()
        if hasattr(self, 'ctrl_tactical_margin'):
            self.engine.ai_broadcast.config.tactical_margin = self.ctrl_tactical_margin.value()
        if hasattr(self, 'chk_anti_jersey'):
            self.engine.ai_broadcast.config.enable_anti_jersey_filter = self.chk_anti_jersey.isChecked()
        if hasattr(self, 'chk_static_leaf'):
            self.engine.ai_broadcast.config.enable_static_leaf_filter = self.chk_static_leaf.isChecked()
        if hasattr(self, 'chk_sideline_filter'):
            self.engine.ai_broadcast.config.enable_sideline_filter = self.chk_sideline_filter.isChecked()
            if hasattr(self, 'chk_sideline_menu'):
                self.chk_sideline_menu.blockSignals(True)
                self.chk_sideline_menu.setChecked(self.chk_sideline_filter.isChecked())
                self.chk_sideline_menu.blockSignals(False)
            if hasattr(self, 'ctrl_pitch_margin'):
                self.ctrl_pitch_margin.setVisible(self.chk_sideline_filter.isChecked())
        if hasattr(self, 'ctrl_pitch_margin'):
            self.engine.ai_broadcast.config.pitch_margin_pct = self.ctrl_pitch_margin.value()
        if hasattr(self, 'chk_strict_ball_center'):
            self.engine.ai_broadcast.config.enable_strict_ball_centering = self.chk_strict_ball_center.isChecked()
        self.refresh_preview()

    def on_pitch_roi_toggled(self, checked: bool):
        """Toggles interactive pitch ROI boundary editing mode."""
        if hasattr(self, 'btn_pitch_roi_tb') and self.btn_pitch_roi_tb is not None and self.btn_pitch_roi_tb.isChecked() != checked:
            self.btn_pitch_roi_tb.blockSignals(True)
            self.btn_pitch_roi_tb.setChecked(checked)
            self.btn_pitch_roi_tb.blockSignals(False)

        if hasattr(self, 'btn_pitch_roi_side') and self.btn_pitch_roi_side.isChecked() != checked:
            self.btn_pitch_roi_side.blockSignals(True)
            self.btn_pitch_roi_side.setChecked(checked)
            self.btn_pitch_roi_side.blockSignals(False)

        # Mutually exclusive with other interactive overlay modes
        if checked:
            if hasattr(self, 'btn_corner_pins_tb') and self.btn_corner_pins_tb is not None and self.btn_corner_pins_tb.isChecked():
                self.btn_corner_pins_tb.setChecked(False)
            if hasattr(self, 'btn_pick_center_tb') and self.btn_pick_center_tb is not None and self.btn_pick_center_tb.isChecked():
                self.btn_pick_center_tb.setChecked(False)

        self.viewport.set_pitch_roi_mode(checked)

    def on_pitch_corner_spin_changed(self, idx: int):
        """Updates pitch polygon corners from UI spinboxes."""
        if hasattr(self, 'spin_pitch_corners') and len(self.spin_pitch_corners) == 6:
            corners = []
            for sx, sy in self.spin_pitch_corners:
                corners.append([sx.value() / 100.0, sy.value() / 100.0])
            self.engine.ai_broadcast.config.pitch_corners = corners
            self.viewport.set_pitch_corners(corners)
            self.trigger_smooth_preview()

    def on_viewport_pitch_corners_changed(self, corners: list):
        """Updates UI spinboxes and engine when user drags pitch points in Viewport."""
        if hasattr(self, 'spin_pitch_corners') and len(self.spin_pitch_corners) == 6:
            for idx, (sx, sy) in enumerate(self.spin_pitch_corners):
                if idx < len(corners):
                    sx.blockSignals(True)
                    sy.blockSignals(True)
                    sx.setValue(corners[idx][0] * 100.0)
                    sy.setValue(corners[idx][1] * 100.0)
                    sx.blockSignals(False)
                    sy.blockSignals(False)
        self.engine.ai_broadcast.config.pitch_corners = [[float(c[0]), float(c[1])] for c in corners]
        self.trigger_smooth_preview()

    def reset_single_pitch_corner(self, idx: int, def_x: float, def_y: float):
        """Resets a single pitch point to its default coordinates."""
        if hasattr(self, 'spin_pitch_corners') and idx < len(self.spin_pitch_corners):
            self.spin_pitch_corners[idx][0].setValue(def_x)
            self.spin_pitch_corners[idx][1].setValue(def_y)

    def reset_all_pitch_corners(self):
        """Resets all 6 pitch points to full video frame boundaries."""
        defaults = [(0.0, 0.0), (50.0, 0.0), (100.0, 0.0), (100.0, 100.0), (50.0, 100.0), (0.0, 100.0)]
        if hasattr(self, 'spin_pitch_corners'):
            for idx, (dx, dy) in enumerate(defaults):
                if idx < len(self.spin_pitch_corners):
                    self.spin_pitch_corners[idx][0].setValue(dx)
                    self.spin_pitch_corners[idx][1].setValue(dy)

    def apply_preset_curve_warp(self):
        """Applies typical panorama lens sag correction preset (straightening curved sidelines)."""
        preset = [
            (0.0, 0.0),    # TL
            (50.0, 8.0),   # TC (Wölbung oben)
            (100.0, 0.0),  # TR
            (100.0, 100.0),# BR
            (50.0, 92.0),  # BC (Wölbung unten)
            (0.0, 100.0)   # BL
        ]
        if hasattr(self, 'spin_pitch_corners'):
            for idx, (dx, dy) in enumerate(preset):
                if idx < len(self.spin_pitch_corners):
                    self.spin_pitch_corners[idx][0].setValue(dx)
                    self.spin_pitch_corners[idx][1].setValue(dy)
        self.status_bar.showMessage("🏟️ Preset aktiviert: Mittellinien-Wölbung entzerrt (Panorama-Seitenlinien begradigt)!")

    def apply_preset_youth_crossfield(self):
        """Applies Querfeld / Jugendspiel preset (half-pitch / cross-pitch field setup)."""
        preset = [
            (5.0, 15.0),   # TL (Tor links)
            (50.0, 12.0),  # TC (obere Begrenzung)
            (95.0, 15.0),  # TR (Tor rechts)
            (95.0, 52.0),  # BR (untere Ecke rechts)
            (50.0, 58.0),  # BC (Wölbung / Pylone unten)
            (5.0, 52.0)    # BL (untere Ecke links)
        ]
        if hasattr(self, 'spin_pitch_corners'):
            for idx, (dx, dy) in enumerate(preset):
                if idx < len(self.spin_pitch_corners):
                    self.spin_pitch_corners[idx][0].setValue(dx)
                    self.spin_pitch_corners[idx][1].setValue(dy)
        self.status_bar.showMessage("⚽ Preset aktiviert: Querfeld / Jugendspiel (untere Seitenlinie bei ~55%, Trainer im Vordergrund ignoriert)!")

    def apply_preset_fullscreen_warp(self):
        """Resets tactical warp to 100% full frame."""
        preset = [
            (0.0, 0.0),
            (50.0, 0.0),
            (100.0, 0.0),
            (100.0, 100.0),
            (50.0, 100.0),
            (0.0, 100.0)
        ]
        if hasattr(self, 'spin_pitch_corners'):
            for idx, (dx, dy) in enumerate(preset):
                if idx < len(self.spin_pitch_corners):
                    self.spin_pitch_corners[idx][0].setValue(dx)
                    self.spin_pitch_corners[idx][1].setValue(dy)
        self.status_bar.showMessage("↺ 100% Vollbild wiederhergestellt.")

    def auto_detect_pitch_roi(self):
        """Automatically detects pitch grass area using HSV green segmentation on current frame."""
        if not self.has_active_video():
            QMessageBox.warning(self, "Spielfeld erkennen", "Bitte zuerst ein Video laden!")
            return

        frame, _, _, _ = self.engine.render_preview_frame(self.current_frame_idx, preview_width=1280, preview_height=360)
        if frame is None:
            return

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_green = np.array([25, 25, 25])
        upper_green = np.array([88, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)

        # Morphological closing
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        mask_clean = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Horizontal and vertical profile sums
        h_proj = np.sum(mask_clean, axis=0) # sum across y (length = w)
        v_proj = np.sum(mask_clean, axis=1) # sum across x (length = h)

        h, w = mask.shape
        max_h = np.max(h_proj) if np.max(h_proj) > 0 else 1
        max_v = np.max(v_proj) if np.max(v_proj) > 0 else 1

        valid_x = np.where(h_proj > (max_h * 0.15))[0]
        valid_y = np.where(v_proj > (max_v * 0.15))[0]

        if len(valid_x) > 0 and len(valid_y) > 0:
            det_left = float(np.clip(valid_x[0] / w, 0.0, 0.35))
            det_right = float(np.clip(valid_x[-1] / w, 0.65, 1.0))
            det_top = float(np.clip(valid_y[0] / h, 0.0, 0.45))
            det_bottom = float(np.clip(valid_y[-1] / h, 0.55, 1.0))
            mid_x = (det_left + det_right) * 0.5

            corners = [
                [det_left, det_top],
                [mid_x, det_top],
                [det_right, det_top],
                [det_right, det_bottom],
                [mid_x, det_bottom],
                [det_left, det_bottom]
            ]
            self.on_viewport_pitch_corners_changed(corners)
            self.viewport.set_pitch_corners(corners)
            QMessageBox.information(self, "Spielfeld erkannt", f"Spielfeld-Grenzen erfolgreich ermittelt:\nLinks: {det_left*100:.1f}%, Mitte: {mid_x*100:.1f}%, Rechts: {det_right*100:.1f}%\nOben: {det_top*100:.1f}%, Unten: {det_bottom*100:.1f}%")
        else:
            QMessageBox.warning(self, "Spielfeld erkennen", "Konnte kein eindeutiges grünes Spielfeld im aktuellen Bild erkennen.")

    def on_ai_engine_changed(self):
        """Switches the active detector backend (YOLO11, RT-DETR, YOLO-Pose, Autocam ONNX, or Custom)."""
        if hasattr(self, 'combo_ai_engine') and hasattr(self.engine, 'ai_broadcast'):
            backend_id = self.combo_ai_engine.currentData() or "yolo11"
            if backend_id == "custom_browse":
                file_path, _ = QFileDialog.getOpenFileName(
                    self, "Eigenes KI-Modell wählen (.onnx oder .pt)", "",
                    "KI-Modelle (*.onnx *.pt);;ONNX Modelle (*.onnx);;PyTorch Modelle (*.pt);;Alle Dateien (*.*)"
                )
                if file_path:
                    fname = os.path.basename(file_path)
                    custom_id = f"custom_{fname}"
                    # Insert right before the last 'browse' entry
                    idx = max(0, self.combo_ai_engine.count() - 1)
                    self.combo_ai_engine.insertItem(idx, f"📁 {fname} (Benutzerdefiniert)", custom_id)
                    self.combo_ai_engine.blockSignals(True)
                    self.combo_ai_engine.setCurrentIndex(idx)
                    self.combo_ai_engine.blockSignals(False)
                    backend_id = custom_id
                    if hasattr(self.engine.ai_broadcast, 'config'):
                        self.engine.ai_broadcast.config.custom_model_path = file_path
                else:
                    # Cancelled -> revert to previous or default
                    idx0 = self.combo_ai_engine.findData("yolo11")
                    if idx0 >= 0:
                        self.combo_ai_engine.blockSignals(True)
                        self.combo_ai_engine.setCurrentIndex(idx0)
                        self.combo_ai_engine.blockSignals(False)
                    return

            self.engine.ai_broadcast.set_backend(backend_id)
            if hasattr(self, 'status_bar'):
                self.status_bar.showMessage(f"🤖 KI-Modell aktiv: {self.combo_ai_engine.currentText()}", 3500)

            # Synchronize with Stepper Wizard if available
            if hasattr(self, 'tab_stepper') and hasattr(self.tab_stepper, 'combo_wizard_model'):
                s_idx = self.tab_stepper.combo_wizard_model.findData(backend_id)
                if s_idx >= 0 and self.tab_stepper.combo_wizard_model.currentIndex() != s_idx:
                    self.tab_stepper.combo_wizard_model.blockSignals(True)
                    self.tab_stepper.combo_wizard_model.setCurrentIndex(s_idx)
                    self.tab_stepper.combo_wizard_model.blockSignals(False)
                if hasattr(self.tab_stepper, 'update_summary'):
                    self.tab_stepper.update_summary()

            self.refresh_preview()

    def on_scan_speed_changed(self):
        """Updates lookahead scan speed and FP16 toggle in AI engine."""
        if hasattr(self, 'combo_scan_speed'):
            step_val = self.combo_scan_speed.currentData() or 5
            self.engine.ai_broadcast.config.scan_step = int(step_val)
        if hasattr(self, 'chk_fp16'):
            self.engine.ai_broadcast.config.use_fp16 = self.chk_fp16.isChecked()


    def on_export_format_changed(self):
        self.update_resolution_dropdown()

    def update_resolution_dropdown(self):
        targets = []
        for attr in ('combo_resolution', 'combo_stitch_res'):
            combo = getattr(self, attr, None)
            if combo is not None and combo not in targets:
                targets.append(combo)

        sender = self.sender()
        if sender and hasattr(sender, 'currentData') and sender.currentData():
            fmt = sender.currentData()
        elif hasattr(self, 'combo_stitch_format') and self.combo_stitch_format is not None and self.combo_stitch_format.currentData():
            fmt = self.combo_stitch_format.currentData()
        elif hasattr(self, 'combo_exp_format') and self.combo_exp_format is not None:
            fmt = self.combo_exp_format.currentData()
        else:
            fmt = "32:9"

        items = []
        if fmt in ("16:9_autocam", "16:9_tactical"):
            items.append(("1920 x 1080 (Full HD 1080p - Standard 16:9)", (1920, 1080)))
            items.append(("1280 x 720 (HD 720p - Schnell / Kompakt)", (1280, 720)))
            items.append(("2560 x 1440 (2K QHD 1440p - Hohe Schärfe)", (2560, 1440)))
            items.append(("3840 x 2160 (4K UHD 2160p - Maximale Details)", (3840, 2160)))
        elif fmt == "21:10":
            items.append(("3360 x 1600 (1600p 21:10 - Hohe Schärfe / Empfohlen)", (3360, 1600)))
            items.append(("2520 x 1200 (1200p 21:10 - Standard / Hohe FPS)", (2520, 1200)))
            items.append(("1512 x 720 (720p 21:10 - Schnell / Kompakt)", (1512, 720)))
            items.append(("4200 x 2000 (2K/4K 21:10 - Maximale Details)", (4200, 2000)))
            items.append(("2100 x 1000 (1000p 21:10 - Schnell)", (2100, 1000)))
        elif fmt == "wide_angle":
            w = self.wide_angle_config.target_width if hasattr(self, 'wide_angle_config') else 1920
            h = self.wide_angle_config.target_height if hasattr(self, 'wide_angle_config') else 1080
            items.append((f"{w} x {h} (Aktuelles Weitwinkel-Zielformat)", (w, h)))
            items.append(("1920 x 1080 (16:9 Full HD 1080p)", (1920, 1080)))
            items.append(("3840 x 2160 (16:9 4K UHD)", (3840, 2160)))
            items.append(("2560 x 1080 (21:9 UltraWide Cinema)", (2560, 1080)))
            items.append(("3440 x 1440 (21:9 UWQHD Cinema)", (3440, 1440)))
            items.append(("3840 x 1080 (32:9 SuperWide Panorama)", (3840, 1080)))
            items.append(("2880 x 2160 (4:3 Vollsensor)", (2880, 2160)))
        elif fmt == "both":
            if self.is_4_3_mode():
                items.append(("1920 x 1080 (1080p Broadcast + 2880x1080 8:3 Master)", (1920, 1080)))
                items.append(("1280 x 720 (720p Broadcast + 1920x720 8:3 Master - Schnell)", (1280, 720)))
                items.append(("2560 x 1440 (1440p Broadcast + 3840x1440 8:3 Master)", (2560, 1440)))
                items.append(("3840 x 2160 (4K Broadcast + 5760x2160 8:3 Master)", (3840, 2160)))
            else:
                items.append(("1920 x 1080 (1080p Broadcast + 3840x1080 Master)", (1920, 1080)))
                items.append(("1280 x 720 (720p Broadcast + 2560x720 Master - Schnell)", (1280, 720)))
                items.append(("2560 x 1440 (1440p Broadcast + 5120x1440 Master)", (2560, 1440)))
                items.append(("3840 x 2160 (4K Broadcast + 7680x2160 Master)", (3840, 2160)))
        elif fmt == "custom_crop":
            norm = getattr(self.viewport, 'crop_rect_norm', [0.0, 0.0, 1.0, 1.0]) if hasattr(self, 'viewport') else [0.0, 0.0, 1.0, 1.0]
            cw = max(320, int(3840 * norm[2]))
            ch = max(180, int(1080 * norm[3]))
            cw = (cw // 2) * 2
            ch = (ch // 2) * 2
            items.append((f"✂️ {cw} x {ch} (Nativer Zuschnitt)", (cw, ch)))
            items.append(("📺 1920 x 1080 (Skaliert Full HD)", (1920, 1080)))
            items.append(("🌟 3840 x 2160 (Skaliert 4K)", (3840, 2160)))
        else: # Panorama (8:3 for 4:3 mode or 32:9 for 16:9 mode)
            if self.is_4_3_mode():
                items.append(("2880 x 1080 (1080p 8:3 Panorama - Standard)", (2880, 1080)))
                items.append(("3840 x 1440 (1440p 8:3 Panorama - Empfohlen)", (3840, 1440)))
                items.append(("1920 x 720 (720p 8:3 Panorama - Schnell / Kompakt)", (1920, 720)))
                items.append(("5760 x 2160 (4K 8:3 Panorama - Maximale Qualität)", (5760, 2160)))
                items.append(("5376 x 2016 (2.7K 8:3 Panorama - Nativ Action 4)", (5376, 2016)))
            else:
                items.append(("5120 x 1440 (1440p 32:9 Panorama - Empfohlen)", (5120, 1440)))
                items.append(("3840 x 1080 (1080p 32:9 Panorama - Standard)", (3840, 1080)))
                items.append(("2560 x 720 (720p 32:9 Panorama - Schnell / Kompakt)", (2560, 720)))
                items.append(("7680 x 2160 (4K 32:9 Panorama - Maximale Qualität)", (7680, 2160)))

        for target in targets:
            curr_data = target.currentData()
            target.clear()
            for label, data in items:
                target.addItem(label, data)
            if curr_data is not None:
                idx = target.findData(curr_data)
                if idx >= 0:
                    target.setCurrentIndex(idx)

    def on_crop_mode_toggled(self, checked: bool):
        self.crop_mode = checked
        if hasattr(self, 'viewport'):
            self.viewport.set_crop_mode(checked)
            if checked:
                self.status_bar.showMessage("✂️ Photoshop Zuschnitt aktiv: Ziehe an den 8 Anfassern oder verschiebe das Rechteck.")
            else:
                self.status_bar.showMessage("Photoshop Zuschnitt deaktiviert.")

    def on_crop_ratio_changed(self, idx: int):
        if hasattr(self, 'combo_crop_ratio') and hasattr(self, 'viewport'):
            ratio = self.combo_crop_ratio.itemData(idx)
            self.viewport.set_crop_aspect_ratio(ratio)

    def on_crop_spin_changed(self):
        if not hasattr(self, 'spin_crop_x') or not hasattr(self, 'viewport'):
            return
        x = self.spin_crop_x.value() / 100.0
        y = self.spin_crop_y.value() / 100.0
        w = self.spin_crop_w.value() / 100.0
        h = self.spin_crop_h.value() / 100.0
        self.viewport.set_crop_rect(x, y, w, h)

    def on_crop_rect_from_viewport(self, x: float, y: float, w: float, h: float):
        for sp in [getattr(self, 'spin_crop_x', None), getattr(self, 'spin_crop_y', None),
                   getattr(self, 'spin_crop_w', None), getattr(self, 'spin_crop_h', None)]:
            if sp: sp.blockSignals(True)
        if hasattr(self, 'spin_crop_x'): self.spin_crop_x.setValue(x * 100.0)
        if hasattr(self, 'spin_crop_y'): self.spin_crop_y.setValue(y * 100.0)
        if hasattr(self, 'spin_crop_w'): self.spin_crop_w.setValue(w * 100.0)
        if hasattr(self, 'spin_crop_h'): self.spin_crop_h.setValue(h * 100.0)
        for sp in [getattr(self, 'spin_crop_x', None), getattr(self, 'spin_crop_y', None),
                   getattr(self, 'spin_crop_w', None), getattr(self, 'spin_crop_h', None)]:
            if sp: sp.blockSignals(False)

    def on_transform_stretch_changed(self):
        sx = self.ctrl_stretch_x.value() if hasattr(self, 'ctrl_stretch_x') else 1.0
        sy = self.ctrl_stretch_y.value() if hasattr(self, 'ctrl_stretch_y') else 1.0
        self.transform_stretch_x = sx
        self.transform_stretch_y = sy
        if hasattr(self, 'viewport'):
            self.viewport.set_transform_stretch(sx, sy)

    def reset_crop_and_transform(self):
        self.transform_stretch_x = 1.0
        self.transform_stretch_y = 1.0
        if hasattr(self, 'ctrl_stretch_x'):
            self.ctrl_stretch_x.setValue(1.0)
        if hasattr(self, 'ctrl_stretch_y'):
            self.ctrl_stretch_y.setValue(1.0)
        if hasattr(self, 'combo_crop_ratio'):
            self.combo_crop_ratio.setCurrentIndex(0)
        if hasattr(self, 'viewport'):
            self.viewport.set_transform_stretch(1.0, 1.0)
            self.viewport.set_crop_rect(0.0, 0.0, 1.0, 1.0)
            self.viewport.set_crop_aspect_ratio(None)
        if hasattr(self, 'spin_crop_x'): self.spin_crop_x.setValue(0.0)
        if hasattr(self, 'spin_crop_y'): self.spin_crop_y.setValue(0.0)
        if hasattr(self, 'spin_crop_w'): self.spin_crop_w.setValue(100.0)
        if hasattr(self, 'spin_crop_h'): self.spin_crop_h.setValue(100.0)
        if hasattr(self, 'btn_crop_mode'):
            self.btn_crop_mode.setChecked(False)

    def trigger_smooth_preview(self):
        """Instant draft update during dragging (<2ms) + debounced high-res render."""
        if not self.has_active_video():
            return
        self.viewport.set_rig(self.rig)
        self.viewport.is_4_3 = self.is_4_3_mode()
        vm = getattr(self.viewport, 'view_mode', '32:9')
        if vm in ("wide_angle", "wide_angle_raw"):
            pw = 1280
            ar = self.wide_angle_config.aspect_ratio if hasattr(self, 'wide_angle_config') else (16.0 / 9.0)
            ph = int(pw / max(0.1, ar))
            if ph % 2 != 0:
                ph += 1
        elif vm == "21:10":
            pw, ph = 1260, 600
        elif vm == "8:3" or (vm != "32:9" and self.is_4_3_mode()):
            pw, ph = 1280, 480
        else:
            pw, ph = 1280, 360
        frame, autocam_box, ball_px, cur_zoom = self.engine.render_preview_frame(
            self.current_frame_idx, preview_width=pw, preview_height=ph, view_mode=vm,
            wide_angle_rectifier=self.wide_angle_rectifier
        )
        ai_tracker = getattr(self.engine, 'ai_broadcast', None)
        players = ai_tracker.active_players if ai_tracker else None
        ball_norm = ai_tracker.current_ball_norm if ai_tracker else None
        cam_span = ai_tracker.get_camera_span() if ai_tracker else None
        offside_line = getattr(ai_tracker, 'current_offside_line', None) if ai_tracker else None
        detections = getattr(ai_tracker, 'active_detections', None) if ai_tracker else None
        ball_spd = ai_tracker.get_ball_speed() if hasattr(ai_tracker, 'get_ball_speed') else None
        self.viewport.set_frame(frame, autocam_box, ball_px, cur_zoom, players=players, ball_norm=ball_norm, cam_span=cam_span, offside_line=offside_line, detections=detections, ball_speed_kmh=ball_spd)
        self.hq_preview_timer.start()

    def render_high_res_preview(self):
        """Crisp preview rendered once slider motion settles."""
        if not self.has_active_video():
            return
        self.viewport.set_rig(self.rig)
        self.viewport.is_4_3 = self.is_4_3_mode()
        vm = getattr(self.viewport, 'view_mode', '32:9')
        if vm in ("wide_angle", "wide_angle_raw"):
            pw = 1920
            ar = self.wide_angle_config.aspect_ratio if hasattr(self, 'wide_angle_config') else (16.0 / 9.0)
            ph = int(pw / max(0.1, ar))
            if ph % 2 != 0:
                ph += 1
        elif vm == "21:10":
            pw, ph = 2100, 1000
        elif vm == "8:3" or (vm != "32:9" and self.is_4_3_mode()):
            pw, ph = 1920, 720
        else:
            pw, ph = 2560, 720
        frame, autocam_box, ball_px, cur_zoom = self.engine.render_preview_frame(
            self.current_frame_idx, preview_width=pw, preview_height=ph, view_mode=vm,
            wide_angle_rectifier=self.wide_angle_rectifier
        )
        ai_tracker = getattr(self.engine, 'ai_broadcast', None)
        players = ai_tracker.active_players if ai_tracker else None
        ball_norm = ai_tracker.current_ball_norm if ai_tracker else None
        cam_span = ai_tracker.get_camera_span() if ai_tracker else None
        offside_line = getattr(ai_tracker, 'current_offside_line', None) if ai_tracker else None
        detections = getattr(ai_tracker, 'active_detections', None) if ai_tracker else None
        ball_spd = ai_tracker.get_ball_speed() if hasattr(ai_tracker, 'get_ball_speed') else None
        self.viewport.set_frame(frame, autocam_box, ball_px, cur_zoom, players=players, ball_norm=ball_norm, cam_span=cam_span, offside_line=offside_line, detections=detections, ball_speed_kmh=ball_spd)

    def refresh_preview(self):
        if not self.has_active_video():
            self.viewport.update()
            return
        self.viewport.set_rig(self.rig)
        self.viewport.is_4_3 = self.is_4_3_mode()
        vm = getattr(self.viewport, 'view_mode', '32:9')
        if vm in ("wide_angle", "wide_angle_raw"):
            pw = 1920
            ar = self.wide_angle_config.aspect_ratio if hasattr(self, 'wide_angle_config') else (16.0 / 9.0)
            ph = int(pw / max(0.1, ar))
            if ph % 2 != 0:
                ph += 1
        elif vm == "21:10":
            pw, ph = 2100, 1000
        elif vm == "8:3" or (vm != "32:9" and self.is_4_3_mode()):
            pw, ph = 1920, 720
        else:
            pw, ph = 2560, 720
        frame, autocam_box, ball_px, cur_zoom = self.engine.render_preview_frame(
            self.current_frame_idx, preview_width=pw, preview_height=ph, view_mode=vm,
            wide_angle_rectifier=self.wide_angle_rectifier
        )
        ai_tracker = getattr(self.engine, 'ai_broadcast', None)
        players = ai_tracker.active_players if ai_tracker else None
        ball_norm = ai_tracker.current_ball_norm if ai_tracker else None
        cam_span = ai_tracker.get_camera_span() if ai_tracker else None
        offside_line = getattr(ai_tracker, 'current_offside_line', None) if ai_tracker else None
        detections = getattr(ai_tracker, 'active_detections', None) if ai_tracker else None
        ball_spd = ai_tracker.get_ball_speed() if hasattr(ai_tracker, 'get_ball_speed') else None
        self.viewport.set_frame(frame, autocam_box, ball_px, cur_zoom, players=players, ball_norm=ball_norm, cam_span=cam_span, offside_line=offside_line, detections=detections, ball_speed_kmh=ball_spd)

    def get_current_settings_dict(self) -> dict:
        """Serializes all rig, camera, AI autocam, export, and UI settings into a dictionary."""
        fmt = self.combo_exp_format.currentData() if hasattr(self, 'combo_exp_format') else "32:9"
        res_data = self.combo_resolution.currentData() if hasattr(self, 'combo_resolution') else (3840, 1080)
        w, h = res_data if isinstance(res_data, (tuple, list)) and len(res_data) == 2 else (3840, 1080)
        codec = self.combo_codec.currentData() if hasattr(self, 'combo_codec') else "hevc_nvenc"
        bitrate = int(self.ctrl_bitrate.value()) if hasattr(self, 'ctrl_bitrate') else 50
        trim_only = self.chk_export_trim_only.isChecked() if hasattr(self, 'chk_export_trim_only') else True
        audio_src = self.combo_audio_source.currentData() if hasattr(self, 'combo_audio_source') else "left"

        # Sync / Offset
        frame_offset = self.spin_offset.value() if hasattr(self, 'spin_offset') else 0

        # AutoCam config
        ac_cfg = self.engine.ai_broadcast.config
        autocam_dict = {
            "ai_tracking": ac_cfg.ai_tracking,
            "tracking_mode": self.combo_ai_model.currentData() if hasattr(self, 'combo_ai_model') else ac_cfg.tracking_mode,
            "ball_weight": self.ctrl_ball_weight.value() if hasattr(self, 'ctrl_ball_weight') else getattr(ac_cfg, 'ball_weight', 0.70),
            "players_weight": self.ctrl_players_weight.value() if hasattr(self, 'ctrl_players_weight') else getattr(ac_cfg, 'players_weight', 0.30),
            "enable_dynamic_zoom": self.chk_dynamic_zoom.isChecked() if hasattr(self, 'chk_dynamic_zoom') else ac_cfg.enable_dynamic_zoom,
            "fixed_zoom_factor": self.ctrl_fixed_zoom.value() if hasattr(self, 'ctrl_fixed_zoom') else ac_cfg.fixed_zoom_factor,
            "min_zoom": self.ctrl_min_zoom.value() if hasattr(self, 'ctrl_min_zoom') else ac_cfg.min_zoom,
            "max_zoom": self.ctrl_max_zoom.value() if hasattr(self, 'ctrl_max_zoom') else ac_cfg.max_zoom,
            "zoom_speed": self.ctrl_zoom_speed.value() if hasattr(self, 'ctrl_zoom_speed') else ac_cfg.zoom_speed,
            "anticipation_lead": self.ctrl_ac_lead.value() if hasattr(self, 'ctrl_ac_lead') else ac_cfg.anticipation_lead,
            "smoothing_factor": self.ctrl_ac_smooth.value() if hasattr(self, 'ctrl_ac_smooth') else ac_cfg.smoothing_factor,
            "deadband_width": self.ctrl_ac_deadband.value() if hasattr(self, 'ctrl_ac_deadband') else ac_cfg.deadband_width,
            "max_pan_speed": self.ctrl_ac_speed.value() if hasattr(self, 'ctrl_ac_speed') else ac_cfg.max_pan_speed,
            "vertical_center_bias": self.ctrl_ac_vpos.value() if hasattr(self, 'ctrl_ac_vpos') else ac_cfg.vertical_center_bias,
            "enable_zone_damping": self.chk_zone_damping.isChecked() if hasattr(self, 'chk_zone_damping') else getattr(ac_cfg, 'enable_zone_damping', True),
            "zone_boost_factor": self.ctrl_zone_boost.value() if hasattr(self, 'ctrl_zone_boost') else getattr(ac_cfg, 'zone_boost_factor', 1.80),
            "tactical_margin": self.ctrl_tactical_margin.value() if hasattr(self, 'ctrl_tactical_margin') else getattr(ac_cfg, 'tactical_margin', 0.0),
            "pitch_corners": [[sx.value() / 100.0, sy.value() / 100.0] for sx, sy in self.spin_pitch_corners] if hasattr(self, 'spin_pitch_corners') and len(self.spin_pitch_corners) == 6 else getattr(ac_cfg, 'pitch_corners', [[0.0, 0.05], [0.5, 0.05], [1.0, 0.05], [1.0, 0.95], [0.5, 0.95], [0.0, 0.95]]),
            "scan_step": self.combo_scan_speed.currentData() if hasattr(self, 'combo_scan_speed') else ac_cfg.scan_step,
            "use_fp16": self.chk_fp16.isChecked() if hasattr(self, 'chk_fp16') else ac_cfg.use_fp16,
            "enable_radar": self.chk_show_radar.isChecked() if hasattr(self, 'chk_show_radar') else getattr(ac_cfg, 'enable_radar', True),
            "radar_position": self.combo_radar_pos.currentData() if hasattr(self, 'combo_radar_pos') else getattr(ac_cfg, 'radar_position', 'top_right'),
            "radar_width": int(self.ctrl_radar_width.value()) if hasattr(self, 'ctrl_radar_width') else getattr(ac_cfg, 'radar_width', 320),
            "radar_opacity": float(self.ctrl_radar_opacity.value()) if hasattr(self, 'ctrl_radar_opacity') else getattr(ac_cfg, 'radar_opacity', 0.85),
            "burn_radar_in_export": self.chk_burn_radar.isChecked() if hasattr(self, 'chk_burn_radar') else getattr(ac_cfg, 'burn_radar_in_export', False),
            "ai_backend": self.combo_ai_engine.currentData() if hasattr(self, 'combo_ai_engine') else getattr(ac_cfg, 'ai_backend', 'yolo11'),
            "enable_offside_line": self.chk_show_offside.isChecked() if hasattr(self, 'chk_show_offside') else getattr(ac_cfg, 'enable_offside_line', True),
            "enable_skeletons": self.chk_show_skeletons.isChecked() if hasattr(self, 'chk_show_skeletons') else getattr(ac_cfg, 'enable_skeletons', False),
            "enable_speed_badges": self.chk_show_speed.isChecked() if hasattr(self, 'chk_show_speed') else getattr(ac_cfg, 'enable_speed_badges', False),
            "enable_anti_jersey_filter": self.chk_anti_jersey.isChecked() if hasattr(self, 'chk_anti_jersey') else getattr(ac_cfg, 'enable_anti_jersey_filter', True),
            "enable_static_leaf_filter": self.chk_static_leaf.isChecked() if hasattr(self, 'chk_static_leaf') else getattr(ac_cfg, 'enable_static_leaf_filter', True),
            "enable_sideline_filter": self.chk_sideline_filter.isChecked() if hasattr(self, 'chk_sideline_filter') else getattr(ac_cfg, 'enable_sideline_filter', True),
            "pitch_margin_pct": self.ctrl_pitch_margin.value() if hasattr(self, 'ctrl_pitch_margin') else getattr(ac_cfg, 'pitch_margin_pct', 1.0),
            "enable_strict_ball_centering": self.chk_strict_ball_center.isChecked() if hasattr(self, 'chk_strict_ball_center') else getattr(ac_cfg, 'enable_strict_ball_centering', True)
        }

        # UI preferences
        ui_dict = {
            "view_mode": self.combo_view_mode.currentData() if hasattr(self, 'combo_view_mode') else "32:9",
            "show_frame": self.chk_show_frame.isChecked() if hasattr(self, 'chk_show_frame') else True,
            "frame_opacity": self.ctrl_frame_opacity.value() if hasattr(self, 'ctrl_frame_opacity') else 50.0,
            "show_center_line": self.chk_show_center_line.isChecked() if hasattr(self, 'chk_show_center_line') else True,
            "show_autocam": self.chk_show_autocam.isChecked() if hasattr(self, 'chk_show_autocam') else True,
            "show_ball": self.chk_show_ball.isChecked() if hasattr(self, 'chk_show_ball') else True,
            "show_radar": self.chk_show_radar.isChecked() if hasattr(self, 'chk_show_radar') else True,
            "show_offside_line": self.chk_show_offside.isChecked() if hasattr(self, 'chk_show_offside') else True,
            "show_skeletons": self.chk_show_skeletons.isChecked() if hasattr(self, 'chk_show_skeletons') else False,
            "show_speed": self.chk_show_speed.isChecked() if hasattr(self, 'chk_show_speed') else False,
            "show_sideline_filter": self.chk_sideline_menu.isChecked() if hasattr(self, 'chk_sideline_menu') else True,
            "show_seam": self.chk_show_seam.isChecked() if hasattr(self, 'chk_show_seam') else True,
            "show_grid": self.chk_show_grid.isChecked() if hasattr(self, 'chk_show_grid') else True,
            "sync_angles": getattr(self, 'sync_angles', True),
            "multi_frame_calib": self.chk_multi_frame_calib.isChecked() if hasattr(self, 'chk_multi_frame_calib') else True
        }

        return {
            "version": "1.3",
            "name": self.rig.name,
            "rig": self.rig.to_dict(),
            "sync": {"frame_offset_right": frame_offset},
            "autocam": autocam_dict,
            "export": {
                "format": fmt,
                "resolution_width": w,
                "resolution_height": h,
                "codec": codec,
                "bitrate_mbps": bitrate,
                "trim_only": trim_only,
                "audio_source": audio_src
            },
            "cloud": {
                "server_url": self.cloud_client.server_url or self.saved_cloud_server_url,
                "team_id": self.combo_cloud_team.currentData() if hasattr(self, 'combo_cloud_team') else self.saved_cloud_team_id,
                "category": self.combo_cloud_category.currentData() if hasattr(self, 'combo_cloud_category') else "Punktspiel",
                "auto_upload": self.chk_cloud_auto_upload.isChecked() if hasattr(self, 'chk_cloud_auto_upload') else False,
                "notify_team": self.chk_cloud_notify.isChecked() if hasattr(self, 'chk_cloud_notify') else True,
                "fast_hls": self.chk_cloud_fast_hls.isChecked() if hasattr(self, 'chk_cloud_fast_hls') else False,
                "concurrency": self.combo_cloud_concurrency.currentData() if hasattr(self, 'combo_cloud_concurrency') else 3
            },
            "livestream": {
                "server_url": self.txt_rtmp_server_url.text().strip() if hasattr(self, 'txt_rtmp_server_url') else "rtmp://matchtrack.de:1935/live",
                "resolution": self.combo_stream_resolution.currentData() if hasattr(self, 'combo_stream_resolution') else "1080p",
                "stream_key_c1": self.txt_stream_key_c1.text().strip() if hasattr(self, 'txt_stream_key_c1') else "broadcast_1",
                "stream_key_c2": self.txt_stream_key_c2.text().strip() if hasattr(self, 'txt_stream_key_c2') else "panorama_1",
                "stream_key_c3": self.txt_stream_key_c3.text().strip() if hasattr(self, 'txt_stream_key_c3') else "cam1",
                "stream_key_c4": self.txt_stream_key_c4.text().strip() if hasattr(self, 'txt_stream_key_c4') else "cam2",
                "source_c1": self.combo_feed_c1.currentData() if hasattr(self, 'combo_feed_c1') else "broadcast_16x9",
                "source_c2": self.combo_feed_c2.currentData() if hasattr(self, 'combo_feed_c2') else "panorama_8x3",
                "source_c3": self.combo_feed_c3.currentData() if hasattr(self, 'combo_feed_c3') else "cam1",
                "source_c4": self.combo_feed_c4.currentData() if hasattr(self, 'combo_feed_c4') else "cam2",
                "chan_c1_enabled": self.chk_chan_1.isChecked() if hasattr(self, 'chk_chan_1') else True,
                "chan_c2_enabled": self.chk_chan_2.isChecked() if hasattr(self, 'chk_chan_2') else True,
                "chan_c3_enabled": self.chk_chan_3.isChecked() if hasattr(self, 'chk_chan_3') else False,
                "chan_c4_enabled": self.chk_chan_4.isChecked() if hasattr(self, 'chk_chan_4') else False,
                "encoder": self.combo_stream_encoder.currentText() if hasattr(self, 'combo_stream_encoder') else "h264_nvenc",
                "bitrate_kbps": self.obs_settings_widget.spn_bitrate.value() if hasattr(self, 'obs_settings_widget') else 3000,
                "wifi_ssid": self.txt_dji_wifi_ssid.text().strip() if hasattr(self, 'txt_dji_wifi_ssid') else "",
                "obs_settings": self.obs_settings_widget.get_settings() if hasattr(self, 'obs_settings_widget') else {}
            },
            "wide_angle": self.wide_angle_config.to_dict() if hasattr(self, 'wide_angle_config') else {},
            "ui_view": ui_dict
        }

    def apply_settings_dict(self, data: dict):
        """Restores full application configuration from a dictionary."""
        if not isinstance(data, dict):
            return

        # 1. Rig settings (support both full profile dictionary and raw rig dictionary)
        rig_data = data.get("rig", data)
        if isinstance(rig_data, dict) and "left_camera" in rig_data and "left_pose" in rig_data:
            self.rig = RigConfiguration.from_dict(rig_data)
            self.engine.rig = self.rig
            
            # Update Rig UI Controls safely without triggering cascading signals
            for ctrl in [
                self.ctrl_left_yaw, self.ctrl_right_yaw, self.ctrl_master_yaw,
                self.ctrl_cam_pitch, self.ctrl_left_roll, self.ctrl_right_roll,
                self.ctrl_global_pitch, self.ctrl_h_offset, self.ctrl_pano_hfov,
                self.ctrl_v_offset, self.ctrl_squeeze, self.ctrl_blend_width,
                self.ctrl_cam_fov, self.ctrl_safety_margin, self.chk_auto_crop
            ]:
                ctrl.blockSignals(True)

            self.ctrl_left_yaw.setValue(self.rig.left_pose.yaw)
            self.ctrl_right_yaw.setValue(self.rig.right_pose.yaw)
            self.ctrl_master_yaw.setValue(abs(self.rig.left_pose.yaw) + abs(self.rig.right_pose.yaw))
            self.ctrl_cam_pitch.setValue(self.rig.left_pose.pitch)
            self.ctrl_left_roll.setValue(self.rig.left_pose.roll)
            self.ctrl_right_roll.setValue(self.rig.right_pose.roll)
            self.ctrl_global_pitch.setValue(self.rig.global_pitch_correction)
            self.ctrl_h_offset.setValue(getattr(self.rig, 'global_yaw_center', 0.0))
            self.ctrl_pano_hfov.setValue(self.rig.pano_hfov)
            self.ctrl_v_offset.setValue(self.rig.vertical_crop_offset)
            self.ctrl_squeeze.setValue(getattr(self.rig, 'horizontal_squeeze', 1.0))
            self.ctrl_blend_width.setValue(self.rig.blend_width_deg)
            self.ctrl_cam_fov.setValue(self.rig.left_camera.hfov_deg)
            self.ctrl_safety_margin.setValue(self.rig.lir_safety_margin * 100.0)
            self.chk_auto_crop.setChecked(self.rig.auto_crop_lir)

            if hasattr(self, 'ctrl_curvature'):
                self.ctrl_curvature.blockSignals(True)
                self.ctrl_curvature.setValue(getattr(self.rig, 'field_curvature_correction', 0.0))
                self.ctrl_curvature.blockSignals(False)

            if hasattr(self, 'combo_projection'):
                self.combo_projection.blockSignals(True)
                proj = getattr(self.rig, 'projection_type', 'cylindrical')
                idx = self.combo_projection.findData(proj)
                if idx >= 0:
                    self.combo_projection.setCurrentIndex(idx)
                self.combo_projection.blockSignals(False)

            if hasattr(self, 'chk_auto_color_match'):
                self.chk_auto_color_match.blockSignals(True)
                val_acm = getattr(self.rig, 'auto_color_match', True)
                self.chk_auto_color_match.setChecked(val_acm)
                if hasattr(self, 'engine'):
                    self.engine.color_matcher.enabled = val_acm
                self.chk_auto_color_match.blockSignals(False)

            if hasattr(self, 'ctrl_color_match_strength'):
                self.ctrl_color_match_strength.blockSignals(True)
                val_cms = getattr(self.rig, 'color_match_strength', 1.0)
                self.ctrl_color_match_strength.setValue(val_cms * 100.0)
                if hasattr(self, 'engine'):
                    self.engine.color_matcher.set_strength(val_cms)
                self.ctrl_color_match_strength.blockSignals(False)

            if hasattr(self, 'ctrl_bottom_taper_x'):
                self.ctrl_bottom_taper_x.blockSignals(True)
                self.ctrl_bottom_taper_x.setValue(getattr(self.rig, 'bottom_seam_taper_x', 0.0))
                self.ctrl_bottom_taper_x.blockSignals(False)

            if hasattr(self, 'ctrl_bottom_taper_y'):
                self.ctrl_bottom_taper_y.blockSignals(True)
                self.ctrl_bottom_taper_y.setValue(getattr(self.rig, 'bottom_seam_taper_y', 0.0))
                self.ctrl_bottom_taper_y.blockSignals(False)

            if hasattr(self, 'chk_smart_seam'):
                self.chk_smart_seam.blockSignals(True)
                self.chk_smart_seam.setChecked(getattr(self.rig, 'smart_seam_enabled', False))
                self.chk_smart_seam.blockSignals(False)

            for ctrl in [
                self.ctrl_left_yaw, self.ctrl_right_yaw, self.ctrl_master_yaw,
                self.ctrl_cam_pitch, self.ctrl_left_roll, self.ctrl_right_roll,
                self.ctrl_global_pitch, self.ctrl_h_offset, self.ctrl_pano_hfov,
                self.ctrl_v_offset, self.ctrl_squeeze, self.ctrl_blend_width,
                self.ctrl_cam_fov, self.ctrl_safety_margin, self.chk_auto_crop
            ]:
                ctrl.blockSignals(False)

            self.update_corner_spinboxes_from_rig()
            self.viewport.set_rig(self.rig)

            if hasattr(self, 'combo_preset'):
                cam_name = self.rig.left_camera.name
                w = self.rig.left_camera.image_width
                h = self.rig.left_camera.image_height
                fov = self.rig.left_camera.hfov_deg
                for idx in range(self.combo_preset.count()):
                    k = self.combo_preset.itemData(idx)
                    if k in CAMERA_PRESETS:
                        p = CAMERA_PRESETS[k]
                        if p.name == cam_name or (p.image_width == w and p.image_height == h and abs(p.hfov_deg - fov) < 1.0):
                            self.combo_preset.blockSignals(True)
                            self.combo_preset.setCurrentIndex(idx)
                            self.combo_preset.blockSignals(False)
                            break

        # 2. Sync / Frame offset
        if "sync" in data and isinstance(data["sync"], dict):
            sync_offset = data["sync"].get("frame_offset_right", 0)
            if hasattr(self, 'spin_offset'):
                self.spin_offset.blockSignals(True)
                self.spin_offset.setValue(sync_offset)
                self.spin_offset.blockSignals(False)
            self.engine.frame_offset_right = sync_offset

        # 3. AutoCam / AI Broadcast Settings
        if "autocam" in data and isinstance(data["autocam"], dict):
            ac = data["autocam"]
            if hasattr(self, 'chk_ai_yolo') and "ai_tracking" in ac:
                self.chk_ai_yolo.setChecked(bool(ac["ai_tracking"]))
                self.engine.ai_broadcast.config.ai_tracking = bool(ac["ai_tracking"])
            
            if hasattr(self, 'combo_ai_model') and "tracking_mode" in ac:
                idx = self.combo_ai_model.findData(ac["tracking_mode"])
                if idx >= 0:
                    self.combo_ai_model.setCurrentIndex(idx)
                    self.engine.ai_broadcast.config.tracking_mode = ac["tracking_mode"]

            if hasattr(self, 'ctrl_ball_weight') and "ball_weight" in ac:
                self.ctrl_ball_weight.setValue(float(ac["ball_weight"]))
                self.engine.ai_broadcast.config.ball_weight = float(ac["ball_weight"])

            if hasattr(self, 'ctrl_players_weight') and "players_weight" in ac:
                self.ctrl_players_weight.setValue(float(ac["players_weight"]))
                self.engine.ai_broadcast.config.players_weight = float(ac["players_weight"])
            
            if hasattr(self, 'chk_dynamic_zoom') and "enable_dynamic_zoom" in ac:
                self.chk_dynamic_zoom.setChecked(bool(ac["enable_dynamic_zoom"]))
                self.engine.ai_broadcast.config.enable_dynamic_zoom = bool(ac["enable_dynamic_zoom"])

            if hasattr(self, 'ctrl_fixed_zoom') and "fixed_zoom_factor" in ac:
                self.ctrl_fixed_zoom.setValue(float(ac["fixed_zoom_factor"]))
                self.engine.ai_broadcast.config.fixed_zoom_factor = float(ac["fixed_zoom_factor"])

            if hasattr(self, 'ctrl_min_zoom') and "min_zoom" in ac:
                self.ctrl_min_zoom.setValue(float(ac["min_zoom"]))
                self.engine.ai_broadcast.config.min_zoom = float(ac["min_zoom"])

            if hasattr(self, 'ctrl_max_zoom') and "max_zoom" in ac:
                self.ctrl_max_zoom.setValue(float(ac["max_zoom"]))
                self.engine.ai_broadcast.config.max_zoom = float(ac["max_zoom"])

            if hasattr(self, 'ctrl_zoom_speed') and "zoom_speed" in ac:
                self.ctrl_zoom_speed.setValue(float(ac["zoom_speed"]))
                self.engine.ai_broadcast.config.zoom_speed = float(ac["zoom_speed"])

            if hasattr(self, 'ctrl_ac_lead') and "anticipation_lead" in ac:
                self.ctrl_ac_lead.setValue(float(ac["anticipation_lead"]))
                self.engine.ai_broadcast.config.anticipation_lead = float(ac["anticipation_lead"])

            if hasattr(self, 'ctrl_ac_smooth') and "smoothing_factor" in ac:
                self.ctrl_ac_smooth.setValue(float(ac["smoothing_factor"]))
                self.engine.ai_broadcast.config.smoothing_factor = float(ac["smoothing_factor"])

            if hasattr(self, 'ctrl_ac_deadband') and "deadband_width" in ac:
                self.ctrl_ac_deadband.setValue(float(ac["deadband_width"]))
                self.engine.ai_broadcast.config.deadband_width = float(ac["deadband_width"])

            if hasattr(self, 'ctrl_ac_speed') and "max_pan_speed" in ac:
                self.ctrl_ac_speed.setValue(float(ac["max_pan_speed"]))
                self.engine.ai_broadcast.config.max_pan_speed = float(ac["max_pan_speed"])

            if hasattr(self, 'ctrl_ac_vpos') and "vertical_center_bias" in ac:
                self.ctrl_ac_vpos.setValue(float(ac["vertical_center_bias"]))
                self.engine.ai_broadcast.config.vertical_center_bias = float(ac["vertical_center_bias"])

            if hasattr(self, 'chk_zone_damping') and "enable_zone_damping" in ac:
                self.chk_zone_damping.setChecked(bool(ac["enable_zone_damping"]))
                self.engine.ai_broadcast.config.enable_zone_damping = bool(ac["enable_zone_damping"])

            if hasattr(self, 'ctrl_zone_boost') and "zone_boost_factor" in ac:
                self.ctrl_zone_boost.setValue(float(ac["zone_boost_factor"]))
                self.engine.ai_broadcast.config.zone_boost_factor = float(ac["zone_boost_factor"])

            if hasattr(self, 'ctrl_tactical_margin') and "tactical_margin" in ac:
                self.ctrl_tactical_margin.setValue(float(ac["tactical_margin"]))
                self.engine.ai_broadcast.config.tactical_margin = float(ac["tactical_margin"])

            if "pitch_corners" in ac and isinstance(ac["pitch_corners"], list):
                corners = ac["pitch_corners"]
                # Convert legacy 4-corner configs to 6-point configs
                if len(corners) == 4:
                    tl, tr, br, bl = corners
                    tc = [(tl[0] + tr[0]) * 0.5, (tl[1] + tr[1]) * 0.5]
                    bc = [(bl[0] + br[0]) * 0.5, (bl[1] + br[1]) * 0.5]
                    corners = [tl, tc, tr, br, bc, bl]
                if hasattr(self, 'spin_pitch_corners') and len(self.spin_pitch_corners) == 6:
                    for idx, (sx, sy) in enumerate(self.spin_pitch_corners):
                        if idx < len(corners):
                            sx.blockSignals(True)
                            sy.blockSignals(True)
                            sx.setValue(float(corners[idx][0]) * 100.0)
                            sy.setValue(float(corners[idx][1]) * 100.0)
                            sx.blockSignals(False)
                            sy.blockSignals(False)
                self.engine.ai_broadcast.config.pitch_corners = [[float(c[0]), float(c[1])] for c in corners]
                if hasattr(self, 'viewport'):
                    self.viewport.set_pitch_corners(self.engine.ai_broadcast.config.pitch_corners)

            if hasattr(self, 'combo_scan_speed') and "scan_step" in ac:
                idx = self.combo_scan_speed.findData(int(ac["scan_step"]))
                if idx >= 0:
                    self.combo_scan_speed.setCurrentIndex(idx)
                    self.engine.ai_broadcast.config.scan_step = int(ac["scan_step"])

            if hasattr(self, 'chk_fp16') and "use_fp16" in ac:
                self.chk_fp16.setChecked(bool(ac["use_fp16"]))
                self.engine.ai_broadcast.config.use_fp16 = bool(ac["use_fp16"])

            if hasattr(self, 'chk_show_radar') and "enable_radar" in ac:
                self.chk_show_radar.setChecked(bool(ac["enable_radar"]))
                self.viewport.set_radar_enabled(bool(ac["enable_radar"]))
                self.engine.ai_broadcast.config.enable_radar = bool(ac["enable_radar"])

            if hasattr(self, 'combo_radar_pos') and "radar_position" in ac:
                idx = self.combo_radar_pos.findData(ac["radar_position"])
                if idx >= 0:
                    self.combo_radar_pos.setCurrentIndex(idx)
                    self.viewport.set_radar_position(ac["radar_position"])
                    self.engine.ai_broadcast.config.radar_position = ac["radar_position"]

            if hasattr(self, 'ctrl_radar_width') and "radar_width" in ac:
                self.ctrl_radar_width.setValue(float(ac["radar_width"]))
                self.viewport.set_radar_width(int(ac["radar_width"]))
                self.engine.ai_broadcast.config.radar_width = int(ac["radar_width"])

            if hasattr(self, 'ctrl_radar_opacity') and "radar_opacity" in ac:
                self.ctrl_radar_opacity.setValue(float(ac["radar_opacity"]))
                self.viewport.set_radar_opacity(float(ac["radar_opacity"]))
                self.engine.ai_broadcast.config.radar_opacity = float(ac["radar_opacity"])

            if "burn_radar_in_export" in ac:
                burn_val = bool(ac["burn_radar_in_export"])
                if hasattr(self, 'chk_burn_radar'):
                    self.chk_burn_radar.setChecked(burn_val)
                if hasattr(self, 'chk_exp_burn_radar'):
                    self.chk_exp_burn_radar.setChecked(burn_val)
                self.engine.ai_broadcast.config.burn_radar_in_export = burn_val

            if hasattr(self, 'combo_ai_engine') and "ai_backend" in ac:
                idx = self.combo_ai_engine.findData(ac["ai_backend"])
                if idx >= 0:
                    self.combo_ai_engine.setCurrentIndex(idx)
                    self.engine.ai_broadcast.set_backend(ac["ai_backend"])

            if "enable_offside_line" in ac:
                self.toggle_offside_line(bool(ac["enable_offside_line"]))

            if "enable_skeletons" in ac:
                self.toggle_skeletons(bool(ac["enable_skeletons"]))

            if "enable_speed_badges" in ac:
                self.toggle_speed_badges(bool(ac["enable_speed_badges"]))

            if hasattr(self, 'chk_anti_jersey') and "enable_anti_jersey_filter" in ac:
                self.chk_anti_jersey.setChecked(bool(ac["enable_anti_jersey_filter"]))
                self.engine.ai_broadcast.config.enable_anti_jersey_filter = bool(ac["enable_anti_jersey_filter"])

            if hasattr(self, 'chk_static_leaf') and "enable_static_leaf_filter" in ac:
                self.chk_static_leaf.setChecked(bool(ac["enable_static_leaf_filter"]))
                self.engine.ai_broadcast.config.enable_static_leaf_filter = bool(ac["enable_static_leaf_filter"])

            if hasattr(self, 'chk_sideline_filter') and "enable_sideline_filter" in ac:
                val_sf = bool(ac["enable_sideline_filter"])
                self.chk_sideline_filter.setChecked(val_sf)
                self.engine.ai_broadcast.config.enable_sideline_filter = val_sf
                if hasattr(self, 'chk_sideline_menu'):
                    self.chk_sideline_menu.blockSignals(True)
                    self.chk_sideline_menu.setChecked(val_sf)
                    self.chk_sideline_menu.blockSignals(False)

            if hasattr(self, 'ctrl_pitch_margin') and "pitch_margin_pct" in ac:
                self.ctrl_pitch_margin.setValue(float(ac["pitch_margin_pct"]))
                self.engine.ai_broadcast.config.pitch_margin_pct = float(ac["pitch_margin_pct"])

            if hasattr(self, 'chk_strict_ball_center') and "enable_strict_ball_centering" in ac:
                self.chk_strict_ball_center.setChecked(bool(ac["enable_strict_ball_centering"]))
                self.engine.ai_broadcast.config.enable_strict_ball_centering = bool(ac["enable_strict_ball_centering"])

        # 4. Export Settings
        if "export" in data and isinstance(data["export"], dict):
            exp = data["export"]
            if hasattr(self, 'combo_exp_format') and "format" in exp:
                idx = self.combo_exp_format.findData(exp["format"])
                if idx >= 0:
                    self.combo_exp_format.setCurrentIndex(idx)
            
            if hasattr(self, 'combo_resolution') and "resolution_width" in exp and "resolution_height" in exp:
                w, h = exp["resolution_width"], exp["resolution_height"]
                idx = self.combo_resolution.findData((w, h))
                if idx >= 0:
                    self.combo_resolution.setCurrentIndex(idx)
            
            if hasattr(self, 'combo_codec') and "codec" in exp:
                idx = self.combo_codec.findData(exp["codec"])
                if idx >= 0:
                    self.combo_codec.setCurrentIndex(idx)

            if hasattr(self, 'ctrl_bitrate') and "bitrate_mbps" in exp:
                self.ctrl_bitrate.setValue(float(exp["bitrate_mbps"]))

            if hasattr(self, 'chk_export_trim_only') and "trim_only" in exp:
                self.chk_export_trim_only.setChecked(bool(exp["trim_only"]))

            if hasattr(self, 'combo_audio_source') and "audio_source" in exp:
                idx = self.combo_audio_source.findData(exp["audio_source"])
                if idx >= 0:
                    self.combo_audio_source.setCurrentIndex(idx)

        # 4b. Wide-Angle Settings
        if "wide_angle" in data and isinstance(data["wide_angle"], dict) and hasattr(self, 'wide_angle_config'):
            try:
                self.wide_angle_config = WideAngleConfig.from_dict(data["wide_angle"])
                self.wide_angle_rectifier.config = self.wide_angle_config
                self.wide_angle_rectifier.update_maps()
                self._sync_wide_angle_ui_from_config()
                if hasattr(self, 'viewport'):
                    self.viewport.set_wide_angle_corners(self.wide_angle_config.corners)
            except Exception as e:
                logger.warning(f"Konnte Weitwinkel-Konfiguration nicht laden: {e}")

        # 4b. Cloud Settings (Non-sensitive preferences only)
        if "cloud" in data and isinstance(data["cloud"], dict):
            cld = data["cloud"]
            if "server_url" in cld and cld["server_url"]:
                self.saved_cloud_server_url = str(cld["server_url"])
            if "team_id" in cld and cld["team_id"]:
                self.saved_cloud_team_id = str(cld["team_id"])
            if hasattr(self, 'combo_cloud_category') and "category" in cld:
                idx = self.combo_cloud_category.findData(cld["category"])
                if idx >= 0:
                    self.combo_cloud_category.setCurrentIndex(idx)
            if hasattr(self, 'chk_cloud_auto_upload') and "auto_upload" in cld:
                self.chk_cloud_auto_upload.setChecked(bool(cld["auto_upload"]))
            if hasattr(self, 'chk_cloud_notify') and "notify_team" in cld:
                self.chk_cloud_notify.setChecked(bool(cld["notify_team"]))
            if hasattr(self, 'chk_cloud_fast_hls') and "fast_hls" in cld:
                self.chk_cloud_fast_hls.setChecked(bool(cld["fast_hls"]))
            if hasattr(self, 'combo_cloud_concurrency') and "concurrency" in cld:
                idx = self.combo_cloud_concurrency.findData(cld["concurrency"])
                if idx >= 0:
                    self.combo_cloud_concurrency.setCurrentIndex(idx)

        # 4c. Live-Stream Preferences
        if "livestream" in data and isinstance(data["livestream"], dict):
            ls = data["livestream"]
            if hasattr(self, 'txt_rtmp_server_url') and "server_url" in ls and ls["server_url"]:
                self.txt_rtmp_server_url.setText(str(ls["server_url"]))

            if hasattr(self, 'combo_stream_resolution') and "resolution" in ls:
                idx = self.combo_stream_resolution.findData(ls["resolution"])
                if idx >= 0:
                    self.combo_stream_resolution.setCurrentIndex(idx)

            # Keys
            k1 = ls.get("stream_key_c1", ls.get("stream_key_bcast", "broadcast_1"))
            k2 = ls.get("stream_key_c2", ls.get("stream_key_pano", "panorama_1"))
            k3 = ls.get("stream_key_c3", ls.get("stream_key_cam1", "cam1"))
            k4 = ls.get("stream_key_c4", ls.get("stream_key_cam2", "cam2"))
            if hasattr(self, 'txt_stream_key_c1'): self.txt_stream_key_c1.setText(str(k1))
            if hasattr(self, 'txt_stream_key_c2'): self.txt_stream_key_c2.setText(str(k2))
            if hasattr(self, 'txt_stream_key_c3'): self.txt_stream_key_c3.setText(str(k3))
            if hasattr(self, 'txt_stream_key_c4'): self.txt_stream_key_c4.setText(str(k4))

            # Feeds / Sources
            for combo, key, default in [
                (getattr(self, 'combo_feed_c1', None), "source_c1", "broadcast_16x9"),
                (getattr(self, 'combo_feed_c2', None), "source_c2", "panorama_8x3"),
                (getattr(self, 'combo_feed_c3', None), "source_c3", "cam1"),
                (getattr(self, 'combo_feed_c4', None), "source_c4", "cam2")
            ]:
                if combo is not None:
                    src = ls.get(key, default)
                    idx = combo.findData(src)
                    if idx >= 0:
                        combo.setCurrentIndex(idx)

            # Channels Enabled
            e1 = ls.get("chan_c1_enabled", ls.get("chan_bcast_enabled", True))
            e2 = ls.get("chan_c2_enabled", ls.get("chan_pano_enabled", True))
            e3 = ls.get("chan_c3_enabled", ls.get("chan_cam1_enabled", False))
            e4 = ls.get("chan_c4_enabled", ls.get("chan_cam2_enabled", False))
            if hasattr(self, 'chk_chan_1'): self.chk_chan_1.setChecked(bool(e1))
            if hasattr(self, 'chk_chan_2'): self.chk_chan_2.setChecked(bool(e2))
            if hasattr(self, 'chk_chan_3'): self.chk_chan_3.setChecked(bool(e3))
            if hasattr(self, 'chk_chan_4'): self.chk_chan_4.setChecked(bool(e4))

            if hasattr(self, 'txt_dji_wifi_ssid') and "wifi_ssid" in ls:
                self.txt_dji_wifi_ssid.setText(str(ls["wifi_ssid"]))
            if hasattr(self, 'obs_settings_widget'):
                if "obs_settings" in ls and isinstance(ls["obs_settings"], dict):
                    self.obs_settings_widget.set_settings(ls["obs_settings"])
                elif "bitrate_kbps" in ls:
                    self.obs_settings_widget.spn_bitrate.setValue(int(ls["bitrate_kbps"]))
            elif hasattr(self, 'combo_stream_bitrate') and "bitrate_kbps" in ls:
                idx = self.combo_stream_bitrate.findData(ls["bitrate_kbps"])
                if idx >= 0:
                    self.combo_stream_bitrate.setCurrentIndex(idx)

        # 5. UI Preferences & Overlays
        if "ui_view" in data and isinstance(data["ui_view"], dict):
            ui = data["ui_view"]
            if hasattr(self, 'combo_view_mode') and "view_mode" in ui:
                idx = self.combo_view_mode.findData(ui["view_mode"])
                if idx >= 0:
                    self.combo_view_mode.setCurrentIndex(idx)
            
            if hasattr(self, 'chk_show_frame') and "show_frame" in ui:
                self.chk_show_frame.setChecked(bool(ui["show_frame"]))
            if hasattr(self, 'ctrl_frame_opacity') and "frame_opacity" in ui:
                self.ctrl_frame_opacity.setValue(float(ui["frame_opacity"]))
            if hasattr(self, 'chk_show_center_line') and "show_center_line" in ui:
                self.chk_show_center_line.setChecked(bool(ui["show_center_line"]))
            if hasattr(self, 'chk_show_autocam') and "show_autocam" in ui:
                self.chk_show_autocam.setChecked(bool(ui["show_autocam"]))
            if hasattr(self, 'chk_show_ball') and "show_ball" in ui:
                self.chk_show_ball.setChecked(bool(ui["show_ball"]))
            if "show_radar" in ui:
                r_show = bool(ui["show_radar"])
                if hasattr(self, 'chk_show_radar'):
                    self.chk_show_radar.setChecked(r_show)
                if hasattr(self, 'chk_show_radar_menu'):
                    self.chk_show_radar_menu.setChecked(r_show)
                self.viewport.set_radar_enabled(r_show)
            if "show_offside_line" in ui:
                self.toggle_offside_line(bool(ui["show_offside_line"]))
            if "show_skeletons" in ui:
                self.toggle_skeletons(bool(ui["show_skeletons"]))
            if "show_speed" in ui:
                self.toggle_speed_badges(bool(ui["show_speed"]))
            if "show_sideline_filter" in ui:
                self.toggle_sideline_filter(bool(ui["show_sideline_filter"]))
            if hasattr(self, 'chk_show_seam') and "show_seam" in ui:
                self.chk_show_seam.setChecked(bool(ui["show_seam"]))
            if hasattr(self, 'chk_show_grid') and "show_grid" in ui:
                self.chk_show_grid.setChecked(bool(ui["show_grid"]))
            if hasattr(self, 'chk_sync_yaw') and "sync_angles" in ui:
                self.chk_sync_yaw.setChecked(bool(ui["sync_angles"]))
                self.sync_angles = bool(ui["sync_angles"])
            if hasattr(self, 'chk_multi_frame_calib') and "multi_frame_calib" in ui:
                self.chk_multi_frame_calib.setChecked(bool(ui["multi_frame_calib"]))

        self.engine.invalidate_luts()
        self._update_format_combo_labels()
        self.update_resolution_dropdown()
        self.refresh_preview()

    def save_full_profile_as(self):
        """Opens file dialog and saves complete application settings to JSON."""
        file_path, _ = QFileDialog.getSaveFileName(self, "Gesamte Einstellungen & Profil speichern", "matchtrack_profil.json", "MatchTrack Profil (*.json)")
        if file_path:
            settings_dict = self.get_current_settings_dict()
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(settings_dict, f, indent=4, ensure_ascii=False)
                logger.info(f"Gesamte Einstellungen erfolgreich gespeichert in: {file_path}")
                self.status_bar.showMessage(f"💾 Einstellungen gespeichert: {os.path.basename(file_path)}")
                QMessageBox.information(self, "Profil gespeichert", f"Alle Einstellungen (Rig, KI-AutoCam, Export & UI) wurden erfolgreich gespeichert in:\n{file_path}")
            except Exception as e:
                logger.error(f"Fehler beim Speichern des Profils {file_path}: {e}", exc_info=True)
                QMessageBox.critical(self, "Fehler", f"Konnte Einstellungen nicht speichern:\n{e}")

    def load_full_profile_from(self):
        """Opens file dialog and loads complete application settings from JSON."""
        file_path, _ = QFileDialog.getOpenFileName(self, "Einstellungen & Profil laden", "", "JSON Konfiguration (*.json)")
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self.apply_settings_dict(data)
                logger.info(f"Gesamte Einstellungen erfolgreich geladen aus: {file_path}")
                self.status_bar.showMessage(f"📁 Einstellungen geladen: {os.path.basename(file_path)}")
                QMessageBox.information(self, "Profil geladen", f"Einstellungen wurden erfolgreich angewendet aus:\n{os.path.basename(file_path)}")
            except Exception as e:
                logger.error(f"Fehler beim Laden des Profils {file_path}: {e}", exc_info=True)
                QMessageBox.critical(self, "Fehler", f"Konnte Einstellungen nicht laden:\n{e}")

    def save_as_default_settings(self):
        """Saves current settings to default_settings.json for automatic startup loading."""
        default_path = get_default_settings_path()
        settings_dict = self.get_current_settings_dict()
        try:
            os.makedirs(os.path.dirname(os.path.abspath(default_path)), exist_ok=True)
            with open(default_path, 'w', encoding='utf-8') as f:
                json.dump(settings_dict, f, indent=4, ensure_ascii=False)
            logger.info(f"Aktuelle Einstellungen als Start-Standard gespeichert in: {default_path}")
            self.status_bar.showMessage("⭐ Einstellungen als Standard für nächste Starts gespeichert!")
            QMessageBox.information(self, "Standard gespeichert", "⭐ Deine aktuellen Einstellungen wurden als Standard gespeichert!\n\nSie werden beim nächsten Start automatisch wiederhergestellt.")
        except Exception as e:
            logger.error(f"Fehler beim Speichern der Standard-Einstellungen: {e}", exc_info=True)
            QMessageBox.critical(self, "Fehler", f"Konnte Standard-Einstellungen nicht speichern:\n{e}")

    def restore_factory_defaults(self):
        """Restores factory default settings."""
        res = QMessageBox.question(
            self,
            "Werkseinstellungen wiederherstellen",
            "Möchten Sie alle Einstellungen wirklich auf die Werkseinstellungen (DJI Action 4 80° Standard) zurücksetzen?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        if res == QMessageBox.Yes:
            default_rig = RigConfiguration()
            default_dict = {
                "rig": default_rig.to_dict(),
                "sync": {"frame_offset_right": 0},
                "autocam": {
                    "ai_tracking": False,
                    "tracking_mode": "hybrid_fusion",
                    "enable_dynamic_zoom": True,
                    "fixed_zoom_factor": 1.0,
                    "min_zoom": 1.15,
                    "max_zoom": 1.60,
                    "zoom_speed": 0.04,
                    "anticipation_lead": 0.30,
                    "smoothing_factor": 0.92,
                    "deadband_width": 0.04,
                    "max_pan_speed": 0.06,
                    "vertical_center_bias": 0.55
                },
                "export": {
                    "format": "32:9",
                    "resolution_width": 3840,
                    "resolution_height": 1080,
                    "codec": "hevc_nvenc",
                    "bitrate_mbps": 50,
                    "trim_only": True
                },
                "ui_view": {
                    "view_mode": "32:9",
                    "show_frame": True,
                    "frame_opacity": 50.0,
                    "show_center_line": True,
                    "show_autocam": True,
                    "show_ball": True,
                    "show_seam": True,
                    "show_grid": True,
                    "sync_angles": True,
                    "multi_frame_calib": True
                }
            }
            self.apply_settings_dict(default_dict)
            logger.info("Werkseinstellungen wiederhergestellt.")
            self.status_bar.showMessage("↺ Werkseinstellungen wiederhergestellt.")

    def load_default_startup_settings(self):
        """Loads user default settings if available, else standard rig."""
        default_path = get_default_settings_path()
        if os.path.exists(default_path):
            try:
                with open(default_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self.apply_settings_dict(data)
                logger.info(f"Benutzer-Standardeinstellungen automatisch geladen: {default_path}")
                return
            except Exception as e:
                logger.warning(f"Konnte Standard-Einstellungen nicht laden ({e}), verwende Werkseinstellungen.")

        # Fallback to default_rig_action4_80deg.json if present
        bundled_default = get_resource_path("default_rig_action4_80deg.json")
        if os.path.exists(bundled_default):
            try:
                with open(bundled_default, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self.apply_settings_dict(data)
                logger.info(f"Standard Rig-Profil geladen: {bundled_default}")
            except Exception:
                pass

    def closeEvent(self, event):
        """Auto-save current settings on clean close."""
        try:
            if hasattr(self, 'gui_handler') and hasattr(self, '_log_sub_cb'):
                self.gui_handler.unsubscribe(self._log_sub_cb)
            elif hasattr(self, 'gui_handler'):
                self.gui_handler.unsubscribe(self.append_log_message)
            default_path = get_default_settings_path()
            settings_dict = self.get_current_settings_dict()
            os.makedirs(os.path.dirname(os.path.abspath(default_path)), exist_ok=True)
            with open(default_path, 'w', encoding='utf-8') as f:
                json.dump(settings_dict, f, indent=4, ensure_ascii=False)
            logger.info("Aktuelle Einstellungen beim Beenden automatisch als Standard gesichert.")
        except Exception as e:
            logger.debug(f"Auto-Save beim Beenden: {e}")
        super().closeEvent(event)

    # Legacy aliases
    def save_rig_profile(self):
        self.save_full_profile_as()

    def load_rig_profile(self):
        self.load_full_profile_from()

    def toggle_frame_overlay(self, checked: bool):
        self.viewport.set_transparent_frame_enabled(checked)

    def on_frame_opacity_changed(self, val: float):
        self.viewport.set_frame_opacity(val / 100.0)

    def on_corner_pins_toggled(self, checked: bool):
        if hasattr(self, 'btn_corner_pins_tb') and self.btn_corner_pins_tb is not None and self.btn_corner_pins_tb.isChecked() != checked:
            self.btn_corner_pins_tb.blockSignals(True)
            self.btn_corner_pins_tb.setChecked(checked)
            self.btn_corner_pins_tb.blockSignals(False)

        if hasattr(self, 'btn_corner_pins_side') and self.btn_corner_pins_side.isChecked() != checked:
            self.btn_corner_pins_side.blockSignals(True)
            self.btn_corner_pins_side.setChecked(checked)
            self.btn_corner_pins_side.blockSignals(False)

        self.viewport.set_corner_pin_mode(checked)
        if checked:
            self.status_bar.showMessage("📐 Ecken-Verzerrung aktiviert: Ziehen Sie die runden Eckpunkte im Bild per Drag-and-Drop...")
        else:
            self.status_bar.showMessage("Bereit")

    def on_corner_cam_changed(self, idx: int):
        cam_side = self.combo_corner_cam.currentData()
        self.viewport.set_corner_pin_camera(cam_side)
        if hasattr(self, 'tab_corners'):
            if cam_side == "left":
                self.tab_corners.setCurrentIndex(0)
            elif cam_side == "right":
                self.tab_corners.setCurrentIndex(1)

    def on_viewport_corner_dragged(self, side: str, idx: int, norm_x: float, norm_y: float):
        """Called live in real-time when user drags a corner handle in the viewport."""
        if side == "left":
            self.rig.left_pose.corners[idx] = [norm_x, norm_y]
            if hasattr(self, 'spin_corners_l') and idx < len(self.spin_corners_l):
                sx, sy = self.spin_corners_l[idx]
                sx.blockSignals(True)
                sy.blockSignals(True)
                sx.setValue(norm_x * 100.0)
                sy.setValue(norm_y * 100.0)
                sx.blockSignals(False)
                sy.blockSignals(False)
        else:
            self.rig.right_pose.corners[idx] = [norm_x, norm_y]
            if hasattr(self, 'spin_corners_r') and idx < len(self.spin_corners_r):
                sx, sy = self.spin_corners_r[idx]
                sx.blockSignals(True)
                sy.blockSignals(True)
                sx.setValue(norm_x * 100.0)
                sy.setValue(norm_y * 100.0)
                sx.blockSignals(False)
                sy.blockSignals(False)

        self.viewport.set_rig(self.rig)
        self.engine.invalidate_luts()
        self.trigger_smooth_preview()

    def on_viewport_corner_drag_finished(self):
        self.render_high_res_preview()

    def on_corner_spin_changed(self, side: str, idx: int):
        """Called when user manually edits X/Y spinboxes in the side panel."""
        if side == "left":
            if hasattr(self, 'spin_corners_l') and idx < len(self.spin_corners_l):
                sx, sy = self.spin_corners_l[idx]
                self.rig.left_pose.corners[idx] = [sx.value() / 100.0, sy.value() / 100.0]
        else:
            if hasattr(self, 'spin_corners_r') and idx < len(self.spin_corners_r):
                sx, sy = self.spin_corners_r[idx]
                self.rig.right_pose.corners[idx] = [sx.value() / 100.0, sy.value() / 100.0]

        self.viewport.set_rig(self.rig)
        self.engine.invalidate_luts()
        self.trigger_smooth_preview()

    def reset_single_corner(self, side: str, idx: int, def_x: float, def_y: float):
        if side == "left":
            if hasattr(self, 'spin_corners_l') and idx < len(self.spin_corners_l):
                sx, sy = self.spin_corners_l[idx]
                sx.setValue(def_x)
                sy.setValue(def_y)
        else:
            if hasattr(self, 'spin_corners_r') and idx < len(self.spin_corners_r):
                sx, sy = self.spin_corners_r[idx]
                sx.setValue(def_x)
                sy.setValue(def_y)

    def reset_camera_corners(self, side: str):
        if side == "left":
            self.rig.left_pose.reset_corners()
            defaults = [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)]
            for idx, (dx, dy) in enumerate(defaults):
                if hasattr(self, 'spin_corners_l') and idx < len(self.spin_corners_l):
                    sx, sy = self.spin_corners_l[idx]
                    sx.blockSignals(True)
                    sy.blockSignals(True)
                    sx.setValue(dx)
                    sy.setValue(dy)
                    sx.blockSignals(False)
                    sy.blockSignals(False)
        else:
            self.rig.right_pose.reset_corners()
            defaults = [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)]
            for idx, (dx, dy) in enumerate(defaults):
                if hasattr(self, 'spin_corners_r') and idx < len(self.spin_corners_r):
                    sx, sy = self.spin_corners_r[idx]
                    sx.blockSignals(True)
                    sy.blockSignals(True)
                    sx.setValue(dx)
                    sy.setValue(dy)
                    sx.blockSignals(False)
                    sy.blockSignals(False)

        self.viewport.set_rig(self.rig)
        self.engine.invalidate_luts()
        self._update_format_combo_labels()
        self.update_resolution_dropdown()
        self.refresh_preview()
        self.status_bar.showMessage(f"Ecken für Kamera {side.capitalize()} zurückgesetzt.")

    def reset_all_camera_corners(self):
        self.reset_camera_corners('left')
        self.reset_camera_corners('right')
        self.status_bar.showMessage("Alle Kamera-Ecken auf Standard zurückgesetzt.")

    def reset_all_corners(self):
        self.reset_all_camera_corners()

    def update_corner_spinboxes_from_rig(self):
        """Synchronizes spinboxes with current rig corner values."""
        if hasattr(self, 'spin_corners_l'):
            for idx, (cx, cy) in enumerate(self.rig.left_pose.corners):
                if idx < len(self.spin_corners_l):
                    sx, sy = self.spin_corners_l[idx]
                    sx.blockSignals(True)
                    sy.blockSignals(True)
                    sx.setValue(cx * 100.0)
                    sy.setValue(cy * 100.0)
                    sx.blockSignals(False)
                    sy.blockSignals(False)

        if hasattr(self, 'spin_corners_r'):
            for idx, (cx, cy) in enumerate(self.rig.right_pose.corners):
                if idx < len(self.spin_corners_r):
                    sx, sy = self.spin_corners_r[idx]
                    sx.blockSignals(True)
                    sy.blockSignals(True)
                    sx.setValue(cx * 100.0)
                    sy.setValue(cy * 100.0)
                    sx.blockSignals(False)
                    sy.blockSignals(False)

    def start_video_export(self, custom_out_path: Optional[str] = None, 
                           custom_w: Optional[int] = None, 
                           custom_h: Optional[int] = None, 
                           mode: Optional[str] = None,
                           custom_start_frame: Optional[int] = None,
                           custom_end_frame: Optional[int] = None,
                           use_lookahead: Optional[bool] = None):
        if not self.has_active_video():
            QMessageBox.warning(self, "Export", "Bitte zuerst Videos laden (2 Kameras oder 32:9 Panorama)!")
            return

        fmt = mode if mode is not None else (self.combo_exp_format.currentData() if hasattr(self, 'combo_exp_format') else "16:9_autocam")
        if not self.engine.is_panorama_mode() and (not self.engine.video_left or not self.engine.video_right):
            if fmt == "wide_angle" and (self.engine.video_left or self.engine.video_right):
                pass
            else:
                QMessageBox.warning(self, "Export", "Bitte zuerst beide Kameras öffnen oder ein 32:9 Panorama laden!")
                return

        if custom_out_path:
            out_path = custom_out_path
        else:
            if fmt == "wide_angle":
                default_name = "weitwinkel_begradigt.mp4"
            elif fmt == "32:9":
                default_name = "fussball_panorama_8x3.mp4" if self.is_4_3_mode() else "fussball_panorama_32x9.mp4"
            elif fmt == "21:10":
                default_name = "fussball_panorama_21x10.mp4"
            elif fmt == "both":
                default_name = "fussball_broadcast_16x9.mp4"
            else:
                default_name = "fussball_broadcast_16x9.mp4"

            out_path, _ = QFileDialog.getSaveFileName(self, "Video speichern", default_name, "MP4 Video (*.mp4)")
            if not out_path:
                return

        if custom_w is not None and custom_h is not None:
            w, h = custom_w, custom_h
        else:
            res_val = self.combo_resolution.currentData() if hasattr(self, 'combo_resolution') else (1920, 1080)
            w, h = res_val if isinstance(res_val, (tuple, list)) else (1920, 1080)

        codec = self.combo_codec.currentData() if hasattr(self, 'combo_codec') else "hevc_nvenc"
        bitrate = int(self.ctrl_bitrate.value()) if hasattr(self, 'ctrl_bitrate') else 16
        audio_src = self.combo_audio_source.currentData() if hasattr(self, 'combo_audio_source') else "left"
        if use_lookahead is None:
            use_lookahead = self.chk_lookahead.isChecked() if hasattr(self, 'chk_lookahead') else True

        total_avail = self.engine.get_max_duration_frames()
        if custom_start_frame is not None and custom_end_frame is not None:
            start_frame = max(0, custom_start_frame)
            end_frame = min(custom_end_frame, total_avail) if total_avail > 0 else custom_end_frame
        else:
            chk_trim = getattr(self, 'chk_export_trim_only', None)
            if chk_trim and chk_trim.isChecked():
                start_frame = self.in_point_frame
                # End frame is exclusive in range(), so we pass out_point_frame + 1
                end_frame = min(self.out_point_frame + 1, total_avail)
            else:
                start_frame = 0
                end_frame = total_avail

        self.btn_start_render.setEnabled(False)
        self.btn_cancel_render.setEnabled(True)
        self.progress_bar.setValue(0)
        self.lbl_render_stats.setText("Export wird vorbereitet...")

        self.render_worker = RenderWorker(
            self.engine, 
            out_path, 
            w, 
            h, 
            fmt, 
            codec, 
            bitrate, 
            start_frame=start_frame, 
            end_frame=end_frame, 
            audio_source=audio_src,
            use_lookahead=use_lookahead,
            wide_angle_rectifier=self.wide_angle_rectifier
        )
        self.render_worker.progress_signal.connect(self.on_render_progress)
        self.render_worker.finished_signal.connect(self.on_render_finished)
        if hasattr(self, 'tab_stepper'):
            self.render_worker.progress_signal.connect(self.tab_stepper.on_export_progress)
            self.render_worker.finished_signal.connect(self.tab_stepper.on_export_finished)
            self.render_worker.preview_signal.connect(self.tab_stepper.on_export_preview_frame)
        self.render_worker.start()

    def on_render_progress(self, processed, total, fps, eta, stage_text=""):
        pct = int((processed / max(total, 1)) * 100)
        self.progress_bar.setValue(pct)
        m, s = divmod(int(eta), 60)
        prefix = f"[{stage_text}] " if stage_text else ""
        self.lbl_render_stats.setText(f"{prefix}{processed}/{total} Frames ({pct}%) | {fps:.1f} FPS | Restzeit: {m:02d}:{s:02d}")

    def on_render_finished(self, success, message):
        self.btn_start_render.setEnabled(True)
        self.btn_cancel_render.setEnabled(False)
        if success:
            self.progress_bar.setValue(100)
            self.lbl_render_stats.setText("Export erfolgreich abgeschlossen! 🎉")
            logger.info("Video-Export erfolgreich beendet.")
            r_16x9 = getattr(self.render_worker, 'rendered_16x9_path', None)
            r_32x9 = getattr(self.render_worker, 'rendered_32x9_path', None)
            if r_16x9 and hasattr(self, 'txt_cloud_video_16x9'):
                self.txt_cloud_video_16x9.setText(r_16x9)
            if r_32x9 and hasattr(self, 'txt_cloud_video_pano'):
                self.txt_cloud_video_pano.setText(r_32x9)
            if hasattr(self, 'chk_cloud_auto_upload') and self.chk_cloud_auto_upload.isChecked():
                logger.info("Starte automatischen Cloud-Upload zu MatchTrack Online nach erfolgreichem Rendering...")
                self._start_cloud_upload(video_16x9_path=r_16x9, video_32x9_path=r_32x9)
            else:
                QMessageBox.information(self, "Export fertig", message)
        else:
            self.lbl_render_stats.setText("Export abgebrochen / Fehler aufgetreten (siehe Log).")
            logger.warning(f"Video-Export nicht erfolgreich: {message}")
            dlg = QMessageBox(self)
            dlg.setIcon(QMessageBox.Warning)
            dlg.setWindowTitle("Export Status")
            dlg.setText(f"{message}\n\nMöchten Sie das System-Log (matchtrack.log) öffnen, um Details zum Fehler zu sehen?")
            btn_log = dlg.addButton("📋 System-Log öffnen", QMessageBox.AcceptRole)
            btn_close = dlg.addButton("Schließen", QMessageBox.RejectRole)
            dlg.exec()
            if dlg.clickedButton() == btn_log:
                log_path = get_log_file_path()
                if os.path.exists(log_path):
                    try:
                        os.startfile(log_path)
                    except Exception as e:
                        logger.error(f"Konnte Logdatei nicht öffnen: {e}")
                else:
                    QMessageBox.information(self, "Logdatei", f"Logdatei noch nicht vorhanden:\n{log_path}")

    def cancel_video_export(self):
        if self.render_worker and self.render_worker.isRunning():
            logger.info("Benutzer hat Export-Abbruch angefordert.")
            self.engine.cancel_render()
            self.lbl_render_stats.setText("Abbruch wird ausgeführt...")
            if hasattr(self, 'tab_stepper') and hasattr(self.tab_stepper, 'on_export_cancelled'):
                self.tab_stepper.on_export_cancelled()

    # =========================================================================
    # MatchTrack Online Cloud Upload Integration
    # =========================================================================

    def _init_stitching_tab(self):
        """Initializes Tab 1: Stitching, Video Inputs, Audio Sync, Naht & Blending."""
        build_stitching_tab(self)

    def _create_codec_combobox(self):
        return create_codec_combobox()

    def _create_audio_combobox(self, include_mix: bool = True):
        return create_audio_combobox(include_mix=include_mix)

    def _create_export_upload_subpage(self, domain: str):
        return create_export_upload_subpage(self, domain)

    def _init_wide_angle_tab(self):
        build_wide_angle_tab(self)

    def _init_autocam_tab(self):
        build_autocam_tab(self)

    def _init_livestream_tab(self):
        build_livestream_tab(self)

    def _init_cloud_tab(self):
        build_cloud_tab(self)

    def _init_landing_page(self):
        self.landing_widget = build_landing_page(self)

    def _init_stepper_wizard(self):
        from .tabs.stepper_wizard import StepperWizardWidget
        self.tab_stepper = StepperWizardWidget(self)
        if hasattr(self, 'viewport'):
            if hasattr(self.viewport, 'field_marking_changed'):
                self.viewport.field_marking_changed.connect(self._on_field_marking_changed)
            if hasattr(self.viewport, 'framing_changed'):
                self.viewport.framing_changed.connect(self._on_framing_changed)

    def _on_field_marking_changed(self, fm):
        pass

    def _on_framing_changed(self, min_z: float, max_z: float, cx: float, cy: float):
        if hasattr(self, 'ctrl_min_zoom'):
            self.ctrl_min_zoom.blockSignals(True)
            self.ctrl_min_zoom.setValue(min_z)
            self.ctrl_min_zoom.blockSignals(False)
        if hasattr(self, 'ctrl_max_zoom'):
            self.ctrl_max_zoom.blockSignals(True)
            self.ctrl_max_zoom.setValue(max_z)
            self.ctrl_max_zoom.blockSignals(False)
        if hasattr(self, 'ctrl_ac_vpos'):
            self.ctrl_ac_vpos.blockSignals(True)
            self.ctrl_ac_vpos.setValue(cy)
            self.ctrl_ac_vpos.blockSignals(False)
        if hasattr(self, 'tab_stepper'):
            st = self.tab_stepper
            if hasattr(st, 'slider_min_zoom'):
                st.slider_min_zoom.blockSignals(True)
                st.slider_min_zoom.setValue(min_z)
                st.slider_min_zoom.blockSignals(False)
            if hasattr(st, 'slider_max_zoom'):
                st.slider_max_zoom.blockSignals(True)
                st.slider_max_zoom.setValue(max_z)
                st.slider_max_zoom.blockSignals(False)
            if hasattr(st, 'slider_vpos'):
                st.slider_vpos.blockSignals(True)
                st.slider_vpos.setValue(cy)
                st.slider_vpos.blockSignals(False)
            if hasattr(st, '_mark_custom_framing'):
                st._mark_custom_framing()
            st.update_framing_previews()

    def open_pre_stitched_panorama(self):
        return self.open_panorama_video()

    def start_export(self, custom_out_path=None, custom_w=None, custom_h=None, mode=None, custom_start_frame=None, custom_end_frame=None, use_lookahead=None):
        return self.start_video_export(custom_out_path, custom_w, custom_h, mode, custom_start_frame, custom_end_frame, use_lookahead)


def launch_gui():
    setup_logging()
    install_crash_handlers()
    logger.info("Starte MatchTrack-Stitcher Benutzeroberfläche...")

    # Install Qt message handler to catch native Qt warnings and errors
    from PySide6.QtCore import qInstallMessageHandler, QtMsgType
    def qt_message_handler(mode, context, message):
        if mode == QtMsgType.QtDebugMsg:
            logger.debug(f"[Qt] {message}")
        elif mode == QtMsgType.QtInfoMsg:
            logger.info(f"[Qt] {message}")
        elif mode == QtMsgType.QtWarningMsg:
            logger.warning(f"[Qt] {message}")
        elif mode == QtMsgType.QtCriticalMsg:
            logger.error(f"[Qt Critical] {message}")
        elif mode == QtMsgType.QtFatalMsg:
            logger.critical(f"[Qt Fatal] {message}")

    qInstallMessageHandler(qt_message_handler)

    # Set Windows Taskbar AppUserModelID for explicit app icon display
    if sys.platform == "win32":
        try:
            import ctypes
            myappid = 'matchtrack.soccer.stitcher.v1'
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
        except Exception:
            pass

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # Set Application Icon
    icon_path = os.path.join(os.path.dirname(__file__), "..", "assets", "icon.ico")
    if not os.path.exists(icon_path):
        icon_path = os.path.join(os.path.dirname(__file__), "..", "assets", "icon.png")
    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))

    window = MainWindow()
    window.show()
    sys.exit(app.exec())
