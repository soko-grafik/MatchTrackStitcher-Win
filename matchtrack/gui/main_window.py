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

from PySide6.QtCore import Qt, QTimer, QThread, Signal, Slot, QPoint, QPointF, QRectF, QObject, QEvent
from PySide6.QtGui import QImage, QPixmap, QIcon, QFont, QColor, QPainter, QPen, QWheelEvent, QMouseEvent, QPolygonF
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QSlider, QSpinBox, QDoubleSpinBox, QComboBox, QAbstractSpinBox,
    QFileDialog, QProgressBar, QGroupBox, QTabWidget, QSplitter,
    QCheckBox, QMessageBox, QFrame, QScrollArea, QStatusBar, QToolButton,
    QPlainTextEdit, QMenu, QWidgetAction
)


import json
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
from .tactics_dialog import TacticsDashboardDialog

logger = get_logger("gui")


class NoWheelSlider(QSlider):
    """
    QSlider that ignores mouse-wheel events so that scrolling over the slider
    scrolls the parent QScrollArea instead of accidentally modifying values.
    """
    def wheelEvent(self, event: QWheelEvent):
        event.ignore()


class NoWheelDoubleSpinBox(QDoubleSpinBox):
    """
    QDoubleSpinBox that ignores mouse-wheel events to prevent accidental value changes.
    """
    def wheelEvent(self, event: QWheelEvent):
        event.ignore()


class NoWheelSpinBox(QSpinBox):
    """
    QSpinBox that ignores mouse-wheel events to prevent accidental value changes.
    """
    def wheelEvent(self, event: QWheelEvent):
        event.ignore()


class NoWheelComboBox(QComboBox):
    """
    QComboBox that ignores mouse-wheel events when popup is closed to prevent accidental selection changes.
    """
    def wheelEvent(self, event: QWheelEvent):
        event.ignore()


class ScrollAreaWheelFilter(QObject):
    """
    Event filter for interactive input controls (Sliders, SpinBoxes, ComboBoxes)
    inside scroll areas: ensures mouse-wheel events scroll the enclosing QScrollArea
    instead of altering input values.
    """
    def eventFilter(self, obj, event):
        if event.type() == QEvent.Type.Wheel:
            if isinstance(obj, (QSlider, QAbstractSpinBox, QComboBox)):
                parent = obj.parentWidget()
                while parent is not None:
                    if isinstance(parent, QScrollArea):
                        QApplication.sendEvent(parent.viewport(), event)
                        return True
                    parent = parent.parentWidget()
                event.ignore()
                return True
        return False


class LabeledSliderSpinBox(QWidget):
    """
    Ergonomic combined widget with:
    - Title Label + precise formatted QDoubleSpinBox
    - Smooth horizontal QSlider underneath with real-time feedback
    - ↺ Quick-Reset Button to return to default value.
    """
    valueChanged = Signal(float)

    def __init__(self, label_text: str, min_val: float, max_val: float, default_val: float, 
                 step: float = 0.1, suffix: str = "°", decimals: int = 1, parent=None):
        super().__init__(parent)
        self.default_val = default_val
        self.step = step
        self.multiplier = int(round(1.0 / step)) if step < 1.0 else 1

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 6)
        layout.setSpacing(3)

        # Header row: Title Label + SpinBox
        header = QHBoxLayout()
        self.lbl_title = QLabel(label_text)
        self.lbl_title.setStyleSheet("font-weight: 600; color: #d4d4d8; font-size: 12px;")
        
        self.spin = NoWheelDoubleSpinBox()
        self.spin.setRange(min_val, max_val)
        self.spin.setValue(default_val)
        self.spin.setSingleStep(step)
        self.spin.setDecimals(decimals)
        if suffix:
            self.spin.setSuffix(f" {suffix}")
        self.spin.setStyleSheet("background-color: #111117; border: 1px solid #282839; border-radius: 6px; padding: 2px 8px; font-weight: 600; color: #60a5fa;")

        header.addWidget(self.lbl_title)
        header.addStretch()
        header.addWidget(self.spin)
        layout.addLayout(header)

        # Slider row: Smooth Slider + ↺ Reset Button
        slider_row = QHBoxLayout()
        slider_row.setSpacing(6)
        
        self.slider = NoWheelSlider(Qt.Horizontal)
        self.slider.setTracking(True)
        self.slider.setRange(int(round(min_val * self.multiplier)), int(round(max_val * self.multiplier)))
        self.slider.setValue(int(round(default_val * self.multiplier)))

        btn_reset = QToolButton()
        btn_reset.setText("↺")
        btn_reset.setToolTip(f"Auf Standardwert ({default_val}{suffix}) zurücksetzen")
        btn_reset.setStyleSheet("background-color: #1e1e28; border: 1px solid #2e2e3e; border-radius: 6px; color: #9ca3af; font-weight: bold; padding: 2px 7px;")
        btn_reset.clicked.connect(self.reset_default)

        slider_row.addWidget(self.slider, 1)
        slider_row.addWidget(btn_reset)
        layout.addLayout(slider_row)

        # Connect signals
        self.spin.valueChanged.connect(self._on_spin_changed)
        self.slider.valueChanged.connect(self._on_slider_changed)

    def _on_spin_changed(self, val: float):
        slider_val = int(round(val * self.multiplier))
        if self.slider.value() != slider_val:
            self.slider.blockSignals(True)
            self.slider.setValue(slider_val)
            self.slider.blockSignals(False)
        self.valueChanged.emit(val)

    def _on_slider_changed(self, s_val: int):
        val = s_val / float(self.multiplier)
        if abs(self.spin.value() - val) > 1e-4:
            self.spin.blockSignals(True)
            self.spin.setValue(val)
            self.spin.blockSignals(False)
        self.valueChanged.emit(val)

    def value(self) -> float:
        return self.spin.value()

    def setValue(self, val: float):
        self.spin.setValue(val)

    def reset_default(self):
        self.setValue(self.default_val)


class LogEmitter(QObject):
    log_signal = Signal(object)


class LogViewerWidget(QWidget):
    """
    Real-Time System Log Viewer with HTML color formatting,
    level filtering, clipboard export, and direct log file opening.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.emitter = LogEmitter()
        self.emitter.log_signal.connect(self._on_log_signal)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        # Header Toolbar
        tb_layout = QHBoxLayout()
        tb_layout.setSpacing(6)

        self.combo_filter = QComboBox()
        self.combo_filter.addItem("Alle Logs (Debug, Info, Warn, Fehler)", "ALL")
        self.combo_filter.addItem("Nur Info, Warnungen & Fehler", "INFO")
        self.combo_filter.addItem("Nur Warnungen & Fehler", "WARNING")
        self.combo_filter.addItem("Nur Fehler (Errors)", "ERROR")
        self.combo_filter.currentIndexChanged.connect(self.refresh_display)

        self.chk_autoscroll = QCheckBox("Auto-Scroll")
        self.chk_autoscroll.setChecked(True)
        self.chk_autoscroll.setStyleSheet("color: #ccc;")

        btn_copy = QPushButton("📋 Kopieren")
        btn_copy.setToolTip("Gesamtes Log in die Zwischenablage kopieren")
        btn_copy.setStyleSheet("padding: 3px 8px; font-size: 11px;")
        btn_copy.clicked.connect(self.copy_to_clipboard)

        btn_open_file = QPushButton("📂 Datei")
        btn_open_file.setToolTip("matchtrack.log im Texteditor öffnen")
        btn_open_file.setStyleSheet("padding: 3px 8px; font-size: 11px;")
        btn_open_file.clicked.connect(self.open_log_file)

        btn_clear = QPushButton("🗑️ Leeren")
        btn_clear.setToolTip("Loganzeige leeren")
        btn_clear.setStyleSheet("padding: 3px 8px; font-size: 11px;")
        btn_clear.clicked.connect(self.clear_logs)

        tb_layout.addWidget(self.combo_filter, 1)
        tb_layout.addWidget(self.chk_autoscroll)
        tb_layout.addWidget(btn_copy)
        tb_layout.addWidget(btn_open_file)
        tb_layout.addWidget(btn_clear)
        layout.addLayout(tb_layout)

        # Log Output Box
        self.txt_log = QPlainTextEdit()
        self.txt_log.setReadOnly(True)
        self.txt_log.setMaximumBlockCount(1500)
        self.txt_log.setStyleSheet("""
            QPlainTextEdit {
                background-color: #0b0f14;
                color: #e0e0e0;
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 11px;
                line-height: 1.3;
                border: 1px solid #233140;
                border-radius: 4px;
                padding: 6px;
            }
        """)
        layout.addWidget(self.txt_log, 1)

        # Subscribe to GuiLogHandler
        self.gui_handler = get_gui_handler()
        self.gui_handler.subscribe(lambda rec: self.emitter.log_signal.emit(rec))

        # Initial populate
        self.refresh_display()

    @Slot(object)
    def _on_log_signal(self, rec: GuiLogRecord):
        if self._matches_filter(rec.level):
            formatted_line = self._format_record_html(rec)
            self.txt_log.appendHtml(formatted_line)
            if self.chk_autoscroll.isChecked():
                self.txt_log.verticalScrollBar().setValue(self.txt_log.verticalScrollBar().maximum())

    def _matches_filter(self, level: str) -> bool:
        mode = self.combo_filter.currentData()
        if mode == "ALL":
            return True
        if mode == "INFO":
            return level in ("INFO", "WARNING", "ERROR", "CRITICAL")
        if mode == "WARNING":
            return level in ("WARNING", "ERROR", "CRITICAL")
        if mode == "ERROR":
            return level in ("ERROR", "CRITICAL")
        return True

    def _format_record_html(self, rec: GuiLogRecord) -> str:
        color = "#a0a0a0"
        tag_bg = "#222"
        if rec.level == "ERROR":
            color = "#ff6b6b"
            tag_bg = "#5c1d1d"
        elif rec.level == "WARNING":
            color = "#feca57"
            tag_bg = "#574218"
        elif rec.level == "INFO":
            color = "#48dbfb"
            tag_bg = "#193c4d"
        elif rec.level == "DEBUG":
            color = "#747d8c"
            tag_bg = "#2f3542"

        msg_safe = rec.message.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        return (
            f'<span style="color: #636e72;">[{rec.timestamp}]</span> '
            f'<span style="background-color: {tag_bg}; color: {color}; font-weight: bold; padding: 1px 4px; border-radius: 2px;">{rec.level:<7}</span> '
            f'<span style="color: #dfe6e9;">{msg_safe}</span>'
        )

    def refresh_display(self):
        self.txt_log.clear()
        records = self.gui_handler.get_all_records()
        for rec in records:
            if self._matches_filter(rec.level):
                self.txt_log.appendHtml(self._format_record_html(rec))
        if self.chk_autoscroll.isChecked():
            self.txt_log.verticalScrollBar().setValue(self.txt_log.verticalScrollBar().maximum())

    def copy_to_clipboard(self):
        text = self.txt_log.toPlainText()
        QApplication.clipboard().setText(text)

    def open_log_file(self):
        log_path = get_log_file_path()
        if os.path.exists(log_path):
            try:
                os.startfile(log_path)
            except Exception:
                import subprocess
                subprocess.Popen(["notepad.exe", log_path])

    def clear_logs(self):
        self.gui_handler.clear()
        self.txt_log.clear()


class RenderWorker(QThread):
    """Background worker for non-blocking video export (32:9 master, 16:9 follow-cam, or both)."""
    progress_signal = Signal(int, int, float, float, str)
    finished_signal = Signal(bool, str)

    def __init__(self, 
                 engine: StitcherEngine, 
                 out_path: str, 
                 width: int, 
                 height: int, 
                 mode: str, 
                 codec: str, 
                 bitrate: int, 
                 start_frame: int = 0, 
                 end_frame: Optional[int] = None, 
                 audio_source: str = "left",
                 use_lookahead: bool = True):
        super().__init__()
        self.engine = engine
        self.out_path = out_path
        self.width = width
        self.height = height
        self.mode = mode
        self.codec = codec
        self.bitrate = bitrate
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.audio_source = audio_source
        self.use_lookahead = use_lookahead

    def run(self):
        try:
            def callback(processed, total, fps, eta, stage_text=""):
                self.progress_signal.emit(processed, total, fps, eta, stage_text)

            if self.engine.is_panorama_mode():
                if self.mode == "21:10":
                    success = self.engine.convert_panorama_to_21x10(
                        output_filepath=self.out_path,
                        out_width=self.width,
                        out_height=self.height,
                        codec=self.codec,
                        bitrate_mbps=self.bitrate,
                        start_frame=self.start_frame,
                        end_frame=self.end_frame,
                        progress_callback=callback
                    )
                elif self.mode == "16:9_autocam":
                    success = self.engine.render_broadcast_from_panorama(
                        output_filepath=self.out_path,
                        out_width=self.width,
                        out_height=self.height,
                        codec=self.codec,
                        bitrate_mbps=self.bitrate,
                        start_frame=self.start_frame,
                        end_frame=self.end_frame,
                        use_lookahead=self.use_lookahead,
                        progress_callback=callback
                    )
                else:
                    success = self.engine.convert_panorama_to_21x10(
                        output_filepath=self.out_path,
                        out_width=self.width,
                        out_height=self.height,
                        codec=self.codec,
                        bitrate_mbps=self.bitrate,
                        start_frame=self.start_frame,
                        end_frame=self.end_frame,
                        progress_callback=callback
                    )
            elif self.mode == "both":
                success = self.engine.render_two_stage_broadcast(
                    output_16x9_filepath=self.out_path,
                    out_16x9_width=self.width,
                    out_16x9_height=self.height,
                    codec=self.codec,
                    bitrate_mbps=self.bitrate,
                    start_frame=self.start_frame,
                    end_frame=self.end_frame,
                    audio_source=self.audio_source,
                    use_lookahead=self.use_lookahead,
                    keep_32x9=True,
                    progress_callback=callback
                )
            elif self.mode == "16:9_autocam":
                success = self.engine.render_two_stage_broadcast(
                    output_16x9_filepath=self.out_path,
                    out_16x9_width=self.width,
                    out_16x9_height=self.height,
                    codec=self.codec,
                    bitrate_mbps=self.bitrate,
                    start_frame=self.start_frame,
                    end_frame=self.end_frame,
                    audio_source=self.audio_source,
                    use_lookahead=self.use_lookahead,
                    keep_32x9=False,
                    progress_callback=callback
                )
            elif self.mode == "21:10":
                def cb_2110(proc, total, fps, eta):
                    self.progress_signal.emit(proc, total, fps, eta, "21:10 Panorama Rendering...")

                success = self.engine.render_video_to_file(
                    output_filepath=self.out_path,
                    out_width=self.width,
                    out_height=self.height,
                    mode="21:10",
                    codec=self.codec,
                    bitrate_mbps=self.bitrate,
                    start_frame=self.start_frame,
                    end_frame=self.end_frame,
                    audio_source=self.audio_source,
                    progress_callback=cb_2110
                )
            else: # 32:9 Panorama
                def cb_329(proc, total, fps, eta):
                    self.progress_signal.emit(proc, total, fps, eta, "32:9 Panorama Rendering...")

                success = self.engine.render_video_to_file(
                    output_filepath=self.out_path,
                    out_width=self.width,
                    out_height=self.height,
                    mode="32:9",
                    codec=self.codec,
                    bitrate_mbps=self.bitrate,
                    start_frame=self.start_frame,
                    end_frame=self.end_frame,
                    audio_source=self.audio_source,
                    progress_callback=cb_329
                )

            if success:
                self.finished_signal.emit(True, "Export erfolgreich abgeschlossen!")
            else:
                self.finished_signal.emit(False, "Export wurde abgebrochen oder es trat ein Fehler auf.")
        except Exception as e:
            logger.critical(f"Kritischer Ausnahmefehler im RenderWorker: {e}", exc_info=True)
            self.finished_signal.emit(False, f"Schwerwiegender interner Fehler: {e}")


class InteractivePanoramaViewport(QWidget):
    """
    High-Resolution Interactive Viewport supporting:
    - 32:9 Tactical Panorama & 16:9 Broadcast
    - Semi-Transparent 32:9 Frame Overlay (Matte/Letterbox) with customizable opacity
    - Photoshop-style Interactive 4-Corner Drag & Drop Distortion (Corner Pinning)
    - Ball Tracking Reticle, Center Pitch Picker, and Horizon Guides.
    """
    framing_dragged = Signal(float, float)
    center_picked = Signal(float)
    corner_dragged = Signal(str, int, float, float) # (camera_side 'left'/'right', corner_idx 0..3, norm_x, norm_y)
    corner_drag_finished = Signal()
    pitch_corners_changed = Signal(list) # 6 freely movable pitch polygon corners [[x_tl, y_tl], [x_tc, y_tc], [x_tr, y_tr], [x_br, y_br], [x_bc, y_bc], [x_bl, y_bl]]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(480, 220)
        self.setMouseTracking(True) # Track mouse motion for instant hover highlights
        self.setStyleSheet("background-color: #0d0d0d; border: 1px solid #333; border-radius: 6px;")
        
        self.current_frame_bgr: Optional[np.ndarray] = None
        self.current_qimage: Optional[QImage] = None
        self.rig_ref: Optional[RigConfiguration] = None
        
        # Display overlays
        self.show_grid = True
        self.show_seam = True
        self.show_autocam_box = True
        self.show_ball_tracking = True
        self.show_center_line = True
        self.pick_center_mode = False

        # 2D Tactical Radar Minimap (FIFA Pitch)
        self.show_radar = True
        self.radar_position = "top_right"
        self.radar_width = 320
        self.radar_opacity = 0.85
        self.radar_players: List[Tuple[float, float, int]] = []
        self.radar_ball: Optional[Tuple[float, float]] = None
        self.radar_cam_span: Optional[Tuple[float, float]] = None

        # Tactical VAR Offside Line & Player Skeletons
        self.show_offside_line = True
        self.show_skeletons = False
        self.show_speed = False
        self.current_offside_line: Optional[Tuple[float, int, str]] = None
        self.current_detections: List[Any] = []
        self.ball_speed_kmh: Optional[float] = None
        
        # Semi-transparent 32:9 Frame Overlay
        self.show_transparent_frame = True
        self.frame_opacity = 0.50 # 50% dark translucent overlay outside the 32:9 frame
        
        # Corner Pinning (Photoshop-style Drag & Drop)
        self.corner_pin_mode = False
        self.corner_pin_camera = "both" # 'left', 'right', 'both'
        self.hovered_corner: Optional[Tuple[str, int]] = None # ('left'|'right', 0..3)
        self.dragging_corner: Optional[Tuple[str, int]] = None
        self.drag_start_pos = QPoint()

        # 6 Freely Movable Pitch Field ROI Points [TL, TC, TR, BR, BC, BL] (Normalized 0.0 - 1.0)
        # Includes Top-Center (TC) and Bottom-Center (BC) on the halfway line (Mittellinie)
        self.show_pitch_roi = True
        self.pitch_roi_mode = False
        self.tactical_mode = "points" # 'points' (setup mode on panorama with live PIP) or 'full' (16:9 fullscreen)
        self.pip_rect: Optional[QRectF] = None
        self.pitch_corners: List[List[float]] = [
            [0.0, 0.0],  # 0: Top-Left (TL - Video-Ecke Oben-Links)
            [0.5, 0.0],  # 1: Top-Center (TC - Video-Mittellinie Oben)
            [1.0, 0.0],  # 2: Top-Right (TR - Video-Ecke Oben-Rechts)
            [1.0, 1.0],  # 3: Bottom-Right (BR - Video-Ecke Unten-Rechts)
            [0.5, 1.0],  # 4: Bottom-Center (BC - Video-Mittellinie Unten)
            [0.0, 1.0]   # 5: Bottom-Left (BL - Video-Ecke Unten-Links)
        ]
        self.hovered_pitch_corner: Optional[int] = None
        self.dragging_pitch_corner: Optional[int] = None
        
        self.autocam_box: Optional[Tuple[int, int, int, int]] = None
        self.ball_px: Optional[Tuple[int, int]] = None
        self.view_mode = "32:9" # '32:9' or '16:9' or '16:9_tactical'
        
        # Zoom & Pan state
        self.zoom_level = 1.0
        self.pan_offset = QPoint(0, 0)
        self.is_panning = False
        self.is_framing_dragging = False
        self.last_mouse_pos = QPoint()
        self.last_framing_pos = QPoint()

        self.dynamic_zoom: float = 1.35

    def set_pitch_corners(self, corners: List[List[float]]):
        """Sets normalized pitch polygon points."""
        if corners and len(corners) == 6:
            self.pitch_corners = [[float(c[0]), float(c[1])] for c in corners]
            self.update()
        elif corners and len(corners) == 4:
            tl, tr, br, bl = corners
            tc = [(tl[0] + tr[0]) * 0.5, (tl[1] + tr[1]) * 0.5]
            bc = [(bl[0] + br[0]) * 0.5, (bl[1] + br[1]) * 0.5]
            self.pitch_corners = [tl, tc, tr, br, bc, bl]
            self.update()

    def set_pitch_roi_mode(self, enabled: bool):
        """Toggles interactive drag-and-drop pitch boundary editing."""
        self.pitch_roi_mode = enabled
        if not enabled:
            self.hovered_pitch_corner = None
            self.dragging_pitch_corner = None
            self.setCursor(Qt.ArrowCursor)
        self.update()

    def set_rig(self, rig: RigConfiguration):
        """Sets reference to current RigConfiguration for reading corner coordinates."""
        self.rig_ref = rig
        self.update()

    def set_corner_pin_mode(self, enabled: bool):
        """Toggles Photoshop-style interactive corner dragging mode."""
        self.corner_pin_mode = enabled
        if not enabled:
            self.hovered_corner = None
            self.dragging_corner = None
            self.setCursor(Qt.ArrowCursor)
        self.update()

    def set_corner_pin_camera(self, cam_side: str):
        """Filters which camera's corner handles are visible ('left', 'right', 'both')."""
        self.corner_pin_camera = cam_side
        self.update()

    def set_transparent_frame_enabled(self, enabled: bool):
        """Toggles semi-transparent 32:9 framing overlay."""
        self.show_transparent_frame = enabled
        self.update()

    def set_frame_opacity(self, opacity: float):
        """Sets opacity of the semi-transparent frame matte (0.0 to 1.0)."""
        self.frame_opacity = max(0.0, min(1.0, opacity))
        self.update()

    def set_pick_center_mode(self, enabled: bool):
        """Toggles interactive click-to-center mode for the soccer pitch center line."""
        self.pick_center_mode = enabled
        if enabled:
            self.setCursor(Qt.CrossCursor)
        else:
            self.setCursor(Qt.ArrowCursor)
        self.update()

    def set_radar_enabled(self, enabled: bool):
        """Toggles 2D tactical radar display."""
        self.show_radar = enabled
        self.update()

    def set_radar_position(self, pos: str):
        """Sets anchor position ('top_right', 'bottom_right', 'bottom_center', 'top_left', 'bottom_left')."""
        self.radar_position = pos
        self.update()

    def set_radar_width(self, width: int):
        """Sets radar width in pixels."""
        self.radar_width = max(180, min(600, int(width)))
        self.update()

    def set_radar_opacity(self, opacity: float):
        """Sets radar opacity (0.2 to 1.0)."""
        self.radar_opacity = max(0.2, min(1.0, float(opacity)))
        self.update()

    def set_offside_line_enabled(self, enabled: bool):
        """Toggles FIFA VAR virtual offside line display."""
        self.show_offside_line = enabled
        self.update()

    def set_skeletons_enabled(self, enabled: bool):
        """Toggles player skeleton keypoint and limb display."""
        self.show_skeletons = enabled
        self.update()

    def set_speed_badges_enabled(self, enabled: bool):
        """Toggles player and ball real-time metric speed badges."""
        self.show_speed = enabled
        self.update()

    def set_frame(self, 
                  frame_bgr: np.ndarray, 
                  autocam_box: Optional[Tuple[int, int, int, int]] = None, 
                  ball_px: Optional[Tuple[int, int]] = None, 
                  dynamic_zoom: float = 1.35,
                  players: Optional[List[Tuple[float, float, int]]] = None,
                  ball_norm: Optional[Tuple[float, float]] = None,
                  cam_span: Optional[Tuple[float, float]] = None,
                  offside_line: Optional[Tuple[float, int, str]] = None,
                  detections: Optional[List[Any]] = None,
                  ball_speed_kmh: Optional[float] = None):
        """Updates the displayed frame, broadcast box, ball reticle, 2D tactical radar, offside line, detections, and speeds."""
        self.current_frame_bgr = frame_bgr
        self.autocam_box = autocam_box
        self.ball_px = ball_px
        self.dynamic_zoom = dynamic_zoom
        self.current_offside_line = offside_line
        self.current_detections = detections if detections is not None else []
        self.ball_speed_kmh = ball_speed_kmh

        if players is not None:
            self.radar_players = players
        if ball_norm is not None:
            self.radar_ball = ball_norm
        elif ball_px is not None and frame_bgr is not None:
            self.radar_ball = (ball_px[0] / max(1, frame_bgr.shape[1]), ball_px[1] / max(1, frame_bgr.shape[0]))
        if cam_span is not None:
            self.radar_cam_span = cam_span

        if frame_bgr is None:
            self.current_qimage = None
            self.update()
            return

        if self.view_mode == "16:9" and autocam_box is not None:
            # Crop to 16:9 broadcast view
            bx, by, bw, bh = autocam_box
            cropped = frame_bgr[by:by+bh, bx:bx+bw]
            h, w, ch = cropped.shape
            bytes_per_line = ch * w
            rgb_frame = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
            self.current_qimage = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()
        else:
            h, w, ch = frame_bgr.shape
            bytes_per_line = ch * w
            rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            self.current_qimage = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()

        self.update()

    def reset_zoom(self):
        self.zoom_level = 1.0
        self.pan_offset = QPoint(0, 0)
        self.update()

    def set_zoom(self, zoom: float):
        self.zoom_level = max(0.5, min(5.0, zoom))
        self.update()

    def _get_target_rect(self) -> Optional[QRectF]:
        if self.current_qimage is None:
            return None
        vw, vh = self.width(), self.height()
        iw, ih = self.current_qimage.width(), self.current_qimage.height()
        scale_fit = min(vw / iw, vh / ih)
        draw_w = iw * scale_fit * self.zoom_level
        draw_h = ih * scale_fit * self.zoom_level
        draw_x = (vw - draw_w) * 0.5 + self.pan_offset.x()
        draw_y = (vh - draw_h) * 0.5 + self.pan_offset.y()
        return QRectF(draw_x, draw_y, draw_w, draw_h)

    def _get_corner_screen_points(self, side: str, target_rect: QRectF) -> list:
        """Computes screen pixel coordinates for 4 corners of left or right camera."""
        if not self.rig_ref:
            corners = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
        elif side == "left":
            corners = self.rig_ref.left_pose.corners
        else:
            corners = self.rig_ref.right_pose.corners

        pts = []
        half_w = target_rect.width() * 0.5
        top = target_rect.top()
        h = target_rect.height()
        x_base = target_rect.left() if side == "left" else (target_rect.left() + half_w)

        for cx, cy in corners:
            px = x_base + cx * half_w
            py = top + cy * h
            pts.append(QPointF(px, py))
        return pts

    def _get_pitch_roi_screen_polygon(self, target_rect: QRectF) -> List[QPointF]:
        """Calculates pixel coordinates on screen for the 4 freely movable pitch corners."""
        pts = []
        for cx, cy in self.pitch_corners:
            px = target_rect.left() + cx * target_rect.width()
            py = target_rect.top() + cy * target_rect.height()
            pts.append(QPointF(px, py))
        return pts

    def _hit_test_pitch_roi(self, pos: QPoint, target_rect: QRectF) -> Optional[int]:
        """Tests if mouse position is within hit radius (32px) or on the badge of any pitch point."""
        pts = self._get_pitch_roi_screen_polygon(target_rect)
        x, y = float(pos.x()), float(pos.y())
        p_qpt = QPointF(x, y)

        # 1. Direct circular hit test with generous 32px tolerance
        tol = 32.0
        for idx, pt in enumerate(pts):
            dist = np.hypot(x - pt.x(), y - pt.y())
            if dist <= tol:
                return idx

        # 2. Check if mouse is on handle label badge
        for idx, pt in enumerate(pts):
            bx = pt.x() + (12 if idx in (0, 4, 5) else -130)
            by = pt.y() + (14 if idx in (0, 1, 2) else -22)
            cbadge = QRectF(bx - 6, by - 6, 137, 30)
            if cbadge.contains(p_qpt):
                return idx

        return None

    def _get_tactical_16x9_screen_handles(self, target_rect: QRectF) -> Tuple[List[QPointF], QRectF]:
        """Calculates screen positions for the 6 control handles and the centered 16:9 target frame rect."""
        canvas_h = target_rect.height()
        canvas_w = target_rect.width()
        frame_h = canvas_h
        frame_w = frame_h * (16.0 / 9.0)
        frame_x0 = target_rect.left() + (canvas_w - frame_w) * 0.5
        frame_x1 = frame_x0 + frame_w
        mid_x = target_rect.left() + canvas_w * 0.5
        top = target_rect.top()
        bottom = target_rect.bottom()

        frame_rect = QRectF(frame_x0, top, frame_w, frame_h)
        handles = [
            QPointF(frame_x0, top),    # 0: TL
            QPointF(mid_x, top),       # 1: TC (Wölbung Oben)
            QPointF(frame_x1, top),    # 2: TR
            QPointF(frame_x1, bottom), # 3: BR
            QPointF(mid_x, bottom),    # 4: BC (Wölbung Unten)
            QPointF(frame_x0, bottom)  # 5: BL
        ]
        return handles, frame_rect

    def _hit_test_tactical_16x9_handles(self, pos: QPoint, target_rect: QRectF) -> Optional[int]:
        """Tests if mouse position is within hit radius (35px) of any of the 6 tactical warp control handles or their badge."""
        handles, _ = self._get_tactical_16x9_screen_handles(target_rect)
        tol = 35.0
        x, y = float(pos.x()), float(pos.y())
        p_qpt = QPointF(x, y)
        for idx, pt in enumerate(handles):
            dist = np.hypot(x - pt.x(), y - pt.y())
            if dist <= tol:
                return idx
        for idx, pt in enumerate(handles):
            bx = pt.x() + (14 if idx in (0, 4, 5) else (-140 if idx in (2, 3) else -65))
            by = pt.y() + (16 if idx in (0, 1, 2) else -24)
            badge = QRectF(bx - 6, by - 6, 142, 32)
            if badge.contains(p_qpt):
                return idx
        return None

    def _hit_test_corners(self, pos: QPoint, target_rect: QRectF) -> Optional[Tuple[str, int]]:
        """Tests if mouse position is within hit radius (14px) of any active corner handle."""
        sides_to_check = []
        if self.corner_pin_camera in ("left", "both"):
            sides_to_check.append("left")
        if self.corner_pin_camera in ("right", "both"):
            sides_to_check.append("right")

        for side in sides_to_check:
            pts = self._get_corner_screen_points(side, target_rect)
            for idx, pt in enumerate(pts):
                dx = pos.x() - pt.x()
                dy = pos.y() - pt.y()
                if (dx * dx + dy * dy) <= (15.0 * 15.0):
                    return (side, idx)
        return None

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)

        painter.fillRect(self.rect(), QColor("#0d0d0d"))

        if self.current_qimage is None:
            painter.setPen(QColor("#666666"))
            painter.setFont(QFont("Segoe UI", 12))
            painter.drawText(self.rect(), Qt.AlignCenter, "Kein Video geladen. Bitte links und rechts DJI Action 4 Videos öffnen.")
            return

        target_rect = self._get_target_rect()
        if target_rect is None:
            return

        vw, vh = self.width(), self.height()
        iw, ih = self.current_qimage.width(), self.current_qimage.height()

        # 1. Draw Stitched Frame Image
        painter.drawImage(target_rect, self.current_qimage)

        # 2. Semi-Transparent 32:9 / 21:10 Frame Overlay (Matte/Letterbox)
        if self.show_transparent_frame and self.view_mode in ("32:9", "21:10"):
            matte_alpha = int(np.clip(self.frame_opacity * 255, 0, 255))
            if matte_alpha > 0:
                mask_color = QColor(0, 0, 0, matte_alpha)
                # Shading regions outside target_rect within viewport
                if target_rect.top() > 0:
                    painter.fillRect(QRectF(0, 0, vw, target_rect.top()), mask_color)
                if target_rect.bottom() < vh:
                    painter.fillRect(QRectF(0, target_rect.bottom(), vw, vh - target_rect.bottom()), mask_color)
                if target_rect.left() > 0:
                    painter.fillRect(QRectF(0, target_rect.top(), target_rect.left(), target_rect.height()), mask_color)
                if target_rect.right() < vw:
                    painter.fillRect(QRectF(target_rect.right(), target_rect.top(), vw - target_rect.right(), target_rect.height()), mask_color)

            # Elegant Glowing Boundary Border
            pen_frame = QPen(QColor(52, 152, 219, 210), 2, Qt.SolidLine)
            painter.setPen(pen_frame)
            painter.drawRect(target_rect)

            # High-Precision Corner Brackets (⌜ ⌝ ⌞ ⌟)
            pen_bracket = QPen(QColor(46, 204, 113, 240), 3, Qt.SolidLine)
            painter.setPen(pen_bracket)
            blen = 18.0
            tl = target_rect.topLeft()
            tr = target_rect.topRight()
            br = target_rect.bottomRight()
            bl = target_rect.bottomLeft()

            # Top-Left Bracket ⌜
            painter.drawLine(QPointF(tl.x(), tl.y() + blen), tl)
            painter.drawLine(tl, QPointF(tl.x() + blen, tl.y()))
            # Top-Right Bracket ⌝
            painter.drawLine(QPointF(tr.x() - blen, tr.y()), tr)
            painter.drawLine(tr, QPointF(tr.x(), tr.y() + blen))
            # Bottom-Right Bracket ⌟
            painter.drawLine(QPointF(br.x(), br.y() - blen), br)
            painter.drawLine(br, QPointF(br.x() - blen, br.y()))
            # Bottom-Left Bracket ⌞
            painter.drawLine(QPointF(bl.x() + blen, bl.y()), bl)
            painter.drawLine(bl, QPointF(bl.x(), bl.y() - blen))

            # Framing Header Tag
            tag_title = "🔲 21:10 Panorama (Gestaucht)" if self.view_mode == "21:10" else "🔲 32:9 Panorama-Ausschnitt"
            tag_frame = QRectF(target_rect.center().x() - 105, target_rect.top() + 6, 210, 20)
            painter.fillRect(tag_frame, QColor(41, 128, 185, 210))
            painter.setPen(QColor(255, 255, 255))
            painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
            painter.drawText(tag_frame, Qt.AlignCenter, tag_title)

        # 3. Overlays for 32:9 / 21:10 mode
        if self.view_mode in ("32:9", "21:10"):
            # Field Center Reference Line (50% Center)
            if self.show_center_line:
                center_x = target_rect.left() + target_rect.width() * 0.5
                pen_center = QPen(QColor(241, 196, 15, 220), 2, Qt.DashDotLine)
                painter.setPen(pen_center)
                painter.drawLine(int(center_x), int(target_rect.top()), int(center_x), int(target_rect.bottom()))
                
                tag_center = QRectF(center_x - 65, target_rect.top() + 6, 130, 20)
                painter.fillRect(tag_center, QColor(241, 196, 15, 210))
                painter.setPen(QColor(0, 0, 0))
                painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
                painter.drawText(tag_center, Qt.AlignCenter, "🎯 Spielfeld-Mitte")

            if self.show_seam:
                seam_x = target_rect.left() + target_rect.width() * 0.5
                pen_seam = QPen(QColor(255, 75, 75, 180), 1, Qt.DotLine)
                painter.setPen(pen_seam)
                painter.drawLine(int(seam_x), int(target_rect.top()), int(seam_x), int(target_rect.bottom()))

            if self.show_grid:
                pen_grid = QPen(QColor(70, 230, 110, 140), 1, Qt.SolidLine)
                painter.setPen(pen_grid)
                horiz_y = target_rect.top() + target_rect.height() * 0.5
                painter.drawLine(int(target_rect.left()), int(horiz_y), int(target_rect.right()), int(horiz_y))
                ground_y = target_rect.top() + target_rect.height() * 0.8
                painter.drawLine(int(target_rect.left()), int(ground_y), int(target_rect.right()), int(ground_y))

            # 16:9 Broadcast Box
            if self.show_autocam_box and self.autocam_box is not None:
                bx, by, bw, bh = self.autocam_box
                scale_x = target_rect.width() / float(iw)
                scale_y = target_rect.height() / float(ih)
                rect_x = target_rect.left() + bx * scale_x
                rect_y = target_rect.top() + by * scale_y
                rect_w = bw * scale_x
                rect_h = bh * scale_y

                # Gold Broadcast Box
                pen_box = QPen(QColor(241, 196, 15, 230), 2, Qt.SolidLine)
                painter.setPen(pen_box)
                painter.drawRect(QRectF(rect_x, rect_y, rect_w, rect_h))

                # Broadcast Tag
                tag_rect = QRectF(rect_x, max(0.0, rect_y - 20.0), 160.0, 20.0)
                painter.fillRect(tag_rect, QColor(241, 196, 15, 210))
                painter.setPen(QColor(0, 0, 0))
                painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
                painter.drawText(tag_rect, Qt.AlignCenter, "🎥 16:9 Ball-Follow Cam")

            # Ball Tracking Reticle
            if self.show_ball_tracking and self.ball_px is not None:
                b_px_x, b_px_y = self.ball_px
                scale_x = target_rect.width() / float(iw)
                scale_y = target_rect.height() / float(ih)
                screen_bx = target_rect.left() + b_px_x * scale_x
                screen_by = target_rect.top() + b_px_y * scale_y

                # Green pulsing target circle
                pen_ball = QPen(QColor(46, 204, 113, 230), 2, Qt.SolidLine)
                painter.setPen(pen_ball)
                painter.drawEllipse(QPoint(int(screen_bx), int(screen_by)), 14, 14)
                painter.drawLine(int(screen_bx - 18), int(screen_by), int(screen_bx + 18), int(screen_by))
                painter.drawLine(int(screen_bx), int(screen_by - 18), int(screen_bx), int(screen_by + 18))

                painter.setPen(QColor(46, 204, 113))
                painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
                ball_txt = "⚽ Ball"
                if self.show_speed and getattr(self, 'ball_speed_kmh', None) is not None and self.ball_speed_kmh > 4.0:
                    ball_txt = f"⚽ {self.ball_speed_kmh:.1f} km/h"
                painter.drawText(int(screen_bx + 16), int(screen_by - 6), ball_txt)

            # 3b. Player Skeletons & Extremities (YOLO-Pose / Tactical Analysis)
            if self.show_skeletons and self.current_detections:
                for det in self.current_detections:
                    if getattr(det, 'keypoints', None) is not None:
                        kpts = det.keypoints
                        team_id = getattr(det, 'team_id', 0)
                        col_limb = QColor(239, 68, 68, 220) if team_id == 1 else (QColor(234, 179, 8, 220) if team_id in (2, -1) else QColor(59, 130, 246, 220))
                        painter.setPen(QPen(col_limb, 2, Qt.SolidLine))
                        # Connect limbs
                        for i1, i2 in SKELETON_CONNECTIONS:
                            if i1 < len(kpts) and i2 < len(kpts):
                                if kpts[i1, 2] > 0.25 and kpts[i2, 2] > 0.25:
                                    p1 = QPointF(target_rect.left() + kpts[i1, 0] * target_rect.width(),
                                                 target_rect.top() + kpts[i1, 1] * target_rect.height())
                                    p2 = QPointF(target_rect.left() + kpts[i2, 0] * target_rect.width(),
                                                 target_rect.top() + kpts[i2, 1] * target_rect.height())
                                    painter.drawLine(p1, p2)
                        # Draw joints
                        painter.setPen(QPen(QColor(255, 255, 255), 1))
                        painter.setBrush(col_limb)
                        for i in range(len(kpts)):
                            if kpts[i, 2] > 0.25:
                                pt = QPointF(target_rect.left() + kpts[i, 0] * target_rect.width(),
                                             target_rect.top() + kpts[i, 1] * target_rect.height())
                                r = 3.0 if i in (15, 16) else 2.0  # Emphasize feet/ankles for VAR
                                painter.drawEllipse(pt, r, r)
                        # Heading direction arrow
                        if getattr(det, 'heading_deg', None) is not None:
                            cx = target_rect.left() + det.center[0] * target_rect.width()
                            cy = target_rect.top() + det.center[1] * target_rect.height()
                            rad = math.radians(det.heading_deg)
                            hx = cx + 18.0 * math.cos(rad)
                            hy = cy + 18.0 * math.sin(rad)
                            painter.setPen(QPen(QColor(255, 255, 255, 220), 1.5, Qt.SolidLine))
                            painter.drawLine(QPointF(cx, cy), QPointF(hx, hy))
                    else:
                        # Draw tactical feet anchor for non-pose backends
                        cx = target_rect.left() + det.center[0] * target_rect.width()
                        bottom_y = target_rect.top() + det.bbox[3] * target_rect.height()
                        col = QColor(239, 68, 68, 200) if getattr(det, 'team_id', 0) == 1 else (QColor(234, 179, 8, 200) if getattr(det, 'team_id', 0) in (2, -1) else QColor(59, 130, 246, 200))
                        painter.setPen(QPen(QColor(255, 255, 255, 180), 1))
                        painter.setBrush(col)
                        painter.drawEllipse(QPointF(cx, bottom_y), 3.0, 3.0)

            # 3c. FIFA VAR Virtual Offside Line
            if self.show_offside_line and self.current_offside_line is not None:
                off_x, off_team, off_label = self.current_offside_line
                ox = target_rect.left() + off_x * target_rect.width()
                if target_rect.left() <= ox <= target_rect.right():
                    col_off = QColor(239, 68, 68, 230) if off_team == 1 else QColor(59, 130, 246, 230)
                    pen_off = QPen(col_off, 2, Qt.DashLine)
                    painter.setPen(pen_off)
                    painter.drawLine(QPointF(ox, target_rect.top()), QPointF(ox, target_rect.bottom()))
                    
                    # VAR Offside Badge Tag
                    badge_w, badge_h = 160.0, 20.0
                    badge_y = target_rect.bottom() - 26.0
                    badge_rect = QRectF(ox - badge_w * 0.5, badge_y, badge_w, badge_h)
                    painter.fillRect(badge_rect, QColor(0, 0, 0, 215))
                    painter.setPen(col_off)
                    painter.drawRect(badge_rect)
                    painter.setFont(QFont("Segoe UI", 7, QFont.Bold))
                    painter.drawText(badge_rect, Qt.AlignCenter, off_label)

            # 3d. Real-Time Player Metric Speed Badges (SkillCorner / TacticAI)
            if self.show_speed and self.current_detections:
                for det in self.current_detections:
                    spd = getattr(det, 'speed_kmh', None)
                    if spd is not None and spd > 1.5:
                        cx = target_rect.left() + det.center[0] * target_rect.width()
                        head_y = target_rect.top() + det.bbox[1] * target_rect.height() - 15.0
                        
                        is_sprint = (spd >= 25.2)
                        badge_txt = f"🔥 {spd:.1f} km/h" if is_sprint else f"⚡ {spd:.1f} km/h"
                        bw = 78.0 if is_sprint else 68.0
                        bh = 17.0
                        rect_badge = QRectF(cx - bw * 0.5, head_y, bw, bh)
                        
                        bg_col = QColor(185, 28, 28, 220) if is_sprint else QColor(15, 23, 42, 210)
                        painter.fillRect(rect_badge, bg_col)
                        painter.setPen(QColor(239, 68, 68) if is_sprint else QColor(56, 189, 248))
                        painter.drawRect(rect_badge)
                        painter.setFont(QFont("Segoe UI", 7, QFont.Bold))
                        painter.drawText(rect_badge, Qt.AlignCenter, badge_txt)

        # 4. Photoshop-style Corner Pinning Handles & Wireframes
        if self.corner_pin_mode:
            banner_rect = QRectF(0, 0, self.width(), 32)
            painter.fillRect(banner_rect, QColor(142, 68, 173, 235))
            painter.setPen(QColor(255, 255, 255))
            painter.setFont(QFont("Segoe UI", 9, QFont.Bold))
            painter.drawText(banner_rect, Qt.AlignCenter, "📐 Ecken-Verzerrung (Corner Pin): Ziehen Sie die runden Eckpunkte (Cyan = Links, Gold = Rechts) per Maus")

            # Left Camera Quad & Handles
            if self.corner_pin_camera in ("left", "both"):
                pts_l = self._get_corner_screen_points("left", target_rect)
                painter.setPen(QPen(QColor(0, 210, 211, 220), 2, Qt.DashLine))
                painter.setBrush(QColor(0, 210, 211, 20))
                painter.drawPolygon(pts_l)

                tag_names = ["TL (Links)", "TR (Mitte)", "BR (Mitte)", "BL (Links)"]
                for idx, pt in enumerate(pts_l):
                    is_active = (self.hovered_corner == ("left", idx) or self.dragging_corner == ("left", idx))
                    r = 13 if is_active else 9

                    # Outer glow / ring
                    painter.setPen(QPen(QColor(0, 210, 211, 255), 3 if is_active else 2))
                    painter.setBrush(QColor(0, 210, 211, 230) if is_active else QColor(10, 25, 35, 220))
                    painter.drawEllipse(pt, r, r)

                    # Center crosshair + dot
                    painter.setPen(QPen(QColor(255, 255, 255), 2))
                    painter.drawEllipse(pt, 3, 3)
                    painter.drawLine(QPointF(pt.x() - 6, pt.y()), QPointF(pt.x() + 6, pt.y()))
                    painter.drawLine(QPointF(pt.x(), pt.y() - 6), QPointF(pt.x(), pt.y() + 6))

                    # Corner Label Badge
                    bx = pt.x() + (14 if idx in (0, 3) else -72)
                    by = pt.y() + (16 if idx in (0, 1) else -22)
                    badge = QRectF(bx, by, 66, 18)
                    painter.fillRect(badge, QColor(0, 0, 0, 210))
                    painter.setPen(QColor(0, 210, 211))
                    painter.setFont(QFont("Segoe UI", 7, QFont.Bold))
                    painter.drawText(badge, Qt.AlignCenter, tag_names[idx])

            # Right Camera Quad & Handles
            if self.corner_pin_camera in ("right", "both"):
                pts_r = self._get_corner_screen_points("right", target_rect)
                painter.setPen(QPen(QColor(255, 159, 67, 220), 2, Qt.DashLine))
                painter.setBrush(QColor(255, 159, 67, 20))
                painter.drawPolygon(pts_r)

                tag_names_r = ["TL (Mitte)", "TR (Rechts)", "BR (Rechts)", "BL (Mitte)"]
                for idx, pt in enumerate(pts_r):
                    is_active = (self.hovered_corner == ("right", idx) or self.dragging_corner == ("right", idx))
                    r = 13 if is_active else 9

                    # Outer glow / ring
                    painter.setPen(QPen(QColor(255, 159, 67, 255), 3 if is_active else 2))
                    painter.setBrush(QColor(255, 159, 67, 230) if is_active else QColor(35, 22, 10, 220))
                    painter.drawEllipse(pt, r, r)

                    # Center crosshair + dot
                    painter.setPen(QPen(QColor(255, 255, 255), 2))
                    painter.drawEllipse(pt, 3, 3)
                    painter.drawLine(QPointF(pt.x() - 6, pt.y()), QPointF(pt.x() + 6, pt.y()))
                    painter.drawLine(QPointF(pt.x(), pt.y() - 6), QPointF(pt.x(), pt.y() + 6))

                    # Corner Label Badge
                    bx = pt.x() + (14 if idx in (0, 3) else -72)
                    by = pt.y() + (16 if idx in (0, 1) else -22)
                    badge = QRectF(bx, by, 66, 18)
                    painter.fillRect(badge, QColor(0, 0, 0, 210))
                    painter.setPen(QColor(255, 159, 67))
                    painter.setFont(QFont("Segoe UI", 7, QFont.Bold))
                    painter.drawText(badge, Qt.AlignCenter, tag_names_r[idx])

        # 5. 16:9 Tactical Warp Overlay & 6-Point Grab Handles on Panorama
        if self.view_mode == "16:9_tactical":
            # A. Calculate Centered 16:9 Target Frame within Panorama Viewport
            box_h = target_rect.height()
            box_w = box_h * (16.0 / 9.0)
            box_x = target_rect.left() + (target_rect.width() - box_w) * 0.5
            box_y = target_rect.top()
            frame_16x9_rect = QRectF(box_x, box_y, box_w, box_h)

            # B. Render Live Rectified 16:9 Video inside the 16:9 Target Frame
            w_dst = int(round(box_w))
            h_dst = int(round(box_h))
            if self.current_frame_bgr is not None and w_dst > 20 and h_dst > 20:
                try:
                    map_x, map_y = generate_tactical_16x9_luts(
                        self.current_frame_bgr.shape[1], self.current_frame_bgr.shape[0],
                        w_dst, h_dst,
                        self.pitch_corners,
                        margin_percent=0.0
                    )
                    warped_bgr = apply_tactical_16x9_warp(self.current_frame_bgr, map_x, map_y)
                    warped_rgb = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2RGB)
                    warped_qimg = QImage(warped_rgb.data, w_dst, h_dst, warped_rgb.strides[0], QImage.Format_RGB888).copy()
                    painter.drawImage(frame_16x9_rect, warped_qimg)
                except Exception:
                    pass

            # C. Draw Translucent Dark Letterbox Outside the 16:9 Target Frame
            left_w = max(0.0, box_x - target_rect.left())
            right_w = max(0.0, target_rect.right() - frame_16x9_rect.right())
            if left_w > 0:
                painter.fillRect(QRectF(target_rect.left(), target_rect.top(), left_w, box_h), QColor(0, 0, 0, 160))
            if right_w > 0:
                painter.fillRect(QRectF(frame_16x9_rect.right(), target_rect.top(), right_w, box_h), QColor(0, 0, 0, 160))

            # D. Glowing 16:9 Target Frame Border & Corner Accents
            painter.setPen(QPen(QColor(52, 152, 219, 240), 2.5, Qt.SolidLine))
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(frame_16x9_rect)

            acc_len = 24.0
            pen_acc = QPen(QColor(241, 196, 15, 255), 3.0)
            painter.setPen(pen_acc)
            # TL
            painter.drawLine(QPointF(frame_16x9_rect.left(), frame_16x9_rect.top()), QPointF(frame_16x9_rect.left() + acc_len, frame_16x9_rect.top()))
            painter.drawLine(QPointF(frame_16x9_rect.left(), frame_16x9_rect.top()), QPointF(frame_16x9_rect.left(), frame_16x9_rect.top() + acc_len))
            # TR
            painter.drawLine(QPointF(frame_16x9_rect.right(), frame_16x9_rect.top()), QPointF(frame_16x9_rect.right() - acc_len, frame_16x9_rect.top()))
            painter.drawLine(QPointF(frame_16x9_rect.right(), frame_16x9_rect.top()), QPointF(frame_16x9_rect.right(), frame_16x9_rect.top() + acc_len))
            # BR
            painter.drawLine(QPointF(frame_16x9_rect.right(), frame_16x9_rect.bottom()), QPointF(frame_16x9_rect.right() - acc_len, frame_16x9_rect.bottom()))
            painter.drawLine(QPointF(frame_16x9_rect.right(), frame_16x9_rect.bottom()), QPointF(frame_16x9_rect.right(), frame_16x9_rect.bottom() - acc_len))
            # BL
            painter.drawLine(QPointF(frame_16x9_rect.left(), frame_16x9_rect.bottom()), QPointF(frame_16x9_rect.left() + acc_len, frame_16x9_rect.bottom()))
            painter.drawLine(QPointF(frame_16x9_rect.left(), frame_16x9_rect.bottom()), QPointF(frame_16x9_rect.left(), frame_16x9_rect.bottom() - acc_len))

            # 16:9 Title Badge
            badge_16x9 = QRectF(frame_16x9_rect.center().x() - 110, frame_16x9_rect.top() + 8, 220, 22)
            painter.fillRect(badge_16x9, QColor(0, 0, 0, 210))
            painter.setPen(QColor(52, 152, 219))
            painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
            painter.drawText(badge_16x9, Qt.AlignCenter, "📺 16:9 Video-Zielbereich (Live-Warp)")

            # E. Draw 6 Draggable Control Points DIRECTLY on the Actual Panorama Video Features
            poly_pts = self._get_pitch_roi_screen_polygon(target_rect)
            if len(poly_pts) == 6:
                qpoly = QPolygonF(poly_pts)
                painter.setPen(QPen(QColor(224, 86, 253, 230), 2, Qt.DashLine))
                painter.setBrush(QColor(224, 86, 253, 15))
                painter.drawPolygon(qpoly)

                # Midline
                pt_tc = poly_pts[1]
                pt_bc = poly_pts[4]
                painter.setPen(QPen(QColor(241, 196, 15, 230), 2, Qt.DashDotLine))
                painter.drawLine(pt_tc, pt_bc)
                
                mid_x = (pt_tc.x() + pt_bc.x()) * 0.5
                mid_y = (pt_tc.y() + pt_bc.y()) * 0.5
                painter.setPen(QPen(QColor(241, 196, 15, 180), 1.5, Qt.DotLine))
                painter.drawEllipse(QPointF(mid_x, mid_y), 24, 24)

                # Drag handles with coordinates
                corner_labels = [
                    f"🚩 Video TL ({int(self.pitch_corners[0][0]*100)}%, {int(self.pitch_corners[0][1]*100)}%)",
                    f"📍 ↕ Wölbung Oben ({int(self.pitch_corners[1][1]*100)}%)",
                    f"🚩 Video TR ({int(self.pitch_corners[2][0]*100)}%, {int(self.pitch_corners[2][1]*100)}%)",
                    f"🚩 Video BR ({int(self.pitch_corners[3][0]*100)}%, {int(self.pitch_corners[3][1]*100)}%)",
                    f"📍 ↕ Wölbung Unten ({int(self.pitch_corners[4][1]*100)}%)",
                    f"🚩 Video BL ({int(self.pitch_corners[5][0]*100)}%, {int(self.pitch_corners[5][1]*100)}%)"
                ]

                for idx, pt in enumerate(poly_pts):
                    is_active = (self.hovered_pitch_corner == idx or self.dragging_pitch_corner == idx)
                    is_mid = (idx in (1, 4))
                    radius = 14 if is_active else 10

                    if is_active:
                        painter.setPen(Qt.NoPen)
                        painter.setBrush(QColor(241, 196, 15, 100) if is_mid else QColor(224, 86, 253, 100))
                        painter.drawEllipse(pt, 26, 26)

                    painter.setPen(QPen(QColor(255, 255, 255), 2))
                    if is_mid:
                        painter.setBrush(QColor(243, 156, 18, 245) if is_active else QColor(241, 196, 15, 220))
                    else:
                        painter.setBrush(QColor(224, 86, 253, 245) if is_active else QColor(190, 46, 221, 220))
                    painter.drawEllipse(pt, radius, radius)

                    painter.setPen(QPen(QColor(0, 0, 0, 220), 1.5))
                    if is_mid:
                        painter.drawLine(QPointF(pt.x(), pt.y() - 5), QPointF(pt.x(), pt.y() + 5))
                        painter.drawLine(QPointF(pt.x() - 3, pt.y() - 2), QPointF(pt.x(), pt.y() - 5))
                        painter.drawLine(QPointF(pt.x() + 3, pt.y() - 2), QPointF(pt.x(), pt.y() - 5))
                        painter.drawLine(QPointF(pt.x() - 3, pt.y() + 2), QPointF(pt.x(), pt.y() + 5))
                        painter.drawLine(QPointF(pt.x() + 3, pt.y() + 2), QPointF(pt.x(), pt.y() + 5))
                    else:
                        painter.drawLine(QPointF(pt.x() - 5, pt.y()), QPointF(pt.x() + 5, pt.y()))
                        painter.drawLine(QPointF(pt.x(), pt.y() - 5), QPointF(pt.x(), pt.y() + 5))

                    bx = pt.x() + (12 if idx in (0, 4, 5) else -135)
                    by = pt.y() + (14 if idx in (0, 1, 2) else -24)
                    cbadge = QRectF(bx, by, 130, 20)
                    painter.fillRect(cbadge, QColor(0, 0, 0, 220))
                    painter.setPen(QColor(241, 196, 15) if is_mid else QColor(224, 86, 253))
                    painter.setFont(QFont("Segoe UI", 7, QFont.Bold))
                    painter.drawText(cbadge, Qt.AlignCenter, corner_labels[idx])

            # Top HUD banner
            banner_rect = QRectF(0, 0, self.width(), 32)
            painter.fillRect(banner_rect, QColor(41, 128, 185, 235))
            painter.setPen(QColor(255, 255, 255))
            painter.setFont(QFont("Segoe UI", 9, QFont.Bold))
            painter.drawText(banner_rect, Qt.AlignCenter, "📐 16:9 Taktik-Warp: Ziehen Sie die 6 Punkte auf den Video-Rändern, um das Panorama in den 16:9 Rahmen einzupassen.")

        # 6. Pitch Field ROI (Action Polygon with Mittellinie) Overlay & Handles in Panorama
        elif self.show_pitch_roi and self.view_mode in ("32:9", "21:10"):
            poly_pts = self._get_pitch_roi_screen_polygon(target_rect)
            if len(poly_pts) >= 4:
                qpoly = QPolygonF(poly_pts)
                painter.setBrush(QColor(46, 204, 113, 20))
                painter.setPen(Qt.NoPen)
                painter.drawPolygon(qpoly)

                pen_roi = QPen(QColor(46, 204, 113, 230), 2, Qt.DashLine)
                painter.setPen(pen_roi)
                painter.setBrush(Qt.NoBrush)
                painter.drawPolygon(qpoly)

                if len(poly_pts) == 6:
                    pt_tc = poly_pts[1]
                    pt_bc = poly_pts[4]
                    painter.setPen(QPen(QColor(241, 196, 15, 220), 2, Qt.DashDotLine))
                    painter.drawLine(pt_tc, pt_bc)

                    mid_x = (pt_tc.x() + pt_bc.x()) * 0.5
                    mid_y = (pt_tc.y() + pt_bc.y()) * 0.5
                    painter.setPen(QPen(QColor(241, 196, 15, 170), 1.5, Qt.DotLine))
                    painter.drawEllipse(QPointF(mid_x, mid_y), 24, 24)

                cx = sum(p.x() for p in poly_pts) / float(len(poly_pts))
                cy = sum(p.y() for p in poly_pts) / float(len(poly_pts))
                badge_w, badge_h = 180.0, 20.0
                badge_rect = QRectF(cx - badge_w * 0.5, cy - badge_h * 0.5, badge_w, badge_h)
                painter.fillRect(badge_rect, QColor(0, 0, 0, 190))
                painter.setPen(QColor(46, 204, 113))
                painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
                painter.drawText(badge_rect, Qt.AlignCenter, "🌱 Spielfeld-Zone (Aktiv)")

                if self.pitch_roi_mode:
                    banner_rect = QRectF(0, 0, self.width(), 32)
                    painter.fillRect(banner_rect, QColor(39, 174, 96, 235))
                    painter.setPen(QColor(255, 255, 255))
                    painter.setFont(QFont("Segoe UI", 9, QFont.Bold))
                    painter.drawText(banner_rect, Qt.AlignCenter, "🌱 Spielfeld-Punkte: Ziehen Sie die 4 Eckfahnen & die 2 Mittellinien-Punkte frei per Maus")

                    if len(poly_pts) == 6:
                        corner_labels = [
                            "🚩 Oben-Links (TL)",
                            "📍 Mittellinie Oben (TC)",
                            "🚩 Oben-Rechts (TR)",
                            "🚩 Unten-Rechts (BR)",
                            "📍 Mittellinie Unten (BC)",
                            "🚩 Unten-Links (BL)"
                        ]
                    else:
                        corner_labels = [f"Punkt {i+1}" for i in range(len(poly_pts))]

                    for idx, pt in enumerate(poly_pts):
                        is_active = (self.hovered_pitch_corner == idx or self.dragging_pitch_corner == idx)
                        is_midpoint = (len(poly_pts) == 6 and idx in (1, 4))
                        radius = 12 if is_active else 8

                        if is_active:
                            painter.setPen(Qt.NoPen)
                            painter.setBrush(QColor(241, 196, 15, 90) if is_midpoint else QColor(46, 204, 113, 80))
                            painter.drawEllipse(pt, 22, 22)

                        painter.setPen(QPen(QColor(255, 255, 255), 2))
                        if is_midpoint:
                            painter.setBrush(QColor(243, 156, 18, 245) if is_active else QColor(241, 196, 15, 220))
                        else:
                            painter.setBrush(QColor(46, 204, 113, 245) if is_active else QColor(39, 174, 96, 220))
                        painter.drawEllipse(pt, radius, radius)

                        painter.setPen(QPen(QColor(0, 0, 0, 220), 1.5))
                        painter.drawLine(QPointF(pt.x() - 5, pt.y()), QPointF(pt.x() + 5, pt.y()))
                        painter.drawLine(QPointF(pt.x(), pt.y() - 5), QPointF(pt.x(), pt.y() + 5))

                        bx = pt.x() + (12 if idx in (0, 4, 5) else -130)
                        by = pt.y() + (14 if idx in (0, 1, 2) else -22)
                        cbadge = QRectF(bx, by, 125, 18)
                        painter.fillRect(cbadge, QColor(0, 0, 0, 215))
                        painter.setPen(QColor(241, 196, 15) if is_midpoint else QColor(46, 204, 113))
                        painter.setFont(QFont("Segoe UI", 7, QFont.Bold))
                        painter.drawText(cbadge, Qt.AlignCenter, corner_labels[idx])

        # Pick Center Banner Mode
        if self.pick_center_mode:
            banner_rect = QRectF(0, 0, self.width(), 32)
            painter.fillRect(banner_rect, QColor(41, 128, 185, 230))
            painter.setPen(QColor(255, 255, 255))
            painter.setFont(QFont("Segoe UI", 10, QFont.Bold))
            painter.drawText(banner_rect, Qt.AlignCenter, "📍 Klicken Sie auf die Spielfeld-Mittellinie / Anstoßkreis im Bild zum sofortigen Zentrieren")

        # 7. 2D Tactical Radar Minimap Overlay (FIFA Standard Pitch)
        if self.show_radar and self.current_frame_bgr is not None:
            SoccerRadarRenderer.paint_radar_qt(
                painter=painter,
                widget_w=self.width(),
                widget_h=self.height(),
                players=self.radar_players,
                ball_norm=self.radar_ball,
                cam_span=self.radar_cam_span,
                radar_w=self.radar_width,
                pos=self.radar_position,
                opacity=self.radar_opacity,
                offside_line=self.current_offside_line if self.show_offside_line else None
            )

        # HUD in top-left
        painter.setPen(QColor(220, 220, 220, 220))
        painter.setFont(QFont("Segoe UI", 9, QFont.Bold))
        if self.view_mode == "32:9":
            mode_str = "🌟 32:9 Taktik-Panorama"
        elif self.view_mode == "21:10":
            mode_str = "📐 21:10 Gestaucht"
        elif self.view_mode == "16:9_tactical":
            mode_str = "📐 16:9 Taktik-Warp (Live-Frame)"
        else:
            mode_str = "🎥 16:9 Auto-Broadcast"
        hud_text = f"{mode_str} ({iw}x{ih}) | 🎬 TV-Zoom: {self.dynamic_zoom:.2f}x | Viewport: {int(self.zoom_level*100)}%"
        painter.drawText(14, 24 if (not self.pick_center_mode and not self.corner_pin_mode and not self.pitch_roi_mode and self.view_mode != "16:9_tactical") else 48, hud_text)

    def wheelEvent(self, event: QWheelEvent):
        delta = event.angleDelta().y()
        if delta > 0:
            self.zoom_level = min(5.0, self.zoom_level * 1.15)
        else:
            self.zoom_level = max(0.8, self.zoom_level / 1.15)
        self.update()

    def mousePressEvent(self, event: QMouseEvent):
        target_rect = self._get_target_rect()

        # 1. Pitch ROI / Tactical 16:9 Corner Dragging
        if (self.pitch_roi_mode or self.view_mode == "16:9_tactical") and target_rect is not None and event.button() == Qt.LeftButton:
            hit = self._hit_test_pitch_roi(event.pos(), target_rect)
            if hit is not None:
                self.dragging_pitch_corner = hit
                self.update()
                return

        # 2. Corner Pin Dragging
        if self.corner_pin_mode and target_rect is not None and event.button() == Qt.LeftButton:
            hit = self._hit_test_corners(event.pos(), target_rect)
            if hit is not None:
                self.dragging_corner = hit
                self.drag_start_pos = event.pos()
                self.update()
                return

        # 3. Pick Center Mode
        if self.pick_center_mode and target_rect is not None and event.button() == Qt.LeftButton:
            if target_rect.contains(event.pos()):
                norm_x = float((event.pos().x() - target_rect.left()) / target_rect.width())
                self.center_picked.emit(norm_x)
                return

        # 4. Normal Pan / Framing Drag
        if event.button() == Qt.LeftButton:
            self.is_panning = True
            self.last_mouse_pos = event.pos()
        elif event.button() in (Qt.RightButton, Qt.MiddleButton):
            self.is_framing_dragging = True
            self.last_framing_pos = event.pos()

    def mouseMoveEvent(self, event: QMouseEvent):
        target_rect = self._get_target_rect()

        # 1. Active Pitch ROI / Tactical 16:9 Corner Dragging
        if self.dragging_pitch_corner is not None and target_rect is not None:
            norm_x = float((event.pos().x() - target_rect.left()) / max(target_rect.width(), 1.0))
            norm_y = float((event.pos().y() - target_rect.top()) / max(target_rect.height(), 1.0))
            idx = int(self.dragging_pitch_corner)

            norm_x = float(np.clip(norm_x, -0.2, 1.2))
            norm_y = float(np.clip(norm_y, -0.2, 1.2))

            self.pitch_corners[idx] = [norm_x, norm_y]
            self.pitch_corners_changed.emit(self.pitch_corners)
            self.update()
            return

        # 2. Active Corner Dragging
        if self.dragging_corner is not None and target_rect is not None:
            side, idx = self.dragging_corner
            half_w = target_rect.width() * 0.5
            x_base = target_rect.left() if side == "left" else (target_rect.left() + half_w)
            
            norm_x = float((event.pos().x() - x_base) / max(half_w, 1.0))
            norm_y = float((event.pos().y() - target_rect.top()) / max(target_rect.height(), 1.0))
            
            norm_x = float(np.clip(norm_x, -0.8, 2.0))
            norm_y = float(np.clip(norm_y, -0.8, 2.0))
            
            self.corner_dragged.emit(side, idx, norm_x, norm_y)
            self.update()
            return

        # 3. Viewport Panning
        if getattr(self, 'is_panning', False):
            diff = event.pos() - self.last_mouse_pos
            self.pan_offset += diff
            self.last_mouse_pos = event.pos()
            self.update()
            return

        # 4. Framing Shift
        if getattr(self, 'is_framing_dragging', False):
            diff = event.pos() - self.last_framing_pos
            dh_deg = -(diff.x() / max(self.width(), 100)) * 40.0
            dv_shift = (diff.y() / max(self.height(), 100)) * 1.5
            self.last_framing_pos = event.pos()
            self.framing_dragged.emit(dh_deg, dv_shift)
            return

        # 5. Pitch ROI / Tactical 16:9 Corner Hover Detection
        if (self.pitch_roi_mode or self.view_mode == "16:9_tactical") and target_rect is not None and self.dragging_pitch_corner is None:
            hit = self._hit_test_pitch_roi(event.pos(), target_rect)
            if hit != self.hovered_pitch_corner:
                self.hovered_pitch_corner = hit
                if hit is not None:
                    self.setCursor(Qt.SizeAllCursor)
                else:
                    self.setCursor(Qt.ArrowCursor)
                self.update()
                return

        # 6. Corner Hover Detection
        if self.corner_pin_mode and target_rect is not None:
            hit = self._hit_test_corners(event.pos(), target_rect)
            if hit != self.hovered_corner:
                self.hovered_corner = hit
                if hit is not None:
                    self.setCursor(Qt.SizeAllCursor)
                else:
                    self.setCursor(Qt.ArrowCursor)
                self.update()

    def mouseReleaseEvent(self, event: QMouseEvent):
        if self.dragging_pitch_corner is not None:
            self.dragging_pitch_corner = None
            self.update()

        if self.dragging_corner is not None:
            self.dragging_corner = None
            self.corner_drag_finished.emit()
            self.update()

        if event.button() == Qt.LeftButton:
            self.is_panning = False
        elif event.button() in (Qt.RightButton, Qt.MiddleButton):
            self.is_framing_dragging = False

    def leaveEvent(self, event):
        if self.hovered_pitch_corner is not None:
            self.hovered_pitch_corner = None
            if not self.pick_center_mode and not self.corner_pin_mode:
                self.setCursor(Qt.ArrowCursor)
        if self.hovered_corner is not None:
            self.hovered_corner = None
            if not self.pick_center_mode and not self.pitch_roi_mode:
                self.setCursor(Qt.ArrowCursor)
            self.update()
        super().leaveEvent(event)

    def mouseDoubleClickEvent(self, event: QMouseEvent):
        self.reset_zoom()




class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MatchTrack-Stitcher | 32:9 Panorama & 16:9 Ball-Follow Broadcast (DJI Action 4)")
        self.resize(1550, 940)
        self.setMinimumSize(1100, 720)

        # Initialize rig, engine and autocam config
        self.rig = RigConfiguration()
        self.engine = StitcherEngine(self.rig)
        self.render_worker: Optional[RenderWorker] = None

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

        # Set Window Icon
        icon_path = os.path.join(os.path.dirname(__file__), "..", "assets", "icon.ico")
        if not os.path.exists(icon_path):
            icon_path = os.path.join(os.path.dirname(__file__), "..", "assets", "icon.png")
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))

        self.init_ui()
        self.apply_dark_theme()
        self.load_default_startup_settings()

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
        self.combo_view_mode.addItem("🌟 32:9 Taktik-Panorama (Gesamt)", "32:9")
        self.combo_view_mode.addItem("📐 21:10 Gestauchtes Panorama (Gesamt)", "21:10")
        self.combo_view_mode.addItem("📐 16:9 Taktik-Warp (Spielfeld entzerrt)", "16:9_tactical")
        self.combo_view_mode.addItem("🎥 16:9 Auto-Broadcast (Follow-Cam)", "16:9")
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
        left_layout.addWidget(self.viewport, 1)

        # Timeline & Playback Controls Frame (One UI Style)
        timeline_frame = QFrame()
        timeline_frame.setObjectName("timelineFrame")
        timeline_frame.setStyleSheet("""
            #timelineFrame {
                background-color: #12121a;
                border: 1px solid #20202e;
                border-radius: 14px;
                padding: 8px 12px;
            }
        """)
        timeline_layout = QVBoxLayout(timeline_frame)
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
        left_layout.addWidget(timeline_frame)

        splitter.addWidget(left_pane)

        # =========================================================================
        # RIGHT PANE: Clean 4-Step Control Panel (Accordion Style)
        # =========================================================================
        self.right_pane = QTabWidget()
        self.right_pane.setMinimumWidth(380)
        self.right_pane.tabBar().setExpanding(True)
        self.right_pane.setDocumentMode(True)
        self.right_pane.currentChanged.connect(self._on_tab_changed)

        # -------------------------------------------------------------------------
        # TAB 1: 📁 Medien
        # -------------------------------------------------------------------------
        self.tab_media = QWidget()
        layout_tab_m = QVBoxLayout(self.tab_media)
        layout_tab_m.setContentsMargins(4, 4, 4, 4)
        scroll_m = QScrollArea()
        scroll_m.setWidgetResizable(True)
        widget_m = QWidget()
        layout_m = QVBoxLayout(widget_m)
        layout_m.setSpacing(12)

        # Card: Video Inputs
        grp_inputs = QGroupBox("🎥 Kamera-Eingänge (Dual-Rig / Panorama)")
        layout_in = QVBoxLayout(grp_inputs)
        layout_in.setSpacing(10)

        lbl_desc_inputs = QLabel("Laden Sie die Aufnahmen beider Kameras (80° Rig) oder ein 32:9 Panoramavideo:")
        lbl_desc_inputs.setStyleSheet("color: #8e8e9c; font-size: 11px;")
        lbl_desc_inputs.setWordWrap(True)
        layout_in.addWidget(lbl_desc_inputs)

        # Camera 1 (Links) Card
        card_l = QFrame()
        card_l.setStyleSheet("QFrame { background-color: #12121c; border: 1px solid #232334; border-radius: 10px; padding: 10px; }")
        layout_card_l = QVBoxLayout(card_l)
        layout_card_l.setSpacing(6)
        row_hdr_l = QHBoxLayout()
        lbl_title_l = QLabel("🎥 Kamera 1 (Links)")
        lbl_title_l.setStyleSheet("font-weight: 700; font-size: 12px; color: #f4f4f6;")
        self.badge_left = QLabel("⚪ Nicht geladen")
        self.badge_left.setStyleSheet("background: #1c1c28; color: #8e8e9c; border-radius: 6px; padding: 2px 8px; font-size: 11px; font-weight: 600;")
        row_hdr_l.addWidget(lbl_title_l)
        row_hdr_l.addStretch()
        row_hdr_l.addWidget(self.badge_left)
        layout_card_l.addLayout(row_hdr_l)

        self.lbl_file_left = QLabel("Keine Videodatei gewählt")
        self.lbl_file_left.setStyleSheet("color: #8e8e9c; font-size: 11px; padding: 2px 0;")
        self.lbl_file_left.setWordWrap(True)
        layout_card_l.addWidget(self.lbl_file_left)

        self.btn_open_left = QPushButton("📂 Linkes Video wählen...")
        self.btn_open_left.setStyleSheet("background-color: #1a1a26; border: 1px solid #2d2d42; font-weight: 600; padding: 7px; color: #f4f4f6; border-radius: 8px;")
        self.btn_open_left.clicked.connect(self.open_left_video)
        layout_card_l.addWidget(self.btn_open_left)
        layout_in.addWidget(card_l)

        # Camera 2 (Rechts) Card
        card_r = QFrame()
        card_r.setStyleSheet("QFrame { background-color: #12121c; border: 1px solid #232334; border-radius: 10px; padding: 10px; }")
        layout_card_r = QVBoxLayout(card_r)
        layout_card_r.setSpacing(6)
        row_hdr_r = QHBoxLayout()
        lbl_title_r = QLabel("🎥 Kamera 2 (Rechts)")
        lbl_title_r.setStyleSheet("font-weight: 700; font-size: 12px; color: #f4f4f6;")
        self.badge_right = QLabel("⚪ Nicht geladen")
        self.badge_right.setStyleSheet("background: #1c1c28; color: #8e8e9c; border-radius: 6px; padding: 2px 8px; font-size: 11px; font-weight: 600;")
        row_hdr_r.addWidget(lbl_title_r)
        row_hdr_r.addStretch()
        row_hdr_r.addWidget(self.badge_right)
        layout_card_r.addLayout(row_hdr_r)

        self.lbl_file_right = QLabel("Keine Videodatei gewählt")
        self.lbl_file_right.setStyleSheet("color: #8e8e9c; font-size: 11px; padding: 2px 0;")
        self.lbl_file_right.setWordWrap(True)
        layout_card_r.addWidget(self.lbl_file_right)

        self.btn_open_right = QPushButton("📂 Rechtes Video wählen...")
        self.btn_open_right.setStyleSheet("background-color: #1a1a26; border: 1px solid #2d2d42; font-weight: 600; padding: 7px; color: #f4f4f6; border-radius: 8px;")
        self.btn_open_right.clicked.connect(self.open_right_video)
        layout_card_r.addWidget(self.btn_open_right)
        layout_in.addWidget(card_r)

        # Panorama Direct Import
        self.btn_open_pano = QPushButton("🎬 Fertiges 32:9 Panoramavideo laden...")
        self.btn_open_pano.setStyleSheet("background-color: #161622; border: 1px solid #28283a; font-weight: 600; padding: 7px; color: #a1a1aa; border-radius: 8px;")
        self.btn_open_pano.setToolTip("Lädt ein einzelnes fertiges 32:9 Panoramavideo direkt für 16:9 Follow-Cam oder Taktik-Warp")
        self.btn_open_pano.clicked.connect(self.open_panorama_video)
        layout_in.addWidget(self.btn_open_pano)

        layout_m.addWidget(grp_inputs)

        # Card: Audio Synchronization & Offset
        grp_sync = QGroupBox("⚡ Audio-Synchronisation & Frame-Offset")
        layout_sync = QVBoxLayout(grp_sync)
        layout_sync.setSpacing(10)

        lbl_sync_info = QLabel("Gleichen Sie den Versatz beider Kameras per Audio-Waveform automatisch ab oder justieren Sie manuell:")
        lbl_sync_info.setStyleSheet("color: #8e8e9c; font-size: 11px;")
        lbl_sync_info.setWordWrap(True)
        layout_sync.addWidget(lbl_sync_info)

        self.btn_auto_sync = QPushButton("⚡ Audio synchronisieren (Auto-Sync)")
        self.btn_auto_sync.setStyleSheet("background-color: #2563eb; font-weight: bold; padding: 9px 14px; color: white; font-size: 12px; border-radius: 9px;")
        self.btn_auto_sync.setToolTip("Automatische Audio-Synchronisation (FFT Cross-Correlation für Frame-Offset)")
        self.btn_auto_sync.clicked.connect(self.run_audio_sync)
        layout_sync.addWidget(self.btn_auto_sync)

        lbl_manual_sync = QLabel("Manuelle Offset-Feinjustage (Kamera Rechts):")
        lbl_manual_sync.setStyleSheet("font-weight: 600; color: #f4f4f6; margin-top: 4px;")
        layout_sync.addWidget(lbl_manual_sync)

        row_offset = QHBoxLayout()
        row_offset.setSpacing(8)

        self.slider_offset = NoWheelSlider(Qt.Horizontal)
        self.slider_offset.setRange(-600, 600)
        self.slider_offset.setValue(0)
        self.slider_offset.valueChanged.connect(self.on_offset_slider_changed)
        row_offset.addWidget(self.slider_offset, 1)

        self.spin_offset = NoWheelSpinBox()
        self.spin_offset.setRange(-5000, 5000)
        self.spin_offset.setValue(0)
        self.spin_offset.setSuffix(" F")
        self.spin_offset.valueChanged.connect(self.on_offset_spin_changed)
        row_offset.addWidget(self.spin_offset)

        btn_offset_reset = QToolButton()
        btn_offset_reset.setText("↺")
        btn_offset_reset.setToolTip("Offset auf 0 zurücksetzen")
        btn_offset_reset.clicked.connect(lambda: self.spin_offset.setValue(0))
        row_offset.addWidget(btn_offset_reset)
        layout_sync.addLayout(row_offset)

        # Quick Frame Step Buttons
        row_steps = QHBoxLayout()
        row_steps.setSpacing(4)
        for delta, txt in [(-5, "-5 F"), (-1, "-1 F"), (0, "0"), (1, "+1 F"), (5, "+5 F")]:
            btn_step = QPushButton(txt)
            btn_step.setStyleSheet("background-color: #1a1a26; border: 1px solid #28283a; border-radius: 6px; padding: 4px 6px; font-size: 11px; font-weight: 600; color: #cbd5e1;")
            if delta == 0:
                btn_step.clicked.connect(lambda chk=False: self.spin_offset.setValue(0))
            else:
                btn_step.clicked.connect(lambda chk=False, d=delta: self.adjust_offset_by(d))
            row_steps.addWidget(btn_step)
        layout_sync.addLayout(row_steps)

        layout_m.addWidget(grp_sync)
        layout_m.addStretch()
        scroll_m.setWidget(widget_m)
        layout_tab_m.addWidget(scroll_m)
        self.right_pane.addTab(self.tab_media, "📁 1. Medien")

        # -------------------------------------------------------------------------
        # TAB 2: 🎯 Stitching & Rig
        # -------------------------------------------------------------------------
        self.tab_calib = QWidget()
        calib_layout = QVBoxLayout(self.tab_calib)
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        scroll_layout.setSpacing(10)

        # 1. Kamera-Profil & KI-Kalibrierung Card
        grp_preset = QGroupBox("📷 Kamera-Profil & KI-Kalibrierung")
        layout_preset = QVBoxLayout(grp_preset)
        layout_preset.setSpacing(8)

        lbl_profile_sel = QLabel("Objektiv-Profil:")
        lbl_profile_sel.setStyleSheet("font-weight: 600; color: #f4f4f6; font-size: 11px;")
        layout_preset.addWidget(lbl_profile_sel)

        self.combo_preset = QComboBox()
        for key, p in CAMERA_PRESETS.items():
            self.combo_preset.addItem(f"{p.name} [{key}]", key)
        self.combo_preset.currentIndexChanged.connect(self.on_preset_changed)
        layout_preset.addWidget(self.combo_preset)

        btn_preset_action4_4x3 = QPushButton("⚡ Quick-Preset: Action 4 (1080p 4:3 RockSteady)")
        btn_preset_action4_4x3.setStyleSheet("background-color: #1e293b; border: 1px solid #38bdf8; font-weight: bold; padding: 7px; color: #38bdf8; border-radius: 8px;")
        btn_preset_action4_4x3.setToolTip("Setzt 1080p 4:3 Dewarp Profil, 82.0° HFOV, ±36.5° Rig-Winkel und Naht-Überlappung mit 1 Klick.")
        btn_preset_action4_4x3.clicked.connect(self.apply_action4_4x3_rocksteady_preset)
        layout_preset.addWidget(btn_preset_action4_4x3)

        self.ctrl_cam_fov = LabeledSliderSpinBox("Objektiv-Blickwinkel (HFOV / Brennweite):", 60.0, 140.0, 92.0, 0.2, "°")
        self.ctrl_cam_fov.valueChanged.connect(self.on_cam_fov_changed)
        layout_preset.addWidget(self.ctrl_cam_fov)

        row_proj = QHBoxLayout()
        lbl_proj = QLabel("Projektions-Typ:")
        lbl_proj.setStyleSheet("font-weight: 600; color: #f4f4f6; font-size: 11px;")
        self.combo_projection = NoWheelComboBox()
        self.combo_projection.addItem("🌟 Zylindrisch (Standard Panorama)", "cylindrical")
        self.combo_projection.addItem("📐 Pannini (Begradigt Linien & Perspektive)", "pannini")
        self.combo_projection.addItem("📏 Rektilinear (100% Gerade)", "rectilinear")
        self.combo_projection.currentIndexChanged.connect(self.on_projection_changed)
        row_proj.addWidget(lbl_proj)
        row_proj.addWidget(self.combo_projection, 1)
        layout_preset.addLayout(row_proj)

        line_c1 = QFrame()
        line_c1.setFrameShape(QFrame.HLine)
        line_c1.setStyleSheet("color: #232332;")
        layout_preset.addWidget(line_c1)

        self.btn_run_ai_calib = QPushButton("🎯 Nahtstelle automatisch kalibrieren")
        self.btn_run_ai_calib.setStyleSheet("background-color: #2563eb; font-weight: bold; padding: 9px 12px; font-size: 12px; color: white; border-radius: 9px;")
        self.btn_run_ai_calib.clicked.connect(self.run_auto_stitch_calibration)
        layout_preset.addWidget(self.btn_run_ai_calib)
        self.btn_auto_stitch = self.btn_run_ai_calib

        self.chk_multi_frame_calib = QCheckBox("Multi-Frame Analyse (5 Video-Frames abtasten)")
        self.chk_multi_frame_calib.setChecked(True)
        self.chk_multi_frame_calib.setStyleSheet("color: #cbd5e1; font-size: 11px;")
        layout_preset.addWidget(self.chk_multi_frame_calib)

        scroll_layout.addWidget(grp_preset)

        # 2. Spielfeld- & Horizont-Begradigung Card
        grp_level = QGroupBox("🌅 Spielfeld- & Horizont-Begradigung")
        layout_level = QVBoxLayout(grp_level)
        layout_level.setSpacing(8)

        self.ctrl_curvature = LabeledSliderSpinBox("Spielfeld-Linien Begradigung (Curvature):", -0.60, 0.60, 0.0, 0.01, "", decimals=2)
        self.ctrl_curvature.setToolTip("Begradigt den V-förmigen Durchhang der Spielfeldlinien bei nach unten geneigten Kameras und zieht die Seitenlinien plan.")
        self.ctrl_curvature.valueChanged.connect(self.on_curvature_changed)
        layout_level.addWidget(self.ctrl_curvature)

        self.ctrl_global_pitch = LabeledSliderSpinBox("Horizont-Ausgleich (+15° Begradigung):", 0.0, 45.0, 15.0, 0.2, "°")
        self.ctrl_global_pitch.valueChanged.connect(self.on_rig_param_changed)
        layout_level.addWidget(self.ctrl_global_pitch)

        self.chk_auto_color_match = QCheckBox("🎨 Automatischer Farb- & Belichtungsabgleich (Kamera L ↔ R)")
        self.chk_auto_color_match.setChecked(True)
        self.chk_auto_color_match.setStyleSheet("color: #38bdf8; font-weight: bold; margin-top: 2px;")
        self.chk_auto_color_match.setToolTip("Gleicht Helligkeit, Belichtung und Weißabgleich beider Kameras automatisch an der Nahtstelle ab. Entfernt die sichtbare Kante in der Bildmitte vollständig.")
        self.chk_auto_color_match.toggled.connect(self.on_auto_color_match_toggled)
        layout_level.addWidget(self.chk_auto_color_match)

        self.ctrl_color_match_strength = LabeledSliderSpinBox("Farbangleich-Stärke:", 0.0, 100.0, 100.0, 5.0, "%", decimals=0)
        self.ctrl_color_match_strength.setToolTip("Bestimmt, wie stark die Farben beider Kameras aufeinander abgestimmt werden (100% = vollständiger Abgleich).")
        self.ctrl_color_match_strength.valueChanged.connect(self.on_color_match_strength_changed)
        layout_level.addWidget(self.ctrl_color_match_strength)

        self.chk_auto_crop = QCheckBox("✅ LIR Auto-Crop (Schwarze Ränder/Bögen entfernen)")
        self.chk_auto_crop.setChecked(True)
        self.chk_auto_crop.setStyleSheet("font-weight: 600; color: #10b981;")
        self.chk_auto_crop.toggled.connect(self.toggle_auto_crop)
        layout_level.addWidget(self.chk_auto_crop)

        self.ctrl_safety_margin = LabeledSliderSpinBox("LIR Sicherheitsrand (Puffer):", 0.0, 5.0, 0.5, 0.1, "%", decimals=1)
        self.ctrl_safety_margin.valueChanged.connect(self.on_safety_margin_changed)
        layout_level.addWidget(self.ctrl_safety_margin)

        self.ctrl_blend_width = LabeledSliderSpinBox("Nahtstellen-Übergangsbreite (Blending):", 1.0, 30.0, 8.0, 0.5, "°")
        self.ctrl_blend_width.valueChanged.connect(self.on_rig_param_changed)
        layout_level.addWidget(self.ctrl_blend_width)

        scroll_layout.addWidget(grp_level)

        # 3. 3D-Gehäuse Ausrichtung (Rig) Card
        grp_rig = QGroupBox("📐 3D-Gehäuse Ausrichtung (Spreizung & Neigung)")
        layout_rig = QVBoxLayout(grp_rig)
        layout_rig.setSpacing(8)

        self.chk_sync_yaw = QCheckBox("Winkel symmetrisch koppeln (± X°)")
        self.chk_sync_yaw.setChecked(True)
        self.chk_sync_yaw.toggled.connect(self.toggle_sync_angles)
        layout_rig.addWidget(self.chk_sync_yaw)

        self.ctrl_master_yaw = LabeledSliderSpinBox("Gesamter Spreizwinkel (80° Rig):", 40.0, 120.0, 80.0, 0.2, "°")
        self.ctrl_master_yaw.valueChanged.connect(self.on_master_yaw_changed)
        layout_rig.addWidget(self.ctrl_master_yaw)

        self.ctrl_left_yaw = LabeledSliderSpinBox("Kamera Links Yaw (Gierwinkel):", -80.0, -10.0, -40.0, 0.2, "°")
        self.ctrl_left_yaw.valueChanged.connect(self.on_left_yaw_changed)
        layout_rig.addWidget(self.ctrl_left_yaw)

        self.ctrl_right_yaw = LabeledSliderSpinBox("Kamera Rechts Yaw (Gierwinkel):", 10.0, 80.0, 40.0, 0.2, "°")
        self.ctrl_right_yaw.valueChanged.connect(self.on_right_yaw_changed)
        layout_rig.addWidget(self.ctrl_right_yaw)

        self.ctrl_cam_pitch = LabeledSliderSpinBox("Rig Neigung nach unten (Pitch):", -45.0, 0.0, -15.0, 0.2, "°")
        self.ctrl_cam_pitch.valueChanged.connect(self.on_rig_param_changed)
        layout_rig.addWidget(self.ctrl_cam_pitch)

        self.ctrl_left_roll = LabeledSliderSpinBox("Roll Feinjustage Links:", -15.0, 15.0, 0.0, 0.1, "°")
        self.ctrl_left_roll.valueChanged.connect(self.on_rig_param_changed)
        layout_rig.addWidget(self.ctrl_left_roll)

        self.ctrl_right_roll = LabeledSliderSpinBox("Roll Feinjustage Rechts:", -15.0, 15.0, 0.0, 0.1, "°")
        self.ctrl_right_roll.valueChanged.connect(self.on_rig_param_changed)
        layout_rig.addWidget(self.ctrl_right_roll)

        scroll_layout.addWidget(grp_rig)

        # 4. 32:9 Panorama Zentrierung & Bildausschnitt Card
        grp_pitch_align = QGroupBox("🏟️ Panorama-Zentrierung & Bildausschnitt")
        layout_pitch = QVBoxLayout(grp_pitch_align)
        layout_pitch.setSpacing(8)

        self.btn_pick_center_calib = QPushButton("🎯 Mittellinie im Bild anklicken (Sofort-Zentrierung)")
        self.btn_pick_center_calib.setCheckable(True)
        self.btn_pick_center_calib.setStyleSheet("background-color: #1e1e2d; border: 1px solid #3b82f6; font-weight: 600; padding: 7px; color: #60a5fa; border-radius: 8px;")
        self.btn_pick_center_calib.setToolTip("Klicken Sie direkt im Panorama-Bild auf den Anstoßpunkt oder die Mittellinie, um das Bild sofort perfekt zu zentrieren.")
        self.btn_pick_center_calib.toggled.connect(self.on_pick_center_toggled)
        layout_pitch.addWidget(self.btn_pick_center_calib)

        self.ctrl_h_offset = LabeledSliderSpinBox("Spielfeld-Mitte Versatz (Links ↔ Rechts Pan):", -60.0, 60.0, 0.0, 0.1, "°", decimals=1)
        self.ctrl_h_offset.valueChanged.connect(self.on_rig_param_changed)
        layout_pitch.addWidget(self.ctrl_h_offset)

        pan_btn_row = QHBoxLayout()
        pan_btn_row.setSpacing(4)
        for d_pan, txt in [(-5.0, "⏪ -5°"), (-1.0, "◀ -1°"), (0.0, "🎯 0°"), (1.0, "▶ +1°"), (5.0, "⏩ +5°")]:
            btn_pan = QPushButton(txt)
            btn_pan.setStyleSheet("background-color: #1a1a26; border: 1px solid #28283a; border-radius: 6px; padding: 4px; font-size: 11px; font-weight: 600; color: #cbd5e1;")
            if d_pan == 0.0:
                btn_pan.clicked.connect(lambda chk=False: self.ctrl_h_offset.setValue(0.0))
            else:
                btn_pan.clicked.connect(lambda chk=False, d=d_pan: self.pan_offset_by(d))
            pan_btn_row.addWidget(btn_pan)
        layout_pitch.addLayout(pan_btn_row)

        self.ctrl_v_offset = LabeledSliderSpinBox("Vertikaler Bildausschnitt (Shift):", -2.0, 2.0, 0.12, 0.02, "", decimals=2)
        self.ctrl_v_offset.valueChanged.connect(self.on_rig_param_changed)
        layout_pitch.addWidget(self.ctrl_v_offset)

        grp_dpad = QGroupBox("🎮 2D Bildausschnitt Feinjustage")
        layout_dpad = QGridLayout(grp_dpad)
        layout_dpad.setSpacing(4)
        btn_dpad_up = QPushButton("⬆️ Oben")
        btn_dpad_up.clicked.connect(lambda: self.shift_framing(0.0, -0.05))
        layout_dpad.addWidget(btn_dpad_up, 0, 1)
        btn_dpad_left = QPushButton("⬅️ Links")
        btn_dpad_left.clicked.connect(lambda: self.shift_framing(-1.5, 0.0))
        layout_dpad.addWidget(btn_dpad_left, 1, 0)
        btn_dpad_center = QPushButton("🎯 Reset")
        btn_dpad_center.clicked.connect(self.reset_framing)
        layout_dpad.addWidget(btn_dpad_center, 1, 1)
        btn_dpad_right = QPushButton("➡️ Rechts")
        btn_dpad_right.clicked.connect(lambda: self.shift_framing(1.5, 0.0))
        layout_dpad.addWidget(btn_dpad_right, 1, 2)
        btn_dpad_down = QPushButton("⬇️ Unten")
        btn_dpad_down.clicked.connect(lambda: self.shift_framing(0.0, 0.05))
        layout_dpad.addWidget(btn_dpad_down, 2, 1)
        layout_pitch.addWidget(grp_dpad)

        self.ctrl_pano_hfov = LabeledSliderSpinBox("32:9 Panorama Weitwinkel (HFOV):", 90.0, 180.0, 145.0, 0.5, "°")
        self.ctrl_pano_hfov.valueChanged.connect(self.on_rig_param_changed)
        layout_pitch.addWidget(self.ctrl_pano_hfov)

        self.ctrl_squeeze = LabeledSliderSpinBox("Gleichmäßige Breiten-Stauchung:", 0.50, 3.00, 1.0, 0.01, "x", decimals=2)
        self.ctrl_squeeze.valueChanged.connect(self.on_rig_param_changed)
        layout_pitch.addWidget(self.ctrl_squeeze)

        row_pitch_presets = QHBoxLayout()
        btn_full_pitch = QPushButton("🏟️ 16:9 Spielfeld (145° Preset)")
        btn_full_pitch.setStyleSheet("background-color: #1e1e2d; border: 1px solid #10b981; font-weight: bold; padding: 7px; color: #34d399; border-radius: 8px;")
        btn_full_pitch.clicked.connect(self.apply_full_pitch_preset)
        row_pitch_presets.addWidget(btn_full_pitch)

        btn_pitch_4x3 = QPushButton("📹 1080p 4:3 RockSteady")
        btn_pitch_4x3.setStyleSheet("background-color: #1e1e2d; border: 1px solid #38bdf8; font-weight: bold; padding: 7px; color: #38bdf8; border-radius: 8px;")
        btn_pitch_4x3.setToolTip("Richtet Rig und Blickfeld passend für 4:3 Dewarp mit RockSteady ein")
        btn_pitch_4x3.clicked.connect(self.apply_action4_4x3_rocksteady_preset)
        row_pitch_presets.addWidget(btn_pitch_4x3)
        layout_pitch.addLayout(row_pitch_presets)

        scroll_layout.addWidget(grp_pitch_align)

        # 5. Ecken-Verzerrung (Corner Pinning) Card
        grp_corner_pin = QGroupBox("📐 Ecken-Verzerrung (Corner Pinning)")
        layout_cp = QVBoxLayout(grp_corner_pin)
        layout_cp.setSpacing(8)

        self.btn_corner_pins_side = QPushButton("✨ Ecken interaktiv im Bild verzerren (Drag & Drop)")
        self.btn_corner_pins_side.setCheckable(True)
        self.btn_corner_pins_side.setStyleSheet("background-color: #2563eb; font-weight: bold; padding: 8px; color: white; border-radius: 8px;")
        self.btn_corner_pins_side.setToolTip("Aktivieren Sie diesen Modus, um die 4 Kamera-Ecken direkt im Panorama per Maus zu ziehen.")
        self.btn_corner_pins_side.toggled.connect(self.on_corner_pins_toggled)
        layout_cp.addWidget(self.btn_corner_pins_side)

        row_cam_sel = QHBoxLayout()
        row_cam_sel.addWidget(QLabel("Anzeige:"))
        self.combo_corner_cam = QComboBox()
        self.combo_corner_cam.addItem("🌟 Beide Kameras (Cyan = Links, Gold = Rechts)", "both")
        self.combo_corner_cam.addItem("🟦 Nur Kamera Links (Cyan)", "left")
        self.combo_corner_cam.addItem("🟧 Nur Kamera Rechts (Gold)", "right")
        self.combo_corner_cam.currentIndexChanged.connect(self.on_corner_cam_changed)
        row_cam_sel.addWidget(self.combo_corner_cam, 1)
        layout_cp.addLayout(row_cam_sel)

        self.ctrl_frame_opacity = LabeledSliderSpinBox("32:9 Rahmen Deckkraft (Abdunklung):", 0.0, 100.0, 50.0, 5.0, "%", decimals=0)
        self.ctrl_frame_opacity.valueChanged.connect(self.on_frame_opacity_changed)
        layout_cp.addWidget(self.ctrl_frame_opacity)

        self.tab_corners = QTabWidget()
        self.tab_corners.setStyleSheet("QTabWidget::pane { border: 1px solid #232332; background: #12121a; border-radius: 8px; } QTabBar::tab { font-size: 11px; padding: 5px 8px; }")

        # TAB LEFT CAMERA
        tab_c_left = QWidget()
        layout_cl = QVBoxLayout(tab_c_left)
        layout_cl.setContentsMargins(6, 6, 6, 6)
        layout_cl.setSpacing(4)

        self.spin_corners_l = []
        labels_l = [
            ("Oben-Links (TL):", 0.0, 0.0),
            ("Oben-Rechts (TR - Nahtstelle):", 100.0, 0.0),
            ("Unten-Rechts (BR - Nahtstelle):", 100.0, 100.0),
            ("Unten-Links (BL):", 0.0, 100.0)
        ]
        for idx, (lbl_txt, def_x, def_y) in enumerate(labels_l):
            row_box = QHBoxLayout()
            lbl = QLabel(lbl_txt)
            lbl.setStyleSheet("font-size: 11px; color: #60a5fa; font-weight: 600; min-width: 130px;")
            
            spin_x = NoWheelDoubleSpinBox()
            spin_x.setRange(-80.0, 180.0)
            spin_x.setValue(def_x)
            spin_x.setSingleStep(0.2)
            spin_x.setDecimals(1)
            spin_x.setPrefix("X: ")
            spin_x.setSuffix("%")
            spin_x.valueChanged.connect(lambda val, s='left', i=idx: self.on_corner_spin_changed(s, i))

            spin_y = NoWheelDoubleSpinBox()
            spin_y.setRange(-80.0, 180.0)
            spin_y.setValue(def_y)
            spin_y.setSingleStep(0.2)
            spin_y.setDecimals(1)
            spin_y.setPrefix("Y: ")
            spin_y.setSuffix("%")
            spin_y.valueChanged.connect(lambda val, s='left', i=idx: self.on_corner_spin_changed(s, i))

            btn_res_c = QToolButton()
            btn_res_c.setText("↺")
            btn_res_c.setToolTip(f"Diese Ecke auf Standard ({def_x:.0f}%, {def_y:.0f}%) zurücksetzen")
            btn_res_c.setStyleSheet("background-color: #1a1a26; border: 1px solid #28283a; border-radius: 4px; color: #94a3b8; padding: 2px 6px;")
            btn_res_c.clicked.connect(lambda checked=False, s='left', i=idx, dx=def_x, dy=def_y: self.reset_single_corner(s, i, dx, dy))

            row_box.addWidget(lbl)
            row_box.addWidget(spin_x, 1)
            row_box.addWidget(spin_y, 1)
            row_box.addWidget(btn_res_c)
            layout_cl.addLayout(row_box)
            self.spin_corners_l.append((spin_x, spin_y))

        btn_res_left_all = QPushButton("↺ Kamera Links Ecken zurücksetzen")
        btn_res_left_all.setStyleSheet("padding: 5px; font-size: 11px; background-color: #1a1a26; border: 1px solid #28283a; border-radius: 6px; color: #94a3b8;")
        btn_res_left_all.clicked.connect(lambda: self.reset_camera_corners('left'))
        layout_cl.addWidget(btn_res_left_all)

        self.tab_corners.addTab(tab_c_left, "🟦 Kamera Links (Cyan)")

        # TAB RIGHT CAMERA
        tab_c_right = QWidget()
        layout_cr = QVBoxLayout(tab_c_right)
        layout_cr.setContentsMargins(6, 6, 6, 6)
        layout_cr.setSpacing(4)

        self.spin_corners_r = []
        labels_r = [
            ("Oben-Links (TL - Nahtstelle):", 0.0, 0.0),
            ("Oben-Rechts (TR):", 100.0, 0.0),
            ("Unten-Rechts (BR):", 100.0, 100.0),
            ("Unten-Links (BL - Nahtstelle):", 0.0, 100.0)
        ]
        for idx, (lbl_txt, def_x, def_y) in enumerate(labels_r):
            row_box = QHBoxLayout()
            lbl = QLabel(lbl_txt)
            lbl.setStyleSheet("font-size: 11px; color: #fbbf24; font-weight: 600; min-width: 130px;")
            
            spin_x = NoWheelDoubleSpinBox()
            spin_x.setRange(-80.0, 180.0)
            spin_x.setValue(def_x)
            spin_x.setSingleStep(0.2)
            spin_x.setDecimals(1)
            spin_x.setPrefix("X: ")
            spin_x.setSuffix("%")
            spin_x.valueChanged.connect(lambda val, s='right', i=idx: self.on_corner_spin_changed(s, i))

            spin_y = NoWheelDoubleSpinBox()
            spin_y.setRange(-80.0, 180.0)
            spin_y.setValue(def_y)
            spin_y.setSingleStep(0.2)
            spin_y.setDecimals(1)
            spin_y.setPrefix("Y: ")
            spin_y.setSuffix("%")
            spin_y.valueChanged.connect(lambda val, s='right', i=idx: self.on_corner_spin_changed(s, i))

            btn_res_c = QToolButton()
            btn_res_c.setText("↺")
            btn_res_c.setToolTip(f"Diese Ecke auf Standard ({def_x:.0f}%, {def_y:.0f}%) zurücksetzen")
            btn_res_c.setStyleSheet("background-color: #1a1a26; border: 1px solid #28283a; border-radius: 4px; color: #94a3b8; padding: 2px 6px;")
            btn_res_c.clicked.connect(lambda checked=False, s='right', i=idx, dx=def_x, dy=def_y: self.reset_single_corner(s, i, dx, dy))

            row_box.addWidget(lbl)
            row_box.addWidget(spin_x, 1)
            row_box.addWidget(spin_y, 1)
            row_box.addWidget(btn_res_c)
            layout_cr.addLayout(row_box)
            self.spin_corners_r.append((spin_x, spin_y))

        btn_res_right_all = QPushButton("↺ Kamera Rechts Ecken zurücksetzen")
        btn_res_right_all.setStyleSheet("padding: 5px; font-size: 11px; background-color: #1a1a26; border: 1px solid #28283a; border-radius: 6px; color: #94a3b8;")
        btn_res_right_all.clicked.connect(lambda: self.reset_camera_corners('right'))
        layout_cr.addWidget(btn_res_right_all)

        self.tab_corners.addTab(tab_c_right, "🟧 Kamera Rechts (Gold)")
        layout_cp.addWidget(self.tab_corners)

        btn_res_all_corners = QPushButton("↺ Alle 8 Ecken auf Standard zurücksetzen")
        btn_res_all_corners.setStyleSheet("padding: 6px; font-weight: 600; background-color: #1a1a26; border: 1px solid #28283a; border-radius: 6px; color: #cbd5e1;")
        btn_res_all_corners.clicked.connect(self.reset_all_corners)
        layout_cp.addWidget(btn_res_all_corners)

        scroll_layout.addWidget(grp_corner_pin)

        # 6. Profile & Standard-Einstellungen Card
        grp_profiles = QGroupBox("💾 Profile & Standard-Einstellungen")
        layout_prof = QVBoxLayout(grp_profiles)
        layout_prof.setSpacing(6)

        row_prof_btns = QHBoxLayout()
        btn_save_full = QPushButton("💾 Profil speichern...")
        btn_save_full.setStyleSheet("background-color: #1e1e2d; border: 1px solid #2d2d42; font-weight: 600; padding: 6px; color: #f4f4f6; border-radius: 8px;")
        btn_save_full.clicked.connect(self.save_full_profile_as)

        btn_load_full = QPushButton("📁 Profil laden...")
        btn_load_full.setStyleSheet("background-color: #1e1e2d; border: 1px solid #2d2d42; font-weight: 600; padding: 6px; color: #f4f4f6; border-radius: 8px;")
        btn_load_full.clicked.connect(self.load_full_profile_from)

        row_prof_btns.addWidget(btn_save_full)
        row_prof_btns.addWidget(btn_load_full)
        layout_prof.addLayout(row_prof_btns)

        btn_set_default = QPushButton("⭐ Als Start-Standard festlegen")
        btn_set_default.setStyleSheet("background-color: #2563eb; font-weight: bold; padding: 7px; color: white; font-size: 11px; border-radius: 8px;")
        btn_set_default.clicked.connect(self.save_as_default_settings)
        layout_prof.addWidget(btn_set_default)

        btn_reset_defaults = QPushButton("↺ Auf Werkseinstellungen zurücksetzen")
        btn_reset_defaults.setStyleSheet("background-color: #161622; border: 1px solid #262638; padding: 5px; font-size: 11px; color: #8e8e9c; border-radius: 8px;")
        btn_reset_defaults.clicked.connect(self.restore_factory_defaults)
        layout_prof.addWidget(btn_reset_defaults)

        scroll_layout.addWidget(grp_profiles)

        scroll_layout.addStretch()
        scroll_area.setWidget(scroll_widget)
        calib_layout.addWidget(scroll_area)
        self.right_pane.addTab(self.tab_calib, "🎯 2. Stitch")

        # -------------------------------------------------------------------------
        # TAB 3: 📐 Taktik & AutoCam (Samsung One UI Style)
        # -------------------------------------------------------------------------
        self.tab_autocam = QWidget()
        autocam_layout = QVBoxLayout(self.tab_autocam)
        autocam_layout.setContentsMargins(4, 4, 4, 4)
        scroll_area_ac = QScrollArea()
        scroll_area_ac.setWidgetResizable(True)
        scroll_widget_ac = QWidget()
        scroll_layout_ac = QVBoxLayout(scroll_widget_ac)
        scroll_layout_ac.setSpacing(12)

        # 16:9 Tactical Mesh Warp Group (6 Points)
        grp_pitch_roi = QGroupBox("📐 16:9 Taktik-Warp (6-Punkte Video-Entzerrung)")
        layout_pitch_roi = QVBoxLayout(grp_pitch_roi)
        layout_pitch_roi.setSpacing(8)

        # Quick View-Switcher buttons
        row_view_quick = QHBoxLayout()
        btn_view_pano = QPushButton("🌟 32:9 Panorama")
        btn_view_pano.setToolTip("Zeigt das vollständige 32:9 Stadion-Panorama")
        btn_view_pano.clicked.connect(lambda: self.set_view_mode_by_id("32:9"))
        btn_view_tactical = QPushButton("📐 16:9 Taktik-Warp")
        btn_view_tactical.setToolTip("Zeigt das entzerrte 16:9 Spielfeld")
        btn_view_tactical.clicked.connect(lambda: self.set_view_mode_by_id("16:9_tactical"))
        row_view_quick.addWidget(btn_view_pano)
        row_view_quick.addWidget(btn_view_tactical)
        layout_pitch_roi.addLayout(row_view_quick)

        lbl_roi_desc = QLabel("Verzerren und passen Sie das Panorama-Video an den 6 Außenpunkten frei an das 16:9 Frame an. Die Mittellinien-Punkte (Wölbung Oben/Unten) begradigen gebogene Seitenlinien.")
        lbl_roi_desc.setStyleSheet("color: #8e8e9c; font-size: 11px;")
        lbl_roi_desc.setWordWrap(True)
        layout_pitch_roi.addWidget(lbl_roi_desc)

        self.btn_pitch_roi_side = QPushButton("📐 6 Punkte im Bild frei verziehen (Drag & Drop)")
        self.btn_pitch_roi_side.setCheckable(True)
        self.btn_pitch_roi_side.setStyleSheet("background-color: #2563eb; font-weight: bold; padding: 8px; color: white; border-radius: 8px;")
        self.btn_pitch_roi_side.setToolTip("Aktivieren Sie diesen Modus, um die 6 Punkte direkt im Vorschaubild per Maus zu ziehen.")
        self.btn_pitch_roi_side.toggled.connect(self.on_pitch_roi_toggled)
        layout_pitch_roi.addWidget(self.btn_pitch_roi_side)

        self.spin_pitch_corners = []
        labels_pitch = [
            ("🚩 Video Oben-Links (TL):", 0.0, 0.0, "#60a5fa"),
            ("📍 ↕ Wölbung Oben (TC):", 50.0, 0.0, "#38bdf8"),
            ("🚩 Video Oben-Rechts (TR):", 100.0, 0.0, "#60a5fa"),
            ("🚩 Video Unten-Rechts (BR):", 100.0, 100.0, "#60a5fa"),
            ("📍 ↕ Wölbung Unten (BC):", 50.0, 100.0, "#38bdf8"),
            ("🚩 Video Unten-Links (BL):", 0.0, 100.0, "#60a5fa")
        ]

        for idx, (lbl_txt, def_x, def_y, col) in enumerate(labels_pitch):
            row_box = QHBoxLayout()
            lbl = QLabel(lbl_txt)
            lbl.setStyleSheet(f"font-size: 11px; color: {col}; font-weight: 600; min-width: 150px;")

            spin_x = NoWheelDoubleSpinBox()
            spin_x.setRange(-20.0, 120.0)
            spin_x.setValue(def_x)
            spin_x.setSingleStep(0.5)
            spin_x.setDecimals(1)
            spin_x.setPrefix("X: ")
            spin_x.setSuffix("%")
            spin_x.valueChanged.connect(lambda val, i=idx: self.on_pitch_corner_spin_changed(i))

            spin_y = NoWheelDoubleSpinBox()
            spin_y.setRange(-20.0, 120.0)
            spin_y.setValue(def_y)
            spin_y.setSingleStep(0.5)
            spin_y.setDecimals(1)
            spin_y.setPrefix("Y: ")
            spin_y.setSuffix("%")
            spin_y.valueChanged.connect(lambda val, i=idx: self.on_pitch_corner_spin_changed(i))

            btn_res_c = QToolButton()
            btn_res_c.setText("↺")
            btn_res_c.setToolTip(f"Diesen Punkt auf Standard ({def_x:.0f}%, {def_y:.0f}%) zurücksetzen")
            btn_res_c.setStyleSheet("background-color: #1a1a26; border: 1px solid #28283a; border-radius: 4px; color: #94a3b8; padding: 2px 6px;")
            btn_res_c.clicked.connect(lambda checked=False, i=idx, dx=def_x, dy=def_y: self.reset_single_pitch_corner(i, dx, dy))

            row_box.addWidget(lbl)
            row_box.addWidget(spin_x, 1)
            row_box.addWidget(spin_y, 1)
            row_box.addWidget(btn_res_c)
            layout_pitch_roi.addLayout(row_box)
            self.spin_pitch_corners.append((spin_x, spin_y))

        row_pitch_btns = QHBoxLayout()
        btn_preset_curve = QPushButton("🏟️ Krümmung ausgleichen")
        btn_preset_curve.setStyleSheet("background-color: #1e1e2d; border: 1px solid #2563eb; font-weight: 600; padding: 6px; color: #60a5fa; border-radius: 8px;")
        btn_preset_curve.setToolTip("Setzt typische Entzerrungswerte für Mittellinien-Wölbung, um gebogene Seitenlinien zu begradigen")
        btn_preset_curve.clicked.connect(self.apply_preset_curve_warp)

        btn_preset_youth = QPushButton("⚽ Querfeld / Jugendspiel")
        btn_preset_youth.setStyleSheet("background-color: #1e293b; border: 1px solid #10b981; font-weight: 600; padding: 6px; color: #34d399; border-radius: 8px;")
        btn_preset_youth.setToolTip("Setzt das 6-Punkte-Polygon sofort passend für Jugendspiele auf dem Querfeld (untere Seitenlinie bei ~55%, Trainer & Bänke werden ignoriert)")
        btn_preset_youth.clicked.connect(self.apply_preset_youth_crossfield)

        btn_reset_pitch = QPushButton("↺ 100% Vollbild")
        btn_reset_pitch.setStyleSheet("background-color: #1a1a26; border: 1px solid #28283a; padding: 6px; font-size: 11px; color: #cbd5e1; border-radius: 8px;")
        btn_reset_pitch.clicked.connect(self.apply_preset_fullscreen_warp)

        row_pitch_btns.addWidget(btn_preset_curve)
        row_pitch_btns.addWidget(btn_preset_youth)
        row_pitch_btns.addWidget(btn_reset_pitch)
        layout_pitch_roi.addLayout(row_pitch_btns)

        self.ctrl_tactical_margin = LabeledSliderSpinBox("16:9 Taktik-Spielfeldrand (Puffer):", 0.0, 10.0, 0.0, 0.5, "%", decimals=1)
        self.ctrl_tactical_margin.setToolTip("Fügt einen Sicherheitsabstand um das entzerrte 16:9 Spielfeld hinzu, um Auslinien nicht am Bildschirmrand abzuschneiden.")
        self.ctrl_tactical_margin.valueChanged.connect(self.on_autocam_param_changed)
        layout_pitch_roi.addWidget(self.ctrl_tactical_margin)

        scroll_layout_ac.addWidget(grp_pitch_roi)

        # 16:9 AutoCam Follow Cam Group
        grp_ac_follow = QGroupBox("🎥 16:9 AutoCam (Automatische TV-Kameraführung)")
        layout_ac_follow = QVBoxLayout(grp_ac_follow)
        layout_ac_follow.setSpacing(8)

        self.chk_ai_yolo = QCheckBox("🤖 Deep Learning Soccer-KI (Spieler-Dichte & Ball)")
        self.chk_ai_yolo.setChecked(True)
        self.chk_ai_yolo.setStyleSheet("font-weight: 600; color: #10b981; margin-bottom: 4px;")
        self.chk_ai_yolo.toggled.connect(self.on_ai_yolo_toggled)
        layout_ac_follow.addWidget(self.chk_ai_yolo)

        lbl_track_mode = QLabel("Kameraführungs-Strategie:")
        lbl_track_mode.setStyleSheet("font-weight: 600; color: #f4f4f6; font-size: 11px;")
        layout_ac_follow.addWidget(lbl_track_mode)

        self.combo_ai_model = QComboBox()
        self.combo_ai_model.addItem("🏆 Soccer-Tracker Fusion (Spieler-Dichte + Ball - Empfohlen)", "hybrid_fusion")
        self.combo_ai_model.addItem("👥 Reine Spielerdichte (100% Team-Cluster, absolut stabil)", "player_density")
        self.combo_ai_model.addItem("⚽ Ball-Zentriert (100% Ball im 16:9 Bildzentrum)", "ball_centric")
        self.combo_ai_model.addItem("🎯 Weite Taktik-Kamera (Ruhige Feldübersicht)", "smooth_tactic")
        self.combo_ai_model.currentIndexChanged.connect(self.on_autocam_param_changed)
        layout_ac_follow.addWidget(self.combo_ai_model)

        # Action Fusion Weights (Ball vs Players)
        self.ctrl_ball_weight = LabeledSliderSpinBox("⚽ Ball-Gewichtung (w_ball):", 0.0, 1.0, 0.70, 0.05, "", decimals=2)
        self.ctrl_ball_weight.setToolTip("Bestimmt den Einfluss der Balltrajektorie auf die Kameraführung (0.70 = 70% Fokus auf den Ball)")
        self.ctrl_ball_weight.valueChanged.connect(self.on_autocam_param_changed)
        layout_ac_follow.addWidget(self.ctrl_ball_weight)

        self.ctrl_players_weight = LabeledSliderSpinBox("👥 Spieler-Gewichtung (w_spieler):", 0.0, 1.0, 0.30, 0.05, "", decimals=2)
        self.ctrl_players_weight.setToolTip("Bestimmt den Einfluss des Spielerschwerpunkts (Vorauseilen bei weiten Pässen)")
        self.ctrl_players_weight.valueChanged.connect(self.on_autocam_param_changed)
        layout_ac_follow.addWidget(self.ctrl_players_weight)

        self.chk_strict_ball_center = QCheckBox("🎯 Ball exakt im 16:9 Zentrum halten (Zero-Lead)")
        self.chk_strict_ball_center.setChecked(True)
        self.chk_strict_ball_center.setStyleSheet("font-weight: 600; color: #f59e0b; margin-top: 4px; margin-bottom: 2px;")
        self.chk_strict_ball_center.setToolTip("Hält den Ball bei Erkennung mathematisch exakt im Mittelpunkt des 16:9 Ausschnitts (horizontal und bei Zoom auch vertikal). Schaltet künstlichen Vorlauf-Versatz ab.")
        self.chk_strict_ball_center.toggled.connect(self.on_autocam_param_changed)
        layout_ac_follow.addWidget(self.chk_strict_ball_center)

        # Smart Zoom Group
        grp_zoom = QGroupBox("Kamera-Zoom")
        layout_zoom = QVBoxLayout(grp_zoom)
        layout_zoom.setSpacing(6)

        self.chk_dynamic_zoom = QCheckBox("🔍 Dynamischen Zoom aktivieren (Bei Pässen & Torschüssen)")
        self.chk_dynamic_zoom.setChecked(False)
        self.chk_dynamic_zoom.setStyleSheet("font-weight: 600; color: #38bdf8; margin-bottom: 4px;")
        self.chk_dynamic_zoom.toggled.connect(self.on_dynamic_zoom_toggled)
        layout_zoom.addWidget(self.chk_dynamic_zoom)

        self.ctrl_fixed_zoom = LabeledSliderSpinBox("Fester Zoom-Faktor (1.0x = Reiner Schwenk, kein Zoom):", 1.00, 2.00, 1.00, 0.05, "x", decimals=2)
        self.ctrl_fixed_zoom.valueChanged.connect(self.on_autocam_param_changed)
        layout_zoom.addWidget(self.ctrl_fixed_zoom)

        self.ctrl_min_zoom = LabeledSliderSpinBox("Min-Zoom (Weitwinkel bei Pässen):", 1.0, 1.4, 1.00, 0.05, "x", decimals=2)
        self.ctrl_min_zoom.valueChanged.connect(self.on_autocam_param_changed)
        self.ctrl_min_zoom.setVisible(False)
        layout_zoom.addWidget(self.ctrl_min_zoom)

        self.ctrl_max_zoom = LabeledSliderSpinBox("Max-Zoom (Nahaufnahme):", 1.2, 2.2, 1.40, 0.05, "x", decimals=2)
        self.ctrl_max_zoom.valueChanged.connect(self.on_autocam_param_changed)
        self.ctrl_max_zoom.setVisible(False)
        layout_zoom.addWidget(self.ctrl_max_zoom)

        self.ctrl_zoom_speed = LabeledSliderSpinBox("Zoom-Geschwindigkeit:", 0.01, 0.15, 0.04, 0.01, "", decimals=2)
        self.ctrl_zoom_speed.valueChanged.connect(self.on_autocam_param_changed)
        self.ctrl_zoom_speed.setVisible(False)
        layout_zoom.addWidget(self.ctrl_zoom_speed)

        layout_ac_follow.addWidget(grp_zoom)

        self.ctrl_ac_lead = LabeledSliderSpinBox("Spiel-Vorlauf / Antizipation (Blick in Spielrichtung):", 0.0, 0.40, 0.15, 0.05, "", decimals=2)
        self.ctrl_ac_lead.valueChanged.connect(self.on_autocam_param_changed)
        layout_ac_follow.addWidget(self.ctrl_ac_lead)

        self.ctrl_ac_smooth = LabeledSliderSpinBox("Kamera-Dämpfung (Broadcast-Sanftheit):", 0.70, 0.99, 0.94, 0.01, "", decimals=2)
        self.ctrl_ac_smooth.valueChanged.connect(self.on_autocam_param_changed)
        layout_ac_follow.addWidget(self.ctrl_ac_smooth)

        self.ctrl_ac_deadband = LabeledSliderSpinBox("Ruhezone / Ruckelfilter (Deadband):", 0.01, 0.25, 0.08, 0.01, "", decimals=2)
        self.ctrl_ac_deadband.valueChanged.connect(self.on_autocam_param_changed)
        layout_ac_follow.addWidget(self.ctrl_ac_deadband)

        self.ctrl_ac_speed = LabeledSliderSpinBox("Max. Schwenkgeschwindigkeit:", 0.01, 0.15, 0.04, 0.005, "", decimals=3)
        self.ctrl_ac_speed.valueChanged.connect(self.on_autocam_param_changed)
        layout_ac_follow.addWidget(self.ctrl_ac_speed)

        self.ctrl_ac_vpos = LabeledSliderSpinBox("Vertikale Ausrichtung (Spielfeld-Zentrum):", 0.35, 0.75, 0.50, 0.02, "", decimals=2)
        self.ctrl_ac_vpos.valueChanged.connect(self.on_autocam_param_changed)
        layout_ac_follow.addWidget(self.ctrl_ac_vpos)

        # Dynamic Zone Damping (Strafraum-Dynamik)
        self.chk_zone_damping = QCheckBox("🏟️ Zonen-Dämpfung (Höhere Agilität im Strafraum)")
        self.chk_zone_damping.setChecked(True)
        self.chk_zone_damping.setStyleSheet("font-weight: 600; color: #38bdf8; margin-top: 4px;")
        self.chk_zone_damping.setToolTip("Erhöht die Kameraschnelligkeit bei Torschüssen und Strafraumszenen automatisch")
        self.chk_zone_damping.toggled.connect(self.on_autocam_param_changed)
        layout_ac_follow.addWidget(self.chk_zone_damping)

        self.ctrl_zone_boost = LabeledSliderSpinBox("⚡ Strafraum-Schub (Agilitäts-Multiplikator):", 1.0, 3.0, 1.80, 0.1, "x", decimals=1)
        self.ctrl_zone_boost.setToolTip("Beschleunigungsfaktor der Kamera in den beiden Strafräumen (1.8x = 80% agilere Reaktion bei Torgefahr)")
        self.ctrl_zone_boost.valueChanged.connect(self.on_autocam_param_changed)
        layout_ac_follow.addWidget(self.ctrl_zone_boost)

        grp_scan_speed = QGroupBox("🤖 KI-Erkennungs-Modell & Performance")
        layout_scan = QVBoxLayout(grp_scan_speed)
        layout_scan.setSpacing(6)

        lbl_engine = QLabel("KI-Tracking-Architektur:")
        lbl_engine.setStyleSheet("font-weight: 600; color: #f4f4f6; font-size: 11px;")
        layout_scan.addWidget(lbl_engine)

        self.combo_ai_engine = QComboBox()
        self.combo_ai_engine.addItem("🏆 YOLO11 Ultra-Fast (Standard / Spezialisiert)", "yolo11")
        self.combo_ai_engine.addItem("🤖 RF-DETR Transformer (Apache 2.0 / Attention)", "rtdetr")
        self.combo_ai_engine.addItem("🦴 YOLO-Pose Skelett (Extremitäten & VAR-Abseits)", "pose")
        self.combo_ai_engine.currentIndexChanged.connect(self.on_ai_engine_changed)
        layout_scan.addWidget(self.combo_ai_engine)

        lbl_scan = QLabel("Berechnungsintervall:")
        lbl_scan.setStyleSheet("font-weight: 600; color: #f4f4f6; font-size: 11px; margin-top: 4px;")
        layout_scan.addWidget(lbl_scan)

        self.combo_scan_speed = QComboBox()
        self.combo_scan_speed.addItem("⚡ Ausgewogen (Jeder 5. Frame – Empfohlen)", 5)
        self.combo_scan_speed.addItem("🚀 Turbo-Modus (Jeder 8. Frame – 5x schneller)", 8)
        self.combo_scan_speed.addItem("🔥 Ultra-Speed (Jeder 12. Frame – 7x schneller)", 12)
        self.combo_scan_speed.addItem("🎯 Maximale Präzision (Jeder 2. Frame)", 2)
        self.combo_scan_speed.currentIndexChanged.connect(self.on_scan_speed_changed)
        layout_scan.addWidget(self.combo_scan_speed)

        self.chk_fp16 = QCheckBox("🚀 GPU FP16 Tensor-Cores (NVIDIA RTX)")
        self.chk_fp16.setChecked(True)
        self.chk_fp16.setStyleSheet("color: #10b981; font-weight: 600;")
        self.chk_fp16.toggled.connect(self.on_scan_speed_changed)
        layout_scan.addWidget(self.chk_fp16)
        layout_ac_follow.addWidget(grp_scan_speed)

        scroll_layout_ac.addWidget(grp_ac_follow)

        # Ball Precision & Sideline False-Positive Filter Group
        grp_precision = QGroupBox("🎯 Ball-Präzision & Spielfeld-Filter")
        layout_precision = QVBoxLayout(grp_precision)
        layout_precision.setSpacing(8)

        self.chk_anti_jersey = QCheckBox("👕 Trikot-Fehlalarm-Filter (Anti-Weißes-Trikot)")
        self.chk_anti_jersey.setChecked(True)
        self.chk_anti_jersey.setStyleSheet("font-weight: 600; color: #10b981;")
        self.chk_anti_jersey.setToolTip("Schließt Ball-Fehlalarme auf Spieleroberkörpern und weißen/hellen Trikots anatomisch aus.")
        self.chk_anti_jersey.toggled.connect(self.on_autocam_param_changed)
        layout_precision.addWidget(self.chk_anti_jersey)

        self.chk_static_leaf = QCheckBox("🍂 Laub- & Bodenflecken-Unterdrückung")
        self.chk_static_leaf.setChecked(True)
        self.chk_static_leaf.setStyleSheet("font-weight: 600; color: #38bdf8;")
        self.chk_static_leaf.setToolTip("Erkennt unbewegliche Flecken (Laubblätter, Rasenschäden) und sperrt diese temporär, wenn kein Spieler in Reichweite ist.")
        self.chk_static_leaf.toggled.connect(self.on_autocam_param_changed)
        layout_precision.addWidget(self.chk_static_leaf)

        self.chk_sideline_filter = QCheckBox("🚶 Trainer- & Auslinien-Filter (Nur aktives Spielfeld)")
        self.chk_sideline_filter.setChecked(True)
        self.chk_sideline_filter.setStyleSheet("font-weight: 600; color: #f59e0b;")
        self.chk_sideline_filter.setToolTip("Filtert Personen außerhalb des 6-Punkte-Polygons (Trainerzone, Auswechselspieler, Zuschauer) anhand präziser Bodenkontaktpunkte (Füße) aus.")
        self.chk_sideline_filter.toggled.connect(self.toggle_sideline_filter)
        layout_precision.addWidget(self.chk_sideline_filter)

        self.ctrl_pitch_margin = LabeledSliderSpinBox("Auslinien-Toleranzpuffer:", 0.0, 5.0, 1.0, 0.5, "%", decimals=1)
        self.ctrl_pitch_margin.setToolTip("Sicherheitsabstand nach innen/außen an der Auslinie (in % der Feldgröße).")
        self.ctrl_pitch_margin.valueChanged.connect(self.on_autocam_param_changed)
        layout_precision.addWidget(self.ctrl_pitch_margin)

        scroll_layout_ac.addWidget(grp_precision)

        # Tactical VAR & Pose Overlays Group
        grp_var = QGroupBox("🚩 Taktik & FIFA-VAR-Entscheidungen")
        layout_var = QVBoxLayout(grp_var)
        layout_var.setSpacing(8)

        self.chk_show_offside = QCheckBox("🚩 Virtuelle Abseitslinie (VAR-Linie)")
        self.chk_show_offside.setChecked(True)
        self.chk_show_offside.setStyleSheet("font-weight: 600; color: #f59e0b;")
        self.chk_show_offside.setToolTip("Berechnet den vorletzten Verteidiger millimetergenau anhand von Fuß- und Körperachsen und projiziert die Abseitslinie auf Rasen und Radar.")
        self.chk_show_offside.toggled.connect(self.toggle_offside_line)
        layout_var.addWidget(self.chk_show_offside)

        self.chk_show_skeletons = QCheckBox("🦴 Spieler-Skelette & Pose (17 Keypoints)")
        self.chk_show_skeletons.setChecked(False)
        self.chk_show_skeletons.setStyleSheet("font-weight: 600; color: #a855f7;")
        self.chk_show_skeletons.setToolTip("Zeigt Gelenke, Extremitäten (Sprunggelenke, Knie, Schultern) und Blick-/Laufrichtung der Spieler.")
        self.chk_show_skeletons.toggled.connect(self.toggle_skeletons)
        layout_var.addWidget(self.chk_show_skeletons)

        scroll_layout_ac.addWidget(grp_var)

        # Scouting & Tactical Metrics Group (SkillCorner & TacticAI)
        grp_scouting = QGroupBox("📊 Metrische Taktik-Analytik & Scouting")
        layout_scouting = QVBoxLayout(grp_scouting)
        layout_scouting.setSpacing(8)

        self.chk_show_speed = QCheckBox("⚡ Live-Geschwindigkeit (km/h) an Spielern einblenden")
        self.chk_show_speed.setChecked(False)
        self.chk_show_speed.setStyleSheet("font-weight: 600; color: #38bdf8;")
        self.chk_show_speed.setToolTip("Zeigt die metrische Momentangeschwindigkeit jedes Spielers und Balls in km/h sowie Sprint-Indikatoren (>25.2 km/h).")
        self.chk_show_speed.toggled.connect(self.toggle_speed_badges)
        layout_scouting.addWidget(self.chk_show_speed)

        btn_open_tactics = QPushButton("📊 Taktik- & Scouting-Dashboard öffnen")
        btn_open_tactics.setStyleSheet("""
            QPushButton { background-color: #2563eb; color: #ffffff; font-weight: 700; padding: 10px 14px; border-radius: 8px; font-size: 12px; }
            QPushButton:hover { background-color: #1d4ed8; }
        """)
        btn_open_tactics.setToolTip("Öffnet das interaktive Taktik-Dashboard mit 2D-Heatmaps, Pitch Control (Dominanz), Zonenverteilung und PDF/PNG-Export.")
        btn_open_tactics.clicked.connect(self.open_tactics_dashboard)
        layout_scouting.addWidget(btn_open_tactics)

        scroll_layout_ac.addWidget(grp_scouting)

        # 2D Tactical Radar Group (FIFA Pitch Mini-Map)
        grp_radar = QGroupBox("🗺️ 2D Taktik-Radar (Mini-Map)")
        layout_radar = QVBoxLayout(grp_radar)
        layout_radar.setSpacing(8)

        self.chk_show_radar = QCheckBox("🗺️ 2D Taktik-Radar im Bild einblenden")
        self.chk_show_radar.setChecked(True)
        self.chk_show_radar.setStyleSheet("font-weight: 600; color: #10b981;")
        self.chk_show_radar.setToolTip("Zeigt eine 2D-Vogelperspektive des FIFA-Spielfelds (105x68m) mit Live-Spielern (Blau/Rot), Ball und 16:9 Kamerasichtfeld")
        self.chk_show_radar.toggled.connect(self.on_radar_param_changed)
        layout_radar.addWidget(self.chk_show_radar)

        lbl_radar_pos = QLabel("Radar-Position im Bild:")
        lbl_radar_pos.setStyleSheet("font-weight: 600; color: #f4f4f6; font-size: 11px;")
        layout_radar.addWidget(lbl_radar_pos)

        self.combo_radar_pos = QComboBox()
        self.combo_radar_pos.addItem("↗️ Oben-Rechts (Standard)", "top_right")
        self.combo_radar_pos.addItem("↘️ Unten-Rechts", "bottom_right")
        self.combo_radar_pos.addItem("⬇️ Unten-Mitte (Klassisch)", "bottom_center")
        self.combo_radar_pos.addItem("↖️ Oben-Links", "top_left")
        self.combo_radar_pos.addItem("↙️ Unten-Links", "bottom_left")
        self.combo_radar_pos.currentIndexChanged.connect(self.on_radar_param_changed)
        layout_radar.addWidget(self.combo_radar_pos)

        self.ctrl_radar_width = LabeledSliderSpinBox("Radar-Größe (Breite):", 200, 520, 320, 10, "px", decimals=0)
        self.ctrl_radar_width.setToolTip("Breite der Mini-Map in Pixeln (Höhe passt sich automatisch dem FIFA 105x68m Seitenverhältnis an)")
        self.ctrl_radar_width.valueChanged.connect(self.on_radar_param_changed)
        layout_radar.addWidget(self.ctrl_radar_width)

        self.ctrl_radar_opacity = LabeledSliderSpinBox("Radar-Deckkraft:", 0.30, 1.00, 0.85, 0.05, "", decimals=2)
        self.ctrl_radar_opacity.setToolTip("Transparenz des Rasens und der Spielfeldmarkierungen im Videobild")
        self.ctrl_radar_opacity.valueChanged.connect(self.on_radar_param_changed)
        layout_radar.addWidget(self.ctrl_radar_opacity)

        self.chk_burn_radar = QCheckBox("🎬 Radar fest in exportiertes Video einbrennen")
        self.chk_burn_radar.setChecked(False)
        self.chk_burn_radar.setStyleSheet("font-weight: 600; color: #38bdf8; margin-top: 4px;")
        self.chk_burn_radar.setToolTip("Rendert die 2D-Minimap fest in das fertige Video beim 16:9- oder 32:9-Export.")
        self.chk_burn_radar.toggled.connect(self.on_burn_radar_toggled)
        layout_radar.addWidget(self.chk_burn_radar)

        scroll_layout_ac.addWidget(grp_radar)

        scroll_layout_ac.addStretch()
        scroll_area_ac.setWidget(scroll_widget_ac)
        autocam_layout.addWidget(scroll_area_ac)
        self.right_pane.addTab(self.tab_autocam, "📐 3. Taktik")

        # -------------------------------------------------------------------------
        # TAB 4: 🚀 Export (Samsung One UI Style)
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
        splitter.addWidget(self.right_pane)

        # Install wheel filter on right pane to prevent slider value changes on scrolling
        self._install_scroll_wheel_filter(self.right_pane)

        root_layout.addWidget(splitter, 1)

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
        root_layout.addWidget(self.log_viewer)

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

    # Video loading & seeking
    def open_left_video(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Linkes Video öffnen (DJI Action 4)", "", "Video Files (*.mp4 *.mov *.mkv *.avi)")
        if file_path:
            self.video_path_l = file_path
            self.lbl_file_left.setText(os.path.basename(file_path))
            self.lbl_file_left.setStyleSheet("color: #60a5fa; font-weight: 600;")
            if hasattr(self, 'badge_left'):
                self.badge_left.setText("🟢 Bereit")
                self.badge_left.setStyleSheet("background: #064e3b; color: #34d399; border-radius: 6px; padding: 2px 8px; font-size: 11px; font-weight: 600;")
            self._try_init_engine()

    def open_right_video(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Rechtes Video öffnen (DJI Action 4)", "", "Video Files (*.mp4 *.mov *.mkv *.avi)")
        if file_path:
            self.video_path_r = file_path
            self.lbl_file_right.setText(os.path.basename(file_path))
            self.lbl_file_right.setStyleSheet("color: #60a5fa; font-weight: 600;")
            if hasattr(self, 'badge_right'):
                self.badge_right.setText("🟢 Bereit")
                self.badge_right.setStyleSheet("background: #064e3b; color: #34d399; border-radius: 6px; padding: 2px 8px; font-size: 11px; font-weight: 600;")
            self._try_init_engine()

    def open_panorama_video(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "32:9 Panorama-Mastervideo öffnen", "", "Video Files (*.mp4 *.mov *.mkv *.avi)")
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
        if self.engine.is_panorama_mode():
            if hasattr(self, 'lbl_video_status'):
                self.lbl_video_status.setText("🎬 32:9 Panorama geladen")
                self.lbl_video_status.setStyleSheet("background: #064e3b; color: #34d399; border: 1px solid #059669; border-radius: 12px; padding: 3px 10px; font-size: 11px; font-weight: bold;")
            self.status_bar.showMessage(f"🎬 32:9 Panorama geladen: {self.engine.video_panorama.width}x{self.engine.video_panorama.height} bei {fps:.2f} FPS. Bereit für 16:9 Broadcast.")
        else:
            if hasattr(self, 'lbl_video_status'):
                self.lbl_video_status.setText("🟢 2 Kameras geladen")
                self.lbl_video_status.setStyleSheet("background: #064e3b; color: #34d399; border: 1px solid #059669; border-radius: 12px; padding: 3px 10px; font-size: 11px; font-weight: bold;")
            self.status_bar.showMessage(f"Videos geladen: {self.engine.video_left.width}x{self.engine.video_left.height} bei {fps:.2f} FPS. Bereit.")
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
            self.combo_exp_format.addItem("🌟 32:9 Panorama Master (Taktik-Ansicht / Ganzes Feld)", "32:9")
            self.combo_exp_format.addItem("📐 21:10 Gestaucht (Taktik-Panorama / Ganzes Feld gestaucht)", "21:10")
            self.combo_exp_format.addItem("🎥 16:9 Broadcast Follow-Cam (2-Stufig aus 32:9 Master)", "16:9_autocam")
            self.combo_exp_format.addItem("🔄 Beides exportieren (32:9 Master + 16:9 Broadcast)", "both")
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

            self.engine.invalidate_luts()
            self.render_high_res_preview()

            self.status_bar.showMessage(
                f"KI-Kalibrierung fertig! {result.num_matches} Matches | "
                f"Nahtfehler: {result.initial_error_px:.1f}px ➔ {result.optimized_error_px:.2f}px ({result.improvement_pct:.1f}% verbessert)"
            )
            QMessageBox.information(self, "KI-Nahtstellen-Kalibrierung", result.message)

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
        if not self.has_active_video():
            return
        if frame_idx is None:
            frame_idx = self.current_frame_idx
        
        max_f = max(0, self.engine.get_max_duration_frames() - 1)
        frame_idx = max(0, min(max_f, frame_idx))
        if frame_idx > self.out_point_frame:
            self.out_point_frame = frame_idx
        self.in_point_frame = frame_idx
        self.update_in_out_display()
        self.status_bar.showMessage(f"🚩 Startpunkt (In) auf Frame {frame_idx} ({self.format_timecode(frame_idx)}) gesetzt.")

    def set_out_point(self, frame_idx: Optional[int] = None):
        if not self.has_active_video():
            return
        if frame_idx is None:
            frame_idx = self.current_frame_idx

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

    def switch_to_step(self, index: int):
        """Switches the right sidebar tab to the designated workflow step and synchronizes step navigation."""
        if hasattr(self, 'right_pane') and 0 <= index < self.right_pane.count():
            self.right_pane.setCurrentIndex(index)
        self._sync_nav_step_buttons(index)

    def _on_tab_changed(self, index: int):
        """Handles sidebar tab switching and updates top step navigation pills."""
        self._sync_nav_step_buttons(index)

    def _sync_nav_step_buttons(self, active_index: int):
        """Synchronizes top bar step buttons with the active tab index."""
        if not hasattr(self, 'nav_step_btns'):
            return
        for i, btn in enumerate(self.nav_step_btns):
            btn.blockSignals(True)
            btn.setChecked(i == active_index)
            btn.blockSignals(False)

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
        """Quick 1-click toggle between 32:9 panorama overview and rectified 16:9 tactical field view."""
        target_mode = "16:9_tactical" if checked else "32:9"
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
        self.rig.left_pose.pitch = self.ctrl_cam_pitch.value()
        self.rig.right_pose.pitch = self.ctrl_cam_pitch.value()
        self.rig.left_pose.roll = self.ctrl_left_roll.value()
        self.rig.right_pose.roll = self.ctrl_right_roll.value()

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
        self.refresh_preview()
        self.status_bar.showMessage("📹 Preset aktiviert: DJI Action 4 1080p 4:3 Dewarp (RockSteady ON, 30fps) - Rig & Stitching optimiert!")

    def toggle_auto_crop(self, checked: bool):
        self.rig.auto_crop_lir = checked
        self.engine.invalidate_luts()
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
        """Switches the active detector backend (YOLO11, RT-DETR, YOLO-Pose)."""
        if hasattr(self, 'combo_ai_engine') and hasattr(self.engine, 'ai_broadcast'):
            backend_id = self.combo_ai_engine.currentData() or "yolo11"
            self.engine.ai_broadcast.set_backend(backend_id)
            if hasattr(self, 'status_bar'):
                self.status_bar.showMessage(f"🤖 KI-Modell gewechselt: {self.combo_ai_engine.currentText()}", 3500)
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
        self.combo_resolution.clear()
        fmt = self.combo_exp_format.currentData()
        if fmt in ("16:9_autocam", "16:9_tactical"):
            self.combo_resolution.addItem("1920 x 1080 (Full HD 1080p - Standard 16:9)", (1920, 1080))
            self.combo_resolution.addItem("1280 x 720 (HD 720p - Schnell / Kompakt)", (1280, 720))
            self.combo_resolution.addItem("2560 x 1440 (2K QHD 1440p - Hohe Schärfe)", (2560, 1440))
            self.combo_resolution.addItem("3840 x 2160 (4K UHD 2160p - Maximale Details)", (3840, 2160))
        elif fmt == "21:10":
            self.combo_resolution.addItem("3360 x 1600 (1600p 21:10 - Hohe Schärfe / Empfohlen)", (3360, 1600))
            self.combo_resolution.addItem("2520 x 1200 (1200p 21:10 - Standard / Hohe FPS)", (2520, 1200))
            self.combo_resolution.addItem("1512 x 720 (720p 21:10 - Schnell / Kompakt)", (1512, 720))
            self.combo_resolution.addItem("4200 x 2000 (2K/4K 21:10 - Maximale Details)", (4200, 2000))
            self.combo_resolution.addItem("2100 x 1000 (1000p 21:10 - Schnell)", (2100, 1000))
        elif fmt == "both":
            self.combo_resolution.addItem("1920 x 1080 (1080p Broadcast + 3840x1080 Master)", (1920, 1080))
            self.combo_resolution.addItem("1280 x 720 (720p Broadcast + 2560x720 Master - Schnell)", (1280, 720))
            self.combo_resolution.addItem("2560 x 1440 (1440p Broadcast + 5120x1440 Master)", (2560, 1440))
            self.combo_resolution.addItem("3840 x 2160 (4K Broadcast + 7680x2160 Master)", (3840, 2160))
        else: # 32:9
            self.combo_resolution.addItem("5120 x 1440 (1440p 32:9 Panorama - Empfohlen)", (5120, 1440))
            self.combo_resolution.addItem("3840 x 1080 (1080p 32:9 Panorama - Standard)", (3840, 1080))
            self.combo_resolution.addItem("2560 x 720 (720p 32:9 Panorama - Schnell / Kompakt)", (2560, 720))
            self.combo_resolution.addItem("7680 x 2160 (4K 32:9 Panorama - Maximale Qualität)", (7680, 2160))

    def trigger_smooth_preview(self):
        """Instant draft update during dragging (<2ms) + debounced high-res render."""
        if not self.has_active_video():
            return
        self.viewport.set_rig(self.rig)
        vm = getattr(self.viewport, 'view_mode', '32:9')
        if vm == "21:10":
            pw, ph = 1260, 600
        else:
            pw, ph = 1280, 360
        frame, autocam_box, ball_px, cur_zoom = self.engine.render_preview_frame(self.current_frame_idx, preview_width=pw, preview_height=ph, view_mode=vm)
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
        vm = getattr(self.viewport, 'view_mode', '32:9')
        if vm == "21:10":
            pw, ph = 2100, 1000
        else:
            pw, ph = 2560, 720
        frame, autocam_box, ball_px, cur_zoom = self.engine.render_preview_frame(self.current_frame_idx, preview_width=pw, preview_height=ph, view_mode=vm)
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
            return
        self.viewport.set_rig(self.rig)
        vm = getattr(self.viewport, 'view_mode', '32:9')
        if vm == "21:10":
            pw, ph = 2100, 1000
        else:
            pw, ph = 2560, 720
        frame, autocam_box, ball_px, cur_zoom = self.engine.render_preview_frame(self.current_frame_idx, preview_width=pw, preview_height=ph, view_mode=vm)
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
                    "ai_tracking": True,
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

    def start_video_export(self):
        if not self.has_active_video():
            QMessageBox.warning(self, "Export", "Bitte zuerst Videos laden (2 Kameras oder 32:9 Panorama)!")
            return

        if not self.engine.is_panorama_mode() and (not self.engine.video_left or not self.engine.video_right):
            QMessageBox.warning(self, "Export", "Bitte zuerst beide Kameras öffnen oder ein 32:9 Panorama laden!")
            return

        fmt = self.combo_exp_format.currentData()
        if fmt == "32:9":
            default_name = "fussball_panorama_32x9.mp4"
        elif fmt == "21:10":
            default_name = "fussball_panorama_21x10.mp4"
        elif fmt == "both":
            default_name = "fussball_broadcast_16x9.mp4"
        else:
            default_name = "fussball_broadcast_16x9.mp4"

        out_path, _ = QFileDialog.getSaveFileName(self, "Video speichern", default_name, "MP4 Video (*.mp4)")
        if not out_path:
            return

        w, h = self.combo_resolution.currentData()
        codec = self.combo_codec.currentData()
        bitrate = int(self.ctrl_bitrate.value())
        audio_src = self.combo_audio_source.currentData() if hasattr(self, 'combo_audio_source') else "left"
        use_lookahead = self.chk_lookahead.isChecked() if hasattr(self, 'chk_lookahead') else True

        total_avail = self.engine.get_max_duration_frames()
        if self.chk_export_trim_only.isChecked():
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
            use_lookahead=use_lookahead
        )
        self.render_worker.progress_signal.connect(self.on_render_progress)
        self.render_worker.finished_signal.connect(self.on_render_finished)
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


