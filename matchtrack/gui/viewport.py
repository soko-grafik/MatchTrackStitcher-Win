"""
InteractivePanoramaViewport:
High-resolution interactive QPainter viewport for MatchTrack-Stitcher.
Supports:
- 32:9 Panorama & 16:9 Broadcast viewport rendering
- Photoshop-style Interactive 4-Corner Pinning drag & drop
- 6-Point Tactical Pitch Warp polygon handles
- 4-Point Wide-Angle Trapezoid alignment handles
- Semi-transparent frame overlay
- Real-time YOLO AI tracking boxes, Ball reticle, Player skeletons, Speed badges
- Offside line overlay & 2D Tactical Minimap Radar
"""

import math
from typing import Optional, List, Tuple, Dict, Any
import numpy as np
import cv2

from PySide6.QtCore import Qt, Signal, QPointF, QRectF, QPoint
from PySide6.QtGui import (
    QPainter, QPen, QColor, QFont, QImage, QPolygonF,
    QMouseEvent, QWheelEvent, QPainterPath
)
from PySide6.QtWidgets import QWidget

from ..rig_geometry import RigConfiguration
from ..radar_minimap import SoccerRadarRenderer
from ..ai_detector import SKELETON_CONNECTIONS
from ..field_calibration import FieldMarking
from ..logger import get_logger

logger = get_logger("gui.viewport")

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
    grid_quadrant_clicked = Signal(int) # 1, 2, 3, 4 (or 0 for toggle back to grid)
    wide_angle_corners_changed = Signal(list) # 4 pitch corners [[x_tl, y_tl], [x_tr, y_tr], [x_br, y_br], [x_bl, y_bl]]
    wide_angle_step_changed = Signal(int) # 0..3 for sequential click step, or -1 when finished
    crop_rect_changed = Signal(float, float, float, float) # norm_x, norm_y, norm_w, norm_h
    field_marking_changed = Signal(object) # emits FieldMarking instance
    framing_changed = Signal(float, float, float, float) # widest_z, tightest_z, center_x, center_y

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

        # 4-Point Wide-Angle Calibration & Trapezoid View (DJI Action 4)
        self.wide_angle_edit_mode = False
        self.wide_angle_step: int = -1  # 0..3 for sequential click placement, -1 when inactive
        self.wide_angle_corners: List[List[float]] = [
            [0.15, 0.25],  # 0: TL (Top-Left, hinten links)
            [0.85, 0.25],  # 1: TR (Top-Right, hinten rechts)
            [0.95, 0.88],  # 2: BR (Bottom-Right, vorne rechts)
            [0.05, 0.88]   # 3: BL (Bottom-Left, vorne links)
        ]
        self.hovered_wa_corner: Optional[int] = None
        self.dragging_wa_corner: Optional[int] = None

        # Photoshop Crop & Transform attributes
        self.crop_mode: bool = False
        self.crop_rect_norm: List[float] = [0.0, 0.0, 1.0, 1.0] # [x, y, w, h] from 0.0 to 1.0
        self.crop_aspect_ratio: Optional[float] = None
        self.hovered_crop_handle: Optional[str] = None
        self.dragging_crop_handle: Optional[str] = None
        self.crop_drag_start_pos: Optional[QPoint] = None
        self.crop_drag_start_rect: Optional[List[float]] = None
        self.transform_stretch_x: float = 1.0
        self.transform_stretch_y: float = 1.0

        # Autocam 10-Point Field Marking attributes
        self.field_marking: FieldMarking = FieldMarking()
        self.hovered_field_point: Optional[int] = None
        self.dragging_field_point: Optional[int] = None

        # Autocam 2-Box Visual Framing attributes
        self.framing_widest_zoom: float = 1.2
        self.framing_tightest_zoom: float = 2.0
        self.framing_center_x: float = 0.5
        self.framing_center_y: float = 0.5
        self.hovered_framing_box: Optional[str] = None
        self.dragging_framing_box: Optional[str] = None
        self.framing_drag_start_pos: Optional[QPoint] = None

        # Workflow Domain for strict tab-isolation ("stitching", "wide_angle", "autocam", "livestream", "cloud", "field_marking", "framing")
        self.workflow_domain: str = "stitching"
        self.placeholder_text: str = "Kein Video geladen. Bitte links und rechts DJI Action 4 Videos öffnen."

    def set_workflow_domain(self, domain: str):
        """Sets the active workflow domain to strictly filter overlays and interaction handles."""
        self.workflow_domain = domain
        if domain == "stitching":
            self.placeholder_text = "Kein Video geladen. Bitte links und rechts Videos laden oder Audio synchronisieren."
        elif domain == "wide_angle":
            self.placeholder_text = "Kein Weitwinkel-Video geladen. Bitte ein 155° Video für Trapez-Begradigung öffnen."
        elif domain == "autocam":
            self.placeholder_text = "Kein Video geladen für KI-Broadcast & Tracking."
        elif domain == "livestream":
            self.placeholder_text = "Live-Stream Studio: Kameras verbinden oder RTMP-Stream starten."
        elif domain == "cloud":
            self.placeholder_text = "MatchTrack Online: Video auswählen oder hochladen."
        elif domain == "field_marking":
            self.placeholder_text = "Kein Video geladen. Bitte Video für die Spielfeld-Markierung (10 Punkte) laden."
        elif domain == "framing":
            self.placeholder_text = "Kein Video geladen für das visuelle 2-Box Framing."
        self.update()

    def clear_preview(self):
        """Clears current preview image and restores clean domain placeholder."""
        self.current_qimage = None
        self.current_frame_bgr = None
        self.update()

    def set_wide_angle_corners(self, corners: List[List[float]]):
        """Sets normalized 4-point pitch corners for wide-angle trapezoid rectification."""
        if corners and len(corners) == 4:
            self.wide_angle_corners = [[float(c[0]), float(c[1])] for c in corners]
            self.update()

    def set_wide_angle_edit_mode(self, enabled: bool):
        """Toggles interactive wide-angle 4-point corner drag and drop."""
        self.wide_angle_edit_mode = enabled
        if not enabled:
            self.hovered_wa_corner = None
            self.dragging_wa_corner = None
            self.wide_angle_step = -1
            self.setCursor(Qt.ArrowCursor)
        self.update()

    def start_wide_angle_marking(self):
        """Initiates sequential 4-point click-to-place calibration (TL -> TR -> BR -> BL)."""
        self.wide_angle_edit_mode = True
        self.wide_angle_step = 0
        self.setCursor(Qt.CrossCursor)
        self.wide_angle_step_changed.emit(0)
        self.update()

    def _get_wa_corner_screen_points(self, target_rect: QRectF) -> List[QPointF]:
        """Calculates viewport screen pixel coordinates for the 4 normalized wide-angle corners."""
        pts = []
        for x_norm, y_norm in self.wide_angle_corners:
            sx = target_rect.left() + x_norm * target_rect.width()
            sy = target_rect.top() + y_norm * target_rect.height()
            pts.append(QPointF(sx, sy))
        return pts

    def _hit_test_wa_corner(self, pos: QPoint, target_rect: QRectF, radius: float = 18.0) -> Optional[int]:
        """Returns corner index (0..3) if mouse click/hover is within threshold radius."""
        pts = self._get_wa_corner_screen_points(target_rect)
        for idx, pt in enumerate(pts):
            dist = math.hypot(pos.x() - pt.x(), pos.y() - pt.y())
            if dist <= radius:
                return idx
        return None

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

    def set_crop_mode(self, enabled: bool):
        """Toggles Photoshop-style interactive crop mode."""
        self.crop_mode = enabled
        if not enabled:
            self.hovered_crop_handle = None
            self.dragging_crop_handle = None
            self.setCursor(Qt.ArrowCursor)
        self.update()

    def set_crop_rect(self, x: float, y: float, w: float, h: float):
        """Sets normalized crop rectangle [x, y, w, h] in 0.0 .. 1.0 range."""
        x = max(0.0, min(0.95, float(x)))
        y = max(0.0, min(0.95, float(y)))
        w = max(0.05, min(1.0 - x, float(w)))
        h = max(0.05, min(1.0 - y, float(h)))
        self.crop_rect_norm = [x, y, w, h]
        self.update()

    def set_crop_aspect_ratio(self, ratio: Optional[float]):
        """Sets fixed aspect ratio for crop tool (e.g. 16/9, 21/10, None for free)."""
        self.crop_aspect_ratio = ratio
        if ratio is not None:
            nx, ny, nw, nh = self.crop_rect_norm
            target_rect = self._get_target_rect()
            if target_rect and target_rect.height() > 0:
                frame_aspect = target_rect.width() / target_rect.height()
                target_norm_ratio = ratio / frame_aspect
                new_h = nw / target_norm_ratio
                if ny + new_h > 1.0:
                    new_h = 1.0 - ny
                    nw = new_h * target_norm_ratio
                self.crop_rect_norm = [nx, ny, max(0.05, nw), max(0.05, new_h)]
                self.crop_rect_changed.emit(*self.crop_rect_norm)
        self.update()

    def set_transform_stretch(self, sx: float, sy: float):
        """Sets anamorphic stretch factors (X and Y scaling)."""
        self.transform_stretch_x = max(0.2, min(3.0, float(sx)))
        self.transform_stretch_y = max(0.2, min(3.0, float(sy)))
        self.update()

    def _get_crop_rect_px(self, target_rect: QRectF) -> QRectF:
        """Calculates pixel QRectF from normalized crop coordinates."""
        nx, ny, nw, nh = self.crop_rect_norm
        px = target_rect.left() + nx * target_rect.width()
        py = target_rect.top() + ny * target_rect.height()
        pw = nw * target_rect.width()
        ph = nh * target_rect.height()
        return QRectF(px, py, pw, ph)

    def _hit_test_crop_handle(self, pos: QPoint, target_rect: QRectF) -> Optional[str]:
        """Returns the handle identifier under mouse position or None."""
        rect = self._get_crop_rect_px(target_rect)
        r = 10.0  # Hit radius in pixels
        mx, my = float(pos.x()), float(pos.y())
        handles = {
            'tl': QPointF(rect.left(), rect.top()),
            'tc': QPointF(rect.center().x(), rect.top()),
            'tr': QPointF(rect.right(), rect.top()),
            'mr': QPointF(rect.right(), rect.center().y()),
            'br': QPointF(rect.right(), rect.bottom()),
            'bc': QPointF(rect.center().x(), rect.bottom()),
            'bl': QPointF(rect.left(), rect.bottom()),
            'ml': QPointF(rect.left(), rect.center().y()),
        }
        for name, pt in handles.items():
            if abs(mx - pt.x()) <= r and abs(my - pt.y()) <= r:
                return name
        if rect.contains(QPointF(mx, my)):
            return 'inside'
        return None

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

    def update_preview_image(self, 
                             frame_bgr: np.ndarray, 
                             autocam_box: Optional[Tuple[int, int, int, int]] = None, 
                             ball_px: Optional[Tuple[int, int]] = None):
        """Helper / alias to update the preview frame from live-streaming or capture engines."""
        self.set_frame(frame_bgr, autocam_box=autocam_box, ball_px=ball_px)

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
        eff_iw = iw * getattr(self, 'transform_stretch_x', 1.0)
        eff_ih = ih * getattr(self, 'transform_stretch_y', 1.0)
        scale_fit = min(vw / eff_iw, vh / eff_ih)
        draw_w = eff_iw * scale_fit * self.zoom_level
        draw_h = eff_ih * scale_fit * self.zoom_level
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
        try:
            self._render_viewport(painter)
        finally:
            painter.end()

    def _render_viewport(self, painter: QPainter):
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)

        painter.fillRect(self.rect(), QColor("#0d0d0d"))

        if self.current_qimage is None:
            painter.setPen(QColor("#666666"))
            painter.setFont(QFont("Segoe UI", 12))
            painter.drawText(self.rect(), Qt.AlignCenter, getattr(self, 'placeholder_text', 'Kein Video geladen.'))
            return

        target_rect = self._get_target_rect()
        if target_rect is None:
            return

        vw, vh = self.width(), self.height()
        iw, ih = self.current_qimage.width(), self.current_qimage.height()

        # 1. Draw Stitched Frame Image
        painter.drawImage(target_rect, self.current_qimage)

        # 2. Semi-Transparent 32:9 / 8:3 / 21:10 Frame Overlay (Matte/Letterbox)
        if self.show_transparent_frame and self.view_mode in ("32:9", "8:3", "21:10") and self.workflow_domain in ("stitching", "autocam"):
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
            is_8x3 = (self.view_mode == "8:3") or getattr(self, 'is_4_3', False) or (self.current_qimage and abs(self.current_qimage.width() / max(1, self.current_qimage.height()) - (8.0 / 3.0)) < 0.25)
            if self.view_mode == "21:10":
                tag_title = "🔲 21:10 Panorama (Gestaucht)"
            elif is_8x3 and self.view_mode != "32:9":
                tag_title = "🔲 8:3 Panorama-Ausschnitt"
            elif self.view_mode == "8:3":
                tag_title = "🔲 8:3 Panorama-Ausschnitt"
            else:
                tag_title = "🔲 32:9 Panorama-Ausschnitt"
            tag_frame = QRectF(target_rect.center().x() - 105, target_rect.top() + 6, 210, 20)
            painter.fillRect(tag_frame, QColor(41, 128, 185, 210))
            painter.setPen(QColor(255, 255, 255))
            painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
            painter.drawText(tag_frame, Qt.AlignCenter, tag_title)

        # 3. Overlays for 32:9 / 8:3 / 21:10 mode
        if self.view_mode in ("32:9", "8:3", "21:10"):
            # Field Center Reference Line (50% Center)
            if self.show_center_line and self.workflow_domain == "stitching":
                center_x = target_rect.left() + target_rect.width() * 0.5
                pen_center = QPen(QColor(241, 196, 15, 220), 2, Qt.DashDotLine)
                painter.setPen(pen_center)
                painter.drawLine(int(center_x), int(target_rect.top()), int(center_x), int(target_rect.bottom()))
                
                tag_center = QRectF(center_x - 65, target_rect.top() + 6, 130, 20)
                painter.fillRect(tag_center, QColor(241, 196, 15, 210))
                painter.setPen(QColor(0, 0, 0))
                painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
                painter.drawText(tag_center, Qt.AlignCenter, "🎯 Spielfeld-Mitte")

            if self.show_seam and self.workflow_domain == "stitching":
                luts_ref = getattr(self, 'luts_ref', None)
                smart_path = getattr(luts_ref, 'smart_seam_path', None) if luts_ref else None

                pen_seam = QPen(QColor(255, 75, 75, 200), 1.5, Qt.DashLine)
                painter.setPen(pen_seam)

                if smart_path is not None and len(smart_path) > 0 and getattr(self.rig_ref, 'smart_seam_enabled', False):
                    # Draw actual curved polyline of smart seam
                    H_lut = len(smart_path)
                    W_lut = getattr(luts_ref, 'out_width', 1)
                    poly = QPolygonF()
                    step = max(1, H_lut // 40)
                    for y_idx in range(0, H_lut, step):
                        px_norm = float(smart_path[y_idx]) / float(W_lut)
                        py_norm = float(y_idx) / float(H_lut)
                        sx = target_rect.left() + target_rect.width() * px_norm
                        sy = target_rect.top() + target_rect.height() * py_norm
                        poly.append(QPointF(sx, sy))
                    sx_end = target_rect.left() + target_rect.width() * (float(smart_path[-1]) / float(W_lut))
                    sy_end = target_rect.bottom()
                    poly.append(QPointF(sx_end, sy_end))
                    
                    pen_smart = QPen(QColor(168, 85, 247, 230), 2.0, Qt.SolidLine)
                    painter.setPen(pen_smart)
                    painter.drawPolyline(poly)

                    tag_seam = QRectF(target_rect.left() + target_rect.width() * 0.5 - 75, target_rect.top() + 28, 150, 18)
                    painter.fillRect(tag_seam, QColor(126, 34, 206, 210))
                    painter.setPen(QColor(255, 255, 255))
                    painter.setFont(QFont("Segoe UI", 7, QFont.Bold))
                    painter.drawText(tag_seam, Qt.AlignCenter, "🧠 Intelligente Kontur-Naht")
                else:
                    seam_offset_deg = getattr(self.rig_ref, 'seam_angle_offset', 0.0) if self.rig_ref else 0.0
                    pano_hfov = getattr(self.rig_ref, 'pano_hfov', 145.0) if self.rig_ref else 145.0
                    seam_norm = 0.5 + (seam_offset_deg / max(1.0, pano_hfov))
                    seam_x = target_rect.left() + target_rect.width() * seam_norm
                    painter.drawLine(int(seam_x), int(target_rect.top()), int(seam_x), int(target_rect.bottom()))

                    tag_seam = QRectF(seam_x - 55, target_rect.top() + 28, 110, 18)
                    painter.fillRect(tag_seam, QColor(220, 38, 38, 200))
                    painter.setPen(QColor(255, 255, 255))
                    painter.setFont(QFont("Segoe UI", 7, QFont.Bold))
                    painter.drawText(tag_seam, Qt.AlignCenter, "⚡ Nahtstelle")

            if self.show_grid and self.workflow_domain in ("stitching", "autocam"):
                pen_grid = QPen(QColor(70, 230, 110, 140), 1, Qt.SolidLine)
                painter.setPen(pen_grid)
                horiz_y = target_rect.top() + target_rect.height() * 0.5
                painter.drawLine(int(target_rect.left()), int(horiz_y), int(target_rect.right()), int(horiz_y))
                ground_y = target_rect.top() + target_rect.height() * 0.8
                painter.drawLine(int(target_rect.left()), int(ground_y), int(target_rect.right()), int(ground_y))

            # 16:9 Broadcast Box
            if self.show_autocam_box and self.autocam_box is not None and self.workflow_domain == "autocam":
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
            if self.show_ball_tracking and self.ball_px is not None and self.workflow_domain == "autocam":
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
            if self.show_skeletons and self.current_detections and self.workflow_domain == "autocam":
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
            if self.show_offside_line and self.current_offside_line is not None and self.workflow_domain == "autocam":
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
            if self.show_speed and self.current_detections and self.workflow_domain == "autocam":
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
        if self.corner_pin_mode and self.workflow_domain == "stitching" and self.view_mode in ("32:9", "8:3", "21:10"):
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

        # 4b. Photoshop-Style Interactive Crop & Freistellungs-Tool
        if self.crop_mode and self.workflow_domain == "stitching" and target_rect is not None:
            crop_px = self._get_crop_rect_px(target_rect)

            # Top notification banner
            banner_rect = QRectF(0, 0, self.width(), 30)
            painter.fillRect(banner_rect, QColor(24, 43, 73, 235))
            painter.setPen(QColor(255, 255, 255))
            painter.setFont(QFont("Segoe UI", 9, QFont.Bold))
            painter.drawText(banner_rect, Qt.AlignCenter, "✂️ Freistellen (Photoshop-Modus): Ziehen Sie die 8 Kanten & Ecken zur Wahl des Bildausschnitts")

            # Dim outer mask (4 areas outside crop rectangle)
            mask_color = QColor(0, 0, 0, 165)
            # Top mask
            painter.fillRect(QRectF(target_rect.left(), target_rect.top(), target_rect.width(), max(0.0, crop_px.top() - target_rect.top())), mask_color)
            # Bottom mask
            painter.fillRect(QRectF(target_rect.left(), crop_px.bottom(), target_rect.width(), max(0.0, target_rect.bottom() - crop_px.bottom())), mask_color)
            # Left mask
            painter.fillRect(QRectF(target_rect.left(), crop_px.top(), max(0.0, crop_px.left() - target_rect.left()), crop_px.height()), mask_color)
            # Right mask
            painter.fillRect(QRectF(crop_px.right(), crop_px.top(), max(0.0, target_rect.right() - crop_px.right()), crop_px.height()), mask_color)

            # White crop boundary
            painter.setPen(QPen(QColor(255, 255, 255, 240), 2, Qt.SolidLine))
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(crop_px)

            # Rule of Thirds grid (1px dashed)
            painter.setPen(QPen(QColor(255, 255, 255, 80), 1, Qt.DashLine))
            w3 = crop_px.width() / 3.0
            h3 = crop_px.height() / 3.0
            painter.drawLine(QPointF(crop_px.left() + w3, crop_px.top()), QPointF(crop_px.left() + w3, crop_px.bottom()))
            painter.drawLine(QPointF(crop_px.left() + 2*w3, crop_px.top()), QPointF(crop_px.left() + 2*w3, crop_px.bottom()))
            painter.drawLine(QPointF(crop_px.left(), crop_px.top() + h3), QPointF(crop_px.right(), crop_px.top() + h3))
            painter.drawLine(QPointF(crop_px.left(), crop_px.top() + 2*h3), QPointF(crop_px.right(), crop_px.top() + 2*h3))

            # 8 Handles (Photoshop style grips)
            handles_pts = {
                'tl': QPointF(crop_px.left(), crop_px.top()),
                'tc': QPointF(crop_px.center().x(), crop_px.top()),
                'tr': QPointF(crop_px.right(), crop_px.top()),
                'mr': QPointF(crop_px.right(), crop_px.center().y()),
                'br': QPointF(crop_px.right(), crop_px.bottom()),
                'bc': QPointF(crop_px.center().x(), crop_px.bottom()),
                'bl': QPointF(crop_px.left(), crop_px.bottom()),
                'ml': QPointF(crop_px.left(), crop_px.center().y()),
            }
            hw = 5.0  # Half handle width
            for h_id, pt in handles_pts.items():
                is_active = (self.hovered_crop_handle == h_id or self.dragging_crop_handle == h_id)
                fill_col = QColor(56, 189, 248) if is_active else QColor(255, 255, 255)
                border_col = QColor(15, 23, 42)
                painter.setPen(QPen(border_col, 1.5))
                painter.setBrush(fill_col)
                painter.drawRect(QRectF(pt.x() - hw, pt.y() - hw, hw * 2, hw * 2))

            # Dimension info badge at bottom-right or top-center
            nx, ny, nw, nh = self.crop_rect_norm
            info_txt = f"{int(nw * 100)}% x {int(nh * 100)}%"
            if self.current_qimage:
                px_w = int(nw * self.current_qimage.width())
                px_h = int(nh * self.current_qimage.height())
                aspect_ratio = (px_w / max(1, px_h))
                info_txt = f"{px_w} x {px_h} px ({aspect_ratio:.2f}:1)"
            
            badge_rect = QRectF(crop_px.left() + 8, crop_px.bottom() - 24, 150, 18)
            painter.fillRect(badge_rect, QColor(15, 23, 42, 220))
            painter.setPen(QColor(226, 232, 240))
            painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
            painter.drawText(badge_rect, Qt.AlignCenter, info_txt)

        # 5. 16:9 Tactical Warp Overlay & 6-Point Grab Handles on Panorama
        if self.view_mode == "16:9_tactical" and self.workflow_domain == "autocam":
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
        elif self.show_pitch_roi and self.view_mode in ("32:9", "8:3", "21:10") and self.workflow_domain == "autocam":
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
        if self.pick_center_mode and self.workflow_domain == "stitching":
            banner_rect = QRectF(0, 0, self.width(), 32)
            painter.fillRect(banner_rect, QColor(41, 128, 185, 230))
            painter.setPen(QColor(255, 255, 255))
            painter.setFont(QFont("Segoe UI", 10, QFont.Bold))
            painter.drawText(banner_rect, Qt.AlignCenter, "📍 Klicken Sie auf die Spielfeld-Mittellinie / Anstoßkreis im Bild zum sofortigen Zentrieren")

        # 7. 2D Tactical Radar Minimap Overlay (FIFA Standard Pitch)
        if self.show_radar and self.current_frame_bgr is not None and self.workflow_domain == "autocam":
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

        # 8. Wide-Angle 4-Point Pitch Trapezoid Overlay & Loupe
        if self.workflow_domain == "wide_angle" and self.view_mode in ("wide_angle", "wide_angle_raw") and (self.wide_angle_edit_mode or self.wide_angle_step >= 0):
            self._render_wide_angle_overlay(painter, target_rect)

        # 9. Autocam 10-Point Field Marking Overlay & Mini-Pitch Radar
        if self.workflow_domain == "field_marking":
            self._render_field_marking_overlay(painter, target_rect)

        # 10. Autocam 2-Box Visual Framing Overlay (Widest vs Tightest 16:9 boxes)
        if self.workflow_domain == "framing":
            self._render_framing_overlay(painter, target_rect)

        # HUD in top-left
        painter.setPen(QColor(220, 220, 220, 220))
        painter.setFont(QFont("Segoe UI", 9, QFont.Bold))
        is_8x3 = (self.view_mode == "8:3") or (self.current_qimage and abs(self.current_qimage.width() / max(1, self.current_qimage.height()) - (8.0 / 3.0)) < 0.25)
        if self.view_mode == "8:3":
            mode_str = "🌟 8:3 Taktik-Panorama"
        elif self.view_mode == "grid":
            mode_str = "🔲 4-Kanal Multiview Grid (1, 2, 3, 4)"
            hud_text = f"{mode_str} | Klick auf Quadrant [1-4] zum Vergrößern | Doppelklick: Zurück zum Grid"
        elif self.view_mode == "cam1":
            mode_str = "📹 Cam 1 Links (Live-Feed)"
            hud_text = f"{mode_str} ({iw}x{ih}) | Doppelklick: Zurück zum 4er-Grid"
        elif self.view_mode == "cam2":
            mode_str = "📹 Cam 2 Rechts (Live-Feed)"
            hud_text = f"{mode_str} ({iw}x{ih}) | Doppelklick: Zurück zum 4er-Grid"
        elif self.view_mode == "wide_angle":
            mode_str = "📐 Weitwinkel Begradigt (DJI Action 4 Trapez)"
            hud_text = f"{mode_str} ({iw}x{ih}) | 4-Punkt Kalibrierung aktiv"
        elif self.view_mode == "wide_angle_raw":
            mode_str = "📷 Weitwinkel Original (4-Punkt Ecken)"
            hud_text = f"{mode_str} ({iw}x{ih}) | Klicken Sie auf die Eckfahnen"
        elif self.view_mode == "32:9":
            mode_str = "🌟 8:3 Taktik-Panorama" if is_8x3 else "🌟 32:9 Taktik-Panorama"
            hud_text = f"{mode_str} ({iw}x{ih}) | 🎬 TV-Zoom: {self.dynamic_zoom:.2f}x | Viewport: {int(self.zoom_level*100)}%"
        elif self.view_mode == "21:10":
            mode_str = "📐 21:10 Gestaucht"
            hud_text = f"{mode_str} ({iw}x{ih}) | 🎬 TV-Zoom: {self.dynamic_zoom:.2f}x | Viewport: {int(self.zoom_level*100)}%"
        elif self.view_mode == "16:9_tactical":
            mode_str = "📐 16:9 Taktik-Warp (Live-Frame)"
            hud_text = f"{mode_str} ({iw}x{ih}) | 🎬 TV-Zoom: {self.dynamic_zoom:.2f}x | Viewport: {int(self.zoom_level*100)}%"
        else:
            mode_str = "🎥 16:9 Auto-Broadcast"
            hud_text = f"{mode_str} ({iw}x{ih}) | 🎬 TV-Zoom: {self.dynamic_zoom:.2f}x | Viewport: {int(self.zoom_level*100)}%"
        painter.drawText(14, 24 if (not self.pick_center_mode and not self.corner_pin_mode and not self.pitch_roi_mode and self.view_mode != "16:9_tactical") else 48, hud_text)

    def _render_wide_angle_overlay(self, painter: QPainter, target_rect: QRectF):
        """Renders the 4-corner pitch calibration polygon, handles, and precision zoom loupe."""
        if target_rect is None:
            return

        pts = self._get_wa_corner_screen_points(target_rect)
        if len(pts) == 4:
            qpoly = QPolygonF(pts)
            # Translucent pitch fill
            painter.setBrush(QColor(56, 189, 248, 25))
            painter.setPen(Qt.NoPen)
            painter.drawPolygon(qpoly)

            # Dashed pitch border
            painter.setPen(QPen(QColor(56, 189, 248, 230), 2, Qt.DashLine))
            painter.setBrush(Qt.NoBrush)
            painter.drawPolygon(qpoly)

            # Halfway line (Mittellinie) between top and bottom sidelines
            pt_tc = QPointF((pts[0].x() + pts[1].x()) * 0.5, (pts[0].y() + pts[1].y()) * 0.5)
            pt_bc = QPointF((pts[3].x() + pts[2].x()) * 0.5, (pts[3].y() + pts[2].y()) * 0.5)
            painter.setPen(QPen(QColor(241, 196, 15, 220), 2, Qt.DashDotLine))
            painter.drawLine(pt_tc, pt_bc)

            # Center circle
            mid_x = (pt_tc.x() + pt_bc.x()) * 0.5
            mid_y = (pt_tc.y() + pt_bc.y()) * 0.5
            painter.setPen(QPen(QColor(241, 196, 15, 170), 1.5, Qt.DotLine))
            painter.drawEllipse(QPointF(mid_x, mid_y), 24, 24)

            # Center badge
            badge_w, badge_h = 190.0, 20.0
            badge_rect = QRectF(mid_x - badge_w * 0.5, mid_y - badge_h * 0.5, badge_w, badge_h)
            painter.fillRect(badge_rect, QColor(0, 0, 0, 200))
            painter.setPen(QColor(56, 189, 248))
            painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
            painter.drawText(badge_rect, Qt.AlignCenter, "📐 Weitwinkel-Trapez (4m Stativ)")

            # Corner configuration: [TL, TR, BR, BL]
            corner_configs = [
                ("🚩 Eckfahne TL (Hinten Links)", QColor(241, 196, 15)),   # Gold/Yellow
                ("🚩 Eckfahne TR (Hinten Rechts)", QColor(56, 189, 248)),  # Cyan
                ("🚩 Eckfahne BR (Vorne Rechts)", QColor(239, 68, 68)),    # Red/Coral
                ("🚩 Eckfahne BL (Vorne Links)", QColor(34, 197, 94)),     # Green
            ]

            for idx, pt in enumerate(pts):
                is_active = (self.hovered_wa_corner == idx or self.dragging_wa_corner == idx or self.wide_angle_step == idx)
                label, col = corner_configs[idx]
                radius = 14 if is_active else 10

                if is_active:
                    painter.setPen(Qt.NoPen)
                    glow_col = QColor(col.red(), col.green(), col.blue(), 100)
                    painter.setBrush(glow_col)
                    painter.drawEllipse(pt, 26, 26)

                painter.setPen(QPen(QColor(255, 255, 255), 2))
                painter.setBrush(col)
                painter.drawEllipse(pt, radius, radius)

                # Crosshair inside dot
                painter.setPen(QPen(QColor(0, 0, 0, 230), 1.5))
                painter.drawLine(QPointF(pt.x() - 5, pt.y()), QPointF(pt.x() + 5, pt.y()))
                painter.drawLine(QPointF(pt.x(), pt.y() - 5), QPointF(pt.x(), pt.y() + 5))

                # Corner Label Badge
                bx = pt.x() + (14 if idx in (0, 3) else -180)
                by = pt.y() + (16 if idx in (0, 1) else -26)
                cbadge = QRectF(bx, by, 168, 20)
                painter.fillRect(cbadge, QColor(0, 0, 0, 215))
                painter.setPen(col)
                painter.setFont(QFont("Segoe UI", 7, QFont.Bold))
                nx, ny = self.wide_angle_corners[idx]
                painter.drawText(cbadge, Qt.AlignCenter, f"{label} ({int(nx*100)}%, {int(ny*100)}%)")

            # Magnifier / Loupe (Zoom Callout)
            active_idx = self.dragging_wa_corner if self.dragging_wa_corner is not None else self.hovered_wa_corner
            if active_idx is not None and self.current_qimage is not None and 0 <= active_idx < 4:
                pt_scr = pts[active_idx]
                nx, ny = self.wide_angle_corners[active_idx]
                iw = self.current_qimage.width()
                ih = self.current_qimage.height()
                src_cx = int(nx * iw)
                src_cy = int(ny * ih)

                # Loupe window geometry
                loupe_diam = 110.0
                lx = pt_scr.x() + (40.0 if pt_scr.x() < self.width() - 160 else -140.0)
                ly = pt_scr.y() - (70.0 if pt_scr.y() > 140 else -40.0)
                loupe_rect = QRectF(lx, ly, loupe_diam, loupe_diam)

                # Crop around source center
                crop_size = 48
                crop_x = max(0, min(iw - crop_size, src_cx - crop_size // 2))
                crop_y = max(0, min(ih - crop_size, src_cy - crop_size // 2))
                cropped_img = self.current_qimage.copy(crop_x, crop_y, crop_size, crop_size)

                painter.save()
                path = QPainterPath()
                path.addEllipse(loupe_rect)
                painter.setClipPath(path)
                painter.drawImage(loupe_rect, cropped_img)
                painter.restore()

                # Loupe border & crosshair
                painter.setPen(QPen(corner_configs[active_idx][1], 2.5))
                painter.setBrush(Qt.NoBrush)
                painter.drawEllipse(loupe_rect)

                center_loupe = loupe_rect.center()
                painter.setPen(QPen(QColor(255, 255, 255, 230), 1.5))
                painter.drawLine(QPointF(center_loupe.x() - 14, center_loupe.y()), QPointF(center_loupe.x() + 14, center_loupe.y()))
                painter.drawLine(QPointF(center_loupe.x(), center_loupe.y() - 14), QPointF(center_loupe.x(), center_loupe.y() + 14))

        # Top sequential step banner
        if self.wide_angle_step >= 0:
            step_texts = [
                "🎯 Klick 1/4: Klicken Sie auf die Eckfahne OBEN LINKS (TL - Hinten Links)",
                "🎯 Klick 2/4: Klicken Sie auf die Eckfahne OBEN RECHTS (TR - Hinten Rechts)",
                "🎯 Klick 3/4: Klicken Sie auf die Eckfahne UNTEN RECHTS (BR - Vorne Rechts)",
                "🎯 Klick 4/4: Klicken Sie auf die Eckfahne UNTEN LINKS (BL - Vorne Links)",
            ]
            banner_txt = step_texts[self.wide_angle_step] if self.wide_angle_step < len(step_texts) else "Markierung abgeschlossen!"
            banner_rect = QRectF(0, 0, self.width(), 36)
            painter.fillRect(banner_rect, QColor(234, 88, 12, 245))  # Orange
            painter.setPen(QColor(255, 255, 255))
            painter.setFont(QFont("Segoe UI", 10, QFont.Bold))
            painter.drawText(banner_rect, Qt.AlignCenter, banner_txt)

    def _render_field_marking_overlay(self, painter: QPainter, target_rect: QRectF):
        """Renders the 10-point sideline field boundary curve and top-left mini-pitch radar."""
        if target_rect is None or not hasattr(self, 'field_marking'):
            return

        tw = target_rect.width()
        th = target_rect.height()
        tl_x = target_rect.left()
        tl_y = target_rect.top()

        pts = self.field_marking.points
        screen_pts = [QPointF(tl_x + p[0] * tw, tl_y + p[1] * th) for p in pts]

        # 1. Closed Field Polygon
        near_curve = self.field_marking.get_spline_curve([0, 1, 2, 3, 4], num_samples=40)
        far_curve = self.field_marking.get_spline_curve([5, 6, 7, 8, 9], num_samples=40)

        poly_path = QPainterPath()
        p0 = near_curve[0]
        poly_path.moveTo(tl_x + p0[0] * tw, tl_y + p0[1] * th)
        for p in near_curve[1:]:
            poly_path.lineTo(tl_x + p[0] * tw, tl_y + p[1] * th)
        for p in far_curve:
            poly_path.lineTo(tl_x + p[0] * tw, tl_y + p[1] * th)
        poly_path.closeSubpath()

        # Fill field interior with transparent emerald tint
        painter.fillPath(poly_path, QColor(16, 185, 129, 30))

        # Glowing perimeter stroke
        pen_glow = QPen(QColor(16, 185, 129, 90), 6, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        painter.strokePath(poly_path, pen_glow)
        pen_outline = QPen(QColor(16, 185, 129, 230), 2.5, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        painter.strokePath(poly_path, pen_outline)

        # 2. Draw 10 Draggable Marker Discs
        for idx, pt in enumerate(screen_pts):
            is_active = (self.hovered_field_point == idx or self.dragging_field_point == idx)
            rad = 12 if is_active else 8

            if is_active:
                painter.setPen(Qt.NoPen)
                painter.setBrush(QColor(16, 185, 129, 90))
                painter.drawEllipse(pt, 24, 24)

            painter.setPen(QPen(QColor(255, 255, 255), 2))
            painter.setBrush(QColor(16, 185, 129) if not is_active else QColor(5, 150, 105))
            painter.drawEllipse(pt, rad, rad)

            num_str = str(idx + 1)
            painter.setPen(QColor(0, 0, 0) if not is_active else QColor(255, 255, 255))
            painter.setFont(QFont("Segoe UI", 7, QFont.Bold))
            painter.drawText(QRectF(pt.x() - rad, pt.y() - rad, rad * 2, rad * 2), Qt.AlignCenter, num_str)

        # 3. Mini-Pitch Schematic Radar in Top-Left (matching Autocam)
        radar_w, radar_h = 160, 95
        pad = 12
        rx = target_rect.left() + pad
        ry = target_rect.top() + pad
        r_box = QRectF(rx, ry, radar_w, radar_h)

        painter.fillRect(r_box, QColor(12, 12, 20, 225))
        painter.setPen(QPen(QColor(30, 30, 45), 1.5))
        painter.drawRoundedRect(r_box, 8, 8)

        m_pad = 10
        p_rect = QRectF(rx + m_pad, ry + m_pad, radar_w - m_pad * 2, radar_h - m_pad * 2)
        painter.setPen(QPen(QColor(148, 163, 184, 180), 1.2))
        painter.drawRect(p_rect)

        mx = p_rect.center().x()
        painter.drawLine(QPointF(mx, p_rect.top()), QPointF(mx, p_rect.bottom()))
        painter.drawEllipse(p_rect.center(), 10, 10)

        dot_fractions = [0.0, 0.25, 0.50, 0.75, 1.0]
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(16, 185, 129))
        for frac in dot_fractions:
            dx = p_rect.left() + frac * p_rect.width()
            painter.drawEllipse(QPointF(dx, p_rect.bottom()), 3, 3)
            painter.drawEllipse(QPointF(dx, p_rect.top()), 3, 3)

    def _render_framing_overlay(self, painter: QPainter, target_rect: QRectF):
        """Renders the TWO 16:9 virtual camera boxes (Widest green vs Tightest dashed white)."""
        if target_rect is None:
            return

        tw = target_rect.width()
        th = target_rect.height()
        tl_x = target_rect.left()
        tl_y = target_rect.top()

        cx = getattr(self, 'framing_center_x', 0.5)
        cy = getattr(self, 'framing_center_y', 0.5)
        widest_z = getattr(self, 'framing_widest_zoom', 1.2)
        tightest_z = getattr(self, 'framing_tightest_zoom', 2.0)

        pano_aspect = tw / max(1.0, th)
        target_aspect = 16.0 / 9.0

        # Widest box
        h_wide_norm = 1.0 / max(0.5, widest_z)
        w_wide_norm = h_wide_norm * (target_aspect / pano_aspect)
        x_wide = tl_x + (cx - w_wide_norm * 0.5) * tw
        y_wide = tl_y + (cy - h_wide_norm * 0.5) * th
        rect_wide = QRectF(x_wide, y_wide, w_wide_norm * tw, h_wide_norm * th)

        # Tightest box
        h_tight_norm = 1.0 / max(0.5, tightest_z)
        w_tight_norm = h_tight_norm * (target_aspect / pano_aspect)
        x_tight = tl_x + (cx - w_tight_norm * 0.5) * tw
        y_tight = tl_y + (cy - h_tight_norm * 0.5) * th
        rect_tight = QRectF(x_tight, y_tight, w_tight_norm * tw, h_tight_norm * th)

        # 1. Draw Widest Box (Solid Green)
        pen_wide = QPen(QColor(16, 185, 129, 240), 2.5, Qt.SolidLine)
        painter.setPen(pen_wide)
        painter.drawRect(rect_wide)

        badge_w = QRectF(rect_wide.left() + 8, rect_wide.top() - 12, 130, 20)
        painter.fillRect(badge_w, QColor(16, 185, 129, 230))
        painter.setPen(QColor(255, 255, 255))
        painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
        painter.drawText(badge_w, Qt.AlignCenter, f"Widest - {widest_z:.1f}x")

        # 2. Draw Tightest Box (Dashed White)
        pen_tight = QPen(QColor(255, 255, 255, 240), 2.5, Qt.DashLine)
        painter.setPen(pen_tight)
        painter.drawRect(rect_tight)

        badge_t = QRectF(rect_tight.left() + 8, rect_tight.top() - 12, 130, 20)
        painter.fillRect(badge_t, QColor(255, 255, 255, 230))
        painter.setPen(QColor(0, 0, 0))
        painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
        painter.drawText(badge_t, Qt.AlignCenter, f"Tightest - {tightest_z:.1f}x")

    def wheelEvent(self, event: QWheelEvent):
        delta = event.angleDelta().y()
        if delta > 0:
            self.zoom_level = min(5.0, self.zoom_level * 1.15)
        else:
            self.zoom_level = max(0.8, self.zoom_level / 1.15)
        self.update()

    def mousePressEvent(self, event: QMouseEvent):
        target_rect = self._get_target_rect()

        # Autocam Field Marking Click & Drag
        if self.workflow_domain == "field_marking" and target_rect is not None and event.button() == Qt.LeftButton:
            pos = event.pos()
            tw = target_rect.width()
            th = target_rect.height()
            tl_x = target_rect.left()
            tl_y = target_rect.top()
            for idx, p in enumerate(self.field_marking.points):
                sx = tl_x + p[0] * tw
                sy = tl_y + p[1] * th
                if math.hypot(pos.x() - sx, pos.y() - sy) <= 18.0:
                    self.field_marking.save_state()
                    self.dragging_field_point = idx
                    self.update()
                    return

        # Autocam Framing Click & Drag
        if self.workflow_domain == "framing" and target_rect is not None and event.button() == Qt.LeftButton:
            self.dragging_framing_box = "center"
            self.framing_drag_start_pos = event.pos()
            return

        # WA-1. Wide-Angle Sequential 4-Click Mode
        if self.workflow_domain == "wide_angle" and self.wide_angle_step >= 0 and target_rect is not None and event.button() == Qt.LeftButton:
            pos = event.pos()
            if target_rect.contains(pos):
                nx = float((pos.x() - target_rect.left()) / max(1.0, target_rect.width()))
                ny = float((pos.y() - target_rect.top()) / max(1.0, target_rect.height()))
                self.wide_angle_corners[self.wide_angle_step] = [
                    float(np.clip(nx, 0.0, 1.0)),
                    float(np.clip(ny, 0.0, 1.0))
                ]
                self.wide_angle_step += 1
                if self.wide_angle_step >= 4:
                    self.wide_angle_step = -1
                    self.setCursor(Qt.ArrowCursor)
                    self.wide_angle_step_changed.emit(-1)
                else:
                    self.wide_angle_step_changed.emit(self.wide_angle_step)
                self.wide_angle_corners_changed.emit(self.wide_angle_corners)
                self.update()
                return

        # WA-2. Wide-Angle Corner Dragging
        if self.workflow_domain == "wide_angle" and (self.wide_angle_edit_mode or self.view_mode in ("wide_angle", "wide_angle_raw")) and target_rect is not None and event.button() == Qt.LeftButton:
            hit = self._hit_test_wa_corner(event.pos(), target_rect)
            if hit is not None:
                self.dragging_wa_corner = hit
                self.update()
                return

        # 0. Grid Quadrant Click (Slot 1, 2, 3, 4 Zoom-In)
        if self.workflow_domain == "livestream" and self.view_mode == "grid" and target_rect is not None and event.button() == Qt.LeftButton:
            pos = event.pos()
            if target_rect.contains(pos):
                rel_x = (pos.x() - target_rect.left()) / max(1.0, target_rect.width())
                rel_y = (pos.y() - target_rect.top()) / max(1.0, target_rect.height())
                quad = 1 if (rel_x < 0.5 and rel_y < 0.5) else (2 if rel_x >= 0.5 and rel_y < 0.5 else (3 if rel_x < 0.5 else 4))
                self.grid_quadrant_clicked.emit(quad)
                return

        # 1. Pitch ROI / Tactical 16:9 Corner Dragging
        if self.workflow_domain == "autocam" and (self.pitch_roi_mode or self.view_mode == "16:9_tactical") and target_rect is not None and event.button() == Qt.LeftButton:
            hit = self._hit_test_pitch_roi(event.pos(), target_rect)
            if hit is not None:
                self.dragging_pitch_corner = hit
                self.update()
                return

        # 2a. Crop Gizmo Press
        if self.workflow_domain == "stitching" and self.crop_mode and target_rect is not None and event.button() == Qt.LeftButton:
            handle = self._hit_test_crop_handle(event.pos(), target_rect)
            if handle is not None:
                self.dragging_crop_handle = handle
                self.crop_drag_start_pos = event.pos()
                self.crop_drag_start_rect = list(self.crop_rect_norm)
                self.update()
                return

        # 2. Corner Pin Dragging
        if self.workflow_domain == "stitching" and self.corner_pin_mode and target_rect is not None and event.button() == Qt.LeftButton:
            hit = self._hit_test_corners(event.pos(), target_rect)
            if hit is not None:
                self.dragging_corner = hit
                self.drag_start_pos = event.pos()
                self.update()
                return

        # 3. Pick Center Mode
        if self.workflow_domain == "stitching" and self.pick_center_mode and target_rect is not None and event.button() == Qt.LeftButton:
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

        # Autocam Field Marking Active Dragging
        if self.dragging_field_point is not None and target_rect is not None:
            nx = (event.pos().x() - target_rect.left()) / max(1.0, target_rect.width())
            ny = (event.pos().y() - target_rect.top()) / max(1.0, target_rect.height())
            self.field_marking.set_point(self.dragging_field_point, nx, ny)
            self.field_marking_changed.emit(self.field_marking)
            self.update()
            return

        # Autocam Framing Active Dragging
        if self.dragging_framing_box is not None and target_rect is not None and self.framing_drag_start_pos is not None:
            dx = (event.pos().x() - self.framing_drag_start_pos.x()) / max(1.0, target_rect.width())
            dy = (event.pos().y() - self.framing_drag_start_pos.y()) / max(1.0, target_rect.height())
            self.framing_center_x = float(np.clip(self.framing_center_x + dx, 0.2, 0.8))
            self.framing_center_y = float(np.clip(self.framing_center_y + dy, 0.2, 0.8))
            self.framing_drag_start_pos = event.pos()
            self.framing_changed.emit(self.framing_widest_zoom, self.framing_tightest_zoom, self.framing_center_x, self.framing_center_y)
            self.update()
            return

        # Autocam Field Marking Hover
        if self.workflow_domain == "field_marking" and target_rect is not None:
            pos = event.pos()
            tw = target_rect.width()
            th = target_rect.height()
            tl_x = target_rect.left()
            tl_y = target_rect.top()
            old_h = self.hovered_field_point
            self.hovered_field_point = None
            for idx, p in enumerate(self.field_marking.points):
                sx = tl_x + p[0] * tw
                sy = tl_y + p[1] * th
                if math.hypot(pos.x() - sx, pos.y() - sy) <= 18.0:
                    self.hovered_field_point = idx
                    self.setCursor(Qt.PointingHandCursor)
                    break
            if self.hovered_field_point is None:
                self.setCursor(Qt.ArrowCursor)
            if old_h != self.hovered_field_point:
                self.update()

        # Crop Gizmo Active Dragging
        if self.dragging_crop_handle is not None and target_rect is not None:
            dx = (event.pos().x() - self.crop_drag_start_pos.x()) / max(1.0, target_rect.width())
            dy = (event.pos().y() - self.crop_drag_start_pos.y()) / max(1.0, target_rect.height())
            sx, sy, sw, sh = self.crop_drag_start_rect

            nx, ny, nw, nh = sx, sy, sw, sh
            handle = self.dragging_crop_handle

            if handle == 'inside':
                nx = float(np.clip(sx + dx, 0.0, 1.0 - sw))
                ny = float(np.clip(sy + dy, 0.0, 1.0 - sh))
            else:
                left = sx
                right = sx + sw
                top = sy
                bottom = sy + sh

                if 'l' in handle:
                    left = float(np.clip(sx + dx, 0.0, right - 0.05))
                if 'r' in handle:
                    right = float(np.clip(sx + sw + dx, left + 0.05, 1.0))
                if 't' in handle:
                    top = float(np.clip(sy + dy, 0.0, bottom - 0.05))
                if 'b' in handle:
                    bottom = float(np.clip(sy + sh + dy, top + 0.05, 1.0))

                nx, ny, nw, nh = left, top, right - left, bottom - top

                if self.crop_aspect_ratio is not None and target_rect.height() > 0:
                    frame_aspect = target_rect.width() / target_rect.height()
                    target_norm_ratio = self.crop_aspect_ratio / frame_aspect
                    if handle in ('tc', 'bc'):
                        nw = float(np.clip(nh * target_norm_ratio, 0.05, 1.0 - nx))
                    else:
                        nh = float(np.clip(nw / target_norm_ratio, 0.05, 1.0 - ny))

            self.crop_rect_norm = [nx, ny, nw, nh]
            self.crop_rect_changed.emit(nx, ny, nw, nh)
            self.update()
            return

        # WA-1. Active Wide-Angle Corner Dragging
        if self.dragging_wa_corner is not None and target_rect is not None:
            norm_x = float((event.pos().x() - target_rect.left()) / max(target_rect.width(), 1.0))
            norm_y = float((event.pos().y() - target_rect.top()) / max(target_rect.height(), 1.0))
            idx = int(self.dragging_wa_corner)
            norm_x = float(np.clip(norm_x, 0.0, 1.0))
            norm_y = float(np.clip(norm_y, 0.0, 1.0))
            self.wide_angle_corners[idx] = [norm_x, norm_y]
            self.wide_angle_corners_changed.emit(self.wide_angle_corners)
            self.update()
            return

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

        # WA-Hover Detection
        if self.workflow_domain == "wide_angle" and (self.wide_angle_edit_mode or self.view_mode in ("wide_angle", "wide_angle_raw")) and target_rect is not None and self.dragging_wa_corner is None:
            hit = self._hit_test_wa_corner(event.pos(), target_rect)
            if hit != self.hovered_wa_corner:
                self.hovered_wa_corner = hit
                if hit is not None:
                    self.setCursor(Qt.SizeAllCursor)
                elif self.wide_angle_step >= 0:
                    self.setCursor(Qt.CrossCursor)
                else:
                    self.setCursor(Qt.ArrowCursor)
                self.update()
                return

        # 5. Pitch ROI / Tactical 16:9 Corner Hover Detection
        if self.workflow_domain == "autocam" and (self.pitch_roi_mode or self.view_mode == "16:9_tactical") and target_rect is not None and self.dragging_pitch_corner is None:
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
        if self.workflow_domain == "stitching" and self.corner_pin_mode and target_rect is not None:
            hit = self._hit_test_corners(event.pos(), target_rect)
            if hit != self.hovered_corner:
                self.hovered_corner = hit
                if hit is not None:
                    self.setCursor(Qt.SizeAllCursor)
                else:
                    self.setCursor(Qt.ArrowCursor)
                self.update()

        # 7. Crop Gizmo Hover
        if self.workflow_domain == "stitching" and self.crop_mode and target_rect is not None and self.dragging_crop_handle is None:
            handle = self._hit_test_crop_handle(event.pos(), target_rect)
            if handle != self.hovered_crop_handle:
                self.hovered_crop_handle = handle
                self.update()

            if handle in ('tl', 'br'):
                self.setCursor(Qt.SizeFDiagCursor)
            elif handle in ('tr', 'bl'):
                self.setCursor(Qt.SizeBDiagCursor)
            elif handle in ('tc', 'bc'):
                self.setCursor(Qt.SizeVerCursor)
            elif handle in ('ml', 'mr'):
                self.setCursor(Qt.SizeHorCursor)
            elif handle == 'inside':
                self.setCursor(Qt.SizeAllCursor)
            else:
                self.setCursor(Qt.ArrowCursor)

    def mouseReleaseEvent(self, event: QMouseEvent):
        if self.dragging_field_point is not None:
            self.dragging_field_point = None
            self.update()

        if self.dragging_framing_box is not None:
            self.dragging_framing_box = None
            self.framing_drag_start_pos = None
            self.update()

        if self.dragging_crop_handle is not None:
            self.dragging_crop_handle = None
            self.crop_drag_start_pos = None
            self.crop_drag_start_rect = None
            self.update()

        if self.dragging_wa_corner is not None:
            self.dragging_wa_corner = None
            self.wide_angle_corners_changed.emit(self.wide_angle_corners)
            self.update()

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
        if self.hovered_crop_handle is not None:
            self.hovered_crop_handle = None
            self.setCursor(Qt.ArrowCursor)
            self.update()
        if self.hovered_wa_corner is not None:
            self.hovered_wa_corner = None
            self.setCursor(Qt.ArrowCursor)
            self.update()
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
        if self.view_mode != "grid":
            self.grid_quadrant_clicked.emit(0)
            return
        self.reset_zoom()




