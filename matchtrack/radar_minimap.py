"""
Soccer Radar Minimap: 2D Tactical Bird's-Eye View of the soccer pitch.
Renders real-time player positions, soccer ball, and the 16:9 Follow-Cam viewport on a FIFA-regulation pitch.
Supports both OpenCV (for video exports) and Qt QPainter (for zero-overhead live viewport display).
"""
import math
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import cv2

try:
    from PySide6.QtGui import QPainter, QColor, QPen, QBrush, QFont, QPainterPath
    from PySide6.QtCore import QRectF, QPointF, Qt
    HAS_QT = True
except Exception:
    HAS_QT = False


class SoccerRadarRenderer:
    """
    Renders a broadcast-quality 2D Tactical Radar overlay of a soccer pitch (105m x 68m).
    """
    # Standard pitch dimensions in meters
    PITCH_LENGTH = 105.0
    PITCH_WIDTH = 68.0
    CENTER_CIRCLE_RADIUS = 9.15
    PENALTY_BOX_DEPTH = 16.5
    PENALTY_BOX_WIDTH = 40.32
    GOAL_AREA_DEPTH = 5.5
    GOAL_AREA_WIDTH = 18.32
    PENALTY_SPOT_DIST = 11.0

    @staticmethod
    def draw_radar_opencv(
        frame_bgr: np.ndarray,
        players: List[Tuple[float, float, int]],  # (norm_x, norm_y, team_id)
        ball_pos: Optional[Tuple[float, float]] = None,  # (norm_x, norm_y)
        cam_span: Optional[Tuple[float, float]] = None,   # (norm_cam_left, norm_cam_right)
        radar_w: int = 340,
        pos: str = "top_right",
        opacity: float = 0.85,
        margin: int = 16
    ) -> np.ndarray:
        """
        Draws the 2D tactical radar onto a BGR numpy frame with smooth alpha blending.
        """
        fh, fw = frame_bgr.shape[:2]
        radar_h = int(round(radar_w * (SoccerRadarRenderer.PITCH_WIDTH / SoccerRadarRenderer.PITCH_LENGTH)))

        # Determine top-left corner
        if pos == "top_left":
            rx, ry = margin, margin
        elif pos == "bottom_left":
            rx, ry = margin, fh - radar_h - margin
        elif pos == "bottom_right":
            rx, ry = fw - radar_w - margin, fh - radar_h - margin
        elif pos == "bottom_center":
            rx, ry = (fw - radar_w) // 2, fh - radar_h - margin
        else: # "top_right" (Default)
            rx, ry = fw - radar_w - margin, margin

        rx = max(0, min(fw - radar_w, rx))
        ry = max(0, min(fh - radar_h, ry))

        # Create overlay image
        overlay = np.zeros((radar_h, radar_w, 3), dtype=np.uint8)
        # Tactical dark grass: BGR(18, 38, 22)
        overlay[:] = (18, 38, 22)

        # Field geometry scaling
        sx = radar_w / SoccerRadarRenderer.PITCH_LENGTH
        sy = radar_h / SoccerRadarRenderer.PITCH_WIDTH

        line_col = (210, 225, 215) # Light sage / white
        line_thick = 1

        # 1. Pitch Outer boundary
        cv2.rectangle(overlay, (2, 2), (radar_w - 3, radar_h - 3), line_col, line_thick)

        # 2. Halfway line
        mid_x = radar_w // 2
        cv2.line(overlay, (mid_x, 2), (mid_x, radar_h - 3), line_col, line_thick)

        # 3. Center circle
        radius_px = int(round(SoccerRadarRenderer.CENTER_CIRCLE_RADIUS * sx))
        mid_y = radar_h // 2
        cv2.circle(overlay, (mid_x, mid_y), radius_px, line_col, line_thick)
        cv2.circle(overlay, (mid_x, mid_y), 2, line_col, -1)

        # 4. Left Penalty Box & Goal Area
        p_box_w = int(round(SoccerRadarRenderer.PENALTY_BOX_DEPTH * sx))
        p_box_h = int(round(SoccerRadarRenderer.PENALTY_BOX_WIDTH * sy))
        p_box_y = (radar_h - p_box_h) // 2
        cv2.rectangle(overlay, (2, p_box_y), (2 + p_box_w, p_box_y + p_box_h), line_col, line_thick)

        g_box_w = int(round(SoccerRadarRenderer.GOAL_AREA_DEPTH * sx))
        g_box_h = int(round(SoccerRadarRenderer.GOAL_AREA_WIDTH * sy))
        g_box_y = (radar_h - g_box_h) // 2
        cv2.rectangle(overlay, (2, g_box_y), (2 + g_box_w, g_box_y + g_box_h), line_col, line_thick)

        # Left Penalty Spot
        pen_spot_x = int(round(SoccerRadarRenderer.PENALTY_SPOT_DIST * sx))
        cv2.circle(overlay, (pen_spot_x, mid_y), 2, line_col, -1)

        # 5. Right Penalty Box & Goal Area
        cv2.rectangle(overlay, (radar_w - 3 - p_box_w, p_box_y), (radar_w - 3, p_box_y + p_box_h), line_col, line_thick)
        cv2.rectangle(overlay, (radar_w - 3 - g_box_w, g_box_y), (radar_w - 3, g_box_y + g_box_h), line_col, line_thick)
        cv2.circle(overlay, (radar_w - 3 - pen_spot_x, mid_y), 2, line_col, -1)

        # 6. Current 16:9 Camera FOV Range (Highlighted Zone)
        if cam_span is not None:
            c_left, c_right = cam_span
            cx1 = int(round(np.clip(c_left, 0.0, 1.0) * radar_w))
            cx2 = int(round(np.clip(c_right, 0.0, 1.0) * radar_w))
            cw = max(4, cx2 - cx1)
            # Tinted highlight for camera sector
            fov_roi = overlay[:, cx1:cx1+cw]
            fov_tint = np.full_like(fov_roi, (40, 80, 50))
            overlay[:, cx1:cx1+cw] = cv2.addWeighted(fov_roi, 0.65, fov_tint, 0.35, 0)
            cv2.rectangle(overlay, (cx1, 2), (cx1 + cw, radar_h - 3), (241, 196, 15), 1)

        # 7. Render Players
        team_colors = {
            0: (235, 130, 50),   # Team A: Cyan / Blue (BGR)
            1: (50, 70, 235),    # Team B: Red / Coral (BGR)
            2: (20, 220, 220),   # Referee: Yellow / Neutral
            -1: (20, 220, 220),  # Referee: Yellow / Neutral
        }

        for p in players:
            norm_x, norm_y = p[0], p[1]
            team_id = p[2] if len(p) > 2 else 0
            px = int(round(np.clip(norm_x, 0.0, 1.0) * (radar_w - 6) + 3))
            py = int(round(np.clip(norm_y, 0.0, 1.0) * (radar_h - 6) + 3))
            col = team_colors.get(team_id, (235, 130, 50))
            # Filled dot with dark border
            cv2.circle(overlay, (px, py), 4, (10, 10, 10), -1, cv2.LINE_AA)
            cv2.circle(overlay, (px, py), 3, col, -1, cv2.LINE_AA)

        # 8. Render Ball
        if ball_pos is not None:
            bx = int(round(np.clip(ball_pos[0], 0.0, 1.0) * (radar_w - 6) + 3))
            by = int(round(np.clip(ball_pos[1], 0.0, 1.0) * (radar_h - 6) + 3))
            # Glowing ball: outer white ring, bright yellow center
            cv2.circle(overlay, (bx, by), 5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.circle(overlay, (bx, by), 4, (0, 245, 255), -1, cv2.LINE_AA)

        # Alpha composite into target frame
        target_roi = frame_bgr[ry:ry+radar_h, rx:rx+radar_w]
        blended = cv2.addWeighted(overlay, opacity, target_roi, 1.0 - opacity, 0)
        # Add subtle dark border around the radar
        cv2.rectangle(blended, (0, 0), (radar_w - 1, radar_h - 1), (60, 60, 70), 1)
        frame_bgr[ry:ry+radar_h, rx:rx+radar_w] = blended
        return frame_bgr

    @staticmethod
    def paint_radar_qt(
        painter: "QPainter",
        widget_w: int,
        widget_h: int,
        players: List[Tuple[float, float, int]],
        ball_norm: Optional[Tuple[float, float]] = None,
        cam_span: Optional[Tuple[float, float]] = None,
        radar_w: int = 320,
        pos: str = "top_right",
        opacity: float = 0.88,
        margin: int = 16
    ):
        """
        Paints high-DPI antialiased 2D tactical radar directly into a Qt QPainter.
        Zero overhead, perfect vector scaling, and instant rendering.
        """
        if not HAS_QT:
            return

        radar_h = int(round(radar_w * (SoccerRadarRenderer.PITCH_WIDTH / SoccerRadarRenderer.PITCH_LENGTH)))

        if pos == "top_left":
            rx, ry = margin, margin
        elif pos == "bottom_left":
            rx, ry = margin, widget_h - radar_h - margin
        elif pos == "bottom_right":
            rx, ry = widget_w - radar_w - margin, widget_h - radar_h - margin
        elif pos == "bottom_center":
            rx, ry = (widget_w - radar_w) // 2, widget_h - radar_h - margin
        else: # "top_right"
            rx, ry = widget_w - radar_w - margin, margin

        painter.save()
        painter.setRenderHint(QPainter.Antialiasing, True)

        rect = QRectF(rx, ry, radar_w, radar_h)

        # 1. Background turf (Translucent dark stadium green)
        path_bg = QPainterPath()
        path_bg.addRoundedRect(rect, 8.0, 8.0)
        bg_alpha = int(round(255 * opacity))
        painter.fillPath(path_bg, QColor(14, 28, 18, bg_alpha))
        painter.setPen(QPen(QColor(40, 60, 48, bg_alpha), 1.5))
        painter.drawPath(path_bg)

        # Scale factors (meters to radar pixels inside margin)
        pad = 6.0
        pitch_rect = QRectF(rx + pad, ry + pad, radar_w - 2 * pad, radar_h - 2 * pad)
        pw = pitch_rect.width()
        ph = pitch_rect.height()

        pen_lines = QPen(QColor(220, 235, 225, int(180 * opacity)), 1.2)
        painter.setPen(pen_lines)
        painter.setBrush(Qt.NoBrush)

        # Outer pitch boundary
        painter.drawRect(pitch_rect)

        # Halfway line
        mid_x = pitch_rect.left() + pw * 0.5
        mid_y = pitch_rect.top() + ph * 0.5
        painter.drawLine(QPointF(mid_x, pitch_rect.top()), QPointF(mid_x, pitch_rect.bottom()))

        # Center circle & spot
        cc_r = (SoccerRadarRenderer.CENTER_CIRCLE_RADIUS / SoccerRadarRenderer.PITCH_LENGTH) * pw
        painter.drawEllipse(QPointF(mid_x, mid_y), cc_r, cc_r)
        painter.setBrush(QColor(220, 235, 225, int(200 * opacity)))
        painter.drawEllipse(QPointF(mid_x, mid_y), 2.0, 2.0)
        painter.setBrush(Qt.NoBrush)

        # Penalty boxes
        pb_w = (SoccerRadarRenderer.PENALTY_BOX_DEPTH / SoccerRadarRenderer.PITCH_LENGTH) * pw
        pb_h = (SoccerRadarRenderer.PENALTY_BOX_WIDTH / SoccerRadarRenderer.PITCH_WIDTH) * ph
        pb_y = pitch_rect.top() + (ph - pb_h) * 0.5

        # Left penalty box
        painter.drawRect(QRectF(pitch_rect.left(), pb_y, pb_w, pb_h))
        # Left goal area
        ga_w = (SoccerRadarRenderer.GOAL_AREA_DEPTH / SoccerRadarRenderer.PITCH_LENGTH) * pw
        ga_h = (SoccerRadarRenderer.GOAL_AREA_WIDTH / SoccerRadarRenderer.PITCH_WIDTH) * ph
        ga_y = pitch_rect.top() + (ph - ga_h) * 0.5
        painter.drawRect(QRectF(pitch_rect.left(), ga_y, ga_w, ga_h))

        # Right penalty box & goal area
        painter.drawRect(QRectF(pitch_rect.right() - pb_w, pb_y, pb_w, pb_h))
        painter.drawRect(QRectF(pitch_rect.right() - ga_w, ga_y, ga_w, ga_h))

        # Penalty spots
        ps_dist = (SoccerRadarRenderer.PENALTY_SPOT_DIST / SoccerRadarRenderer.PITCH_LENGTH) * pw
        painter.setBrush(QColor(220, 235, 225, int(200 * opacity)))
        painter.drawEllipse(QPointF(pitch_rect.left() + ps_dist, mid_y), 1.5, 1.5)
        painter.drawEllipse(QPointF(pitch_rect.right() - ps_dist, mid_y), 1.5, 1.5)

        # 2. 16:9 Camera FOV sector highlight
        if cam_span is not None:
            c_left, c_right = cam_span
            cx1 = pitch_rect.left() + max(0.0, min(1.0, c_left)) * pw
            cx2 = pitch_rect.left() + max(0.0, min(1.0, c_right)) * pw
            fov_w = max(4.0, cx2 - cx1)
            fov_rect = QRectF(cx1, pitch_rect.top(), fov_w, ph)
            painter.fillRect(fov_rect, QColor(241, 196, 15, int(35 * opacity)))
            painter.setPen(QPen(QColor(241, 196, 15, int(220 * opacity)), 1.5))
            painter.drawRect(fov_rect)

        # 3. Players
        pen_dot_border = QPen(QColor(15, 15, 20, 220), 1.2)
        painter.setPen(pen_dot_border)

        for p in players:
            nx, ny = p[0], p[1]
            team_id = p[2] if len(p) > 2 else 0
            px = pitch_rect.left() + max(0.0, min(1.0, nx)) * pw
            py = pitch_rect.top() + max(0.0, min(1.0, ny)) * ph

            if team_id == 1:
                painter.setBrush(QColor(239, 68, 68))   # Team B: Red
            elif team_id in (-1, 2):
                painter.setBrush(QColor(234, 179, 8))   # Ref: Yellow
            else:
                painter.setBrush(QColor(59, 130, 246))  # Team A: Blue

            painter.drawEllipse(QPointF(px, py), 4.0, 4.0)

        # 4. Ball
        if ball_norm is not None:
            bx = pitch_rect.left() + max(0.0, min(1.0, ball_norm[0])) * pw
            by = pitch_rect.top() + max(0.0, min(1.0, ball_norm[1])) * ph
            # Outer glow
            painter.setPen(QPen(QColor(255, 255, 255, 230), 1.5))
            painter.setBrush(QColor(255, 220, 0)) # Bright yellow ball
            painter.drawEllipse(QPointF(bx, by), 4.5, 4.5)

        # Badge tag: "🗺️ 2D Taktik-Radar"
        painter.setPen(QColor(160, 185, 170, int(190 * opacity)))
        painter.setFont(QFont("Segoe UI", 7, QFont.Bold))
        painter.drawText(QRectF(rx + 8, ry + 2, 120, 12), Qt.AlignLeft, "🗺️ Taktik-Radar")

        painter.restore()
