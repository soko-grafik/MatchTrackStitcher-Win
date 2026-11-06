"""
Interactive Tactics & Scouting Dashboard Dialog (Samsung One UI Design).
Displays real-time match analytics, 2D heatmaps, pitch control dominance,
and exports PDF, PNG infographics, and JSON reports.
"""
from typing import Optional, Dict, Any
import os
import cv2
import numpy as np

from PySide6.QtCore import Qt, QRectF, QPointF
from PySide6.QtGui import QPainter, QColor, QFont, QPen, QImage, QPainterPath, QIcon
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTabWidget, QWidget, QFileDialog, QMessageBox, QTableWidget,
    QTableWidgetItem, QHeaderView, QFrame
)

from ..tactics_metrics import MatchTacticsSummary, TacticsReportExporter
from ..logger import get_logger

logger = get_logger("gui.tactics")


class PitchHeatmapCanvas(QWidget):
    """
    Renders an antialiased FIFA 105x68m soccer pitch with a chosen tactical heatmap layer.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(420, 270)
        self.setStyleSheet("background-color: #0b130e; border: 1px solid #1e293b; border-radius: 8px;")
        self.current_heatmap: Optional[np.ndarray] = None
        self.layer_type: str = "team_0" # "team_0", "team_1", "pitch_control", "ball"

    def set_layer(self, heatmap: Optional[np.ndarray], layer_type: str):
        self.current_heatmap = heatmap
        self.layer_type = layer_type
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)

        vw, vh = self.width(), self.height()
        margin = 16.0
        avail_w = vw - 2 * margin
        avail_h = vh - 2 * margin

        # Maintain 105:68 aspect ratio
        pitch_ratio = 68.0 / 105.0
        if avail_w * pitch_ratio <= avail_h:
            pw = avail_w
            ph = pw * pitch_ratio
        else:
            ph = avail_h
            pw = ph / pitch_ratio

        rx = (vw - pw) * 0.5
        ry = (vh - ph) * 0.5
        pitch_rect = QRectF(rx, ry, pw, ph)

        # 1. Background Turf
        painter.fillRect(self.rect(), QColor("#0a0f1d"))
        painter.fillRect(pitch_rect, QColor("#112417"))

        # 2. Render Heatmap if available
        if self.current_heatmap is not None and self.current_heatmap.size > 0:
            pw_i = int(round(pw))
            ph_i = int(round(ph))
            heat_resized = cv2.resize(self.current_heatmap, (pw_i, ph_i), interpolation=cv2.INTER_LINEAR)

            if self.layer_type == "pitch_control":
                # Continuous color: Blue (near 0.0) -> Neutral (0.5) -> Red (near 1.0)
                # Map 0..1 to BGR
                norm = np.clip(heat_resized, 0.0, 1.0)
                # Team 0 (Blue) = (1.0 - norm), Team 1 (Red) = norm
                b_chan = np.clip((1.0 - norm) * 220 + 20, 0, 255).astype(np.uint8)
                r_chan = np.clip(norm * 220 + 20, 0, 255).astype(np.uint8)
                g_chan = np.full_like(b_chan, 30)
                heat_bgr = cv2.merge([b_chan, g_chan, r_chan])
                heat_rgb = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)
                qimg = QImage(heat_rgb.data, pw_i, ph_i, heat_rgb.strides[0], QImage.Format_RGB888)
                painter.setOpacity(0.55)
                painter.drawImage(pitch_rect, qimg)
                painter.setOpacity(1.0)
            else:
                heat_norm = np.clip(heat_resized * 255.0, 0, 255).astype(np.uint8)
                heat_color = cv2.applyColorMap(heat_norm, cv2.COLORMAP_JET)
                heat_rgb = cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB)
                qimg = QImage(heat_rgb.data, pw_i, ph_i, heat_rgb.strides[0], QImage.Format_RGB888)
                painter.setOpacity(0.65)
                painter.drawImage(pitch_rect, qimg)
                painter.setOpacity(1.0)

        # 3. Vector FIFA Pitch Lines
        pen_lines = QPen(QColor(220, 235, 225, 210), 1.5)
        painter.setPen(pen_lines)
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(pitch_rect)

        mid_x = rx + pw * 0.5
        mid_y = ry + ph * 0.5
        painter.drawLine(QPointF(mid_x, ry), QPointF(mid_x, ry + ph))

        # Center circle & spot
        cc_r = (9.15 / 105.0) * pw
        painter.drawEllipse(QPointF(mid_x, mid_y), cc_r, cc_r)
        painter.setBrush(QColor(220, 235, 225, 230))
        painter.drawEllipse(QPointF(mid_x, mid_y), 2.5, 2.5)
        painter.setBrush(Qt.NoBrush)

        # Penalty boxes
        pb_w = (16.5 / 105.0) * pw
        pb_h = (40.32 / 68.0) * ph
        pby = ry + (ph - pb_h) * 0.5
        painter.drawRect(QRectF(rx, pby, pb_w, pb_h))
        painter.drawRect(QRectF(rx + pw - pb_w, pby, pb_w, pb_h))

        # Goal areas
        ga_w = (5.5 / 105.0) * pw
        ga_h = (18.32 / 68.0) * ph
        gay = ry + (ph - ga_h) * 0.5
        painter.drawRect(QRectF(rx, gay, ga_w, ga_h))
        painter.drawRect(QRectF(rx + pw - ga_w, gay, ga_w, ga_h))

        # Penalty spots
        ps_dist = (11.0 / 105.0) * pw
        painter.setBrush(QColor(220, 235, 225, 230))
        painter.drawEllipse(QPointF(rx + ps_dist, mid_y), 2.0, 2.0)
        painter.drawEllipse(QPointF(rx + pw - ps_dist, mid_y), 2.0, 2.0)

        # Subtle pitch dimension labels
        painter.setPen(QColor(140, 160, 150, 180))
        painter.setFont(QFont("Segoe UI", 8))
        painter.drawText(QRectF(rx + 8, ry + 4, 120, 16), Qt.AlignLeft, "← 105 m →")


class TacticsDashboardDialog(QDialog):
    """
    Modern Full-Featured Tactics & Scouting Dashboard Modal Dialog.
    """
    def __init__(self, summary: MatchTacticsSummary, parent=None):
        super().__init__(parent)
        self.summary = summary
        self.setWindowTitle("MatchTrack AI — Taktik- & Scouting-Dashboard")
        self.resize(1080, 750)
        self.setMinimumSize(850, 600)
        self.setStyleSheet("""
            QDialog { background-color: #0f172a; color: #f8fafc; font-family: 'Segoe UI', system-ui; }
            QTabWidget::pane { border: 1px solid #1e293b; background-color: #0f172a; border-radius: 8px; }
            QTabBar::tab { background-color: #1e293b; color: #94a3b8; font-weight: 600; padding: 8px 18px; margin-right: 4px; border-top-left-radius: 6px; border-top-right-radius: 6px; }
            QTabBar::tab:selected { background-color: #3b82f6; color: #ffffff; }
            QTableWidget { background-color: #1e293b; color: #f8fafc; gridline-color: #334155; border: 1px solid #334155; border-radius: 6px; }
            QHeaderView::section { background-color: #0f172a; color: #94a3b8; font-weight: 600; padding: 6px; border: none; }
        """)

        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(18, 18, 18, 18)

        # 1. Header Bar
        header_layout = QHBoxLayout()
        lbl_title = QLabel("⚽ Taktik- & Leistungsbericht (FIFA-Standard)")
        lbl_title.setStyleSheet("font-size: 18px; font-weight: 700; color: #38bdf8;")
        header_layout.addWidget(lbl_title)

        header_layout.addStretch()

        dur_m = int(summary.duration_sec // 60)
        dur_s = int(summary.duration_sec % 60)
        lbl_time = QLabel(f"⏱️ Spieldauer: {dur_m:02d}:{dur_s:02d} ({summary.analyzed_frames} Frames)")
        lbl_time.setStyleSheet("font-size: 12px; font-weight: 600; color: #94a3b8; background-color: #1e293b; padding: 4px 12px; border-radius: 6px;")
        header_layout.addWidget(lbl_time)
        layout.addLayout(header_layout)

        # 2. KPI Cards Row
        kpi_layout = QHBoxLayout()
        kpi_layout.setSpacing(10)

        c1 = self._create_kpi_card("Ballbesitz", f"{summary.team_possession_pct[0]}% : {summary.team_possession_pct[1]}%", "Team Blau vs. Team Rot")
        c2 = self._create_kpi_card("Gesamtdistanz", f"{summary.team_distance_km[0]} : {summary.team_distance_km[1]} km", "Gelaufene Distanz")
        c3 = self._create_kpi_card("Sprints (>25 km/h)", f"{summary.team_sprints[0]} : {summary.team_sprints[1]}", "Intensitäts-Läufe")
        c4 = self._create_kpi_card("Pitch Control", f"{summary.team_pitch_control_pct[0]}% : {summary.team_pitch_control_pct[1]}%", "Territoriale Dominanz")

        kpi_layout.addWidget(c1)
        kpi_layout.addWidget(c2)
        kpi_layout.addWidget(c3)
        kpi_layout.addWidget(c4)
        layout.addLayout(kpi_layout)

        # 3. Main Center Splitter: Left = Heatmap Tabs, Right = Leaderboard & Zones
        body_layout = QHBoxLayout()
        body_layout.setSpacing(12)

        # Left: Interactive Pitch Canvas with Tabs
        self.tab_pitch = QTabWidget()
        self.canvas_blue = PitchHeatmapCanvas()
        self.canvas_blue.set_layer(summary.heatmaps.get("team_0"), "team_0")
        self.tab_pitch.addTab(self.canvas_blue, "🔵 Team Blau Heatmap")

        self.canvas_red = PitchHeatmapCanvas()
        self.canvas_red.set_layer(summary.heatmaps.get("team_1"), "team_1")
        self.tab_pitch.addTab(self.canvas_red, "🔴 Team Rot Heatmap")

        self.canvas_control = PitchHeatmapCanvas()
        self.canvas_control.set_layer(summary.heatmaps.get("pitch_control"), "pitch_control")
        self.tab_pitch.addTab(self.canvas_control, "🌐 Pitch Control (Dominanz)")

        self.canvas_ball = PitchHeatmapCanvas()
        self.canvas_ball.set_layer(summary.heatmaps.get("ball"), "ball")
        self.tab_pitch.addTab(self.canvas_ball, "⚽ Ball-Heatmap")

        body_layout.addWidget(self.tab_pitch, stretch=3)

        # Right: Stats & Top Sprinters
        side_panel = QVBoxLayout()
        side_panel.setSpacing(10)

        # Zone Distribution Box
        box_zones = QFrame()
        box_zones.setStyleSheet("background-color: #1e293b; border-radius: 8px; padding: 10px;")
        layout_bz = QVBoxLayout(box_zones)
        lbl_zt = QLabel("📊 Zonenverteilung (Drittel)")
        lbl_zt.setStyleSheet("font-size: 13px; font-weight: 700; color: #f8fafc;")
        layout_bz.addWidget(lbl_zt)

        z0 = summary.team_zone_distribution[0]
        z1 = summary.team_zone_distribution[1]
        lbl_zb = QLabel(f"Team Blau: {z0['defensive_third']}% Abw. | {z0['middle_third']}% Mid | {z0['attacking_third']}% Angr.")
        lbl_zb.setStyleSheet("color: #38bdf8; font-weight: 600; font-size: 11px;")
        lbl_zr = QLabel(f"Team Rot:  {z1['defensive_third']}% Abw. | {z1['middle_third']}% Mid | {z1['attacking_third']}% Angr.")
        lbl_zr.setStyleSheet("color: #ef4444; font-weight: 600; font-size: 11px;")
        layout_bz.addWidget(lbl_zb)
        layout_bz.addWidget(lbl_zr)
        side_panel.addWidget(box_zones)

        # Sprint Leaderboard Table
        lbl_lead = QLabel("⚡ Top-Sprinter & High-Speed")
        lbl_lead.setStyleSheet("font-size: 13px; font-weight: 700; color: #f8fafc; margin-top: 4px;")
        side_panel.addWidget(lbl_lead)

        self.table_sprinters = QTableWidget()
        self.table_sprinters.setColumnCount(4)
        self.table_sprinters.setHorizontalHeaderLabels(["ID", "Team", "Speed", "Sprints"])
        self.table_sprinters.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table_sprinters.verticalHeader().setVisible(False)
        self.table_sprinters.setRowCount(len(summary.top_sprinters))

        for row, p in enumerate(summary.top_sprinters):
            it_id = QTableWidgetItem(f"#{p['id']}")
            it_team = QTableWidgetItem(p['team'])
            it_spd = QTableWidgetItem(f"{p['max_speed_kmh']} km/h")
            it_sp = QTableWidgetItem(str(p['sprints']))
            
            it_id.setTextAlignment(Qt.AlignCenter)
            it_team.setTextAlignment(Qt.AlignCenter)
            it_spd.setTextAlignment(Qt.AlignCenter)
            it_sp.setTextAlignment(Qt.AlignCenter)

            if p['team'] == "Team Blau":
                it_team.setForeground(QColor("#38bdf8"))
            else:
                it_team.setForeground(QColor("#ef4444"))

            self.table_sprinters.setItem(row, 0, it_id)
            self.table_sprinters.setItem(row, 1, it_team)
            self.table_sprinters.setItem(row, 2, it_spd)
            self.table_sprinters.setItem(row, 3, it_sp)

        side_panel.addWidget(self.table_sprinters)
        body_layout.addLayout(side_panel, stretch=2)
        layout.addLayout(body_layout, stretch=1)

        # 4. Action Buttons Footer
        footer_layout = QHBoxLayout()
        footer_layout.setSpacing(10)

        btn_pdf = QPushButton("📄 PDF-Report exportieren (Druckansicht)")
        btn_pdf.setStyleSheet("background-color: #2563eb; color: #ffffff; font-weight: 700; padding: 9px 18px; border-radius: 8px;")
        btn_pdf.clicked.connect(self.on_export_pdf)
        footer_layout.addWidget(btn_pdf)

        btn_png = QPushButton("🖼️ Infografik speichern (PNG)")
        btn_png.setStyleSheet("background-color: #1e293b; color: #f8fafc; font-weight: 600; padding: 9px 18px; border: 1px solid #334155; border-radius: 8px;")
        btn_png.clicked.connect(self.on_export_png)
        footer_layout.addWidget(btn_png)

        btn_json = QPushButton("💾 JSON-Daten exportieren")
        btn_json.setStyleSheet("background-color: #1e293b; color: #f8fafc; font-weight: 600; padding: 9px 18px; border: 1px solid #334155; border-radius: 8px;")
        btn_json.clicked.connect(self.on_export_json)
        footer_layout.addWidget(btn_json)

        footer_layout.addStretch()

        btn_close = QPushButton("Schließen")
        btn_close.setStyleSheet("background-color: #334155; color: #f8fafc; padding: 9px 22px; border-radius: 8px;")
        btn_close.clicked.connect(self.accept)
        footer_layout.addWidget(btn_close)

        layout.addLayout(footer_layout)

    def _create_kpi_card(self, title: str, value: str, sub: str) -> QFrame:
        card = QFrame()
        card.setStyleSheet("background-color: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 10px;")
        lay = QVBoxLayout(card)
        lay.setSpacing(2)
        lbl_t = QLabel(title)
        lbl_t.setStyleSheet("font-size: 11px; font-weight: 600; color: #94a3b8;")
        lbl_v = QLabel(value)
        lbl_v.setStyleSheet("font-size: 18px; font-weight: 700; color: #ffffff;")
        lbl_s = QLabel(sub)
        lbl_s.setStyleSheet("font-size: 9px; color: #64748b;")
        lay.addWidget(lbl_t)
        lay.addWidget(lbl_v)
        lay.addWidget(lbl_s)
        return card

    def on_export_pdf(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Taktik-PDF exportieren", "matchtrack_taktik_report.pdf", "PDF Dokument (*.pdf)")
        if file_path:
            ok = TacticsReportExporter.export_pdf(self.summary, file_path)
            if ok:
                QMessageBox.information(self, "PDF Exportiert", f"Der Taktik- und Scouting-Bericht wurde erfolgreich exportiert als:\n{file_path}")
            else:
                QMessageBox.critical(self, "Fehler", "PDF konnte nicht exportiert werden.")

    def on_export_png(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Taktik-Infografik speichern", "matchtrack_taktik_infografik.png", "PNG Bild (*.png)")
        if file_path:
            ok = TacticsReportExporter.export_infographic_png(self.summary, file_path)
            if ok:
                QMessageBox.information(self, "Bild Gespeichert", f"Die Taktik-Infografik wurde erfolgreich gespeichert als:\n{file_path}")
            else:
                QMessageBox.critical(self, "Fehler", "Infografik konnte nicht gespeichert werden.")

    def on_export_json(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Taktik-Daten speichern", "matchtrack_taktik_daten.json", "JSON (*.json)")
        if file_path:
            ok = TacticsReportExporter.export_json(self.summary, file_path)
            if ok:
                QMessageBox.information(self, "JSON Gespeichert", f"Taktik-Daten erfolgreich gespeichert als:\n{file_path}")
            else:
                QMessageBox.critical(self, "Fehler", "JSON konnte nicht gespeichert werden.")
