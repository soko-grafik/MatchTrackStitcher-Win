"""
Wide-Angle Optical Dewarping, 4-Point Pitch Trapezoid Calibration Tab for MatchTrack-Stitcher.
"""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QGroupBox, QComboBox,
    QPushButton, QRadioButton, QCheckBox, QFrame, QScrollArea, QStackedWidget
)
from ..widgets import NoWheelSpinBox, LabeledSliderSpinBox
from .export_subpage import create_export_upload_subpage


def build_wide_angle_tab(win) -> QWidget:
    """
    Constructs Tab 2: Wide-Angle Optical Dewarping, 4-Point Trapezoid, and dedicated Export/Upload.
    """
    win.tab_wide_angle = QWidget()
    layout_tab_wa = QVBoxLayout(win.tab_wide_angle)
    layout_tab_wa.setContentsMargins(4, 4, 4, 4)
    layout_tab_wa.setSpacing(6)

    # Sub-Navigation Toggle: [ 📐 Begradigung & Kalibrierung ]  [ 🚀 Export & Upload ]
    sub_nav_wa = QFrame()
    sub_nav_wa.setStyleSheet("QFrame { background: #0c0c14; border: 1px solid #1e293b; border-radius: 8px; padding: 2px; }")
    sub_layout_wa = QHBoxLayout(sub_nav_wa)
    sub_layout_wa.setContentsMargins(2, 2, 2, 2)
    sub_layout_wa.setSpacing(4)

    win.btn_wa_sub_calib = QPushButton("📐 Begradigung & Kalibrierung")
    win.btn_wa_sub_export = QPushButton("🚀 Export & Upload")

    win.btn_wa_sub_calib.clicked.connect(lambda: win.set_wide_angle_subpage(0))
    win.btn_wa_sub_export.clicked.connect(lambda: win.set_wide_angle_subpage(1))

    sub_layout_wa.addWidget(win.btn_wa_sub_calib)
    sub_layout_wa.addWidget(win.btn_wa_sub_export)
    layout_tab_wa.addWidget(sub_nav_wa)

    win.wa_sub_stack = QStackedWidget()

    scroll_wa = QScrollArea()
    scroll_wa.setWidgetResizable(True)
    widget_wa = QWidget()
    wa_layout = QVBoxLayout(widget_wa)
    wa_layout.setSpacing(10)
    wa_layout.setContentsMargins(8, 8, 8, 8)

    # 1. Header Card
    grp_hdr = QGroupBox("📐 WEITWINKEL & SPIELFELD-TRAPEZ")
    grp_hdr.setStyleSheet("QGroupBox { font-weight: bold; color: #38bdf8; border: 1px solid #1e293b; border-radius: 8px; margin-top: 6px; padding-top: 10px; background-color: #0f172a; }")
    hdr_layout = QVBoxLayout(grp_hdr)
    hdr_layout.setSpacing(6)

    lbl_hdr_desc = QLabel(
        "Optische Begradigung extremer Weitwinkel-Objektive (155° Fisheye der DJI Osmo Action 4) "
        "und Entzerrung der Spielfeldlinien zu einem mathematisch perfekten Trapez aus 4m Kamerahöhe."
    )
    lbl_hdr_desc.setStyleSheet("color: #94a3b8; font-size: 11px;")
    lbl_hdr_desc.setWordWrap(True)
    hdr_layout.addWidget(lbl_hdr_desc)
    wa_layout.addWidget(grp_hdr)

    # 2. Source Video & Camera Profile
    grp_source = QGroupBox("📹 Quellvideo & Kamera-Profil")
    source_layout = QVBoxLayout(grp_source)
    source_layout.setSpacing(8)

    win.btn_wa_open_video = QPushButton("📁 155° Ultra-Weitwinkel Video öffnen")
    win.btn_wa_open_video.setStyleSheet("background-color: #0284c7; color: white; font-weight: bold; padding: 9px; font-size: 12px; border-radius: 8px;")
    win.btn_wa_open_video.clicked.connect(lambda: win.open_wide_angle_video())
    source_layout.addWidget(win.btn_wa_open_video)

    win.lbl_wa_video_info = QLabel("Kein Weitwinkel-Video geladen.")
    win.lbl_wa_video_info.setStyleSheet("color: #94a3b8; font-size: 11px; font-style: italic;")
    win.lbl_wa_video_info.setWordWrap(True)
    source_layout.addWidget(win.lbl_wa_video_info)

    source_layout.addWidget(QLabel("Kamera-Profil & Optik-Voreinstellung:"))
    win.combo_wa_preset = QComboBox()
    win.combo_wa_preset.addItem("DJI Osmo Action 4 (155° Ultrawide - 4m Stativ)", "action4_155")
    win.combo_wa_preset.addItem("DJI Osmo Action 3 / 5 Pro (155° Ultrawide)", "action35_155")
    win.combo_wa_preset.addItem("GoPro Hero 11/12 (HyperView / SuperView)", "gopro_hyper")
    win.combo_wa_preset.addItem("Generisches Fisheye (150°-170°)", "generic_fisheye")
    win.combo_wa_preset.addItem("Benutzerdefiniert (Manuelle Parameter)", "custom")
    win.combo_wa_preset.currentIndexChanged.connect(win._on_wa_preset_changed)
    source_layout.addWidget(win.combo_wa_preset)

    wa_layout.addWidget(grp_source)

    # 3. Optical Dewarping Group
    grp_optics = QGroupBox("🔍 1. Optische Begradigung (Fisheye-Entzerrung)")
    optics_layout = QVBoxLayout(grp_optics)
    optics_layout.setSpacing(6)

    win.ctrl_wa_strength = LabeledSliderSpinBox("Fisheye-Begradigung:", 0.0, 1.20, 0.85, 0.01, "", decimals=2)
    win.ctrl_wa_strength.setToolTip("Stärke der optischen Fisheye-Korrektur (0.0 = Aus/Fisheye, 0.85 = DJI Action 4 Ideal)")
    win.ctrl_wa_strength.valueChanged.connect(win._on_wa_param_changed)
    optics_layout.addWidget(win.ctrl_wa_strength)

    win.ctrl_wa_hfov = LabeledSliderSpinBox("Horizontales FOV:", 90.0, 170.0, 138.0, 1.0, "°", decimals=1)
    win.ctrl_wa_hfov.setToolTip("Sichtfeld der Kamera in Grad (DJI Action 4 Ultrawide ca. 138°-145°)")
    win.ctrl_wa_hfov.valueChanged.connect(win._on_wa_param_changed)
    optics_layout.addWidget(win.ctrl_wa_hfov)

    win.ctrl_wa_zoom = LabeledSliderSpinBox("Brennweite / Zoom:", 0.70, 2.00, 1.05, 0.01, "x", decimals=2)
    win.ctrl_wa_zoom.setToolTip("Skaliert das Bild zur Vermeidung schwarzer Ecken nach der Begradigung")
    win.ctrl_wa_zoom.valueChanged.connect(win._on_wa_param_changed)
    optics_layout.addWidget(win.ctrl_wa_zoom)

    win.ctrl_wa_tilt = LabeledSliderSpinBox("Kamera-Neigung (Tilt):", -25.0, 25.0, 0.0, 0.5, "°", decimals=1)
    win.ctrl_wa_tilt.setToolTip("Kompensiert den Neigungswinkel des Stativs nach unten/oben")
    win.ctrl_wa_tilt.valueChanged.connect(win._on_wa_param_changed)
    optics_layout.addWidget(win.ctrl_wa_tilt)

    win.ctrl_wa_roll = LabeledSliderSpinBox("Horizont-Neigung (Roll):", -10.0, 10.0, 0.0, 0.1, "°", decimals=1)
    win.ctrl_wa_roll.setToolTip("Dreht das Bild feinfühlig, um schiefe Stative waagerecht zu stellen")
    win.ctrl_wa_roll.valueChanged.connect(win._on_wa_param_changed)
    optics_layout.addWidget(win.ctrl_wa_roll)

    win.ctrl_wa_v_offset = LabeledSliderSpinBox("Vertikaler Bildversatz:", -0.30, 0.30, 0.0, 0.01, "", decimals=2)
    win.ctrl_wa_v_offset.setToolTip("Verschiebt den Bildmittelpunkt nach oben oder unten")
    win.ctrl_wa_v_offset.valueChanged.connect(win._on_wa_param_changed)
    optics_layout.addWidget(win.ctrl_wa_v_offset)

    wa_layout.addWidget(grp_optics)

    # 4. 4-Point Pitch Trapezoid Calibration Group
    grp_calib = QGroupBox("⚽ 2. 4-Punkt Spielfeld-Kalibrierung (Trapez)")
    calib_layout = QVBoxLayout(grp_calib)
    calib_layout.setSpacing(8)

    row_status = QHBoxLayout()
    win.lbl_wa_calib_status = QLabel("● 4/4 Ecken aktiv")
    win.lbl_wa_calib_status.setStyleSheet("background-color: #064e3b; color: #34d399; font-weight: 700; font-size: 11px; padding: 4px 8px; border-radius: 6px;")
    row_status.addWidget(win.lbl_wa_calib_status)
    row_status.addStretch()
    calib_layout.addLayout(row_status)

    win.btn_wa_mark_corners = QPushButton("🎯 4 Ecken nacheinander anklicken (1-4)")
    win.btn_wa_mark_corners.setStyleSheet("background-color: #f59e0b; color: #1e1e2d; font-weight: 800; padding: 8px; font-size: 11px; border-radius: 8px;")
    win.btn_wa_mark_corners.setToolTip("Klicken Sie nacheinander auf: 1. Eckfahne TL (Gelb) -> 2. TR (Blau) -> 3. BR (Rot) -> 4. BL (Grün)")
    win.btn_wa_mark_corners.clicked.connect(win._on_wa_start_marking)
    calib_layout.addWidget(win.btn_wa_mark_corners)

    row_calib_actions = QHBoxLayout()
    win.btn_wa_reset_corners = QPushButton("↺ Ecken zurücksetzen")
    win.btn_wa_reset_corners.setStyleSheet("background-color: #1a1a28; border: 1px solid #2a2a3e; padding: 6px; font-size: 11px; border-radius: 6px; color: #cbd5e1;")
    win.btn_wa_reset_corners.clicked.connect(win._on_wa_reset_corners)
    row_calib_actions.addWidget(win.btn_wa_reset_corners)

    win.chk_wa_edit_mode = QCheckBox("✏️ Ecken-Ziehen im Bild aktiv")
    win.chk_wa_edit_mode.setChecked(True)
    win.chk_wa_edit_mode.setStyleSheet("font-weight: 600; color: #38bdf8;")
    win.chk_wa_edit_mode.toggled.connect(win._on_wa_edit_toggled)
    row_calib_actions.addWidget(win.chk_wa_edit_mode)
    calib_layout.addLayout(row_calib_actions)

    # Geometry mode
    calib_layout.addWidget(QLabel("Geometrie-Modus der Linien-Entzerrung:"))
    win.radio_geom_trapezoid = QRadioButton("📐 Symmetrisches Trapez (Perspektive aus 4m Stativ)")
    win.radio_geom_trapezoid.setChecked(True)
    win.radio_geom_trapezoid.setStyleSheet("color: #f8fafc; font-weight: 600;")
    win.radio_geom_trapezoid.toggled.connect(win._on_wa_geom_mode_toggled)
    calib_layout.addWidget(win.radio_geom_trapezoid)

    win.radio_geom_planar = QRadioButton("⚽ Planare Aufsicht (FIFA 105 × 68 m Rechteck)")
    win.radio_geom_planar.setStyleSheet("color: #cbd5e1;")
    win.radio_geom_planar.toggled.connect(win._on_wa_geom_mode_toggled)
    calib_layout.addWidget(win.radio_geom_planar)

    win.ctrl_wa_top_ratio = LabeledSliderSpinBox("Trapez-Breite Oben:", 0.40, 1.00, 0.72, 0.01, "", decimals=2)
    win.ctrl_wa_top_ratio.setToolTip("Fluchtverhältnis der hinteren Auslinie gegenüber der vorderen Auslinie (typisch 0.65 - 0.75 bei 4m Höhe)")
    win.ctrl_wa_top_ratio.valueChanged.connect(win._on_wa_param_changed)
    calib_layout.addWidget(win.ctrl_wa_top_ratio)

    win.ctrl_wa_margin = LabeledSliderSpinBox("Spielfeld-Randabstand:", 0.0, 0.30, 0.08, 0.01, "", decimals=2)
    win.ctrl_wa_margin.setToolTip("Sicherheitsabstand um das Feld, damit Trainerbänke und Eckfahnen sichtbar bleiben")
    win.ctrl_wa_margin.valueChanged.connect(win._on_wa_param_changed)
    calib_layout.addWidget(win.ctrl_wa_margin)

    wa_layout.addWidget(grp_calib)

    # 5. Target Aspect Ratio & Framing Group
    grp_target = QGroupBox("📺 3. Zielformat & Framing")
    target_layout = QVBoxLayout(grp_target)
    target_layout.setSpacing(6)

    target_layout.addWidget(QLabel("Ziel-Seitenverhältnis & Bildformat:"))
    win.combo_wa_target_format = QComboBox()
    win.combo_wa_target_format.addItem("📺 16:9 Broadcast Full HD (1920x1080)", (1920, 1080))
    win.combo_wa_target_format.addItem("📺 16:9 Broadcast 4K UHD (3840x2160)", (3840, 2160))
    win.combo_wa_target_format.addItem("🎞️ 21:9 UltraWide Cinema (2560x1080)", (2560, 1080))
    win.combo_wa_target_format.addItem("🎞️ 21:9 UWQHD Cinema (3440x1440)", (3440, 1440))
    win.combo_wa_target_format.addItem("🏟️ 32:9 SuperWide Panorama (3840x1080)", (3840, 1080))
    win.combo_wa_target_format.addItem("📷 4:3 Classic (Vollsensor 2880x2160)", (2880, 2160))
    win.combo_wa_target_format.addItem("⚙️ Benutzerdefiniert (Breite x Höhe)", "custom")
    win.combo_wa_target_format.currentIndexChanged.connect(win._on_wa_target_format_changed)
    target_layout.addWidget(win.combo_wa_target_format)

    win.widget_wa_custom_res = QWidget()
    row_custom_res = QHBoxLayout(win.widget_wa_custom_res)
    row_custom_res.setContentsMargins(0, 0, 0, 0)
    win.spin_wa_custom_w = NoWheelSpinBox()
    win.spin_wa_custom_w.setRange(640, 7680)
    win.spin_wa_custom_w.setValue(1920)
    win.spin_wa_custom_w.setSuffix(" px")
    win.spin_wa_custom_w.valueChanged.connect(win._on_wa_custom_res_changed)
    win.spin_wa_custom_h = NoWheelSpinBox()
    win.spin_wa_custom_h.setRange(360, 4320)
    win.spin_wa_custom_h.setValue(1080)
    win.spin_wa_custom_h.setSuffix(" px")
    win.spin_wa_custom_h.valueChanged.connect(win._on_wa_custom_res_changed)
    row_custom_res.addWidget(QLabel("Breite:"))
    row_custom_res.addWidget(win.spin_wa_custom_w)
    row_custom_res.addWidget(QLabel("Höhe:"))
    row_custom_res.addWidget(win.spin_wa_custom_h)
    win.widget_wa_custom_res.setVisible(False)
    target_layout.addWidget(win.widget_wa_custom_res)

    wa_layout.addWidget(grp_target)

    # 6. Action Buttons
    grp_actions = QGroupBox("🚀 Aktionen & Export")
    actions_layout = QVBoxLayout(grp_actions)
    actions_layout.setSpacing(8)

    win.btn_wa_save_preset = QPushButton("💾 Kalibrierung als Standard sichern")
    win.btn_wa_save_preset.setStyleSheet("background-color: #1e1e2d; border: 1px solid #3b3b54; padding: 8px; font-size: 11px; border-radius: 8px; color: #cbd5e1; font-weight: 600;")
    win.btn_wa_save_preset.clicked.connect(win.save_as_default_settings)
    actions_layout.addWidget(win.btn_wa_save_preset)

    win.btn_wa_export = QPushButton("🚀 Begradigtes Video exportieren")
    win.btn_wa_export.setStyleSheet("background-color: #2563eb; color: white; font-weight: 800; padding: 10px; font-size: 12px; border-radius: 8px;")
    win.btn_wa_export.clicked.connect(win._on_wa_export_clicked)
    actions_layout.addWidget(win.btn_wa_export)

    wa_layout.addWidget(grp_actions)

    wa_layout.addStretch()
    scroll_wa.setWidget(widget_wa)
    win.wa_sub_stack.addWidget(scroll_wa)

    # Subpage 1: Dedicated Weitwinkel Export & Upload
    win.scroll_wa_exp = create_export_upload_subpage(win, "wide_angle")
    win.wa_sub_stack.addWidget(win.scroll_wa_exp)

    layout_tab_wa.addWidget(win.wa_sub_stack, 1)
    win.set_wide_angle_subpage(0)

    return win.tab_wide_angle
