# matchtrack/gui/tabs/stitching_tab.py
"""
Tab 1: Stitching & Calibration Tab Builder.
Quellformat (4:3 vs 16:9), Video-Auswahl, Audio-Sync, Nahtstellen- & Farbkalibrierung,
3D-Gehäuse Rig-Parameter, Ecken-Verzerrung (Corner Pinning) und Profile.
"""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QGroupBox,
    QRadioButton,
    QLabel,
    QFrame,
    QPushButton,
    QStackedWidget,
    QScrollArea,
    QToolButton,
    QComboBox,
    QCheckBox,
    QTabWidget,
)

from ..widgets import (
    NoWheelSlider,
    NoWheelSpinBox,
    NoWheelDoubleSpinBox,
    NoWheelComboBox,
    LabeledSliderSpinBox,
)
from ...rig_geometry import CAMERA_PRESETS


def build_stitching_tab(win) -> QWidget:
    """Builds and attaches Tab 1 (Stitching, Quellformat, Sync & Kalibrierung) to win (MainWindow)."""
    win.tab_stitching = QWidget()
    win.tab_media = win.tab_stitching  # alias for backwards compatibility
    layout_tab_stitch = QVBoxLayout(win.tab_stitching)
    layout_tab_stitch.setContentsMargins(4, 4, 4, 4)
    layout_tab_stitch.setSpacing(6)

    # ---------------------------------------------------------------------
    # Card: Kamera-Quellformat & Stitching-Profil (4:3 vs 16:9)
    # ---------------------------------------------------------------------
    grp_src_format = QGroupBox("📷 Kamera-Quellformat & Stitching-Profil")
    grp_src_format.setStyleSheet(
        "QGroupBox { font-weight: bold; color: #38bdf8; border: 1px solid #282839; border-radius: 10px; margin-top: 6px; padding: 10px; background-color: #12121c; }"
    )
    layout_sf = QVBoxLayout(grp_src_format)
    layout_sf.setSpacing(8)

    row_sf = QHBoxLayout()
    win.radio_src_4_3 = QRadioButton("4:3 Format (DJI Action 4 RockSteady ON)")
    win.radio_src_4_3.setStyleSheet("font-weight: 700; color: #f4f4f6; font-size: 11px;")
    win.radio_src_4_3.setChecked(True)
    win.radio_src_4_3.toggled.connect(win._on_source_format_radio_changed)

    win.radio_src_16_9 = QRadioButton("16:9 Format (Standard Webcams / 16:9 Kameras)")
    win.radio_src_16_9.setStyleSheet("font-weight: 700; color: #f4f4f6; font-size: 11px;")

    row_sf.addWidget(win.radio_src_4_3)
    row_sf.addWidget(win.radio_src_16_9)
    layout_sf.addLayout(row_sf)

    win.lbl_src_format_desc = QLabel("💡 Ausgabeprofil: 8:3 Taktik-Panorama (2560x960) | DJI RockSteady Dewarp aktiv.")
    win.lbl_src_format_desc.setStyleSheet("color: #38bdf8; font-size: 11px; font-weight: 600;")
    layout_sf.addWidget(win.lbl_src_format_desc)

    layout_tab_stitch.addWidget(grp_src_format)

    # Sub-Navigation Toggle: [ 🎥 Videos & Synchronisation ]  [ 🎨 Naht, Winkel & Farbe ]  [ 🚀 Export & Upload ]
    sub_nav_frame = QFrame()
    sub_nav_frame.setStyleSheet("QFrame { background: #0c0c14; border: 1px solid #1e1e2d; border-radius: 8px; padding: 2px; }")
    sub_layout = QHBoxLayout(sub_nav_frame)
    sub_layout.setContentsMargins(2, 2, 2, 2)
    sub_layout.setSpacing(4)

    win.btn_stitch_sub_media = QPushButton("🎥 Videos & Synchronisation")
    win.btn_stitch_sub_calib = QPushButton("🎨 Naht, Winkel & Farbe")
    win.btn_stitch_sub_export = QPushButton("🚀 Export & Upload")

    win.btn_stitch_sub_media.clicked.connect(lambda: win.set_stitching_subpage(0))
    win.btn_stitch_sub_calib.clicked.connect(lambda: win.set_stitching_subpage(1))
    win.btn_stitch_sub_export.clicked.connect(lambda: win.set_stitching_subpage(2))

    sub_layout.addWidget(win.btn_stitch_sub_media)
    sub_layout.addWidget(win.btn_stitch_sub_calib)
    sub_layout.addWidget(win.btn_stitch_sub_export)
    layout_tab_stitch.addWidget(sub_nav_frame)

    win.stitch_sub_stack = QStackedWidget()

    # Sub-Page 0: Media & Sync
    scroll_m = QScrollArea()
    scroll_m.setWidgetResizable(True)
    widget_m = QWidget()
    layout_m = QVBoxLayout(widget_m)
    layout_m.setSpacing(12)

    # Card: Video Inputs
    grp_inputs = QGroupBox("🎥 Kamera-Eingänge (Dual-Cam 80° Rig)")
    layout_in = QVBoxLayout(grp_inputs)
    layout_in.setSpacing(10)

    lbl_desc_inputs = QLabel("Laden Sie die Aufnahmen beider Kameras (Kamera 1 Links und Kamera 2 Rechts):")
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
    win.badge_left = QLabel("⚪ Nicht geladen")
    win.badge_left.setStyleSheet("background: #1c1c28; color: #8e8e9c; border-radius: 6px; padding: 2px 8px; font-size: 11px; font-weight: 600;")
    row_hdr_l.addWidget(lbl_title_l)
    row_hdr_l.addStretch()
    row_hdr_l.addWidget(win.badge_left)
    layout_card_l.addLayout(row_hdr_l)

    win.lbl_file_left = QLabel("Keine Videodatei gewählt")
    win.lbl_file_left.setStyleSheet("color: #8e8e9c; font-size: 11px; padding: 2px 0;")
    win.lbl_file_left.setWordWrap(True)
    layout_card_l.addWidget(win.lbl_file_left)

    win.btn_open_left = QPushButton("📂 Linkes Video wählen...")
    win.btn_open_left.setStyleSheet("background-color: #1a1a26; border: 1px solid #2d2d42; font-weight: 600; padding: 7px; color: #f4f4f6; border-radius: 8px;")
    win.btn_open_left.clicked.connect(win.open_left_video)
    layout_card_l.addWidget(win.btn_open_left)
    layout_in.addWidget(card_l)

    # Camera 2 (Rechts) Card
    card_r = QFrame()
    card_r.setStyleSheet("QFrame { background-color: #12121c; border: 1px solid #232334; border-radius: 10px; padding: 10px; }")
    layout_card_r = QVBoxLayout(card_r)
    layout_card_r.setSpacing(6)
    row_hdr_r = QHBoxLayout()
    lbl_title_r = QLabel("🎥 Kamera 2 (Rechts)")
    lbl_title_r.setStyleSheet("font-weight: 700; font-size: 12px; color: #f4f4f6;")
    win.badge_right = QLabel("⚪ Nicht geladen")
    win.badge_right.setStyleSheet("background: #1c1c28; color: #8e8e9c; border-radius: 6px; padding: 2px 8px; font-size: 11px; font-weight: 600;")
    row_hdr_r.addWidget(lbl_title_r)
    row_hdr_r.addStretch()
    row_hdr_r.addWidget(win.badge_right)
    layout_card_r.addLayout(row_hdr_r)

    win.lbl_file_right = QLabel("Keine Videodatei gewählt")
    win.lbl_file_right.setStyleSheet("color: #8e8e9c; font-size: 11px; padding: 2px 0;")
    win.lbl_file_right.setWordWrap(True)
    layout_card_r.addWidget(win.lbl_file_right)

    win.btn_open_right = QPushButton("📂 Rechtes Video wählen...")
    win.btn_open_right.setStyleSheet("background-color: #1a1a26; border: 1px solid #2d2d42; font-weight: 600; padding: 7px; color: #f4f4f6; border-radius: 8px;")
    win.btn_open_right.clicked.connect(win.open_right_video)
    layout_card_r.addWidget(win.btn_open_right)
    layout_in.addWidget(card_r)

    layout_m.addWidget(grp_inputs)

    # Card: Audio Synchronization & Offset
    grp_sync = QGroupBox("⚡ Audio-Synchronisation & Frame-Offset")
    layout_sync = QVBoxLayout(grp_sync)
    layout_sync.setSpacing(10)

    lbl_sync_info = QLabel("Gleichen Sie den Versatz beider Kameras per Audio-Waveform automatisch ab oder justieren Sie manuell:")
    lbl_sync_info.setStyleSheet("color: #8e8e9c; font-size: 11px;")
    lbl_sync_info.setWordWrap(True)
    layout_sync.addWidget(lbl_sync_info)

    win.btn_auto_sync = QPushButton("⚡ Audio synchronisieren (Auto-Sync)")
    win.btn_auto_sync.setStyleSheet("background-color: #2563eb; font-weight: bold; padding: 9px 14px; color: white; font-size: 12px; border-radius: 9px;")
    win.btn_auto_sync.setToolTip("Automatische Audio-Synchronisation (FFT Cross-Correlation für Frame-Offset)")
    win.btn_auto_sync.clicked.connect(win.run_audio_sync)
    layout_sync.addWidget(win.btn_auto_sync)

    lbl_manual_sync = QLabel("Manuelle Offset-Feinjustage (Kamera Rechts):")
    lbl_manual_sync.setStyleSheet("font-weight: 600; color: #f4f4f6; margin-top: 4px;")
    layout_sync.addWidget(lbl_manual_sync)

    row_offset = QHBoxLayout()
    row_offset.setSpacing(8)

    win.slider_offset = NoWheelSlider(Qt.Horizontal)
    win.slider_offset.setRange(-600, 600)
    win.slider_offset.setValue(0)
    win.slider_offset.valueChanged.connect(win.on_offset_slider_changed)
    row_offset.addWidget(win.slider_offset, 1)

    win.spin_offset = NoWheelSpinBox()
    win.spin_offset.setRange(-5000, 5000)
    win.spin_offset.setValue(0)
    win.spin_offset.setSuffix(" F")
    win.spin_offset.valueChanged.connect(win.on_offset_spin_changed)
    row_offset.addWidget(win.spin_offset)

    btn_offset_reset = QToolButton()
    btn_offset_reset.setText("↺")
    btn_offset_reset.setToolTip("Offset auf 0 zurücksetzen")
    btn_offset_reset.clicked.connect(lambda: win.spin_offset.setValue(0))
    row_offset.addWidget(btn_offset_reset)
    layout_sync.addLayout(row_offset)

    # Quick Frame Step Buttons
    row_steps = QHBoxLayout()
    row_steps.setSpacing(4)
    for delta, txt in [(-5, "-5 F"), (-1, "-1 F"), (0, "0"), (1, "+1 F"), (5, "+5 F")]:
        btn_step = QPushButton(txt)
        btn_step.setStyleSheet("background-color: #1a1a26; border: 1px solid #28283a; border-radius: 6px; padding: 4px 6px; font-size: 11px; font-weight: 600; color: #cbd5e1;")
        if delta == 0:
            btn_step.clicked.connect(lambda chk=False: win.spin_offset.setValue(0))
        else:
            btn_step.clicked.connect(lambda chk=False, d=delta: win.adjust_offset_by(d))
        row_steps.addWidget(btn_step)
    layout_sync.addLayout(row_steps)

    layout_m.addWidget(grp_sync)
    layout_m.addStretch()
    scroll_m.setWidget(widget_m)
    win.stitch_sub_stack.addWidget(scroll_m)

    # -------------------------------------------------------------------------
    # Sub-Page 1: 🎨 Naht, Winkel & Farbe (Kalibrierung & Geometrie)
    # -------------------------------------------------------------------------
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

    win.combo_preset = QComboBox()
    for key, p in CAMERA_PRESETS.items():
        win.combo_preset.addItem(f"{p.name} [{key}]", key)
    win.combo_preset.currentIndexChanged.connect(win.on_preset_changed)
    layout_preset.addWidget(win.combo_preset)

    btn_preset_action4_4x3 = QPushButton("⚡ Quick-Preset: Action 4 (1080p 4:3 RockSteady)")
    btn_preset_action4_4x3.setStyleSheet("background-color: #1e293b; border: 1px solid #38bdf8; font-weight: bold; padding: 7px; color: #38bdf8; border-radius: 8px;")
    btn_preset_action4_4x3.setToolTip("Setzt 1080p 4:3 Dewarp Profil, 82.0° HFOV, ±36.5° Rig-Winkel und Naht-Überlappung mit 1 Klick.")
    btn_preset_action4_4x3.clicked.connect(win.apply_action4_4x3_rocksteady_preset)
    layout_preset.addWidget(btn_preset_action4_4x3)

    win.ctrl_cam_fov = LabeledSliderSpinBox("Objektiv-Blickwinkel (HFOV / Brennweite):", 60.0, 140.0, 92.0, 0.2, "°")
    win.ctrl_cam_fov.valueChanged.connect(win.on_cam_fov_changed)
    layout_preset.addWidget(win.ctrl_cam_fov)

    row_proj = QHBoxLayout()
    lbl_proj = QLabel("Projektions-Typ:")
    lbl_proj.setStyleSheet("font-weight: 600; color: #f4f4f6; font-size: 11px;")
    win.combo_projection = NoWheelComboBox()
    win.combo_projection.addItem("🌟 Zylindrisch (Standard Panorama)", "cylindrical")
    win.combo_projection.addItem("📐 Pannini (Begradigt Linien & Perspektive)", "pannini")
    win.combo_projection.addItem("📏 Rektilinear (100% Gerade)", "rectilinear")
    win.combo_projection.currentIndexChanged.connect(win.on_projection_changed)
    row_proj.addWidget(lbl_proj)
    row_proj.addWidget(win.combo_projection, 1)
    layout_preset.addLayout(row_proj)

    line_c1 = QFrame()
    line_c1.setFrameShape(QFrame.HLine)
    line_c1.setStyleSheet("color: #232332;")
    layout_preset.addWidget(line_c1)

    win.btn_run_ai_calib = QPushButton("🎯 Nahtstelle automatisch kalibrieren (Multi-Zonen & Boden)")
    win.btn_run_ai_calib.setStyleSheet("background-color: #2563eb; font-weight: bold; padding: 9px 12px; font-size: 12px; color: white; border-radius: 9px;")
    win.btn_run_ai_calib.setToolTip("Analysiert Hintergrund und Vordergrund (Rasen/Linien mit CLAHE) getrennt, optimiert Pitch, Roll und Boden-Taper vollautomatisch.")
    win.btn_run_ai_calib.clicked.connect(win.run_auto_stitch_calibration)
    layout_preset.addWidget(win.btn_run_ai_calib)
    win.btn_auto_stitch = win.btn_run_ai_calib

    win.chk_multi_frame_calib = QCheckBox("Multi-Frame Analyse (5 Video-Frames abtasten)")
    win.chk_multi_frame_calib.setChecked(True)
    win.chk_multi_frame_calib.setStyleSheet("color: #cbd5e1; font-size: 11px;")
    layout_preset.addWidget(win.chk_multi_frame_calib)

    scroll_layout.addWidget(grp_preset)

    # 2. Spielfeld- & Horizont-Begradigung Card
    grp_level = QGroupBox("🌅 Spielfeld- & Horizont-Begradigung")
    layout_level = QVBoxLayout(grp_level)
    layout_level.setSpacing(8)

    win.ctrl_curvature = LabeledSliderSpinBox("Spielfeld-Linien Begradigung (Curvature):", -0.60, 0.60, 0.0, 0.01, "", decimals=2)
    win.ctrl_curvature.setToolTip("Begradigt den V-förmigen Durchhang der Spielfeldlinien bei nach unten geneigten Kameras und zieht die Seitenlinien plan.")
    win.ctrl_curvature.valueChanged.connect(win.on_curvature_changed)
    layout_level.addWidget(win.ctrl_curvature)

    win.ctrl_global_pitch = LabeledSliderSpinBox("Horizont-Ausgleich (+15° Begradigung):", 0.0, 45.0, 15.0, 0.2, "°")
    win.ctrl_global_pitch.valueChanged.connect(win.on_rig_param_changed)
    layout_level.addWidget(win.ctrl_global_pitch)

    win.chk_auto_color_match = QCheckBox("🎨 Automatischer Farb- & Belichtungsabgleich (Kamera L ↔ R)")
    win.chk_auto_color_match.setChecked(True)
    win.chk_auto_color_match.setStyleSheet("color: #38bdf8; font-weight: bold; margin-top: 2px;")
    win.chk_auto_color_match.setToolTip("Gleicht Helligkeit, Belichtung und Weißabgleich beider Kameras automatisch an der Nahtstelle ab. Entfernt die sichtbare Kante in der Bildmitte vollständig.")
    win.chk_auto_color_match.toggled.connect(win.on_auto_color_match_toggled)
    layout_level.addWidget(win.chk_auto_color_match)

    win.ctrl_color_match_strength = LabeledSliderSpinBox("Farbangleich-Stärke:", 0.0, 100.0, 100.0, 5.0, "%", decimals=0)
    win.ctrl_color_match_strength.setToolTip("Bestimmt, wie stark die Farben beider Kameras aufeinander abgestimmt werden (100% = vollständiger Abgleich).")
    win.ctrl_color_match_strength.valueChanged.connect(win.on_color_match_strength_changed)
    layout_level.addWidget(win.ctrl_color_match_strength)

    win.chk_auto_crop = QCheckBox("✅ LIR Auto-Crop (Schwarze Ränder/Bögen entfernen)")
    win.chk_auto_crop.setChecked(True)
    win.chk_auto_crop.setStyleSheet("font-weight: 600; color: #10b981;")
    win.chk_auto_crop.toggled.connect(win.toggle_auto_crop)
    layout_level.addWidget(win.chk_auto_crop)

    win.ctrl_safety_margin = LabeledSliderSpinBox("LIR Sicherheitsrand (Puffer):", 0.0, 5.0, 0.5, 0.1, "%", decimals=1)
    win.ctrl_safety_margin.valueChanged.connect(win.on_safety_margin_changed)
    layout_level.addWidget(win.ctrl_safety_margin)

    win.ctrl_blend_width = LabeledSliderSpinBox("Nahtstellen-Übergangsbreite (Blending):", 1.0, 30.0, 8.0, 0.5, "°")
    win.ctrl_blend_width.valueChanged.connect(win.on_rig_param_changed)
    layout_level.addWidget(win.ctrl_blend_width)

    scroll_layout.addWidget(grp_level)

    # 3. 3D-Gehäuse Ausrichtung (Rig) Card
    grp_rig = QGroupBox("📐 3D-Gehäuse Ausrichtung (Spreizung & Neigung)")
    layout_rig = QVBoxLayout(grp_rig)
    layout_rig.setSpacing(8)

    win.chk_sync_yaw = QCheckBox("Winkel symmetrisch koppeln (± X°)")
    win.chk_sync_yaw.setChecked(True)
    win.chk_sync_yaw.toggled.connect(win.toggle_sync_angles)
    layout_rig.addWidget(win.chk_sync_yaw)

    win.ctrl_master_yaw = LabeledSliderSpinBox("Gesamter Spreizwinkel (80° Rig):", 40.0, 120.0, 80.0, 0.2, "°")
    win.ctrl_master_yaw.valueChanged.connect(win.on_master_yaw_changed)
    layout_rig.addWidget(win.ctrl_master_yaw)

    win.ctrl_left_yaw = LabeledSliderSpinBox("Kamera Links Yaw (Gierwinkel):", -80.0, -10.0, -40.0, 0.2, "°")
    win.ctrl_left_yaw.valueChanged.connect(win.on_left_yaw_changed)
    layout_rig.addWidget(win.ctrl_left_yaw)

    win.ctrl_right_yaw = LabeledSliderSpinBox("Kamera Rechts Yaw (Gierwinkel):", 10.0, 80.0, 40.0, 0.2, "°")
    win.ctrl_right_yaw.valueChanged.connect(win.on_right_yaw_changed)
    layout_rig.addWidget(win.ctrl_right_yaw)

    win.ctrl_cam_pitch = LabeledSliderSpinBox("Rig Neigung nach unten (Pitch):", -45.0, 0.0, -15.0, 0.2, "°")
    win.ctrl_cam_pitch.valueChanged.connect(win.on_rig_param_changed)
    layout_rig.addWidget(win.ctrl_cam_pitch)

    win.ctrl_pitch_diff = LabeledSliderSpinBox("Vertikaler Naht-Ausgleich (Pitch Rechts ±):", -10.0, 10.0, 0.0, 0.1, "°")
    win.ctrl_pitch_diff.setToolTip("Gleicht einen Höhenversatz an der Nahtstelle zwischen linker und rechter Kamera pixelgenau aus (hebt oder senkt das rechte Kamerabild).")
    win.ctrl_pitch_diff.valueChanged.connect(win.on_rig_param_changed)
    layout_rig.addWidget(win.ctrl_pitch_diff)

    win.ctrl_left_roll = LabeledSliderSpinBox("Roll Feinjustage Links:", -15.0, 15.0, 0.0, 0.1, "°")
    win.ctrl_left_roll.valueChanged.connect(win.on_rig_param_changed)
    layout_rig.addWidget(win.ctrl_left_roll)

    win.ctrl_right_roll = LabeledSliderSpinBox("Roll Feinjustage Rechts:", -15.0, 15.0, 0.0, 0.1, "°")
    win.ctrl_right_roll.valueChanged.connect(win.on_rig_param_changed)
    layout_rig.addWidget(win.ctrl_right_roll)

    win.ctrl_seam_pos = LabeledSliderSpinBox("Nahtstellen-Position (Links ↔ Rechts Versatz):", -25.0, 25.0, 0.0, 0.2, "°")
    win.ctrl_seam_pos.setToolTip("Verschiebt die Nahtstelle nach links oder rechts, damit sie nicht direkt durch Personen oder Objekte schneidet.")
    win.ctrl_seam_pos.valueChanged.connect(win.on_rig_param_changed)
    layout_rig.addWidget(win.ctrl_seam_pos)

    win.ctrl_bottom_taper_x = LabeledSliderSpinBox("Boden-Nahtstelle Taper X (Links ↔ Rechts ±):", -15.0, 15.0, 0.0, 0.1, "°")
    win.ctrl_bottom_taper_x.setToolTip("Gleicht Parallaxe und Scherung im unteren Bildbereich (Nahfeld/Rasen) horizontal aus, während der obere Bereich (Hintergrund) zu 100% unberührt bleibt.")
    win.ctrl_bottom_taper_x.valueChanged.connect(win.on_rig_param_changed)
    layout_rig.addWidget(win.ctrl_bottom_taper_x)

    win.ctrl_bottom_taper_y = LabeledSliderSpinBox("Boden-Nahtstelle Taper Y (Oben ↕ Unten ±):", -10.0, 10.0, 0.0, 0.1, "°")
    win.ctrl_bottom_taper_y.setToolTip("Gleicht vertikale Höhenunterschiede der Nahtstelle im unteren Bildbereich aus.")
    win.ctrl_bottom_taper_y.valueChanged.connect(win.on_rig_param_changed)
    layout_rig.addWidget(win.ctrl_bottom_taper_y)

    win.chk_smart_seam = QCheckBox("🧠 Intelligente Kontur-Nahtstelle (umgeht Trainer/Personen)")
    win.chk_smart_seam.setChecked(False)
    win.chk_smart_seam.setStyleSheet("font-weight: 600; color: #c084fc; font-size: 11px;")
    win.chk_smart_seam.setToolTip("Sucht per Dynamic Programming den Pfad mit minimalem Farbunterschied durch den Rasen und kurvt automatisch um Personen/Trainer herum, damit niemand in zwei Hälften zerschnitten wird.")
    win.chk_smart_seam.toggled.connect(win.on_smart_seam_toggled)
    layout_rig.addWidget(win.chk_smart_seam)

    scroll_layout.addWidget(grp_rig)

    # 4. 32:9 Panorama Zentrierung & Bildausschnitt Card
    grp_pitch_align = QGroupBox("🏟️ Panorama-Zentrierung & Bildausschnitt")
    layout_pitch = QVBoxLayout(grp_pitch_align)
    layout_pitch.setSpacing(8)

    win.btn_pick_center_calib = QPushButton("🎯 Mittellinie im Bild anklicken (Sofort-Zentrierung)")
    win.btn_pick_center_calib.setCheckable(True)
    win.btn_pick_center_calib.setStyleSheet("background-color: #1e1e2d; border: 1px solid #3b82f6; font-weight: 600; padding: 7px; color: #60a5fa; border-radius: 8px;")
    win.btn_pick_center_calib.setToolTip("Klicken Sie direkt im Panorama-Bild auf den Anstoßpunkt oder die Mittellinie, um das Bild sofort perfekt zu zentrieren.")
    win.btn_pick_center_calib.toggled.connect(win.on_pick_center_toggled)
    layout_pitch.addWidget(win.btn_pick_center_calib)

    win.ctrl_h_offset = LabeledSliderSpinBox("Spielfeld-Mitte Versatz (Links ↔ Rechts Pan):", -60.0, 60.0, 0.0, 0.1, "°", decimals=1)
    win.ctrl_h_offset.valueChanged.connect(win.on_rig_param_changed)
    layout_pitch.addWidget(win.ctrl_h_offset)

    pan_btn_row = QHBoxLayout()
    pan_btn_row.setSpacing(4)
    for d_pan, txt in [(-5.0, "⏪ -5°"), (-1.0, "◀ -1°"), (0.0, "🎯 0°"), (1.0, "▶ +1°"), (5.0, "⏩ +5°")]:
        btn_pan = QPushButton(txt)
        btn_pan.setStyleSheet("background-color: #1a1a26; border: 1px solid #28283a; border-radius: 6px; padding: 4px; font-size: 11px; font-weight: 600; color: #cbd5e1;")
        if d_pan == 0.0:
            btn_pan.clicked.connect(lambda chk=False: win.ctrl_h_offset.setValue(0.0))
        else:
            btn_pan.clicked.connect(lambda chk=False, d=d_pan: win.pan_offset_by(d))
        pan_btn_row.addWidget(btn_pan)
    layout_pitch.addLayout(pan_btn_row)

    win.ctrl_v_offset = LabeledSliderSpinBox("Vertikaler Bildausschnitt (Shift):", -2.0, 2.0, 0.12, 0.02, "", decimals=2)
    win.ctrl_v_offset.valueChanged.connect(win.on_rig_param_changed)
    layout_pitch.addWidget(win.ctrl_v_offset)

    grp_dpad = QGroupBox("🎮 2D Bildausschnitt Feinjustage")
    layout_dpad = QGridLayout(grp_dpad)
    layout_dpad.setSpacing(4)
    btn_dpad_up = QPushButton("⬆️ Oben")
    btn_dpad_up.clicked.connect(lambda: win.shift_framing(0.0, -0.05))
    layout_dpad.addWidget(btn_dpad_up, 0, 1)
    btn_dpad_left = QPushButton("⬅️ Links")
    btn_dpad_left.clicked.connect(lambda: win.shift_framing(-1.5, 0.0))
    layout_dpad.addWidget(btn_dpad_left, 1, 0)
    btn_dpad_center = QPushButton("🎯 Reset")
    btn_dpad_center.clicked.connect(win.reset_framing)
    layout_dpad.addWidget(btn_dpad_center, 1, 1)
    btn_dpad_right = QPushButton("➡️ Rechts")
    btn_dpad_right.clicked.connect(lambda: win.shift_framing(1.5, 0.0))
    layout_dpad.addWidget(btn_dpad_right, 1, 2)
    btn_dpad_down = QPushButton("⬇️ Unten")
    btn_dpad_down.clicked.connect(lambda: win.shift_framing(0.0, 0.05))
    layout_dpad.addWidget(btn_dpad_down, 2, 1)
    layout_pitch.addWidget(grp_dpad)

    win.ctrl_pano_hfov = LabeledSliderSpinBox("32:9 Panorama Weitwinkel (HFOV):", 90.0, 180.0, 145.0, 0.5, "°")
    win.ctrl_pano_hfov.valueChanged.connect(win.on_rig_param_changed)
    layout_pitch.addWidget(win.ctrl_pano_hfov)

    win.ctrl_squeeze = LabeledSliderSpinBox("Gleichmäßige Breiten-Stauchung:", 0.50, 3.00, 1.0, 0.01, "x", decimals=2)
    win.ctrl_squeeze.valueChanged.connect(win.on_rig_param_changed)
    layout_pitch.addWidget(win.ctrl_squeeze)

    row_pitch_presets = QHBoxLayout()
    btn_full_pitch = QPushButton("🏟️ 16:9 Spielfeld (145° Preset)")
    btn_full_pitch.setStyleSheet("background-color: #1e1e2d; border: 1px solid #10b981; font-weight: bold; padding: 7px; color: #34d399; border-radius: 8px;")
    btn_full_pitch.clicked.connect(win.apply_full_pitch_preset)
    row_pitch_presets.addWidget(btn_full_pitch)

    btn_pitch_4x3 = QPushButton("📹 1080p 4:3 RockSteady")
    btn_pitch_4x3.setStyleSheet("background-color: #1e1e2d; border: 1px solid #38bdf8; font-weight: bold; padding: 7px; color: #38bdf8; border-radius: 8px;")
    btn_pitch_4x3.setToolTip("Richtet Rig und Blickfeld passend für 4:3 Dewarp mit RockSteady ein")
    btn_pitch_4x3.clicked.connect(win.apply_action4_4x3_rocksteady_preset)
    row_pitch_presets.addWidget(btn_pitch_4x3)
    layout_pitch.addLayout(row_pitch_presets)

    scroll_layout.addWidget(grp_pitch_align)

    # 5. Ecken-Verzerrung (Corner Pinning) Card
    grp_corner_pin = QGroupBox("📐 Ecken-Verzerrung (Corner Pinning)")
    layout_cp = QVBoxLayout(grp_corner_pin)
    layout_cp.setSpacing(8)

    win.btn_corner_pins_side = QPushButton("✨ Ecken interaktiv im Bild verzerren (Drag & Drop)")
    win.btn_corner_pins_side.setCheckable(True)
    win.btn_corner_pins_side.setStyleSheet("background-color: #2563eb; font-weight: bold; padding: 8px; color: white; border-radius: 8px;")
    win.btn_corner_pins_side.setToolTip("Aktivieren Sie diesen Modus, um die 4 Kamera-Ecken direkt im Panorama per Maus zu ziehen.")
    win.btn_corner_pins_side.toggled.connect(win.on_corner_pins_toggled)
    layout_cp.addWidget(win.btn_corner_pins_side)

    row_cam_sel = QHBoxLayout()
    row_cam_sel.addWidget(QLabel("Anzeige:"))
    win.combo_corner_cam = QComboBox()
    win.combo_corner_cam.addItem("🌟 Beide Kameras (Cyan = Links, Gold = Rechts)", "both")
    win.combo_corner_cam.addItem("🟦 Nur Kamera Links (Cyan)", "left")
    win.combo_corner_cam.addItem("🟧 Nur Kamera Rechts (Gold)", "right")
    win.combo_corner_cam.currentIndexChanged.connect(win.on_corner_cam_changed)
    row_cam_sel.addWidget(win.combo_corner_cam, 1)
    layout_cp.addLayout(row_cam_sel)

    win.ctrl_frame_opacity = LabeledSliderSpinBox("32:9 Rahmen Deckkraft (Abdunklung):", 0.0, 100.0, 50.0, 5.0, "%", decimals=0)
    win.ctrl_frame_opacity.valueChanged.connect(win.on_frame_opacity_changed)
    layout_cp.addWidget(win.ctrl_frame_opacity)

    win.tab_corners = QTabWidget()
    win.tab_corners.setStyleSheet("QTabWidget::pane { border: 1px solid #232332; background: #12121a; border-radius: 8px; } QTabBar::tab { font-size: 11px; padding: 5px 8px; }")

    # TAB LEFT CAMERA
    tab_c_left = QWidget()
    layout_cl = QVBoxLayout(tab_c_left)
    layout_cl.setContentsMargins(6, 6, 6, 6)
    layout_cl.setSpacing(4)

    win.spin_corners_l = []
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
        spin_x.valueChanged.connect(lambda val, s='left', i=idx: win.on_corner_spin_changed(s, i))

        spin_y = NoWheelDoubleSpinBox()
        spin_y.setRange(-80.0, 180.0)
        spin_y.setValue(def_y)
        spin_y.setSingleStep(0.2)
        spin_y.setDecimals(1)
        spin_y.setPrefix("Y: ")
        spin_y.setSuffix("%")
        spin_y.valueChanged.connect(lambda val, s='left', i=idx: win.on_corner_spin_changed(s, i))

        btn_res_c = QToolButton()
        btn_res_c.setText("↺")
        btn_res_c.setToolTip(f"Diese Ecke auf Standard ({def_x:.0f}%, {def_y:.0f}%) zurücksetzen")
        btn_res_c.setStyleSheet("background-color: #1a1a26; border: 1px solid #28283a; border-radius: 4px; color: #94a3b8; padding: 2px 6px;")
        btn_res_c.clicked.connect(lambda checked=False, s='left', i=idx, dx=def_x, dy=def_y: win.reset_single_corner(s, i, dx, dy))

        row_box.addWidget(lbl)
        row_box.addWidget(spin_x, 1)
        row_box.addWidget(spin_y, 1)
        row_box.addWidget(btn_res_c)
        layout_cl.addLayout(row_box)
        win.spin_corners_l.append((spin_x, spin_y))

    btn_res_left_all = QPushButton("↺ Kamera Links Ecken zurücksetzen")
    btn_res_left_all.setStyleSheet("padding: 5px; font-size: 11px; background-color: #1a1a26; border: 1px solid #28283a; border-radius: 6px; color: #94a3b8;")
    btn_res_left_all.clicked.connect(lambda: win.reset_camera_corners('left'))
    layout_cl.addWidget(btn_res_left_all)

    win.tab_corners.addTab(tab_c_left, "🟦 Kamera Links (Cyan)")

    # TAB RIGHT CAMERA
    tab_c_right = QWidget()
    layout_cr = QVBoxLayout(tab_c_right)
    layout_cr.setContentsMargins(6, 6, 6, 6)
    layout_cr.setSpacing(4)

    win.spin_corners_r = []
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
        spin_x.valueChanged.connect(lambda val, s='right', i=idx: win.on_corner_spin_changed(s, i))

        spin_y = NoWheelDoubleSpinBox()
        spin_y.setRange(-80.0, 180.0)
        spin_y.setValue(def_y)
        spin_y.setSingleStep(0.2)
        spin_y.setDecimals(1)
        spin_y.setPrefix("Y: ")
        spin_y.setSuffix("%")
        spin_y.valueChanged.connect(lambda val, s='right', i=idx: win.on_corner_spin_changed(s, i))

        btn_res_c = QToolButton()
        btn_res_c.setText("↺")
        btn_res_c.setToolTip(f"Diese Ecke auf Standard ({def_x:.0f}%, {def_y:.0f}%) zurücksetzen")
        btn_res_c.setStyleSheet("background-color: #1a1a26; border: 1px solid #28283a; border-radius: 4px; color: #94a3b8; padding: 2px 6px;")
        btn_res_c.clicked.connect(lambda checked=False, s='right', i=idx, dx=def_x, dy=def_y: win.reset_single_corner(s, i, dx, dy))

        row_box.addWidget(lbl)
        row_box.addWidget(spin_x, 1)
        row_box.addWidget(spin_y, 1)
        row_box.addWidget(btn_res_c)
        layout_cr.addLayout(row_box)
        win.spin_corners_r.append((spin_x, spin_y))

    btn_res_right_all = QPushButton("↺ Kamera Rechts Ecken zurücksetzen")
    btn_res_right_all.setStyleSheet("padding: 5px; font-size: 11px; background-color: #1a1a26; border: 1px solid #28283a; border-radius: 6px; color: #94a3b8;")
    btn_res_right_all.clicked.connect(lambda: win.reset_camera_corners('right'))
    layout_cr.addWidget(btn_res_right_all)

    win.tab_corners.addTab(tab_c_right, "🟧 Kamera Rechts (Gold)")
    layout_cp.addWidget(win.tab_corners)

    btn_res_all_corners = QPushButton("↺ Alle 8 Ecken auf Standard zurücksetzen")
    btn_res_all_corners.setStyleSheet("padding: 6px; font-weight: 600; background-color: #1a1a26; border: 1px solid #28283a; border-radius: 6px; color: #cbd5e1;")
    btn_res_all_corners.clicked.connect(win.reset_all_corners)
    layout_cp.addWidget(btn_res_all_corners)

    scroll_layout.addWidget(grp_corner_pin)

    # 6. Freistellen & Transformieren (Photoshop-Modus) Card
    grp_crop_transform = QGroupBox("✂️ Freistellen & Transformieren (Photoshop-Modus)")
    layout_ct = QVBoxLayout(grp_crop_transform)
    layout_ct.setSpacing(8)

    lbl_ct_info = QLabel("Freie Formate, Photoshop-Zuschnitt & Bild-Stauchung (X / Y Skalierung):")
    lbl_ct_info.setStyleSheet("color: #8e8e9c; font-size: 11px;")
    lbl_ct_info.setWordWrap(True)
    layout_ct.addWidget(lbl_ct_info)

    win.btn_crop_mode = QPushButton("✂️ Photoshop Zuschnitt aktivieren (8 Griffe & Drittel-Raster)")
    win.btn_crop_mode.setCheckable(True)
    win.btn_crop_mode.setStyleSheet("background-color: #0284c7; font-weight: bold; padding: 8px; color: white; border-radius: 8px;")
    win.btn_crop_mode.setToolTip("Aktiviert das interaktive Photoshop-Crop-Rechteck mit 8 Anfassern und Drittel-Raster direkt im Panorama-Fenster.")
    win.btn_crop_mode.toggled.connect(win.on_crop_mode_toggled)
    layout_ct.addWidget(win.btn_crop_mode)

    row_ratio = QHBoxLayout()
    row_ratio.addWidget(QLabel("Seitenverhältnis:"))
    win.combo_crop_ratio = QComboBox()
    win.combo_crop_ratio.addItem("Frei (Ungebunden)", None)
    win.combo_crop_ratio.addItem("16:9 (Standard Broadcast)", 16.0 / 9.0)
    win.combo_crop_ratio.addItem("21:10 (Gestaucht Taktik)", 21.0 / 10.0)
    win.combo_crop_ratio.addItem("21:9 (Ultrawide)", 21.0 / 9.0)
    win.combo_crop_ratio.addItem("32:9 (Super-Ultrawide Panorama)", 32.0 / 9.0)
    win.combo_crop_ratio.addItem("8:3 (Taktik-Panorama)", 8.0 / 3.0)
    win.combo_crop_ratio.addItem("4:3 (Klassisch)", 4.0 / 3.0)
    win.combo_crop_ratio.addItem("1:1 (Quadratisch)", 1.0)
    win.combo_crop_ratio.currentIndexChanged.connect(win.on_crop_ratio_changed)
    row_ratio.addWidget(win.combo_crop_ratio, 1)
    layout_ct.addLayout(row_ratio)

    grid_crop = QGridLayout()
    grid_crop.setSpacing(6)

    grid_crop.addWidget(QLabel("X-Start:"), 0, 0)
    win.spin_crop_x = NoWheelDoubleSpinBox()
    win.spin_crop_x.setRange(0.0, 99.0)
    win.spin_crop_x.setValue(0.0)
    win.spin_crop_x.setSingleStep(0.5)
    win.spin_crop_x.setDecimals(1)
    win.spin_crop_x.setSuffix(" %")
    win.spin_crop_x.valueChanged.connect(win.on_crop_spin_changed)
    grid_crop.addWidget(win.spin_crop_x, 0, 1)

    grid_crop.addWidget(QLabel("Y-Start:"), 0, 2)
    win.spin_crop_y = NoWheelDoubleSpinBox()
    win.spin_crop_y.setRange(0.0, 99.0)
    win.spin_crop_y.setValue(0.0)
    win.spin_crop_y.setSingleStep(0.5)
    win.spin_crop_y.setDecimals(1)
    win.spin_crop_y.setSuffix(" %")
    win.spin_crop_y.valueChanged.connect(win.on_crop_spin_changed)
    grid_crop.addWidget(win.spin_crop_y, 0, 3)

    grid_crop.addWidget(QLabel("Breite:"), 1, 0)
    win.spin_crop_w = NoWheelDoubleSpinBox()
    win.spin_crop_w.setRange(1.0, 100.0)
    win.spin_crop_w.setValue(100.0)
    win.spin_crop_w.setSingleStep(0.5)
    win.spin_crop_w.setDecimals(1)
    win.spin_crop_w.setSuffix(" %")
    win.spin_crop_w.valueChanged.connect(win.on_crop_spin_changed)
    grid_crop.addWidget(win.spin_crop_w, 1, 1)

    grid_crop.addWidget(QLabel("Höhe:"), 1, 2)
    win.spin_crop_h = NoWheelDoubleSpinBox()
    win.spin_crop_h.setRange(1.0, 100.0)
    win.spin_crop_h.setValue(100.0)
    win.spin_crop_h.setSingleStep(0.5)
    win.spin_crop_h.setDecimals(1)
    win.spin_crop_h.setSuffix(" %")
    win.spin_crop_h.valueChanged.connect(win.on_crop_spin_changed)
    grid_crop.addWidget(win.spin_crop_h, 1, 3)

    layout_ct.addLayout(grid_crop)

    win.ctrl_stretch_x = LabeledSliderSpinBox("Breiten-Skalierung (Stretch X):", 0.30, 2.00, 1.00, 0.01, "x", decimals=2)
    win.ctrl_stretch_x.setToolTip("Staucht oder dehnt das Bild horizontal (Transformieren).")
    win.ctrl_stretch_x.valueChanged.connect(win.on_transform_stretch_changed)
    layout_ct.addWidget(win.ctrl_stretch_x)

    win.ctrl_stretch_y = LabeledSliderSpinBox("Höhen-Skalierung (Stretch Y):", 0.30, 2.00, 1.00, 0.01, "x", decimals=2)
    win.ctrl_stretch_y.setToolTip("Staucht oder dehnt das Bild vertikal (Transformieren).")
    win.ctrl_stretch_y.valueChanged.connect(win.on_transform_stretch_changed)
    layout_ct.addWidget(win.ctrl_stretch_y)

    win.btn_reset_crop = QPushButton("↺ Zuschnitt & Transformation zurücksetzen")
    win.btn_reset_crop.setStyleSheet("padding: 6px; font-weight: 600; background-color: #1a1a26; border: 1px solid #28283a; border-radius: 6px; color: #cbd5e1;")
    win.btn_reset_crop.clicked.connect(win.reset_crop_and_transform)
    layout_ct.addWidget(win.btn_reset_crop)

    scroll_layout.addWidget(grp_crop_transform)

    # 7. Profile & Standard-Einstellungen Card
    grp_profiles = QGroupBox("💾 Profile & Standard-Einstellungen")
    layout_prof = QVBoxLayout(grp_profiles)
    layout_prof.setSpacing(6)

    row_prof_btns = QHBoxLayout()
    btn_save_full = QPushButton("💾 Profil speichern...")
    btn_save_full.setStyleSheet("background-color: #1e1e2d; border: 1px solid #2d2d42; font-weight: 600; padding: 6px; color: #f4f4f6; border-radius: 8px;")
    btn_save_full.clicked.connect(win.save_full_profile_as)

    btn_load_full = QPushButton("📁 Profil laden...")
    btn_load_full.setStyleSheet("background-color: #1e1e2d; border: 1px solid #2d2d42; font-weight: 600; padding: 6px; color: #f4f4f6; border-radius: 8px;")
    btn_load_full.clicked.connect(win.load_full_profile_from)

    row_prof_btns.addWidget(btn_save_full)
    row_prof_btns.addWidget(btn_load_full)
    layout_prof.addLayout(row_prof_btns)

    btn_set_default = QPushButton("⭐ Als Start-Standard festlegen")
    btn_set_default.setStyleSheet("background-color: #2563eb; font-weight: bold; padding: 7px; color: white; font-size: 11px; border-radius: 8px;")
    btn_set_default.clicked.connect(win.save_as_default_settings)
    layout_prof.addWidget(btn_set_default)

    btn_reset_defaults = QPushButton("↺ Auf Werkseinstellungen zurücksetzen")
    btn_reset_defaults.setStyleSheet("background-color: #161622; border: 1px solid #262638; padding: 5px; font-size: 11px; color: #8e8e9c; border-radius: 8px;")
    btn_reset_defaults.clicked.connect(win.restore_factory_defaults)
    layout_prof.addWidget(btn_reset_defaults)

    scroll_layout.addWidget(grp_profiles)

    scroll_layout.addStretch()
    scroll_area.setWidget(scroll_widget)
    win.stitch_sub_stack.addWidget(scroll_area)

    # Sub-Page 2: Stitching Export & Upload
    win.scroll_stitch_exp = win._create_export_upload_subpage("stitching")
    win.stitch_sub_stack.addWidget(win.scroll_stitch_exp)

    layout_tab_stitch.addWidget(win.stitch_sub_stack, 1)
    win.tab_calib = scroll_area
    win.set_stitching_subpage(0)

    return win.tab_stitching
