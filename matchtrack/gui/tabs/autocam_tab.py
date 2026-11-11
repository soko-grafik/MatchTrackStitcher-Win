# matchtrack/gui/tabs/autocam_tab.py
"""
Tab 3: KI-Broadcast & Follow-Cam Tab Builder.
16:9 Taktik-Warp (6-Punkte Video-Entzerrung), YOLO Soccer-KI, Smart Zoom,
Kamera-Dämpfung, Taktik & FIFA-VAR Abseitslinie, 2D Radar und KI-Export/Upload.
"""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGroupBox,
    QLabel,
    QFrame,
    QPushButton,
    QStackedWidget,
    QScrollArea,
    QToolButton,
    QComboBox,
    QCheckBox,
)

from ..widgets import (
    NoWheelDoubleSpinBox,
    LabeledSliderSpinBox,
)


def build_autocam_tab(win) -> QWidget:
    """Builds and attaches Tab 3 (KI-Broadcast & Follow-Cam) to win (MainWindow)."""
    win.tab_autocam = QWidget()
    autocam_layout = QVBoxLayout(win.tab_autocam)
    autocam_layout.setContentsMargins(4, 4, 4, 4)
    autocam_layout.setSpacing(6)

    # Sub-Navigation Toggle: [ 🎯 KI-Tracking & Broadcast ]  [ 🚀 Export & Upload ]
    sub_nav_ai = QFrame()
    sub_nav_ai.setStyleSheet("QFrame { background: #0c0c14; border: 1px solid #1e293b; border-radius: 8px; padding: 2px; }")
    sub_layout_ai = QHBoxLayout(sub_nav_ai)
    sub_layout_ai.setContentsMargins(2, 2, 2, 2)
    sub_layout_ai.setSpacing(4)

    win.btn_ai_sub_main = QPushButton("🎯 KI-Tracking & Broadcast")
    win.btn_ai_sub_export = QPushButton("🚀 Export & Upload")

    win.btn_ai_sub_main.clicked.connect(lambda: win.set_autocam_subpage(0))
    win.btn_ai_sub_export.clicked.connect(lambda: win.set_autocam_subpage(1))

    sub_layout_ai.addWidget(win.btn_ai_sub_main)
    sub_layout_ai.addWidget(win.btn_ai_sub_export)
    autocam_layout.addWidget(sub_nav_ai)

    win.ai_sub_stack = QStackedWidget()

    scroll_area_ac = QScrollArea()
    scroll_area_ac.setWidgetResizable(True)
    scroll_widget_ac = QWidget()
    scroll_layout_ac = QVBoxLayout(scroll_widget_ac)
    scroll_layout_ac.setSpacing(12)

    # -----------------------------------------------------------------
    # Card 0: 🎬 Panorama-Mastervideo (Eingangsquelle für KI-Broadcast)
    # -----------------------------------------------------------------
    grp_pano_in = QGroupBox("🎬 Panorama-Mastervideo (Quelle für KI-Broadcast)")
    layout_pano_in = QVBoxLayout(grp_pano_in)
    layout_pano_in.setSpacing(8)

    lbl_pano_desc = QLabel(
        "Laden Sie ein zuvor erstelltes 32:9 oder 8:3 Panoramavideo (z.B. aus dem Stitching-Tab). "
        "Um maximale Rechenleistung und Systemstabilität zu gewährleisten, analysiert die KI "
        "das fertige Panoramavideo direkt und führt das Ball- und Spielertracking ohne gleichzeitiges Stitchen aus."
    )
    lbl_pano_desc.setStyleSheet("color: #8e8e9c; font-size: 11px;")
    lbl_pano_desc.setWordWrap(True)
    layout_pano_in.addWidget(lbl_pano_desc)

    card_pano_file = QFrame()
    card_pano_file.setStyleSheet(
        "QFrame { background-color: #12121c; border: 1px solid #232334; border-radius: 10px; padding: 10px; }"
    )
    layout_card_pano = QVBoxLayout(card_pano_file)
    layout_card_pano.setSpacing(6)

    row_hdr_pano = QHBoxLayout()
    lbl_title_pano = QLabel("🎬 Aktives Panoramavideo")
    lbl_title_pano.setStyleSheet("font-weight: 700; font-size: 12px; color: #f4f4f6;")
    win.badge_ai_pano = QLabel("⚪ Nicht geladen")
    win.badge_ai_pano.setStyleSheet(
        "background: #1c1c28; color: #8e8e9c; border-radius: 6px; padding: 2px 8px; font-size: 11px; font-weight: 600;"
    )
    row_hdr_pano.addWidget(lbl_title_pano)
    row_hdr_pano.addStretch()
    row_hdr_pano.addWidget(win.badge_ai_pano)
    layout_card_pano.addLayout(row_hdr_pano)

    win.lbl_file_ai_pano = QLabel("Keine Videodatei gewählt — bitte fertiges Panoramavideo laden")
    win.lbl_file_ai_pano.setStyleSheet("color: #8e8e9c; font-size: 11px; padding: 2px 0;")
    win.lbl_file_ai_pano.setWordWrap(True)
    layout_card_pano.addWidget(win.lbl_file_ai_pano)

    win.lbl_info_ai_pano = QLabel("")
    win.lbl_info_ai_pano.setStyleSheet("color: #60a5fa; font-size: 11px; font-weight: 600;")
    layout_card_pano.addWidget(win.lbl_info_ai_pano)

    row_pano_btn = QHBoxLayout()
    win.btn_open_ai_pano = QPushButton("🎬 Fertiges Panoramavideo laden (32:9 / 8:3)...")
    win.btn_open_ai_pano.setStyleSheet(
        "background-color: #1e293b; border: 1px solid #3b82f6; font-weight: 700; "
        "padding: 8px 12px; color: #60a5fa; border-radius: 8px; font-size: 12px;"
    )
    win.btn_open_ai_pano.setToolTip("Lädt ein fertiges 32:9 oder 8:3 Panoramavideo für Follow-Cam, VAR-Abseits und 2D-Radar")
    win.btn_open_ai_pano.clicked.connect(win.open_panorama_video)
    win.btn_open_pano = win.btn_open_ai_pano  # Legacy alias
    row_pano_btn.addWidget(win.btn_open_ai_pano)
    layout_card_pano.addLayout(row_pano_btn)

    layout_pano_in.addWidget(card_pano_file)
    scroll_layout_ac.addWidget(grp_pano_in)

    # 16:9 Tactical Mesh Warp Group (6 Points)
    grp_pitch_roi = QGroupBox("📐 16:9 Taktik-Warp (6-Punkte Video-Entzerrung)")
    layout_pitch_roi = QVBoxLayout(grp_pitch_roi)
    layout_pitch_roi.setSpacing(8)

    # Quick View-Switcher buttons
    row_view_quick = QHBoxLayout()
    win.btn_view_pano = QPushButton("🌟 32:9 Panorama")
    win.btn_view_pano.setToolTip("Zeigt das vollständige 32:9 bzw. 8:3 Stadion-Panorama")
    win.btn_view_pano.clicked.connect(lambda: win.set_view_mode_by_id("8:3" if win.is_4_3_mode() else "32:9"))
    btn_view_tactical = QPushButton("📐 16:9 Taktik-Warp")
    btn_view_tactical.setToolTip("Zeigt das entzerrte 16:9 Spielfeld")
    btn_view_tactical.clicked.connect(lambda: win.set_view_mode_by_id("16:9_tactical"))
    row_view_quick.addWidget(win.btn_view_pano)
    row_view_quick.addWidget(btn_view_tactical)
    layout_pitch_roi.addLayout(row_view_quick)

    lbl_roi_desc = QLabel("Verzerren und passen Sie das Panorama-Video an den 6 Außenpunkten frei an das 16:9 Frame an. Die Mittellinien-Punkte (Wölbung Oben/Unten) begradigen gebogene Seitenlinien.")
    lbl_roi_desc.setStyleSheet("color: #8e8e9c; font-size: 11px;")
    lbl_roi_desc.setWordWrap(True)
    layout_pitch_roi.addWidget(lbl_roi_desc)

    win.btn_pitch_roi_side = QPushButton("📐 6 Punkte im Bild frei verziehen (Drag & Drop)")
    win.btn_pitch_roi_side.setCheckable(True)
    win.btn_pitch_roi_side.setStyleSheet("background-color: #2563eb; font-weight: bold; padding: 8px; color: white; border-radius: 8px;")
    win.btn_pitch_roi_side.setToolTip("Aktivieren Sie diesen Modus, um die 6 Punkte direkt im Vorschaubild per Maus zu ziehen.")
    win.btn_pitch_roi_side.toggled.connect(win.on_pitch_roi_toggled)
    layout_pitch_roi.addWidget(win.btn_pitch_roi_side)

    win.spin_pitch_corners = []
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
        spin_x.valueChanged.connect(lambda val, i=idx: win.on_pitch_corner_spin_changed(i))

        spin_y = NoWheelDoubleSpinBox()
        spin_y.setRange(-20.0, 120.0)
        spin_y.setValue(def_y)
        spin_y.setSingleStep(0.5)
        spin_y.setDecimals(1)
        spin_y.setPrefix("Y: ")
        spin_y.setSuffix("%")
        spin_y.valueChanged.connect(lambda val, i=idx: win.on_pitch_corner_spin_changed(i))

        btn_res_c = QToolButton()
        btn_res_c.setText("↺")
        btn_res_c.setToolTip(f"Diesen Punkt auf Standard ({def_x:.0f}%, {def_y:.0f}%) zurücksetzen")
        btn_res_c.setStyleSheet("background-color: #1a1a26; border: 1px solid #28283a; border-radius: 4px; color: #94a3b8; padding: 2px 6px;")
        btn_res_c.clicked.connect(lambda checked=False, i=idx, dx=def_x, dy=def_y: win.reset_single_pitch_corner(i, dx, dy))

        row_box.addWidget(lbl)
        row_box.addWidget(spin_x, 1)
        row_box.addWidget(spin_y, 1)
        row_box.addWidget(btn_res_c)
        layout_pitch_roi.addLayout(row_box)
        win.spin_pitch_corners.append((spin_x, spin_y))

    row_pitch_btns = QHBoxLayout()
    btn_preset_curve = QPushButton("🏟️ Krümmung ausgleichen")
    btn_preset_curve.setStyleSheet("background-color: #1e1e2d; border: 1px solid #2563eb; font-weight: 600; padding: 6px; color: #60a5fa; border-radius: 8px;")
    btn_preset_curve.setToolTip("Setzt typische Entzerrungswerte für Mittellinien-Wölbung, um gebogene Seitenlinien zu begradigen")
    btn_preset_curve.clicked.connect(win.apply_preset_curve_warp)

    btn_preset_youth = QPushButton("⚽ Querfeld / Jugendspiel")
    btn_preset_youth.setStyleSheet("background-color: #1e293b; border: 1px solid #10b981; font-weight: 600; padding: 6px; color: #34d399; border-radius: 8px;")
    btn_preset_youth.setToolTip("Setzt das 6-Punkte-Polygon sofort passend für Jugendspiele auf dem Querfeld (untere Seitenlinie bei ~55%, Trainer & Bänke werden ignoriert)")
    btn_preset_youth.clicked.connect(win.apply_preset_youth_crossfield)

    btn_reset_pitch = QPushButton("↺ 100% Vollbild")
    btn_reset_pitch.setStyleSheet("background-color: #1a1a26; border: 1px solid #28283a; padding: 6px; font-size: 11px; color: #cbd5e1; border-radius: 8px;")
    btn_reset_pitch.clicked.connect(win.apply_preset_fullscreen_warp)

    row_pitch_btns.addWidget(btn_preset_curve)
    row_pitch_btns.addWidget(btn_preset_youth)
    row_pitch_btns.addWidget(btn_reset_pitch)
    layout_pitch_roi.addLayout(row_pitch_btns)

    win.ctrl_tactical_margin = LabeledSliderSpinBox("16:9 Taktik-Spielfeldrand (Puffer):", 0.0, 10.0, 0.0, 0.5, "%", decimals=1)
    win.ctrl_tactical_margin.setToolTip("Fügt einen Sicherheitsabstand um das entzerrte 16:9 Spielfeld hinzu, um Auslinien nicht am Bildschirmrand abzuschneiden.")
    win.ctrl_tactical_margin.valueChanged.connect(win.on_autocam_param_changed)
    layout_pitch_roi.addWidget(win.ctrl_tactical_margin)

    scroll_layout_ac.addWidget(grp_pitch_roi)

    # 16:9 AutoCam Follow Cam Group
    grp_ac_follow = QGroupBox("🎥 16:9 AutoCam (Automatische TV-Kameraführung)")
    layout_ac_follow = QVBoxLayout(grp_ac_follow)
    layout_ac_follow.setSpacing(8)

    win.chk_ai_yolo = QCheckBox("🤖 Deep Learning Soccer-KI (Spieler-Dichte & Ball)")
    win.chk_ai_yolo.setChecked(False)
    win.chk_ai_yolo.setStyleSheet("font-weight: 600; color: #10b981; margin-bottom: 4px;")
    win.chk_ai_yolo.toggled.connect(win.on_ai_yolo_toggled)
    layout_ac_follow.addWidget(win.chk_ai_yolo)

    lbl_track_mode = QLabel("Kameraführungs-Strategie:")
    lbl_track_mode.setStyleSheet("font-weight: 600; color: #f4f4f6; font-size: 11px;")
    layout_ac_follow.addWidget(lbl_track_mode)

    win.combo_ai_model = QComboBox()
    win.combo_ai_model.addItem("🏆 Soccer-Tracker Fusion (Spieler-Dichte + Ball - Empfohlen)", "hybrid_fusion")
    win.combo_ai_model.addItem("👥 Reine Spielerdichte (100% Team-Cluster, absolut stabil)", "player_density")
    win.combo_ai_model.addItem("⚽ Ball-Zentriert (100% Ball im 16:9 Bildzentrum)", "ball_centric")
    win.combo_ai_model.addItem("🎯 Weite Taktik-Kamera (Ruhige Feldübersicht)", "smooth_tactic")
    win.combo_ai_model.currentIndexChanged.connect(win.on_autocam_param_changed)
    layout_ac_follow.addWidget(win.combo_ai_model)

    # Action Fusion Weights (Ball vs Players)
    win.ctrl_ball_weight = LabeledSliderSpinBox("⚽ Ball-Gewichtung (w_ball):", 0.0, 1.0, 0.70, 0.05, "", decimals=2)
    win.ctrl_ball_weight.setToolTip("Bestimmt den Einfluss der Balltrajektorie auf die Kameraführung (0.70 = 70% Fokus auf den Ball)")
    win.ctrl_ball_weight.valueChanged.connect(win.on_autocam_param_changed)
    layout_ac_follow.addWidget(win.ctrl_ball_weight)

    win.ctrl_players_weight = LabeledSliderSpinBox("👥 Spieler-Gewichtung (w_spieler):", 0.0, 1.0, 0.30, 0.05, "", decimals=2)
    win.ctrl_players_weight.setToolTip("Bestimmt den Einfluss des Spielerschwerpunkts (Vorauseilen bei weiten Pässen)")
    win.ctrl_players_weight.valueChanged.connect(win.on_autocam_param_changed)
    layout_ac_follow.addWidget(win.ctrl_players_weight)

    win.chk_strict_ball_center = QCheckBox("🎯 Ball exakt im 16:9 Zentrum halten (Zero-Lead)")
    win.chk_strict_ball_center.setChecked(True)
    win.chk_strict_ball_center.setStyleSheet("font-weight: 600; color: #f59e0b; margin-top: 4px; margin-bottom: 2px;")
    win.chk_strict_ball_center.setToolTip("Hält den Ball bei Erkennung mathematisch exakt im Mittelpunkt des 16:9 Ausschnitts (horizontal und bei Zoom auch vertikal). Schaltet künstlichen Vorlauf-Versatz ab.")
    win.chk_strict_ball_center.toggled.connect(win.on_autocam_param_changed)
    layout_ac_follow.addWidget(win.chk_strict_ball_center)

    # Smart Zoom Group
    grp_zoom = QGroupBox("Kamera-Zoom")
    layout_zoom = QVBoxLayout(grp_zoom)
    layout_zoom.setSpacing(6)

    win.chk_dynamic_zoom = QCheckBox("🔍 Dynamischen Zoom aktivieren (Bei Pässen & Torschüssen)")
    win.chk_dynamic_zoom.setChecked(False)
    win.chk_dynamic_zoom.setStyleSheet("font-weight: 600; color: #38bdf8; margin-bottom: 4px;")
    win.chk_dynamic_zoom.toggled.connect(win.on_dynamic_zoom_toggled)
    layout_zoom.addWidget(win.chk_dynamic_zoom)

    win.ctrl_fixed_zoom = LabeledSliderSpinBox("Fester Zoom-Faktor (1.0x = Reiner Schwenk, kein Zoom):", 1.00, 2.00, 1.00, 0.05, "x", decimals=2)
    win.ctrl_fixed_zoom.valueChanged.connect(win.on_autocam_param_changed)
    layout_zoom.addWidget(win.ctrl_fixed_zoom)

    win.ctrl_min_zoom = LabeledSliderSpinBox("Min-Zoom (Weitwinkel bei Pässen):", 1.0, 1.4, 1.00, 0.05, "x", decimals=2)
    win.ctrl_min_zoom.valueChanged.connect(win.on_autocam_param_changed)
    win.ctrl_min_zoom.setVisible(False)
    layout_zoom.addWidget(win.ctrl_min_zoom)

    win.ctrl_max_zoom = LabeledSliderSpinBox("Max-Zoom (Nahaufnahme):", 1.2, 2.2, 1.40, 0.05, "x", decimals=2)
    win.ctrl_max_zoom.valueChanged.connect(win.on_autocam_param_changed)
    win.ctrl_max_zoom.setVisible(False)
    layout_zoom.addWidget(win.ctrl_max_zoom)

    win.ctrl_zoom_speed = LabeledSliderSpinBox("Zoom-Geschwindigkeit:", 0.01, 0.15, 0.04, 0.01, "", decimals=2)
    win.ctrl_zoom_speed.valueChanged.connect(win.on_autocam_param_changed)
    win.ctrl_zoom_speed.setVisible(False)
    layout_zoom.addWidget(win.ctrl_zoom_speed)

    layout_ac_follow.addWidget(grp_zoom)

    win.ctrl_ac_lead = LabeledSliderSpinBox("Spiel-Vorlauf / Antizipation (Blick in Spielrichtung):", 0.0, 0.40, 0.15, 0.05, "", decimals=2)
    win.ctrl_ac_lead.valueChanged.connect(win.on_autocam_param_changed)
    layout_ac_follow.addWidget(win.ctrl_ac_lead)

    win.ctrl_ac_smooth = LabeledSliderSpinBox("Kamera-Dämpfung (Broadcast-Sanftheit):", 0.70, 0.99, 0.94, 0.01, "", decimals=2)
    win.ctrl_ac_smooth.valueChanged.connect(win.on_autocam_param_changed)
    layout_ac_follow.addWidget(win.ctrl_ac_smooth)

    win.ctrl_ac_deadband = LabeledSliderSpinBox("Ruhezone / Ruckelfilter (Deadband):", 0.01, 0.25, 0.08, 0.01, "", decimals=2)
    win.ctrl_ac_deadband.valueChanged.connect(win.on_autocam_param_changed)
    layout_ac_follow.addWidget(win.ctrl_ac_deadband)

    win.ctrl_ac_speed = LabeledSliderSpinBox("Max. Schwenkgeschwindigkeit:", 0.01, 0.15, 0.04, 0.005, "", decimals=3)
    win.ctrl_ac_speed.valueChanged.connect(win.on_autocam_param_changed)
    layout_ac_follow.addWidget(win.ctrl_ac_speed)

    win.ctrl_ac_vpos = LabeledSliderSpinBox("Vertikale Ausrichtung (Spielfeld-Zentrum):", 0.35, 0.75, 0.50, 0.02, "", decimals=2)
    win.ctrl_ac_vpos.valueChanged.connect(win.on_autocam_param_changed)
    layout_ac_follow.addWidget(win.ctrl_ac_vpos)

    # Dynamic Zone Damping (Strafraum-Dynamik)
    win.chk_zone_damping = QCheckBox("🏟️ Zonen-Dämpfung (Höhere Agilität im Strafraum)")
    win.chk_zone_damping.setChecked(True)
    win.chk_zone_damping.setStyleSheet("font-weight: 600; color: #38bdf8; margin-top: 4px;")
    win.chk_zone_damping.setToolTip("Erhöht die Kameraschnelligkeit bei Torschüssen und Strafraumszenen automatisch")
    win.chk_zone_damping.toggled.connect(win.on_autocam_param_changed)
    layout_ac_follow.addWidget(win.chk_zone_damping)

    win.ctrl_zone_boost = LabeledSliderSpinBox("⚡ Strafraum-Schub (Agilitäts-Multiplikator):", 1.0, 3.0, 1.80, 0.1, "x", decimals=1)
    win.ctrl_zone_boost.setToolTip("Beschleunigungsfaktor der Kamera in den beiden Strafräumen (1.8x = 80% agilere Reaktion bei Torgefahr)")
    win.ctrl_zone_boost.valueChanged.connect(win.on_autocam_param_changed)
    layout_ac_follow.addWidget(win.ctrl_zone_boost)

    grp_scan_speed = QGroupBox("🤖 KI-Erkennungs-Modell & Performance")
    layout_scan = QVBoxLayout(grp_scan_speed)
    layout_scan.setSpacing(6)

    lbl_engine = QLabel("KI-Tracking-Architektur & Modell (Ball / Spieler):")
    lbl_engine.setStyleSheet("font-weight: 600; color: #f4f4f6; font-size: 11px;")
    layout_scan.addWidget(lbl_engine)

    from ...paths import get_available_ai_models
    win.combo_ai_engine = QComboBox()
    available_ai_models = get_available_ai_models()
    for m in available_ai_models:
        win.combo_ai_engine.addItem(m["name"], m["id"])
    win.combo_ai_engine.addItem("📁 Eigenes Modell (.onnx / .pt) wählen...", "custom_browse")
    win.combo_ai_engine.currentIndexChanged.connect(win.on_ai_engine_changed)
    layout_scan.addWidget(win.combo_ai_engine)

    lbl_scan = QLabel("Berechnungsintervall:")
    lbl_scan.setStyleSheet("font-weight: 600; color: #f4f4f6; font-size: 11px; margin-top: 4px;")
    layout_scan.addWidget(lbl_scan)

    win.combo_scan_speed = QComboBox()
    win.combo_scan_speed.addItem("⚡ Ausgewogen (Jeder 5. Frame – Empfohlen)", 5)
    win.combo_scan_speed.addItem("🚀 Turbo-Modus (Jeder 8. Frame – 5x schneller)", 8)
    win.combo_scan_speed.addItem("🔥 Ultra-Speed (Jeder 12. Frame – 7x schneller)", 12)
    win.combo_scan_speed.addItem("🎯 Maximale Präzision (Jeder 2. Frame)", 2)
    win.combo_scan_speed.currentIndexChanged.connect(win.on_scan_speed_changed)
    layout_scan.addWidget(win.combo_scan_speed)

    win.chk_fp16 = QCheckBox("🚀 GPU FP16 Tensor-Cores (NVIDIA RTX)")
    win.chk_fp16.setChecked(True)
    win.chk_fp16.setStyleSheet("color: #10b981; font-weight: 600;")
    win.chk_fp16.toggled.connect(win.on_scan_speed_changed)
    layout_scan.addWidget(win.chk_fp16)
    layout_ac_follow.addWidget(grp_scan_speed)

    scroll_layout_ac.addWidget(grp_ac_follow)

    # Ball Precision & Sideline False-Positive Filter Group
    grp_precision = QGroupBox("🎯 Ball-Präzision & Spielfeld-Filter")
    layout_precision = QVBoxLayout(grp_precision)
    layout_precision.setSpacing(8)

    win.chk_anti_jersey = QCheckBox("👕 Trikot-Fehlalarm-Filter (Anti-Weißes-Trikot)")
    win.chk_anti_jersey.setChecked(True)
    win.chk_anti_jersey.setStyleSheet("font-weight: 600; color: #10b981;")
    win.chk_anti_jersey.setToolTip("Schließt Ball-Fehlalarme auf Spieleroberkörpern und weißen/hellen Trikots anatomisch aus.")
    win.chk_anti_jersey.toggled.connect(win.on_autocam_param_changed)
    layout_precision.addWidget(win.chk_anti_jersey)

    win.chk_static_leaf = QCheckBox("🍂 Laub- & Bodenflecken-Unterdrückung")
    win.chk_static_leaf.setChecked(True)
    win.chk_static_leaf.setStyleSheet("font-weight: 600; color: #38bdf8;")
    win.chk_static_leaf.setToolTip("Erkennt unbewegliche Flecken (Laubblätter, Rasenschäden) und sperrt diese temporär, wenn kein Spieler in Reichweite ist.")
    win.chk_static_leaf.toggled.connect(win.on_autocam_param_changed)
    layout_precision.addWidget(win.chk_static_leaf)

    win.chk_sideline_filter = QCheckBox("🚶 Trainer- & Auslinien-Filter (Nur aktives Spielfeld)")
    win.chk_sideline_filter.setChecked(True)
    win.chk_sideline_filter.setStyleSheet("font-weight: 600; color: #f59e0b;")
    win.chk_sideline_filter.setToolTip("Filtert Personen außerhalb des 6-Punkte-Polygons (Trainerzone, Auswechselspieler, Zuschauer) anhand präziser Bodenkontaktpunkte (Füße) aus.")
    win.chk_sideline_filter.toggled.connect(win.toggle_sideline_filter)
    layout_precision.addWidget(win.chk_sideline_filter)

    win.ctrl_pitch_margin = LabeledSliderSpinBox("Auslinien-Toleranzpuffer:", 0.0, 5.0, 1.0, 0.5, "%", decimals=1)
    win.ctrl_pitch_margin.setToolTip("Sicherheitsabstand nach innen/außen an der Auslinie (in % der Feldgröße).")
    win.ctrl_pitch_margin.valueChanged.connect(win.on_autocam_param_changed)
    layout_precision.addWidget(win.ctrl_pitch_margin)

    scroll_layout_ac.addWidget(grp_precision)

    # Tactical VAR & Pose Overlays Group
    grp_var = QGroupBox("🚩 Taktik & FIFA-VAR-Entscheidungen")
    layout_var = QVBoxLayout(grp_var)
    layout_var.setSpacing(8)

    win.chk_show_offside = QCheckBox("🚩 Virtuelle Abseitslinie (VAR-Linie)")
    win.chk_show_offside.setChecked(True)
    win.chk_show_offside.setStyleSheet("font-weight: 600; color: #f59e0b;")
    win.chk_show_offside.setToolTip("Berechnet den vorletzten Verteidiger millimetergenau anhand von Fuß- und Körperachsen und projiziert die Abseitslinie auf Rasen und Radar.")
    win.chk_show_offside.toggled.connect(win.toggle_offside_line)
    layout_var.addWidget(win.chk_show_offside)

    win.chk_show_skeletons = QCheckBox("🦴 Spieler-Skelette & Pose (17 Keypoints)")
    win.chk_show_skeletons.setChecked(False)
    win.chk_show_skeletons.setStyleSheet("font-weight: 600; color: #a855f7;")
    win.chk_show_skeletons.setToolTip("Zeigt Gelenke, Extremitäten (Sprunggelenke, Knie, Schultern) und Blick-/Laufrichtung der Spieler.")
    win.chk_show_skeletons.toggled.connect(win.toggle_skeletons)
    layout_var.addWidget(win.chk_show_skeletons)

    scroll_layout_ac.addWidget(grp_var)

    # Scouting & Tactical Metrics Group (SkillCorner & TacticAI)
    grp_scouting = QGroupBox("📊 Metrische Taktik-Analytik & Scouting")
    layout_scouting = QVBoxLayout(grp_scouting)
    layout_scouting.setSpacing(8)

    win.chk_show_speed = QCheckBox("⚡ Live-Geschwindigkeit (km/h) an Spielern einblenden")
    win.chk_show_speed.setChecked(False)
    win.chk_show_speed.setStyleSheet("font-weight: 600; color: #38bdf8;")
    win.chk_show_speed.setToolTip("Zeigt die metrische Momentangeschwindigkeit jedes Spielers und Balls in km/h sowie Sprint-Indikatoren (>25.2 km/h).")
    win.chk_show_speed.toggled.connect(win.toggle_speed_badges)
    layout_scouting.addWidget(win.chk_show_speed)

    btn_open_tactics = QPushButton("📊 Taktik- & Scouting-Dashboard öffnen")
    btn_open_tactics.setStyleSheet("""
        QPushButton { background-color: #2563eb; color: #ffffff; font-weight: 700; padding: 10px 14px; border-radius: 8px; font-size: 12px; }
        QPushButton:hover { background-color: #1d4ed8; }
    """)
    btn_open_tactics.setToolTip("Öffnet das interaktive Taktik-Dashboard mit 2D-Heatmaps, Pitch Control (Dominanz), Zonenverteilung und PDF/PNG-Export.")
    btn_open_tactics.clicked.connect(win.open_tactics_dashboard)
    layout_scouting.addWidget(btn_open_tactics)

    scroll_layout_ac.addWidget(grp_scouting)

    # 2D Tactical Radar Group (FIFA Pitch Mini-Map)
    grp_radar = QGroupBox("🗺️ 2D Taktik-Radar (Mini-Map)")
    layout_radar = QVBoxLayout(grp_radar)
    layout_radar.setSpacing(8)

    win.chk_show_radar = QCheckBox("🗺️ 2D Taktik-Radar im Bild einblenden")
    win.chk_show_radar.setChecked(True)
    win.chk_show_radar.setStyleSheet("font-weight: 600; color: #10b981;")
    win.chk_show_radar.setToolTip("Zeigt eine 2D-Vogelperspektive des FIFA-Spielfelds (105x68m) mit Live-Spielern (Blau/Rot), Ball und 16:9 Kamerasichtfeld")
    win.chk_show_radar.toggled.connect(win.on_radar_param_changed)
    layout_radar.addWidget(win.chk_show_radar)

    lbl_radar_pos = QLabel("Radar-Position im Bild:")
    lbl_radar_pos.setStyleSheet("font-weight: 600; color: #f4f4f6; font-size: 11px;")
    layout_radar.addWidget(lbl_radar_pos)

    win.combo_radar_pos = QComboBox()
    win.combo_radar_pos.addItem("↗️ Oben-Rechts (Standard)", "top_right")
    win.combo_radar_pos.addItem("↘️ Unten-Rechts", "bottom_right")
    win.combo_radar_pos.addItem("⬇️ Unten-Mitte (Klassisch)", "bottom_center")
    win.combo_radar_pos.addItem("↖️ Oben-Links", "top_left")
    win.combo_radar_pos.addItem("↙️ Unten-Links", "bottom_left")
    win.combo_radar_pos.currentIndexChanged.connect(win.on_radar_param_changed)
    layout_radar.addWidget(win.combo_radar_pos)

    win.ctrl_radar_width = LabeledSliderSpinBox("Radar-Größe (Breite):", 200, 520, 320, 10, "px", decimals=0)
    win.ctrl_radar_width.setToolTip("Breite der Mini-Map in Pixeln (Höhe passt sich automatisch dem FIFA 105x68m Seitenverhältnis an)")
    win.ctrl_radar_width.valueChanged.connect(win.on_radar_param_changed)
    layout_radar.addWidget(win.ctrl_radar_width)

    win.ctrl_radar_opacity = LabeledSliderSpinBox("Radar-Deckkraft:", 0.30, 1.00, 0.85, 0.05, "", decimals=2)
    win.ctrl_radar_opacity.setToolTip("Transparenz des Rasens und der Spielfeldmarkierungen im Videobild")
    win.ctrl_radar_opacity.valueChanged.connect(win.on_radar_param_changed)
    layout_radar.addWidget(win.ctrl_radar_opacity)

    win.chk_burn_radar = QCheckBox("🎬 Radar fest in exportiertes Video einbrennen")
    win.chk_burn_radar.setChecked(False)
    win.chk_burn_radar.setStyleSheet("font-weight: 600; color: #38bdf8; margin-top: 4px;")
    win.chk_burn_radar.setToolTip("Rendert die 2D-Minimap fest in das fertige Video beim 16:9- oder 32:9-Export.")
    win.chk_burn_radar.toggled.connect(win.on_burn_radar_toggled)
    layout_radar.addWidget(win.chk_burn_radar)

    scroll_layout_ac.addWidget(grp_radar)

    scroll_layout_ac.addStretch()
    scroll_area_ac.setWidget(scroll_widget_ac)
    win.ai_sub_stack.addWidget(scroll_area_ac)

    # Subpage 1: Dedicated KI-Broadcast Export & Upload
    win.scroll_ai_exp = win._create_export_upload_subpage("autocam")
    win.ai_sub_stack.addWidget(win.scroll_ai_exp)

    autocam_layout.addWidget(win.ai_sub_stack, 1)
    win.set_autocam_subpage(0)

    return win.tab_autocam

