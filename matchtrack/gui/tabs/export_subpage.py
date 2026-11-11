"""
Reusable Export & Cloud-Upload Subpage Builder for Stitching, Weitwinkel, and KI-Broadcast.
"""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QGroupBox, QComboBox,
    QPushButton, QCheckBox, QLineEdit, QProgressBar, QScrollArea
)
from ..widgets import NoWheelSpinBox, NoWheelComboBox, LabeledSliderSpinBox


def create_codec_combobox() -> QComboBox:
    combo = QComboBox()
    combo.addItem("🚀 NVIDIA NVENC H.264 8-Bit (RTX 3070 / 2070 Ultra Fast - Empfohlen)", "h264_nvenc")
    combo.addItem("🌟 NVIDIA NVENC H.264 8-Bit HQ (Hohe Qualität p6)", "h264_nvenc_hq")
    combo.addItem("🔋 Intel QuickSync H.264 8-Bit (Iris Xe / Core i7 HW)", "h264_qsv")
    combo.addItem("💻 Software CPU H.264 / x264 (Multi-Core 8-Bit yuv420p)", "libx264")
    return combo


def create_audio_combobox(include_mix: bool = True) -> QComboBox:
    combo = QComboBox()
    if include_mix:
        combo.addItem("🎤 Kamera Links (Audioquelle Video A)", "left")
        combo.addItem("🎤 Kamera Rechts (Audioquelle Video B, synchronisiert)", "right")
        combo.addItem("🎚️ Beide Kameras mischen (Audio A + B)", "mix")
    else:
        combo.addItem("🎤 Originalton aus Video übernehmen", "left")
    combo.addItem("🔇 Stumm (Kein Audio)", "none")
    return combo


def create_export_upload_subpage(win, domain: str) -> QScrollArea:
    """
    Creates a dedicated Export & Cloud-Upload subpage for a specific processing domain:
    - domain == 'stitching': Panorama export (8:3 / 32:9 / 21:10)
    - domain == 'wide_angle': Optical dewarped & trapezoid rectified video export
    - domain == 'autocam': 16:9 TV Broadcast with follow-cam & ball tracking
    """
    scroll_area = QScrollArea()
    scroll_area.setWidgetResizable(True)
    content = QWidget()
    layout = QVBoxLayout(content)
    layout.setSpacing(12)
    layout.setContentsMargins(6, 6, 6, 6)

    # -----------------------------------------------------------------
    # Card 1: ✂️ Export-Schnittbereich (In/Out-Trim)
    # -----------------------------------------------------------------
    grp_trim = QGroupBox("✂️ Export-Schnittbereich")
    layout_trim = QVBoxLayout(grp_trim)
    layout_trim.setSpacing(8)

    chk_trim = QCheckBox("Nur ausgewählten In/Out-Schnittbereich exportieren")
    chk_trim.setChecked(True)
    chk_trim.setStyleSheet("font-weight: 600; color: #10b981;")

    row_in = QHBoxLayout()
    lbl_in = QLabel("Start (In):")
    lbl_in.setStyleSheet("font-weight: 600; min-width: 60px; color: #f4f4f6;")
    spin_in = NoWheelSpinBox()
    spin_in.setRange(0, 10000000)
    spin_in.setValue(getattr(win, 'in_point_frame', 0))
    spin_in.setSuffix(" F")
    lbl_in_time = QLabel("00:00:00.00")
    lbl_in_time.setStyleSheet("color: #60a5fa; font-weight: bold; min-width: 80px;")
    btn_in_curr = QPushButton("📍 Playhead")
    btn_in_curr.setStyleSheet("background-color: #1a1a26; border: 1px solid #28283a; border-radius: 6px; padding: 4px 8px; font-size: 11px;")
    btn_in_curr.clicked.connect(lambda: win.set_in_point())
    row_in.addWidget(lbl_in)
    row_in.addWidget(spin_in)
    row_in.addWidget(lbl_in_time)
    row_in.addWidget(btn_in_curr)
    layout_trim.addLayout(row_in)

    row_out = QHBoxLayout()
    lbl_out = QLabel("Ende (Out):")
    lbl_out.setStyleSheet("font-weight: 600; min-width: 60px; color: #f4f4f6;")
    spin_out = NoWheelSpinBox()
    spin_out.setRange(0, 10000000)
    spin_out.setValue(getattr(win, 'out_point_frame', 0))
    spin_out.setSuffix(" F")
    lbl_out_time = QLabel("00:00:00.00")
    lbl_out_time.setStyleSheet("color: #60a5fa; font-weight: bold; min-width: 80px;")
    btn_out_curr = QPushButton("📍 Playhead")
    btn_out_curr.setStyleSheet("background-color: #1a1a26; border: 1px solid #28283a; border-radius: 6px; padding: 4px 8px; font-size: 11px;")
    btn_out_curr.clicked.connect(lambda: win.set_out_point())
    row_out.addWidget(lbl_out)
    row_out.addWidget(spin_out)
    row_out.addWidget(lbl_out_time)
    row_out.addWidget(btn_out_curr)
    layout_trim.addLayout(row_out)

    row_dur = QHBoxLayout()
    lbl_dur = QLabel("Render-Dauer: 00:00:00 (0 Frames)")
    lbl_dur.setStyleSheet("color: #10b981; font-weight: 600; font-size: 11px;")
    btn_reset_dur = QPushButton("↺ Alles")
    btn_reset_dur.setStyleSheet("background-color: #1a1a26; border: 1px solid #28283a; border-radius: 6px; padding: 4px 8px; font-size: 11px;")
    btn_reset_dur.clicked.connect(win.reset_in_out_points)
    row_dur.addWidget(lbl_dur, 1)
    row_dur.addWidget(btn_reset_dur)
    layout_trim.addLayout(row_dur)

    spin_in.valueChanged.connect(win.on_spin_export_in_changed)
    spin_out.valueChanged.connect(win.on_spin_export_out_changed)
    chk_trim.toggled.connect(win.on_export_trim_toggled)

    layout.addWidget(grp_trim)

    # -----------------------------------------------------------------
    # Card 2: ⚙️ Render- & Export-Einstellungen
    # -----------------------------------------------------------------
    grp_settings = QGroupBox("⚙️ Render- & Export-Einstellungen")
    layout_settings = QVBoxLayout(grp_settings)
    layout_settings.setSpacing(8)

    if domain == "stitching":
        layout_settings.addWidget(QLabel("Panorama-Format:"))
        combo_format = QComboBox()
        is_4_3 = win.is_4_3_mode() if hasattr(win, 'is_4_3_mode') else True
        pano_label = "🌟 8:3 Panorama Master (Taktik-Ansicht)" if is_4_3 else "🌟 32:9 Panorama Master (Taktik-Ansicht)"
        both_label = "🔄 Beides exportieren (8:3 Master + 16:9 Broadcast)" if is_4_3 else "🔄 Beides exportieren (32:9 Master + 16:9 Broadcast)"
        combo_format.addItem(pano_label, "32:9")
        combo_format.addItem("📐 21:10 Gestaucht (Taktik-Panorama / 32:9 zu 21:10)", "21:10")
        combo_format.addItem(both_label, "both")
        combo_format.addItem("✂️ Freistellen / Zuschnitt (Aktiver Photoshop Crop & Transform)", "custom_crop")
        combo_format.currentIndexChanged.connect(win.on_export_format_changed)
        layout_settings.addWidget(combo_format)

        layout_settings.addWidget(QLabel("Ausgabe-Auflösung:"))
        combo_res = QComboBox()
        layout_settings.addWidget(combo_res)

        layout_settings.addWidget(QLabel("Hardware Video-Encoder:"))
        combo_codec = create_codec_combobox()
        layout_settings.addWidget(combo_codec)

        ctrl_bitrate = LabeledSliderSpinBox("Export-Bitrate:", 10.0, 150.0, 50.0, 1.0, "Mbps", decimals=0)
        layout_settings.addWidget(ctrl_bitrate)

        layout_settings.addWidget(QLabel("Audio-Quelle:"))
        combo_audio = create_audio_combobox(include_mix=True)
        layout_settings.addWidget(combo_audio)

        # Store references on win
        win.chk_trim_stitch = chk_trim
        win.spin_in_stitch = spin_in
        win.spin_out_stitch = spin_out
        win.lbl_in_time_stitch = lbl_in_time
        win.lbl_out_time_stitch = lbl_out_time
        win.lbl_dur_stitch = lbl_dur
        win.btn_in_curr_stitch = btn_in_curr
        win.btn_out_curr_stitch = btn_out_curr
        win.combo_stitch_format = combo_format
        win.combo_stitch_res = combo_res
        win.combo_stitch_codec = combo_codec
        win.ctrl_stitch_bitrate = ctrl_bitrate
        win.combo_stitch_audio = combo_audio

        # Forward compatibility aliases
        win.combo_exp_format = combo_format
        win.combo_resolution = combo_res
        win.combo_codec = combo_codec
        win.ctrl_bitrate = ctrl_bitrate
        win.combo_audio_source = combo_audio
        win.chk_export_trim_only = chk_trim
        win.spin_export_in = spin_in
        win.spin_export_out = spin_out
        win.lbl_export_in_time = lbl_in_time
        win.lbl_export_out_time = lbl_out_time
        win.lbl_export_duration_info = lbl_dur
        win.btn_export_set_in_curr = btn_in_curr
        win.btn_export_set_out_curr = btn_out_curr

    elif domain == "wide_angle":
        layout_settings.addWidget(QLabel("Ziel-Seitenverhältnis & Bildformat:"))
        combo_format = QComboBox()
        combo_format.addItem("📺 16:9 Broadcast Full HD (1920x1080)", (1920, 1080))
        combo_format.addItem("📺 16:9 Broadcast 4K UHD (3840x2160)", (3840, 2160))
        combo_format.addItem("🎞️ 21:9 UltraWide Cinema (2560x1080)", (2560, 1080))
        combo_format.addItem("🎞️ 21:9 UWQHD Cinema (3440x1440)", (3440, 1440))
        combo_format.addItem("🏟️ 32:9 SuperWide Panorama (3840x1080)", (3840, 1080))
        combo_format.addItem("📷 4:3 Classic (Vollsensor 2880x2160)", (2880, 2160))
        combo_format.addItem("⚙️ Benutzerdefiniert (Breite x Höhe)", "custom")
        combo_format.currentIndexChanged.connect(win._on_wa_target_format_changed)
        layout_settings.addWidget(combo_format)

        layout_settings.addWidget(QLabel("Hardware Video-Encoder:"))
        combo_codec = create_codec_combobox()
        layout_settings.addWidget(combo_codec)

        ctrl_bitrate = LabeledSliderSpinBox("Export-Bitrate:", 10.0, 150.0, 50.0, 1.0, "Mbps", decimals=0)
        layout_settings.addWidget(ctrl_bitrate)

        layout_settings.addWidget(QLabel("Audio-Quelle:"))
        combo_audio = create_audio_combobox(include_mix=False)
        layout_settings.addWidget(combo_audio)

        # Store references on win
        win.chk_trim_wa = chk_trim
        win.spin_in_wa = spin_in
        win.spin_out_wa = spin_out
        win.lbl_in_time_wa = lbl_in_time
        win.lbl_out_time_wa = lbl_out_time
        win.lbl_dur_wa = lbl_dur
        win.btn_in_curr_wa = btn_in_curr
        win.btn_out_curr_wa = btn_out_curr
        win.combo_wa_exp_format = combo_format
        win.combo_wa_codec = combo_codec
        win.ctrl_wa_bitrate = ctrl_bitrate
        win.combo_wa_audio = combo_audio

    else: # autocam
        layout_settings.addWidget(QLabel("Broadcast-Format:"))
        combo_format = QComboBox()
        combo_format.addItem("🎥 16:9 AutoCam (TV-Broadcast mit Ball-Tracking)", "16:9_autocam")
        combo_format.addItem("📐 16:9 Taktik-Warp (Entzerrtes Spielfeld ohne Schwenk)", "16:9_tactical")
        layout_settings.addWidget(combo_format)

        layout_settings.addWidget(QLabel("Ausgabe-Auflösung:"))
        combo_res = QComboBox()
        combo_res.addItem("1920 x 1080 (Full HD 1080p - Standard 16:9)", (1920, 1080))
        combo_res.addItem("3840 x 2160 (4K UHD 2160p - Maximale Details)", (3840, 2160))
        combo_res.addItem("1280 x 720 (HD 720p - Schnell / Kompakt)", (1280, 720))
        layout_settings.addWidget(combo_res)

        layout_settings.addWidget(QLabel("Hardware Video-Encoder:"))
        combo_codec = create_codec_combobox()
        layout_settings.addWidget(combo_codec)

        ctrl_bitrate = LabeledSliderSpinBox("Export-Bitrate:", 10.0, 150.0, 50.0, 1.0, "Mbps", decimals=0)
        layout_settings.addWidget(ctrl_bitrate)

        layout_settings.addWidget(QLabel("Audio-Quelle:"))
        combo_audio = create_audio_combobox(include_mix=True)
        layout_settings.addWidget(combo_audio)

        chk_lookahead = QCheckBox("🎬 Filmreife Lookahead-Glättung (2-Pass antizipatorisch)")
        chk_lookahead.setChecked(True)
        chk_lookahead.setStyleSheet("font-weight: 600; color: #f59e0b;")
        layout_settings.addWidget(chk_lookahead)

        chk_burn_radar = QCheckBox("🗺️ 2D Taktik-Radar in Video einbrennen")
        chk_burn_radar.setChecked(False)
        chk_burn_radar.setStyleSheet("font-weight: 600; color: #38bdf8;")
        chk_burn_radar.toggled.connect(win.on_burn_radar_toggled)
        layout_settings.addWidget(chk_burn_radar)

        # Store references on win
        win.chk_trim_ai = chk_trim
        win.spin_in_ai = spin_in
        win.spin_out_ai = spin_out
        win.lbl_in_time_ai = lbl_in_time
        win.lbl_out_time_ai = lbl_out_time
        win.lbl_dur_ai = lbl_dur
        win.btn_in_curr_ai = btn_in_curr
        win.btn_out_curr_ai = btn_out_curr
        win.combo_ai_exp_format = combo_format
        win.combo_ai_res = combo_res
        win.combo_ai_codec = combo_codec
        win.ctrl_ai_bitrate = ctrl_bitrate
        win.combo_ai_audio = combo_audio
        win.chk_ai_lookahead = chk_lookahead
        win.chk_ai_burn_radar = chk_burn_radar
        win.chk_lookahead = chk_lookahead
        win.chk_exp_burn_radar = chk_burn_radar

    layout.addWidget(grp_settings)

    # -----------------------------------------------------------------
    # Card 3: ☁️ MatchTrack Online – Cloud-Upload
    # -----------------------------------------------------------------
    grp_cloud = QGroupBox("☁️ MatchTrack Online – Cloud-Upload")
    layout_cloud = QVBoxLayout(grp_cloud)
    layout_cloud.setSpacing(8)

    chk_auto_upload = QCheckBox("🚀 Nach dem Rendern automatisch in Cloud hochladen")
    chk_auto_upload.setChecked(False)
    chk_auto_upload.setStyleSheet("font-weight: 700; color: #10b981;")
    layout_cloud.addWidget(chk_auto_upload)

    layout_cloud.addWidget(QLabel("Spielname / Begegnung:"))
    txt_match_name = QLineEdit()
    txt_match_name.setPlaceholderText("z.B. SG Musterstadt vs. FC Beispiel")
    txt_match_name.setStyleSheet("background-color: #111118; border: 1px solid #282839; border-radius: 8px; padding: 6px 10px; color: #f4f4f6;")
    layout_cloud.addWidget(txt_match_name)

    row_team_cat = QHBoxLayout()
    col_t = QVBoxLayout()
    col_t.addWidget(QLabel("Mannschaft:"))
    combo_team = NoWheelComboBox()
    combo_team.addItem("— Bitte erst verbinden —", "")
    col_t.addWidget(combo_team)

    col_c = QVBoxLayout()
    col_c.addWidget(QLabel("Kategorie:"))
    combo_cat = NoWheelComboBox()
    for c_name in ["Punktspiel", "Testspiel", "Pokalspiel", "Turnier", "Training", "Trainingslager"]:
        combo_cat.addItem(c_name, c_name)
    col_c.addWidget(combo_cat)

    row_team_cat.addLayout(col_t, 3)
    row_team_cat.addLayout(col_c, 2)
    layout_cloud.addLayout(row_team_cat)

    row_status = QHBoxLayout()
    lbl_status = QLabel("☁️ Status: Nicht angemeldet")
    lbl_status.setStyleSheet("color: #94a3b8; font-size: 11px;")
    btn_goto_cloud = QPushButton("☁️ Cloud-Zentrale öffnen ➔")
    btn_goto_cloud.setStyleSheet("background-color: #1e1e2d; border: 1px solid #3b82f6; font-size: 11px; padding: 4px 8px; border-radius: 6px; color: #60a5fa; font-weight: 600;")
    btn_goto_cloud.clicked.connect(lambda: win.show_workspace(4))
    row_status.addWidget(lbl_status, 1)
    row_status.addWidget(btn_goto_cloud)
    layout_cloud.addLayout(row_status)

    layout.addWidget(grp_cloud)

    if domain == "stitching":
        win.chk_cloud_auto_stitch = chk_auto_upload
        win.txt_cloud_name_stitch = txt_match_name
        win.combo_cloud_team_stitch = combo_team
        win.combo_cloud_cat_stitch = combo_cat
        win.lbl_cloud_status_stitch = lbl_status
    elif domain == "wide_angle":
        win.chk_cloud_auto_wa = chk_auto_upload
        win.txt_cloud_name_wa = txt_match_name
        win.combo_cloud_team_wa = combo_team
        win.combo_cloud_cat_wa = combo_cat
        win.lbl_cloud_status_wa = lbl_status
    else:
        win.chk_cloud_auto_ai = chk_auto_upload
        win.txt_cloud_name_ai = txt_match_name
        win.combo_cloud_team_ai = combo_team
        win.combo_cloud_cat_ai = combo_cat
        win.lbl_cloud_status_ai = lbl_status

    # -----------------------------------------------------------------
    # Card 4: 🚀 GPU Video-Export & Rendern
    # -----------------------------------------------------------------
    grp_render = QGroupBox("🚀 GPU Video-Export")
    render_box = QVBoxLayout(grp_render)
    render_box.setSpacing(8)

    if domain == "stitching":
        btn_title = "🚀 Panorama jetzt Rendern & Exportieren"
    elif domain == "wide_angle":
        btn_title = "🚀 Begradigtes Video jetzt Rendern & Exportieren"
    else:
        btn_title = "🚀 16:9 Broadcast jetzt Rendern & Exportieren"

    btn_start = QPushButton(btn_title)
    btn_start.setStyleSheet("background-color: #2563eb; font-weight: bold; padding: 12px; font-size: 13px; color: white; border-radius: 10px;")
    btn_start.clicked.connect(lambda chk=False, d=domain: win.start_targeted_export(d))
    render_box.addWidget(btn_start)

    progress_bar = QProgressBar()
    progress_bar.setValue(0)
    render_box.addWidget(progress_bar)

    lbl_stats = QLabel("Bereit zum Exportieren.")
    lbl_stats.setStyleSheet("color: #8e8e9c; font-size: 11px;")
    render_box.addWidget(lbl_stats)

    btn_cancel = QPushButton("Abbrechen")
    btn_cancel.setEnabled(False)
    btn_cancel.setStyleSheet("background-color: #1a1a26; border: 1px solid #28283a; border-radius: 8px; padding: 6px; font-size: 11px; color: #94a3b8;")
    btn_cancel.clicked.connect(win.cancel_video_export)
    render_box.addWidget(btn_cancel)

    layout.addWidget(grp_render)
    layout.addStretch()

    if domain == "stitching":
        win.btn_stitch_start_render = btn_start
        win.progress_stitch_bar = progress_bar
        win.lbl_stitch_render_stats = lbl_stats
        win.btn_stitch_cancel_render = btn_cancel
        win.btn_start_render = btn_start
        win.progress_bar = progress_bar
        win.lbl_render_stats = lbl_stats
        win.btn_cancel_render = btn_cancel
    elif domain == "wide_angle":
        win.btn_wa_start_render = btn_start
        win.progress_wa_bar = progress_bar
        win.lbl_wa_render_stats = lbl_stats
        win.btn_wa_cancel_render = btn_cancel
    else:
        win.btn_ai_start_render = btn_start
        win.progress_ai_bar = progress_bar
        win.lbl_ai_render_stats = lbl_stats
        win.btn_ai_cancel_render = btn_cancel

    scroll_area.setWidget(content)
    return scroll_area

