"""
MatchTrack Online Cloud-Upload and Match Management Tab for MatchTrack-Stitcher.
"""

from PySide6.QtCore import Qt, QDate
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QGroupBox, QComboBox,
    QPushButton, QToolButton, QCheckBox, QLineEdit, QDateEdit,
    QProgressBar, QFrame, QScrollArea
)
from ..widgets import NoWheelComboBox


def build_cloud_tab(win) -> QWidget:
    """
    Constructs dedicated Tab 5: MatchTrack Online Cloud-Upload & Match-Zuweisung.
    """
    win.tab_cloud = QWidget()
    layout_tab_cloud = QVBoxLayout(win.tab_cloud)
    layout_tab_cloud.setContentsMargins(4, 4, 4, 4)

    scroll_cloud = QScrollArea()
    scroll_cloud.setWidgetResizable(True)
    widget_cloud = QWidget()
    cloud_tab_layout = QVBoxLayout(widget_cloud)
    cloud_tab_layout.setSpacing(12)

    grp_cloud = QGroupBox("☁️ MatchTrack Online – Direkt-Upload & Match-Zuweisung")
    cloud_box = QVBoxLayout(grp_cloud)
    cloud_box.setSpacing(8)

    # Connection Status Row
    conn_row = QHBoxLayout()
    win.lbl_cloud_status = QLabel("🔴 Nicht verbunden")
    win.lbl_cloud_status.setStyleSheet(
        "background-color: #1a1520; color: #f87171; border: 1px solid #3f202b; "
        "border-radius: 8px; padding: 5px 10px; font-size: 11px; font-weight: 600;"
    )
    win.btn_cloud_login = QPushButton("🔑 Verbinden")
    win.btn_cloud_login.setStyleSheet(
        "background-color: #1e293b; border: 1px solid #3b82f6; color: #60a5fa; "
        "font-weight: bold; padding: 5px 10px; border-radius: 8px; font-size: 11px;"
    )
    win.btn_cloud_login.clicked.connect(win.open_cloud_login_dialog)

    win.btn_cloud_refresh = QToolButton()
    win.btn_cloud_refresh.setText("🔄")
    win.btn_cloud_refresh.setToolTip("Mannschaften & Spieleliste vom Server neu laden")
    win.btn_cloud_refresh.setStyleSheet(
        "background-color: #1a1a26; border: 1px solid #28283a; border-radius: 8px; padding: 5px 8px;"
    )
    win.btn_cloud_refresh.setEnabled(False)
    win.btn_cloud_refresh.clicked.connect(win.refresh_cloud_teams_and_matches)

    win.btn_cloud_logout = QPushButton("Abmelden")
    win.btn_cloud_logout.setStyleSheet(
        "background-color: #1a1a26; border: 1px solid #28283a; color: #94a3b8; "
        "padding: 5px 8px; border-radius: 8px; font-size: 11px;"
    )
    win.btn_cloud_logout.setVisible(False)
    win.btn_cloud_logout.clicked.connect(win.logout_cloud_session)

    conn_row.addWidget(win.lbl_cloud_status, 1)
    conn_row.addWidget(win.btn_cloud_login)
    conn_row.addWidget(win.btn_cloud_refresh)
    conn_row.addWidget(win.btn_cloud_logout)
    cloud_box.addLayout(conn_row)

    # Target Mode: New Match vs Existing Match
    cloud_box.addWidget(QLabel("Ziel auf Plattform:"))
    win.combo_cloud_target_mode = NoWheelComboBox()
    win.combo_cloud_target_mode.addItem("✨ Neues Spiel auf MatchTrack Online anlegen", "new")
    win.combo_cloud_target_mode.addItem("➕ Zu bestehendem Spiel hinzufügen / Perspektive ergänzen", "existing")
    win.combo_cloud_target_mode.currentIndexChanged.connect(win.on_cloud_target_mode_changed)
    cloud_box.addWidget(win.combo_cloud_target_mode)

    win.lbl_cloud_existing = QLabel("Bestehendes Spiel auswählen:")
    win.lbl_cloud_existing.setVisible(False)
    cloud_box.addWidget(win.lbl_cloud_existing)

    win.combo_cloud_existing_match = NoWheelComboBox()
    win.combo_cloud_existing_match.setVisible(False)
    win.combo_cloud_existing_match.currentIndexChanged.connect(win.on_cloud_existing_match_selected)
    cloud_box.addWidget(win.combo_cloud_existing_match)

    # Match Title
    cloud_box.addWidget(QLabel("Spielname / Begegnung:"))
    win.txt_cloud_match_name = QLineEdit()
    win.txt_cloud_match_name.setPlaceholderText("z.B. SG Musterstadt vs. FC Beispiel")
    win.txt_cloud_match_name.setStyleSheet(
        "background-color: #111118; border: 1px solid #282839; border-radius: 8px; padding: 6px 10px; color: #f4f4f6;"
    )
    cloud_box.addWidget(win.txt_cloud_match_name)

    # Team & Category Row
    row_team_cat = QHBoxLayout()
    col_team = QVBoxLayout()
    col_team.addWidget(QLabel("Mannschaft:"))
    win.combo_cloud_team = NoWheelComboBox()
    win.combo_cloud_team.addItem("— Bitte erst verbinden —", "")
    col_team.addWidget(win.combo_cloud_team)

    col_cat = QVBoxLayout()
    col_cat.addWidget(QLabel("Kategorie:"))
    win.combo_cloud_category = NoWheelComboBox()
    for cat_name in ["Punktspiel", "Testspiel", "Pokalspiel", "Turnier", "Training", "Trainingslager"]:
        win.combo_cloud_category.addItem(cat_name, cat_name)
    col_cat.addWidget(win.combo_cloud_category)

    row_team_cat.addLayout(col_team, 3)
    row_team_cat.addLayout(col_cat, 2)
    cloud_box.addLayout(row_team_cat)

    # Recording Date Row
    row_rec_date = QHBoxLayout()
    row_rec_date.addWidget(QLabel("Spieltag / Aufnahmedatum:"))
    win.date_cloud_recording = QDateEdit()
    win.date_cloud_recording.setCalendarPopup(True)
    win.date_cloud_recording.setDisplayFormat("dd.MM.yyyy")
    win.date_cloud_recording.setDate(QDate.currentDate())
    win.date_cloud_recording.setStyleSheet(
        "background-color: #111118; border: 1px solid #282839; border-radius: 8px; padding: 5px 8px; color: #f4f4f6;"
    )
    row_rec_date.addWidget(win.date_cloud_recording, 1)
    cloud_box.addLayout(row_rec_date)

    # Upload Options Checkboxes
    win.chk_cloud_auto_upload = QCheckBox("🚀 Nach dem Rendern automatisch zu MatchTrack Online hochladen")
    win.chk_cloud_auto_upload.setChecked(False)
    win.chk_cloud_auto_upload.setStyleSheet("font-weight: 700; color: #10b981; margin-top: 4px;")
    cloud_box.addWidget(win.chk_cloud_auto_upload)

    win.chk_cloud_notify = QCheckBox("🔔 Mannschaftsmitglieder nach Upload benachrichtigen")
    win.chk_cloud_notify.setChecked(True)
    win.chk_cloud_notify.setStyleSheet("color: #cbd5e1;")
    cloud_box.addWidget(win.chk_cloud_notify)

    win.chk_cloud_fast_hls = QCheckBox("⚡ Schnelle Web-Bereitstellung (Direktes HLS ohne Re-Encode falls H.264)")
    win.chk_cloud_fast_hls.setChecked(False)
    win.chk_cloud_fast_hls.setStyleSheet("color: #38bdf8;")
    win.chk_cloud_fast_hls.setToolTip(
        "Wenn aktiviert, überspringt der Server die erneute H.264-Umkodierung vor der HLS-Segmentierung."
    )
    cloud_box.addWidget(win.chk_cloud_fast_hls)

    # Password protection option
    pw_prot_row = QHBoxLayout()
    win.chk_cloud_password = QCheckBox("🔒 Spiel mit Passwort schützen:")
    win.chk_cloud_password.setStyleSheet("color: #f59e0b; font-weight: 600;")
    win.txt_cloud_password = QLineEdit()
    win.txt_cloud_password.setPlaceholderText("Spiel-Passwort...")
    win.txt_cloud_password.setVisible(False)
    win.txt_cloud_password.setStyleSheet(
        "background-color: #111118; border: 1px solid #282839; border-radius: 6px; padding: 4px 8px; color: #f4f4f6;"
    )
    win.chk_cloud_password.toggled.connect(win.txt_cloud_password.setVisible)
    pw_prot_row.addWidget(win.chk_cloud_password)
    pw_prot_row.addWidget(win.txt_cloud_password, 1)
    cloud_box.addLayout(pw_prot_row)

    # Multi-Stream Concurrency (Upload-Geschwindigkeit)
    row_concurrency = QHBoxLayout()
    lbl_concurrency = QLabel("⚡ Upload-Streams:")
    lbl_concurrency.setStyleSheet("color: #38bdf8; font-weight: 600; font-size: 11px;")
    win.combo_cloud_concurrency = QComboBox()
    win.combo_cloud_concurrency.addItem("⚡ 3 Streams (Empfohlen - bis 50 Mbit/s)", 3)
    win.combo_cloud_concurrency.addItem("🚀 4 Streams (High-Speed - 100+ Mbit/s)", 4)
    win.combo_cloud_concurrency.addItem("⚡ 2 Streams (Ausgewogen)", 2)
    win.combo_cloud_concurrency.addItem("🐢 1 Stream (Schonend)", 1)
    win.combo_cloud_concurrency.setStyleSheet(
        "background-color: #111118; border: 1px solid #282839; border-radius: 6px; padding: 4px 8px; color: #f4f4f6;"
    )
    win.combo_cloud_concurrency.setToolTip(
        "Parallele Upload-Streams nutzen deine Internetleitung voll aus und verhindern Bandbreiten-Drosselungen einzelner TCP-Verbindungen."
    )
    row_concurrency.addWidget(lbl_concurrency)
    row_concurrency.addWidget(win.combo_cloud_concurrency, 1)
    cloud_box.addLayout(row_concurrency)

    # Dedicated file pickers for finished Broadcast (16:9) & Panorama (32:9 / 8:3) videos
    files_frame = QFrame()
    files_frame.setStyleSheet(
        "QFrame { background-color: #101018; border: 1px solid #242436; border-radius: 8px; padding: 6px; }"
    )
    files_layout = QVBoxLayout(files_frame)
    files_layout.setContentsMargins(6, 6, 6, 6)
    files_layout.setSpacing(6)

    lbl_files_hdr = QLabel("🎬 Fertige Videos für Upload (einzeln oder beide zusammen):")
    lbl_files_hdr.setStyleSheet("color: #93c5fd; font-weight: 700; font-size: 11px; border: none; padding: 0;")
    files_layout.addWidget(lbl_files_hdr)

    # Row 1: 16:9 Broadcast Video
    row_f16 = QHBoxLayout()
    row_f16.setSpacing(4)
    lbl_f16 = QLabel("🎥 Broadcast (16:9):")
    lbl_f16.setFixedWidth(122)
    lbl_f16.setStyleSheet("color: #cbd5e1; font-size: 11px; border: none;")
    win.txt_cloud_video_16x9 = QLineEdit()
    win.txt_cloud_video_16x9.setPlaceholderText("Optional: Fertiges 16:9 Broadcast-Video (.mp4)...")
    win.txt_cloud_video_16x9.setStyleSheet(
        "background-color: #0b0b10; border: 1px solid #282839; border-radius: 6px; padding: 4px 8px; color: #e2e8f0; font-size: 11px;"
    )
    btn_pick_16x9 = QPushButton("📂 Wählen…")
    btn_pick_16x9.setStyleSheet(
        "background-color: #1e293b; border: 1px solid #3b82f6; color: #93c5fd; padding: 4px 8px; border-radius: 6px; font-size: 11px; font-weight: 600;"
    )
    btn_pick_16x9.clicked.connect(win.pick_cloud_16x9_video)
    btn_clr_16x9 = QToolButton()
    btn_clr_16x9.setText("✕")
    btn_clr_16x9.setToolTip("Auswahl für 16:9 Broadcast-Video leeren")
    btn_clr_16x9.setStyleSheet("background-color: #1a1a26; border: 1px solid #28283a; border-radius: 6px; padding: 3px 6px; color: #94a3b8;")
    btn_clr_16x9.clicked.connect(lambda: win.txt_cloud_video_16x9.clear())
    row_f16.addWidget(lbl_f16)
    row_f16.addWidget(win.txt_cloud_video_16x9, 1)
    row_f16.addWidget(btn_pick_16x9)
    row_f16.addWidget(btn_clr_16x9)
    files_layout.addLayout(row_f16)

    # Row 2: Panorama Video (32:9 / 8:3 / 21:10)
    row_fpano = QHBoxLayout()
    row_fpano.setSpacing(4)
    lbl_fpano = QLabel("🌟 Panorama (32:9/8:3):")
    lbl_fpano.setFixedWidth(122)
    lbl_fpano.setStyleSheet("color: #cbd5e1; font-size: 11px; border: none;")
    win.txt_cloud_video_pano = QLineEdit()
    win.txt_cloud_video_pano.setPlaceholderText("Optional: Fertiges Panorama-Video (.mp4)...")
    win.txt_cloud_video_pano.setStyleSheet(
        "background-color: #0b0b10; border: 1px solid #282839; border-radius: 6px; padding: 4px 8px; color: #e2e8f0; font-size: 11px;"
    )
    btn_pick_pano = QPushButton("📂 Wählen…")
    btn_pick_pano.setStyleSheet(
        "background-color: #1e293b; border: 1px solid #3b82f6; color: #93c5fd; padding: 4px 8px; border-radius: 6px; font-size: 11px; font-weight: 600;"
    )
    btn_pick_pano.clicked.connect(win.pick_cloud_pano_video)
    btn_clr_pano = QToolButton()
    btn_clr_pano.setText("✕")
    btn_clr_pano.setToolTip("Auswahl für Panorama-Video leeren")
    btn_clr_pano.setStyleSheet("background-color: #1a1a26; border: 1px solid #28283a; border-radius: 6px; padding: 3px 6px; color: #94a3b8;")
    btn_clr_pano.clicked.connect(lambda: win.txt_cloud_video_pano.clear())
    row_fpano.addWidget(lbl_fpano)
    row_fpano.addWidget(win.txt_cloud_video_pano, 1)
    row_fpano.addWidget(btn_pick_pano)
    row_fpano.addWidget(btn_clr_pano)
    files_layout.addLayout(row_fpano)

    cloud_box.addWidget(files_frame)

    # Manual Upload Action Button
    win.btn_cloud_upload_now = QPushButton("☁️ Fertige Videos (Broadcast & Panorama) jetzt hochladen")
    win.btn_cloud_upload_now.setStyleSheet(
        "background-color: #059669; font-weight: bold; padding: 10px; font-size: 12px; color: white; border-radius: 10px;"
    )
    win.btn_cloud_upload_now.clicked.connect(win.start_manual_cloud_upload_dialog)
    cloud_box.addWidget(win.btn_cloud_upload_now)

    win.cloud_progress_bar = QProgressBar()
    win.cloud_progress_bar.setValue(0)
    cloud_box.addWidget(win.cloud_progress_bar)

    win.lbl_cloud_upload_stats = QLabel("Bereit für Cloud-Upload.")
    win.lbl_cloud_upload_stats.setStyleSheet("color: #8e8e9c; font-size: 11px;")
    win.lbl_cloud_upload_stats.setWordWrap(True)
    cloud_box.addWidget(win.lbl_cloud_upload_stats)

    cloud_btn_bottom = QHBoxLayout()
    win.btn_cloud_cancel = QPushButton("Upload abbrechen")
    win.btn_cloud_cancel.setEnabled(False)
    win.btn_cloud_cancel.setStyleSheet(
        "background-color: #1a1a26; border: 1px solid #28283a; border-radius: 8px; padding: 6px; font-size: 11px; color: #94a3b8;"
    )
    win.btn_cloud_cancel.clicked.connect(win.cancel_cloud_upload)

    win.btn_cloud_open_browser = QPushButton("🌐 Spiel im Browser öffnen")
    win.btn_cloud_open_browser.setEnabled(False)
    win.btn_cloud_open_browser.setStyleSheet(
        "background-color: #1e293b; border: 1px solid #3b82f6; border-radius: 8px; padding: 6px; font-size: 11px; color: #60a5fa; font-weight: bold;"
    )
    win.btn_cloud_open_browser.clicked.connect(win.open_last_uploaded_match_in_browser)

    cloud_btn_bottom.addWidget(win.btn_cloud_cancel)
    cloud_btn_bottom.addWidget(win.btn_cloud_open_browser)
    cloud_box.addLayout(cloud_btn_bottom)

    grp_cloud.setMaximumWidth(960)
    cloud_tab_layout.addWidget(grp_cloud, 0, Qt.AlignHCenter)
    cloud_tab_layout.addStretch()

    scroll_cloud.setAlignment(Qt.AlignHCenter | Qt.AlignTop)
    scroll_cloud.setWidget(widget_cloud)
    layout_tab_cloud.addWidget(scroll_cloud)

    return win.tab_cloud
