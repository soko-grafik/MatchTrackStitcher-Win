"""
Start-Zentrale / Workflow Hub Landing Page for MatchTrack-Stitcher.
"""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel,
    QPushButton, QFrame, QScrollArea
)


def build_landing_page(win) -> QWidget:
    """
    Constructs the Start-Zentrale Landing Page widget with:
    - 1. Stitching & Panorama (4:3 DJI Action 4 and 16:9 Standard)
    - 2. Ultra-Weitwinkel & Spielfeld-Trapez Begradigung
    - 3. KI-Broadcast & Follow-Cam (Ball- & Spielertracking)
    - 4. Live-Streaming Studio
    - 5. MatchTrack Online Cloud-Upload
    - Bottom quick navigation bar
    """
    landing_widget = QWidget()
    layout_outer = QVBoxLayout(landing_widget)
    layout_outer.setContentsMargins(0, 0, 0, 0)

    scroll_area = QScrollArea()
    scroll_area.setWidgetResizable(True)
    scroll_area.setFrameShape(QFrame.NoFrame)
    scroll_area.setStyleSheet("QScrollArea { background-color: #0b0b12; border: none; }")

    content = QWidget()
    content.setStyleSheet("QWidget { background-color: #0b0b12; }")
    layout_main = QVBoxLayout(content)
    layout_main.setContentsMargins(36, 24, 36, 24)
    layout_main.setSpacing(20)

    # -------------------------------------------------------------
    # HERO CARD: Autocam 5-Schritte Assistent
    # -------------------------------------------------------------
    hero_card = QFrame()
    hero_card.setStyleSheet("""
        QFrame {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #064e3b, stop:0.5 #0f172a, stop:1 #111827);
            border: 2px solid #10b981;
            border-radius: 16px;
            padding: 20px;
        }
    """)
    hero_layout = QHBoxLayout(hero_card)
    hero_layout.setSpacing(20)

    hero_info = QVBoxLayout()
    hero_badge = QLabel("NEU: ONCE SPORT AUTOCAM 4.1 WORKFLOW")
    hero_badge.setStyleSheet("color: #34d399; font-weight: 800; font-size: 11px; letter-spacing: 1px;")
    hero_title = QLabel("⚡ Autocam 5-Schritte Assistent")
    hero_title.setStyleSheet("font-size: 22px; font-weight: 900; color: #ffffff;")
    hero_desc = QLabel(
        "Linearer Produktions-Workflow nach Vorbild von Once Sport Autocam: "
        "Setup ➔ Dual-Cam Source ➔ 10-Punkte Spline-Spielfeldmarkierung ➔ "
        "Dual Live-Framing (Widest & Tightest) ➔ H.264 Broadcast-Export."
    )
    hero_desc.setWordWrap(True)
    hero_desc.setStyleSheet("color: #cbd5e1; font-size: 13px; margin-top: 4px;")

    hero_info.addWidget(hero_badge)
    hero_info.addWidget(hero_title)
    hero_info.addWidget(hero_desc)

    hero_layout.addLayout(hero_info, 1)

    btn_launch_wizard = QPushButton("🚀 Autocam-Assistent starten ➔")
    btn_launch_wizard.setStyleSheet("""
        QPushButton {
            background-color: #10b981;
            color: #064e3b;
            font-weight: 900;
            font-size: 14px;
            padding: 14px 28px;
            border-radius: 10px;
            border: none;
        }
        QPushButton:hover {
            background-color: #34d399;
            color: #022c22;
        }
    """)
    btn_launch_wizard.clicked.connect(lambda: win.show_workspace("wizard"))
    hero_layout.addWidget(btn_launch_wizard, 0, Qt.AlignVCenter)

    layout_main.addWidget(hero_card)

    # -------------------------------------------------------------
    # Action Cards (2x2 Grid)
    # -------------------------------------------------------------
    grid = QGridLayout()
    grid.setSpacing(16)

    card_style_base = """
        QFrame {
            background-color: #12121c;
            border: 1px solid #202034;
            border-radius: 14px;
            padding: 16px;
        }
        QFrame:hover {
            border-color: #3b82f6;
            background-color: #151524;
        }
    """

    # ------------------ CARD 1: Stitching & Panorama ------------------
    card1 = QFrame()
    card1.setStyleSheet(card_style_base)
    card1_layout = QVBoxLayout(card1)
    card1_layout.setSpacing(10)

    c1_head = QHBoxLayout()
    lbl_c1_icon = QLabel("🧵")
    lbl_c1_icon.setStyleSheet("font-size: 26px; border: none; background: transparent;")
    c1_title_box = QVBoxLayout()
    lbl_c1_title = QLabel("1. Video-Stitching & Panorama")
    lbl_c1_title.setStyleSheet("font-size: 16px; font-weight: 800; color: #f8fafc; border: none; background: transparent;")
    lbl_c1_sub = QLabel("Dual-Cam Stitching • Horizont & Nahtkorrektur")
    lbl_c1_sub.setStyleSheet("font-size: 11px; color: #38bdf8; font-weight: 600; border: none; background: transparent;")
    c1_title_box.addWidget(lbl_c1_title)
    c1_title_box.addWidget(lbl_c1_sub)
    c1_head.addWidget(lbl_c1_icon)
    c1_head.addLayout(c1_title_box, 1)
    card1_layout.addLayout(c1_head)

    lbl_c1_desc = QLabel(
        "Fügt zwei Kameravideos nahtlos zu einem hochauflösenden Taktik-Panorama zusammen. "
        "Wählen Sie das Quellformat Ihrer Kameras für automatische Voreinstellungen:"
    )
    lbl_c1_desc.setWordWrap(True)
    lbl_c1_desc.setStyleSheet("font-size: 12px; color: #94a3b8; border: none; background: transparent;")
    card1_layout.addWidget(lbl_c1_desc)

    lbl_fmt_pick = QLabel("📷 Quellformat wählen & direkt starten:")
    lbl_fmt_pick.setStyleSheet("font-size: 11px; font-weight: 700; color: #cbd5e1; margin-top: 4px; border: none; background: transparent;")
    card1_layout.addWidget(lbl_fmt_pick)

    row_c1_btns = QHBoxLayout()
    btn_start_4_3 = QPushButton("📹 4:3 DJI Action 4\n(8:3 Taktik-Panorama)")
    btn_start_4_3.setStyleSheet("""
        QPushButton {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #0284c7, stop:1 #0369a1);
            color: #ffffff;
            font-weight: 800;
            font-size: 11px;
            padding: 10px 8px;
            border-radius: 9px;
            border: none;
            text-align: center;
        }
        QPushButton:hover { background: #0ea5e9; }
    """)
    btn_start_4_3.clicked.connect(lambda: win._launch_workflow_stitching("4:3"))

    btn_start_16_9 = QPushButton("📷 16:9 Standard Cams\n(32:9 Ultra-Panorama)")
    btn_start_16_9.setStyleSheet("""
        QPushButton {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #059669, stop:1 #047857);
            color: #ffffff;
            font-weight: 800;
            font-size: 11px;
            padding: 10px 8px;
            border-radius: 9px;
            border: none;
            text-align: center;
        }
        QPushButton:hover { background: #10b981; }
    """)
    btn_start_16_9.clicked.connect(lambda: win._launch_workflow_stitching("16:9"))

    row_c1_btns.addWidget(btn_start_4_3)
    row_c1_btns.addWidget(btn_start_16_9)
    card1_layout.addLayout(row_c1_btns)

    card1_layout.addStretch()
    grid.addWidget(card1, 0, 0)

    # ------------------ CARD 2: Autocam 5-Schritte Follow-Cam ------------------
    card2 = QFrame()
    card2.setStyleSheet(card_style_base)
    card2_layout = QVBoxLayout(card2)
    card2_layout.setSpacing(10)

    c2_head = QHBoxLayout()
    lbl_c2_icon = QLabel("⚡")
    lbl_c2_icon.setStyleSheet("font-size: 26px; border: none; background: transparent;")
    c2_title_box = QVBoxLayout()
    lbl_c2_title = QLabel("2. Autocam 5-Schritte Follow-Cam")
    lbl_c2_title.setStyleSheet("font-size: 16px; font-weight: 800; color: #f8fafc; border: none; background: transparent;")
    lbl_c2_sub = QLabel("Once Autocam Workflow • Dual-Preview & KI-Tracking")
    lbl_c2_sub.setStyleSheet("font-size: 11px; color: #10b981; font-weight: 600; border: none; background: transparent;")
    c2_title_box.addWidget(lbl_c2_title)
    c2_title_box.addWidget(lbl_c2_sub)
    c2_head.addWidget(lbl_c2_icon)
    c2_head.addLayout(c2_title_box, 1)
    card2_layout.addLayout(c2_head)

    lbl_c2_desc = QLabel(
        "Geführter 5-Schritte Produktions-Workflow nach Once Sport Autocam Vorbild: "
        "10-Punkte Spline-Spielfeldmarkierung, synchrones 16:9 Dual-Zoom Framing und KI-Balltracking."
    )
    lbl_c2_desc.setWordWrap(True)
    lbl_c2_desc.setStyleSheet("font-size: 12px; color: #94a3b8; border: none; background: transparent;")
    card2_layout.addWidget(lbl_c2_desc)

    badges_c2 = QHBoxLayout()
    for b_text in ["5-Schritte Assistent", "10-Pkt Spline", "Dual 16:9 Monitore", "ONNX & YOLO"]:
        b_lbl = QLabel(b_text)
        b_lbl.setStyleSheet("background: #181829; border: 1px solid #282844; border-radius: 5px; color: #94a3b8; font-size: 10px; font-weight: 600; padding: 2px 6px;")
        badges_c2.addWidget(b_lbl)
    badges_c2.addStretch()
    card2_layout.addLayout(badges_c2)

    card2_layout.addStretch()

    row_c2_btns = QHBoxLayout()
    row_c2_btns.setSpacing(6)

    btn_open_pano_lp = QPushButton("🎬 Panoramavideo laden...")
    btn_open_pano_lp.setStyleSheet("""
        QPushButton {
            background-color: #1a1a26;
            border: 1px solid #28283a;
            color: #f4f4f6;
            font-weight: 700;
            font-size: 11px;
            padding: 10px 8px;
            border-radius: 9px;
            text-align: center;
        }
        QPushButton:hover { background-color: #242436; border-color: #10b981; }
    """)
    btn_open_pano_lp.clicked.connect(lambda: (win.open_panorama_video(), win.show_workspace("wizard")))

    btn_c2_action = QPushButton("⚡ Autocam Assistent starten ➔")
    btn_c2_action.setStyleSheet("""
        QPushButton {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #10b981, stop:1 #059669);
            color: #ffffff;
            font-weight: 800;
            font-size: 11px;
            padding: 10px 8px;
            border-radius: 9px;
            border: none;
            text-align: center;
        }
        QPushButton:hover { background: #34d399; color: #022c22; }
    """)
    btn_c2_action.clicked.connect(lambda: win.show_workspace("wizard"))

    row_c2_btns.addWidget(btn_open_pano_lp)
    row_c2_btns.addWidget(btn_c2_action)
    card2_layout.addLayout(row_c2_btns)

    grid.addWidget(card2, 0, 1)

    # ------------------ CARD 3: Live-Streaming ------------------
    card3 = QFrame()
    card3.setStyleSheet(card_style_base)
    card3_layout = QVBoxLayout(card3)
    card3_layout.setSpacing(10)

    c3_head = QHBoxLayout()
    lbl_c3_icon = QLabel("🔴")
    lbl_c3_icon.setStyleSheet("font-size: 26px; border: none; background: transparent;")
    c3_title_box = QVBoxLayout()
    lbl_c3_title = QLabel("3. Live-Streaming Studio")
    lbl_c3_title.setStyleSheet("font-size: 16px; font-weight: 800; color: #f8fafc; border: none; background: transparent;")
    lbl_c3_sub = QLabel("OBS-Style RTMP Multi-Stream • Ohne Mimo App")
    lbl_c3_sub.setStyleSheet("font-size: 11px; color: #f43f5e; font-weight: 600; border: none; background: transparent;")
    c3_title_box.addWidget(lbl_c3_title)
    c3_title_box.addWidget(lbl_c3_sub)
    c3_head.addWidget(lbl_c3_icon)
    c3_head.addLayout(c3_title_box, 1)
    card3_layout.addLayout(c3_head)

    lbl_c3_desc = QLabel(
        "Direktes Live-Streaming an MatchTrack Cloud, YouTube oder Twitch. Unterstützt DJI BLE-Kopplung, "
        "normale USB-Webcams, LTE 720p/480p Presets und flexible Feed-Kanal-Zuweisung."
    )
    lbl_c3_desc.setWordWrap(True)
    lbl_c3_desc.setStyleSheet("font-size: 12px; color: #94a3b8; border: none; background: transparent;")
    card3_layout.addWidget(lbl_c3_desc)

    badges_c3 = QHBoxLayout()
    for b_text in ["Direkt-RTMP", "USB-Cams & DJI BLE", "LTE 720p/480p", "Multi-Kanal"]:
        b_lbl = QLabel(b_text)
        b_lbl.setStyleSheet("background: #181829; border: 1px solid #282844; border-radius: 5px; color: #94a3b8; font-size: 10px; font-weight: 600; padding: 2px 6px;")
        badges_c3.addWidget(b_lbl)
    badges_c3.addStretch()
    card3_layout.addLayout(badges_c3)

    card3_layout.addStretch()

    btn_c3_action = QPushButton("🔴 Live-Streaming Einrichten ➔")
    btn_c3_action.setStyleSheet("""
        QPushButton {
            background-color: #e11d48;
            color: #ffffff;
            font-weight: 800;
            font-size: 12px;
            padding: 10px;
            border-radius: 9px;
            border: none;
        }
        QPushButton:hover { background-color: #f43f5e; }
    """)
    btn_c3_action.clicked.connect(lambda: win.show_workspace("live"))
    card3_layout.addWidget(btn_c3_action)

    grid.addWidget(card3, 1, 0)

    # ------------------ CARD 4: Cloud-Upload ------------------
    card4 = QFrame()
    card4.setStyleSheet(card_style_base)
    card4_layout = QVBoxLayout(card4)
    card4_layout.setSpacing(10)

    c4_head = QHBoxLayout()
    lbl_c4_icon = QLabel("☁️")
    lbl_c4_icon.setStyleSheet("font-size: 26px; border: none; background: transparent;")
    c4_title_box = QVBoxLayout()
    lbl_c4_title = QLabel("4. MatchTrack Cloud-Upload")
    lbl_c4_title.setStyleSheet("font-size: 16px; font-weight: 800; color: #f8fafc; border: none; background: transparent;")
    lbl_c4_sub = QLabel("Fertige Videos hochladen & Matches verwalten")
    lbl_c4_sub.setStyleSheet("font-size: 11px; color: #34d399; font-weight: 600; border: none; background: transparent;")
    c4_title_box.addWidget(lbl_c4_title)
    c4_title_box.addWidget(lbl_c4_sub)
    c4_head.addWidget(lbl_c4_icon)
    c4_head.addLayout(c4_title_box, 1)
    card4_layout.addLayout(c4_head)

    lbl_c4_desc = QLabel(
        "Laden Sie fertige Broadcast- und Panorama-Videos direkt auf die Vereins-Plattform hoch. "
        "Automatische Spielanlage, Team-Zuweisung und Highspeed Multi-Stream Upload."
    )
    lbl_c4_desc.setWordWrap(True)
    lbl_c4_desc.setStyleSheet("font-size: 12px; color: #94a3b8; border: none; background: transparent;")
    card4_layout.addWidget(lbl_c4_desc)

    badges_c4 = QHBoxLayout()
    for b_text in ["Highspeed Multi-Stream", "Automatische Spielanlage", "Team-Zuordnung"]:
        b_lbl = QLabel(b_text)
        b_lbl.setStyleSheet("background: #181829; border: 1px solid #282844; border-radius: 5px; color: #94a3b8; font-size: 10px; font-weight: 600; padding: 2px 6px;")
        badges_c4.addWidget(b_lbl)
    badges_c4.addStretch()
    card4_layout.addLayout(badges_c4)

    card4_layout.addStretch()

    btn_c4_action = QPushButton("☁️ Cloud-Upload Öffnen ➔")
    btn_c4_action.setStyleSheet("""
        QPushButton {
            background-color: #059669;
            color: #ffffff;
            font-weight: 800;
            font-size: 12px;
            padding: 10px;
            border-radius: 9px;
            border: none;
        }
        QPushButton:hover { background-color: #10b981; }
    """)
    btn_c4_action.clicked.connect(lambda: win.show_workspace("cloud"))
    card4_layout.addWidget(btn_c4_action)

    grid.addWidget(card4, 1, 1)

    layout_main.addLayout(grid)

    # -------------------------------------------------------------
    # 3. Bottom Quick Navigation Bar
    # -------------------------------------------------------------
    bottom_frame = QFrame()
    bottom_frame.setStyleSheet("""
        QFrame {
            background-color: #12121a;
            border: 1px solid #1e1e2d;
            border-radius: 12px;
            padding: 12px 18px;
        }
    """)
    bottom_layout = QHBoxLayout(bottom_frame)
    bottom_layout.setSpacing(12)

    lbl_quick = QLabel("⚡ Schnelleinstieg:")
    lbl_quick.setStyleSheet("font-weight: 700; color: #cbd5e1; font-size: 11px; border: none; background: transparent;")
    bottom_layout.addWidget(lbl_quick)

    btn_quick_prof = QPushButton("📁 Profil / Projekt öffnen")
    btn_quick_prof.setStyleSheet("background-color: #1c1c28; border: 1px solid #2a2a3e; font-size: 11px; padding: 7px 14px; border-radius: 8px; color: #cbd5e1; font-weight: 600;")
    btn_quick_prof.clicked.connect(win.load_full_profile_from)
    bottom_layout.addWidget(btn_quick_prof)

    btn_quick_work = QPushButton("🖥️ Direkt zur Arbeitsfläche wechseln (Alle Werkzeuge)")
    btn_quick_work.setStyleSheet("background-color: #1e293b; border: 1px solid #3b82f6; font-size: 11px; padding: 7px 14px; border-radius: 8px; color: #60a5fa; font-weight: 700;")
    btn_quick_work.clicked.connect(lambda: win.show_workspace(0))
    bottom_layout.addWidget(btn_quick_work)

    bottom_layout.addStretch()

    layout_main.addWidget(bottom_frame)
    layout_main.addStretch()

    scroll_area.setWidget(content)
    layout_outer.addWidget(scroll_area)

    return landing_widget
