# MatchTrack-Stitcher

<div align="center">

![MatchTrack-Stitcher Banner](matchtrack/assets/icon.png)

### Professioneller 32:9 Fußball-Panorama Stitcher & KI-gestützte 16:9 Broadcast Follow-Cam

[![Release](https://img.shields.io/badge/Release-v1.5.5-blue.svg)](https://github.com/soko-grafik/MatchTrackStitcher-Win/releases/tag/v1.5.5)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PySide6](https://img.shields.io/badge/GUI-PySide6%20%2F%20Qt6-41CD52.svg?logo=qt&logoColor=white)](https://pypi.org/project/PySide6/)
[![OpenCV](https://img.shields.io/badge/Computer%20Vision-OpenCV-5C3EE8.svg?logo=opencv&logoColor=white)](https://opencv.org/)
[![YOLO11](https://img.shields.io/badge/AI%20Tracking-YOLO11%20%2F%20v8-00FFFF.svg)](https://docs.ultralytics.com/)
[![FFmpeg NVENC](https://img.shields.io/badge/Hardware%20Accel-NVIDIA%20NVENC-76B900.svg?logo=nvidia&logoColor=white)](https://developer.nvidia.com/video-encode-decode-gpu-support-matrix)
[![Platform](https://img.shields.io/badge/Platform-Windows%2010%20%2F%2011%20(x64)-0078D6.svg?logo=windows&logoColor=white)](https://github.com/soko-grafik/MatchTrackStitcher-Win/releases)
[![License](https://img.shields.io/badge/License-Proprietary-red.svg)](#lizenz)

[**Download Windows Installer (v1.5.5)**](https://github.com/soko-grafik/MatchTrackStitcher-Win/releases/tag/v1.5.5) • [**Dokumentation**](#-workflow-anleitung) • [**CLI Reference**](#-cli-batch-modus) • [**Release Notes**](#-release-notes--changelog)

</div>

---

## 📖 Inhaltsverzeichnis

- [Über das Projekt](#-über-das-projekt)
- [Hauptfunktionen](#-hauptfunktionen)
- [Systemanforderungen](#-systemanforderungen)
- [Installation & Schnellstart](#-installation--schnellstart)
  - [Option A: Windows Setup Installer (Empfohlen)](#option-a-windows-setup-installer-empfohlen)
  - [Option B: Ausführen aus dem Quellcode](#option-b-ausführen-aus-dem-quellcode)
- [Der 4-Schritte-Workflow](#-der-4-schritte-workflow)
  - [1. Medien & Sync](#1--medien--sync)
  - [2. Stitching & Rig](#2--stitching--rig)
  - [3. Taktik & AutoCam](#3--taktik--autocam)
  - [4. Export](#4--export)
- [CLI-Headless- & Batch-Modus](#-cli-batch-modus)
- [DJI Action 4 Dual-Rig Empfehlungen](#-dji-action-4-dual-rig-empfehlungen)
- [Projektstruktur](#-projektstruktur)
- [Build & Packaging](#-build--packaging)
- [Release Notes / Changelog](#-release-notes--changelog)
- [Lizenz](#-lizenz)

---

## ⚽ Über das Projekt

**MatchTrack-Stitcher** ist eine hochperformante Desktop-Anwendung und Batch-Pipeline zur automatischen und manuellen Zusammenführung (Stitching) von zwei Action-Cam-Videos (z. B. 2× **DJI Osmo Action 4**, GoPro etc.) zu einem ultraweiten **32:9 Panorama-Gesamtbild** eines gesamten Fußballplatzes.

Zusätzlich beinhaltet das System eine **KI-gestützte 16:9 Broadcast Follow-Cam (AutoCam)**: Mittels modernster **YOLO11-Objekterkennung** werden Fußball und Spieler in Echtzeit lokalisiert, über einen **2D-Kalman-Filter mit physikalischer Kinematik** geglättet und über ein vorausschauendes **2-Pass Lookahead-Verfahren** in eine flüssige, fernsehreife Kameraschwenk-Perspektive übertragen – vollautomatisch, ohne Kameramann vor Ort.

---

## ✨ Hauptfunktionen

### 🎥 Nahtloses Dual-Kamera Stitching
- **Mathematisch präzise Projektion:** 3D-Zylinder- und sphärische Projektion mit individueller Ausrichtung beider Kameras (Yaw, Pitch, Roll, Konvergenz, vertikaler Versatz und optisches Zentrum $c_x, c_y$).
- **Objektivprofile & Dewarping:** Vordefinierte Profile für DJI Osmo Action 4 (1080p, 2.7K, 4K Standard/Dewarp, Wide, Ultrawide) sowie Unterstützung für Fisheye-Entzerrung (Kannala-Brandt) und Pinhole-Modelle.
- **Interaktives 4-Punkt-Corner-Mesh:** Direktes Verschieben der Ecken im Viewport per Drag & Drop für perfekte manuelle Feinjustierung.
- **Automatisches Feature-Matching:** Automatisierte Ausrichtung über SIFT/ORB-Merkmalsabgleich im Überlappungsbereich.
- **Multi-Band & Linear Blending:** Seidenweiche Überblendung im Überlappungsbereich ohne sichtbare Nähte oder Geisterbilder.

### ⚡ Akustische Audio-Synchronisation
- **Automatische Kreuzkorrelation (Cross-Correlation):** Gleicht den Audio-Versatz beider Kameras auf die Millisekunde genau ab.
- **Manuelle Offset-Korrektur:** Millisekunden- und Frame-genaue Feinjustierung.
- **Audio-Routing:** Wählbar zwischen linker Audiospur, rechter Audiospur, 50/50 Mixdown oder Stummschaltung (Mute).

### 🤖 KI-Ball- & Spielertracking (YOLO11)
- **Moderne KI-Modelle:** Nahtlos integrierte Modelle (`yolo11m_football_player.pt`, `yolo11n_football_ball.pt`).
- **Physik-gefilterter Ball-Tracker:** 2D-Kalman-Filter mit Modellierung des aerodynamischen Luftwiderstands und Validierung plausibler Ballgeschwindigkeiten (verhindert Sprünge durch Fehlmessungen).
- **Spielerdichte-Gewichtung:** Intelligentes Ausrichten der Kamera am Ballgeschehen unter Berücksichtigung der Spielerballung.

### 🎬 AutoCam / Follow-Cam (16:9 Broadcast)
- **Human-Like Motion Smoothing:** 2-Pass Lookahead-Filterung antizipiert Spielzüge und erzeugt kinoreife Schwenks ohne Ruckeln oder Hektik.
- **Deadband-Zone & Zoom-Stabilität:** Horizontale Schwenks mit einstellbarer Reaktionsschwelle und sanfter Dämpfung.
- **Ball-Fadenkreuz (Reticle):** Zuschaltbare visuelle Markierung zur Analyse und Verfolgung des Spielgeräts.

### 📐 16:9 Tactical Field Warp
- **Spielfeld-Begradigung:** Entzerrt gekrümmte oder trapezförmige Spielfeldkanten über ein 6-Punkt-Polygon (TL, TC, TR, BR, BC, BL).
- **Planare Taktikansicht:** Erzeugt eine gleichmäßige, rechtwinklige 16:9-Taktikperspektive des gesamten Platzes für Trainer und Videoanalysten.

### 🚀 Hardware-beschleunigter Export (NVENC)
- **NVIDIA GPU Beschleunigung:** Superschneller Export via `hevc_nvenc` (H.265) und `h264_nvenc`, sowie CPU-Fallback (`libx264`).
- **Vielseitige Export-Modi:**
  - `32:9` Panorama Master (z. B. 3840×1080, 5760×1620, 7680×2160)
  - `16:9` Broadcast Follow-Cam (Full HD 1080p, 4K)
  - `21:10` Squeezed Panorama (optimiert für Ultrawide-Monitore)
  - `both` Dual-Pass Export (erzeugt 32:9 Master & 16:9 Broadcast in einem optimierten Batch-Durchlauf)
- **Live-Rendering-Metriken:** Fortschrittsanzeige mit verarbeiteten Frames, Render-FPS und genauer Restzeit (ETA).

### 🖥️ Moderne PySide6 Benutzeroberfläche
- **Ergonomisches Dark-Theme:** Professionelle Optik optimiert für lange Analyse-Sessions.
- **Zero-Lag Slider Engine:** Schnelle Voransicht im Draft-Modus mit 45ms Debounce-Rendering für gestochen scharfe High-Res-Aktualisierung.
- **Strukturierter 4-Schritte-Workflow:** Übersichtlich aufgeteilt in Medien & Sync, Stitching & Rig, Taktik & AutoCam sowie Export.
- **Integrierte Log-Konsole:** Echtzeit-Statusmeldungen mit Filterfunktion nach Info, Warnung und Fehler.

---

## 💻 Systemanforderungen

| Komponente | Mindestanforderung | Empfohlen |
| :--- | :--- | :--- |
| **Betriebssystem** | Windows 10 (64-bit) | Windows 11 (64-bit) |
| **Prozessor (CPU)** | Intel Core i5 / AMD Ryzen 5 (4 Kerne) | Intel Core i7 / AMD Ryzen 7 (8+ Kerne) |
| **Grafikkarte (GPU)** | NVIDIA GTX 1060 (4 GB VRAM) | NVIDIA RTX 3060 / 4070 oder besser (CUDA & NVENC) |
| **Arbeitsspeicher (RAM)** | 16 GB | 32 GB DDR4 / DDR5 |
| **Speicherplatz** | 2 GB freier Speicherplatz | Schnelle NVMe SSD für Video-Rendering |
| **Kameras** | 2× Action-Kameras (z. B. DJI Action 4) | Dual-Rig mit ca. 80° Öffnungswinkel |

---

## 📦 Installation & Schnellstart

### Option A: Windows Setup Installer (Empfohlen)

Für Endanwender und Analysten steht ein vorkompilierter Standalone-Installer zur Verfügung, der sämtliche Abhängigkeiten, FFmpeg und KI-Modelle bündelt:

1. Laden Sie die Datei **`MatchTrack-Stitcher-Setup-v1.5.5.exe`** unter [**Releases**](https://github.com/soko-grafik/MatchTrackStitcher-Win/releases/tag/v1.5.5) herunter.
2. Führen Sie die Setup-Datei aus und folgen Sie dem Installationsassistenten.
3. Starten Sie **MatchTrack Stitcher** bequem über das Startmenü oder die Desktop-Verknüpfung.

---

### Option B: Ausführen aus dem Quellcode

Für Entwickler oder zur Anpassung von Modellen und Algorithmen:

#### 1. Repository klonen
```bash
git clone https://github.com/soko-grafik/matchtrack-stitcher.git
cd matchtrack-stitcher
```

#### 2. Virtuelle Python-Umgebung erstellen
```bash
python -m venv .venv
.venv\Scripts\activate
```

#### 3. Abhängigkeiten installieren
```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install ultralytics torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

#### 4. FFmpeg bereitstellen
Stellen Sie sicher, dass sich `ffmpeg.exe` und `ffprobe.exe` entweder im Systempfad (`PATH`) oder im Unterordner `bin/` des Projekts befinden.

#### 5. Anwendung starten
```bash
python main.py
```

---

## 🔄 Der 4-Schritte-Workflow

Die Benutzeroberfläche führt Sie logisch und strukturiert in 4 Schritten vom Rohmaterial zum fertigen Video:

```mermaid
flowchart LR
    Step1["📁 1. Medien & Sync\n• Kameras laden\n• Ton synchronisieren"] --> Step2["🎯 2. Stitching & Rig\n• Rig-Geometrie\n• Ecken/Mesh anpassen\n• Blending"]
    Step2 --> Step3["📐 3. Taktik & AutoCam\n• 16:9 Taktik-Warp\n• KI-Tracking\n• Follow-Cam Glättung"]
    Step3 --> Step4["🚀 4. Export\n• Auflösung & Codec\n• NVENC Rendering\n• 32:9 / 16:9"]
```

### 1. 📁 Medien & Sync
- **Videoeingänge wählen:** Laden Sie entweder die linken und rechten Kameravideos (Dual-Cam Modus) oder öffnen Sie direkt ein bereits fertiges 32:9 Panoramavideo.
- **Audio-Sync:** Klicken Sie auf **"Automatisch via Audio abgleichen"**, um den Versatz über die Tonspuren beider Kameras zu ermitteln, oder justieren Sie den Versatz manuell in Frames bzw. Millisekunden.
- **Audioquelle:** Legen Sie fest, ob für den finalen Export der Ton von Kamera 1 (Links), Kamera 2 (Rechts) oder ein gemischter Stereomixdown genutzt werden soll.

### 2. 🎯 Stitching & Rig
- **Kameraprofil:** Wählen Sie das passende Profil für Ihre Kamera (z. B. *DJI Action 4 1080p Standard/Dewarp*).
- **Rig-Geometrie:** Justieren Sie Konvergenzwinkel (Yaw), Neigung (Pitch), Drehung (Roll) und FOV mit den Schiebereglern oder dem Reset-Button `↺`.
- **Interaktives Mesh:** Im Viewport können die 4 Eckpunkte beider Kameras direkt mit der Maus verschoben werden.
- **Auto-Kalibrierung:** Nutzen Sie den Button **"Rig automatisch kalibrieren (SIFT)"**, um Rotationswinkel automatisch anhand markanter Punkte im Überlappungsbereich zu berechnen.
- **Naht & Blending:** Passen Sie die Nahtposition und die Überblendungsbreite für einen unsichtbaren Übergang an.

### 3. 📐 Taktik & AutoCam
- **16:9 Taktik-Warp:** Aktivieren Sie den Taktik-Modus und platzieren Sie die 6 Referenzpunkte auf den Schnittpunkten der Spielfeldlinien (4 Platzecken + 2 Mittellinienpunkte), um ein perfekt rechtwinkliges Taktikbild zu erhalten.
- **KI-Tracking:** Aktivieren Sie die Ball- und Spieler-Erkennung. Passen Sie Vertrauensschwellen (Confidence Threshold) und Reaktionsgeschwindigkeiten an.
- **Lookahead Follow-Cam:** Stellen Sie die Dämpfung, die Toleranzzone (Deadband) und den Lookahead-Horizont ein, um butterweiche Kameraschwenks zu erzielen.

### 4. 🚀 Export
- **Export-Modus:**
  - `32:9 Panorama` – Vollständige Breitbildübersicht (z. B. 3840×1080).
  - `16:9 Follow-Cam` – Dynamischer Kameraschnitt auf Spielhöhe (z. B. 1920×1080).
  - `21:10 Squeezed` – Gleichmäßig komprimierte Übersicht für 21:9 Bildschirme.
  - `Beide (32:9 & 16:9)` – Effizientes 2-Stufen-Rendering beider Formate in einem Durchgang.
- **Codec & Bitrate:** Wählen Sie `hevc_nvenc` für beste Qualität und minimale Renderzeit auf NVIDIA-Karten oder `h264_nvenc` / `libx264`.
- **Trimming:** Begrenzen Sie optional Start- und Endframe zur schnellen Ausgabe von Highlights.

---

## ⚡ CLI-Headless- & Batch-Modus

MatchTrack-Stitcher kann vollständig ohne Benutzeroberfläche über die Kommandozeile betrieben werden – ideal für automatisierte Skripte, Server oder Batch-Verarbeitung:

### Befehlsparameter

```
usage: main.py [-h] [--left LEFT] [--right RIGHT] [--panorama PANORAMA]
               [--out OUT] [--mode {32:9,21:10,16:9,both}] [--config CONFIG]
               [--resolution RESOLUTION] [--codec CODEC] [--bitrate BITRATE]
               [--audio-source {left,right,mix,none}] [--start-frame START_FRAME]
               [--end-frame END_FRAME] [--no-lookahead] [--cli]
```

| Parameter | Beschreibung | Standard |
| :--- | :--- | :--- |
| `--left <pfad>` | Pfad zum linken Kameravideo | - |
| `--right <pfad>` | Pfad zum rechten Kameravideo | - |
| `--panorama <pfad>` | Pfad zu fertigem 32:9 Panorama (Einzel-Videomodus) | - |
| `--out <pfad>` | Zielpfad der exportierten Videodatei (**Erforderlich**) | - |
| `--mode` | Export-Modus: `32:9`, `21:10`, `16:9`, `both` | `32:9` |
| `--config <pfad>` | Pfad zur Rig-Konfigurationsdatei (`.json`) | `default_rig` |
| `--resolution` | Ausgabeauflösung im Format `BxH` (z. B. `1920x1080` oder `3840x1080`) | Modusabhängig |
| `--codec` | Video-Encoder (`hevc_nvenc`, `h264_nvenc`, `libx264`) | `hevc_nvenc` |
| `--bitrate` | Ziel-Bitrate in Megabit pro Sekunde (Mbps) | `50` |
| `--audio-source` | Audioquelle (`left`, `right`, `mix`, `none`) | `left` |
| `--start-frame` | Startframe für Trimming | `0` |
| `--end-frame` | Endframe für Trimming | Videoende |
| `--no-lookahead` | Deaktiviert die 2-Pass Lookahead-Glättung | Aktiviert |
| `--cli` | Erzwingt den Headless-Modus ohne GUI | Automatisch |

### Praxisbeispiele

#### 1. Dual-Kamera Video zu 32:9 Panorama (4K) mit NVENC rendern
```bash
python main.py --left "cam_left.mp4" --right "cam_right.mp4" --out "panorama_32x9.mp4" --mode 32:9 --resolution 3840x1080 --codec hevc_nvenc --bitrate 60
```

#### 2. Dual-Kamera direkt zu 16:9 KI-Follow-Cam rendern
```bash
python main.py --left "cam_left.mp4" --right "cam_right.mp4" --out "broadcast_16x9.mp4" --mode 16:9 --resolution 1920x1080 --config "dji_action4_1080p_dewarp_rig.json"
```

#### 3. Bestehendes Panorama zu 16:9 Broadcast Follow-Cam verarbeiten
```bash
python main.py --panorama "full_panorama.mp4" --out "game_followcam.mp4" --mode 16:9 --resolution 1920x1080 --codec hevc_nvenc
```

---

## 📷 DJI Action 4 Dual-Rig Empfehlungen

Um beste Stitching-Ergebnisse ohne Parallaxenfehler zu erzielen, beachten Sie folgende Setup-Empfehlungen:

1. **Rig-Winkel:**
   - Montieren Sie beide Kameras nebeneinander auf einer stabilen Schiene mit einem Konvergenzwinkel von ca. **75° bis 85°** zueinander.
   - Ein Überlappungsbereich von **15% bis 25%** in der Spielfeldmitte liefert optimale Ergebnisse für Naht- und Farbangleich.
2. **Kameraeinstellungen:**
   - **Sichtfeld (FOV):** *Standard (Entzerrung / Dewarp)* oder *Breit* (RockSteady deaktiviert für konsistente Randbereiche).
   - **Belichtung & Weißabgleich:** Manuell fest einstellen (z. B. 5500K), um Farbsprünge zwischen linker und rechter Bildhälfte zu vermeiden.
   - **Auflösung:** 1080p60, 2.7K60 oder 4K60 (identische Bildrate bei beiden Kameras erforderlich).
3. **Aufstellung:**
   - Auf Höhe der Mittellinie auf einem Stativ in 4 bis 7 Metern Höhe für die beste Perspektive auf das gesamte Spielfeld.

---

## 📁 Projektstruktur

```
matchtrack-stitcher/
├── bin/                          # Lokale FFmpeg/FFprobe Binaries
├── installer_output/             # Vorkompilierte Windows Setup Installer (.exe)
├── matchtrack/                   # Kernpaket der Anwendung
│   ├── assets/                   # Icons, Logo-Grafiken und Styling
│   ├── gui/                      # PySide6 Desktop-Oberfläche
│   │   ├── __init__.py
│   │   └── main_window.py        # Modernes Hauptfenster (4-Schritte-Workflow)
│   ├── ai_tracker.py             # YOLO11 Ball- & Spielertracking, Kalman-Filter
│   ├── audio_sync.py             # Akustische Synchronisation (FFmpeg + FFT)
│   ├── auto_stitch.py            # Feature-Matching (SIFT/ORB) & Rig-Optimierung
│   ├── autocam.py                # Broadcast Follow-Cam & Lookahead-Algorithmus
│   ├── camera_model.py           # Objektivkalibrierung, Intrinsics & Presets
│   ├── color_matcher.py          # Farbabgleich zwischen Kameras
│   ├── logger.py                 # Logging-System & GUI-Handler
│   ├── lut_generator.py          # GPU/OpenCV Remap-LUTs & Blending-Engine
│   ├── paths.py                  # Dynamische Pfadverwaltung für Dev & Freeze-Modus
│   ├── rig_geometry.py           # 3D-Geometrie, Rotationsmatrizen & Projektion
│   ├── stitcher_engine.py        # Video-Pipeline, Multi-Threading & NVENC-Renderer
│   └── tactical_warp.py          # 16:9 Taktik-Warp & 6-Punkt Homographie
├── shaders/                      # GLSL Shader für GPU-beschleunigtes Blending
├── tests/                        # Automatisierte Unit- und Integrationstests
├── build_exe.py                  # PyInstaller & Inno Setup Packaging Script
├── installer.iss                 # Inno Setup Windows Compiler-Skript
├── MatchTrack-Stitcher.spec      # PyInstaller Spezifikationsdatei
├── main.py                       # Haupteinstiegspunkt (GUI & CLI Dispatcher)
├── requirements.txt              # Python-Abhängigkeiten
└── README.md                     # Projektdokumentation
```

---

## 🔨 Build & Packaging

Ein kompletter Windows-Installer kann vollautomatisch mit Inno Setup und PyInstaller erstellt werden:

```bash
python build_exe.py
```

Das Skript führt folgende Schritte automatisch aus:
1. Bereinigen alter `build/`, `dist/` und `installer_output/` Verzeichnisse.
2. Kompilieren aller Python-Module und GUI-Ressourcen mit **PyInstaller**.
3. Bündeln von `ffmpeg.exe`, `ffprobe.exe` sowie aller **YOLO-KI-Modelle** (`.pt`).
4. Generierung des signierfähigen Setup-Installers mit **Inno Setup 6**.
5. Ausgabe der fertigen Datei in `installer_output/MatchTrack-Stitcher-Setup-v1.5.5.exe`.

---

## 📝 Release Notes / Changelog

### Version 1.5.5 (Aktuell)
- **DJI Osmo Action 4 (155° Ultrawide) Begradigung & Spielfeld-Trapez:**
  - **Optische Fisheye-Begradigung:** Mathematisches Entzerrungsmodell (Kannala-Brandt / Äquidistant) für extreme Weitwinkelaufnahmen (155° FOV) eines einzelnen stationären 4m-Stativs.
  - **Interaktive 4-Punkt Kalibrierung im Viewport:** Geführter 4-Klick Assistent (TL ➔ TR ➔ BR ➔ BL) zum punktgenauen Setzen der Eckfahnen im Videobild.
  - **Präzisions-Lupe (2.5x Zoom):** Kreisrundes Lupen-Callout mit Fadenkreuz beim Platzieren und Ziehen jedes Eckpunkts.
  - **Symmetrisches Spielfeld-Trapezoid:** Transformiert die Auslinien so, dass Seiten- und Torauslinien auf dem Video ein mathematisch fluchtendes, waagerechtes Trapez bilden (oder optional eine FIFA 105×68 m Planar-Aufsicht).
  - **Single-Pass Remap LUT:** Kombiniert Linsenbegradigung und Perspektivtransformation in einer 32-Bit Float-Lookup-Table (< 1.0 ms pro Frame für 60+ FPS Echtzeit-Vorschau).
  - **Flexibles Zielformat:** Frei wählbar zwischen 16:9 Broadcast (1080p / 4K), 21:9 UltraWide Cinema, 32:9 SuperWide Panorama, 4:3 Vollsensor oder benutzerdefinierten Auflösungen.
  - **Neuer Arbeitsbereich `📐 2. Weitwinkel`:** Eigener Tab mit Optik-Reglern (Stärke, HFOV, Zoom, Tilt, Roll, Offset), Trapezoid-Feineinstellung, Preset-Manager (`DJI Action 4`, `Action 3/5 Pro`, `GoPro`, `Generisch`) und Schnelltaste `Ctrl+2`.
  - **Standalone Video-Export:** Hardwarebeschleunigter Export (`render_wide_angle_video`) via NVIDIA NVENC / Intel QSV / CPU auch für Einzelkameras ohne Dual-Cam-Zwang.

### Version 1.5.4
- **1:1 OBS Studio Ausgabe- & Kodierer-Engine:**
  - Komplette OBS-Studio Ausgabe- und Kodierereinstellungen (Erweitert & Einfach) mit originalgetreuem Layout.
  - Volle Unterstützung für Hardware-Encoder (**NVIDIA NVENC H.264 / HEVC**, **Intel QuickSync QSV**) sowie **x264 Software CPU**.
  - Präzise Kodierer-Parameter: Qualitätsregulierungsmethode (VBR, CBR, CQP, Verlustfrei), Bitrate & Maximalbitrate, Keyframe-Intervall, Voreinstellungen P1–P7, Tuning (Hohe Qualität, Niedrige Latenz, Lossless), Multipass-Modus (QRes / FullRes), Profile (main, high, baseline), Lookahead (32 Frames), Adaptive Psycho-Visual Quantisierung, B-Frames und benutzerdefinierte CLI-Optionen.
  - Eigenständiges OBS-Studio Einstellungsfenster (`🗗 OBS-Fenster`) für übersichtliche Großansicht.
- **Multi-Stream & 4-Kanal Live-Multiview Grid:**
  - Paralleles Streaming von bis zu 4 RTMP-Kanälen (16:9 KI-Regie, Taktik-Panorama, Cam 1 Direkt, Cam 2 Direkt).
  - Interaktives 4er-Multiview-Grid im Live-Viewport zur gleichzeitigen Überwachung aller Feeds inklusive Tally-Indikatoren.
  - Dynamischer Live-Modus-Umschalter: Dual-Stitching vs. Einzelkameras (Cam 1 / Cam 2) vs. Dual-Direct ohne CPU/GPU-Stitching-Last.
- **Optimierte Ballerkennung (YOLO-Football-Modell):**
  - Dediziertes neuronales Netzwerk für Fußbälle zur Vermeidung von Ballverlusten bei schnellen Fernschüssen und Flanken.
- **Workflow-Zentrale & UI-Bereinigung:**
  - Zentrale Startseite für geführten Workflow (Stitching, KI-Broadcast, Live-Streaming, Cloud-Upload) und bereinigte Kopfnavigation.
- **QBackingStore & Render-Stabilitätsfix:**
  - Saubere Ressourcenfreigabe aller nativen Qt-Painter (`try: ... finally: painter.end()`) zur Beseitigung von PaintEvent-Warnungen.

### Version 1.5.3
- Vorbereitende Live-Streaming-Module und DJI Direct-UVC-Kopplung.

### Version 1.5.2
- **Metrische Taktik-Analytik & Scouting (SkillCorner & Google DeepMind TacticAI-Prinzip):**
  - **Spielfeld-Homographie:** Nahtlose Umrechnung der 6-Punkt-Panoramakoordinaten in reale metrische FIFA-Spielfeldmaße ($105\text{ m} \times 68\text{ m}$) via Dual-Quad-Homographie.
  - **Live km/h Speed Badges:** Echtzeitanzeige der Geschwindigkeit für Spieler (`🔥 XX.X km/h` Sprint, `⚡ XX.X km/h` Lauf) und Ball (`⚽ XX.X km/h`) im Live-Viewport.
  - **Interaktives Taktik-Dashboard:** Modernes One UI Dashboard mit KPI-Karten (Laufdistanz beider Teams, Ballbesitz %, Raumkontrolle %, Top-Speed), interaktiver 2D Heatmap (Team Blau, Team Rot, Ball, Voronoi-Dominanz) und Sprint-Rangliste.
  - **Scouting-Exporte:** Druckreife A4-Vektor-PDF-Trainingsberichte (300 DPI via `QPdfWriter`), 1080p Infografiken (PNG) und strukturierte Scouting-JSONs.
- **Pluggable AI-Backends & FIFA VAR-Abseitslinie:**
  - Beliebiger Modellwechsel während der Laufzeit: **YOLO11 / v8**, **RF-DETR Transformer** (Real-Time Detection Transformer mit Deformable Attention) und **YOLO-Pose** (17-Keypoint Skelettschätzung).
  - Virtuelle VAR-Abseitslinie mit automatischer Erkennung des vorletzten Verteidigers, millimetergenauen Sprunggelenks-Keypoints und Live-Badge.
- **ByteTrack Multi-Object Tracking & 2D Taktik-Radar:**
  - Robustes Tracking über Verdeckungen hinweg mit automatischer Trikot-Farberkennung (Team Blau, Team Rot, Schiedsrichter).
  - Interaktives 2D Taktik-Radar Overlay mit Live-Kamera-Sichtfeld (FOV-Sektor), flexiblen Positionen, Skalierung, Transparenz und Einbrennfunktion für den Videoexport.
- **Follow-Cam V2:**
  - Feinjustierbare Gewichtungsregler für Ball vs. Spielertraube ($w_{\text{ball}}/w_{\text{spieler}}$) sowie dynamische Zonen-Dämpfung für rasante Strafraumszenen und ruhige Mittelfeldfahrten.

### Version 1.5.1
- **Samsung One UI Design:** Vollständig überarbeitete, moderne Benutzeroberfläche inspiriert vom One UI Design Kit (Segmented Controls, abgerundete Karten, optimierte Abstände und Typografie, One-UI-Farbakzente).
- **Bereinigte Navigation & Viewport:** Entfernung redundanter Navigationselemente im Header und Entrümpelung der Viewport-Toolbar.
- **Mouse-Wheel-Schutz für Sidebar-Regler:** Schieberegler (`WheelFilteredSlider`, `WheelFilteredSpinBox`) reagieren nicht mehr versehentlich auf das Scrollrad beim vertikalen Durchscrollen der Seitenleiste.
- **Schritt-Buttons für Feineinstellungen:** Zusätzliche One-Click-Buttons (`-1`, `+1`) für sekundenschnelles Justieren von Audio-Offset und horizontaler Zentrierung.
- **Automatischer Farbabgleich (Auto Color Matching):** Glättung von Helligkeits- und Farbunterschieden (Histogramm- und LAB-Angleich) zwischen linker und rechter Kamera zur Eliminierung sichtbarer Schnittkanten in der Bildmitte.
- **720p HD Export-Presets:** Schnelle, ressourcenschonende Rendering-Presets für 720p (32:9 Panorama, 16:9 Broadcast und 21:10 Squeezed).
- **Thread-sicheres Logging & Crash-Watchdog:** Hintergrund-Logging auf sichere Qt-Signale umgestellt sowie automatischer Faulthandler zur Vermeidung nativer Abstürze bei langen Render-Vorgängen.

### Version 1.5.0
- **UI/UX Reorganisation:** Vollständige Entrümpelung der Oberfläche und Umstrukturierung in den intuitiven **4-Schritte-Workflow** (*Medien & Sync*, *Stitching & Rig*, *Taktik & AutoCam*, *Export*).
- **YOLO11 Integration:** Aktualisierte neuronale Netze für präzisere Fußball- und Spielererkennung bei schwierigen Lichtverhältnissen.
- **16:9 Tactical Field Warp:** Piecewise Dual-Quad Homographie mit 6 Ankerpunkten zur planaren Entzerrung gekrümmter Spielfelder.
- **Zero-Lag Slider Engine:** Sofortige Vorschau im Draft-Modus mit 45ms Debounce-Rendering für unterbrechungsfreies Justieren aller Rig-Werte.
- **Erweitertes Audio-Routing:** Wählbare Tonspuren (Links, Rechts, Mixdown, Mute) für alle Exportformate.
- **Robuste Pfadverwaltung:** Zuverlässiges Laden von Assets und Modellen im kompilieren Frozen-Modus via `paths.py`.

### Version 1.3.0
- Einführung der Hardware-beschleunigten NVENC-Exportpipeline (`hevc_nvenc`).
- Akustische Ton-Synchronisation via Audio-Kreuzkorrelation.
- Automatischer SIFT/ORB Merkmalsabgleich zur schnellen Kamerawinkel-Justage.

---

## 📄 Lizenz

Dieses Projekt ist urheberrechtlich geschützt. Alle Rechte vorbehalten.  
Copyright © 2026 MatchTrack / soko-grafik.


## ☕ Support & Spenden

**MatchTrack Online** ist ein unabhängiges Projekt, das mit viel Herzblut für Trainer und Vereine entwickelt wird. Der Betrieb von Testservern, die Bereitstellung von Updates und die kontinuierliche Entwicklung neuer Funktionen erfordern jedoch viel Zeit und laufende Ressourcen.

Wenn dir die Plattform bei deiner Videoanalyse, Spielvorbereitung und Trainingsarbeit hilft und du die Weiterentwicklung unterstützen möchtest, freue ich mich riesig über deine Unterstützung und einen Kaffee!

<p align="center">
  <a href="https://paypal.me/soko21061983" target="_blank">
    <img src="https://img.shields.io/badge/Donate-PayPal.Me-00457C?style=for-the-badge&logo=paypal&logoColor=white" alt="Donate via PayPal.Me" />
  </a>
  &nbsp;&nbsp;
  <a href="https://paypal.me/soko21061983" target="_blank">
    <img src="https://img.shields.io/badge/Buy_Me_A_Coffee-FFDD00?style=for-the-badge&logo=buy-me-a-coffee&logoColor=black" alt="Buy Me A Coffee" />
  </a>
</p>

---

<p align="center">
  Made with ⚽ for coaches and teams.
</p>
