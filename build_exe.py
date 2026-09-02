"""
Automated Build & Packaging Script for MatchTrack-Stitcher.
Compiles standalone binaries with PyInstaller and builds a single-file Windows Setup Installer with Inno Setup.
"""
import os
import sys
import shutil
import subprocess
import glob

# Ensure UTF-8 output encoding on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def find_iscc() -> str:
    """Finds the Inno Setup Compiler executable ISCC.exe on the system."""
    candidates = [
        r"D:\Users\SoKo\AppData\Local\Programs\Inno Setup 6\ISCC.exe",
        os.path.expandvars(r"%LOCALAPPDATA%\Programs\Inno Setup 6\ISCC.exe"),
        r"C:\Program Files (x86)\Inno Setup 6\ISCC.exe",
        r"C:\Program Files\Inno Setup 6\ISCC.exe",
        r"C:\Program Files (x86)\Inno Setup 5\ISCC.exe",
    ]
    for cand in candidates:
        if os.path.exists(cand):
            return cand
    
    which_iscc = shutil.which("ISCC.exe")
    if which_iscc:
        return which_iscc

    local_app = os.environ.get("LOCALAPPDATA", "")
    if local_app:
        matches = glob.glob(os.path.join(local_app, "Programs", "**", "ISCC.exe"), recursive=True)
        if matches:
            return matches[0]

    return ""


def build():
    print("==========================================================")
    print("     Building MatchTrack-Stitcher Windows Installer       ")
    print("==========================================================")

    root_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(root_dir)

    # 1. Clean previous builds
    for d in ["build", "dist", "installer_output"]:
        if os.path.exists(d):
            print(f"Cleaning {d}/ directory...")
            shutil.rmtree(d, ignore_errors=True)

    # 2. Run PyInstaller
    print("\n[Step 1/2] Compiling standalone application with PyInstaller...")
    cmd = [sys.executable, "-m", "PyInstaller", "--clean", "--noconfirm", "MatchTrack-Stitcher.spec"]
    res = subprocess.run(cmd)
    if res.returncode != 0:
        print("\n[ERROR] Build failed during PyInstaller execution.")
        sys.exit(1)

    dist_app = os.path.join(root_dir, "dist", "MatchTrack-Stitcher")
    if not os.path.exists(dist_app):
        print(f"\n[ERROR] Expected output directory not found: {dist_app}")
        sys.exit(1)

    # 3. Verify key bundled files
    print("\nVerifying bundled application assets...")
    exe_path = os.path.join(dist_app, "MatchTrack-Stitcher.exe")
    assert os.path.exists(exe_path), f"Executable missing: {exe_path}"
    print(f"[OK] Executable verified: {exe_path}")

    # Ensure ffmpeg.exe is in app folder
    ffmpeg_target = os.path.join(dist_app, "ffmpeg.exe")
    if not os.path.exists(ffmpeg_target):
        if os.path.exists(r"bin\ffmpeg.exe"):
            shutil.copy2(r"bin\ffmpeg.exe", ffmpeg_target)
            print("[OK] Bundled bin\\ffmpeg.exe into app folder")
        elif os.path.exists(r"C:\ffmpeg\bin\ffmpeg.exe"):
            shutil.copy2(r"C:\ffmpeg\bin\ffmpeg.exe", ffmpeg_target)
            print("[OK] Bundled C:\\ffmpeg\\bin\\ffmpeg.exe into app folder")

    # Ensure ffprobe.exe is in app folder
    ffprobe_target = os.path.join(dist_app, "ffprobe.exe")
    if not os.path.exists(ffprobe_target):
        if os.path.exists(r"bin\ffprobe.exe"):
            shutil.copy2(r"bin\ffprobe.exe", ffprobe_target)
            print("[OK] Bundled bin\\ffprobe.exe into app folder")
        elif os.path.exists(r"C:\ffmpeg\bin\ffprobe.exe"):
            shutil.copy2(r"C:\ffmpeg\bin\ffprobe.exe", ffprobe_target)
            print("[OK] Bundled C:\\ffmpeg\\bin\\ffprobe.exe into app folder")

    # Ensure all AI models are in app folder
    for pt_model in ["yolo11m_football_player.pt", "yolo11n_football_ball.pt", "yolo11n.pt", "yolov8n.pt"]:
        model_src = os.path.join(root_dir, pt_model)
        model_target = os.path.join(dist_app, pt_model)
        if os.path.exists(model_src) and not os.path.exists(model_target):
            shutil.copy2(model_src, model_target)
            print(f"[OK] Bundled {pt_model} into app folder")

    # Ensure JSON configs are in app folder
    for json_file in ["default_rig_action4_80deg.json", "dji_action4_1080p_dewarp_rig.json", "user_default_settings.json", "dji_action4_1080p_2.json"]:
        src_json = os.path.join(root_dir, json_file)
        dst_json = os.path.join(dist_app, json_file)
        if os.path.exists(src_json) and not os.path.exists(dst_json):
            shutil.copy2(src_json, dst_json)
            print(f"[OK] Bundled {json_file}")

    # 4. Build Windows Setup Installer with Inno Setup
    print("\n[Step 2/2] Compiling Windows Setup Installer with Inno Setup...")
    iscc_bin = find_iscc()
    if not iscc_bin:
        print("[ERROR] Inno Setup compiler (ISCC.exe) not found.")
        sys.exit(1)

    print(f"Using Inno Setup Compiler: {iscc_bin}")
    iss_script = os.path.join(root_dir, "installer.iss")
    res_inno = subprocess.run([iscc_bin, iss_script])
    if res_inno.returncode != 0:
        print("\n[ERROR] Inno Setup compilation failed.")
        sys.exit(1)

    setup_exes = glob.glob(os.path.join(root_dir, "installer_output", "MatchTrack-Stitcher-Setup-*.exe"))
    if not setup_exes:
        print(f"\n[ERROR] Expected setup installer not found in installer_output")
        sys.exit(1)

    setup_exe = setup_exes[0]
    setup_size_mb = os.path.getsize(setup_exe) / (1024 * 1024)
    print("\n==========================================================")
    print("WINDOWS INSTALLER ERFOLGREICH ERSTELLT!")
    print(f"Setup Installer Datei: {setup_exe}")
    print(f"Dateigroesse:          {setup_size_mb:.1f} MB")
    print(f"Enthaelt: Python-Runtime, PySide6 GUI, PyTorch, YOLOv8-KI, OpenCV, FFmpeg HW-Encoder")
    print("==========================================================")


if __name__ == "__main__":
    build()

