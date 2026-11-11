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
    for d in ["build", "dist"]:
        if os.path.exists(d):
            print(f"Cleaning {d}/ directory...")
            shutil.rmtree(d, ignore_errors=True)
    os.makedirs("installer_output", exist_ok=True)

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

    # 3. Verify key bundled files and structured directories
    print("\nVerifying bundled application assets...")
    exe_path = os.path.join(dist_app, "MatchTrack-Stitcher.exe")
    assert os.path.exists(exe_path), f"Executable missing: {exe_path}"
    print(f"[OK] Executable verified: {exe_path}")

    # Create structured subdirectories: bin/, models/, configs/, logs/
    for sub in ["bin", "models", "configs", "logs"]:
        os.makedirs(os.path.join(dist_app, sub), exist_ok=True)

    # Ensure bin/ contains ffmpeg.exe and ffprobe.exe
    for b_name in ["ffmpeg.exe", "ffprobe.exe"]:
        dst_b = os.path.join(dist_app, "bin", b_name)
        if not os.path.exists(dst_b):
            src_b = os.path.join(root_dir, "bin", b_name)
            if os.path.exists(src_b):
                shutil.copy2(src_b, dst_b)
                print(f"[OK] Bundled bin/{b_name}")

    # Ensure models/ contains all .pt and .onnx models
    models_dir = os.path.join(root_dir, "models")
    if os.path.exists(models_dir):
        for m in os.listdir(models_dir):
            src_m = os.path.join(models_dir, m)
            dst_m = os.path.join(dist_app, "models", m)
            if os.path.isfile(src_m) and not os.path.exists(dst_m):
                shutil.copy2(src_m, dst_m)
                print(f"[OK] Bundled models/{m}")

    # Ensure configs/ contains all JSON presets and settings
    configs_dir = os.path.join(root_dir, "configs")
    if os.path.exists(configs_dir):
        for c in os.listdir(configs_dir):
            src_c = os.path.join(configs_dir, c)
            dst_c = os.path.join(dist_app, "configs", c)
            if os.path.isfile(src_c) and not os.path.exists(dst_c):
                shutil.copy2(src_c, dst_c)
                print(f"[OK] Bundled configs/{c}")

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

    setup_exes = sorted(
        glob.glob(os.path.join(root_dir, "installer_output", "MatchTrack-Stitcher-Setup-*.exe")),
        key=os.path.getmtime,
        reverse=True
    )
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

