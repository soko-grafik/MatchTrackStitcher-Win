"""
Path and resource resolution utilities for MatchTrack-Stitcher.
Handles development runs and PyInstaller frozen standalone bundles.
"""
import os
import sys
import shutil


def get_base_dir() -> str:
    """Returns the base application directory."""
    if getattr(sys, 'frozen', False):
        return getattr(sys, '_MEIPASS', os.path.dirname(sys.executable))
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_resource_path(relative_path: str) -> str:
    """
    Resolves relative paths to assets, models, or config files in dev and frozen modes.
    Intelligently checks dedicated subdirectories:
    - models/ (for .pt and .onnx models)
    - configs/ (for .json presets and default configurations)
    - bin/ (for helper executables)
    - matchtrack/assets/ (for UI icons and branding)
    """
    candidates_sub = [
        relative_path,
        os.path.join("models", relative_path),
        os.path.join("configs", relative_path),
        os.path.join("bin", relative_path),
        os.path.join("matchtrack", "assets", relative_path),
    ]

    # 1. Check in PyInstaller bundle directory or installed executable directory
    if getattr(sys, 'frozen', False):
        base = getattr(sys, '_MEIPASS', os.path.dirname(sys.executable))
        app_dir = os.path.dirname(sys.executable)
        for root_cand in [app_dir, base]:
            for sub in candidates_sub:
                p = os.path.join(root_cand, sub)
                if os.path.exists(p):
                    return p

    # 2. Check in Dev Workspace Root
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for sub in candidates_sub:
        p = os.path.join(root, sub)
        if os.path.exists(p):
            return p

    p_pkg = os.path.join(os.path.dirname(os.path.abspath(__file__)), relative_path)
    if os.path.exists(p_pkg):
        return p_pkg

    return os.path.join(root, relative_path)


def get_ffmpeg_path() -> str:
    """Returns the absolute path to ffmpeg executable, checking bin/ subdirectories first."""
    # 1. Check PyInstaller _MEIPASS or executable directory
    if getattr(sys, 'frozen', False):
        base = getattr(sys, '_MEIPASS', os.path.dirname(sys.executable))
        app_dir = os.path.dirname(sys.executable)
        for cand in [
            os.path.join(app_dir, "bin", "ffmpeg.exe"),
            os.path.join(app_dir, "ffmpeg.exe"),
            os.path.join(base, "bin", "ffmpeg.exe"),
            os.path.join(base, "ffmpeg.exe"),
        ]:
            if os.path.exists(cand):
                return cand

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for cand in [
        os.path.join(root, "bin", "ffmpeg.exe"),
        os.path.join(root, "ffmpeg.exe"),
        r"C:\ffmpeg\bin\ffmpeg.exe"
    ]:
        if os.path.exists(cand):
            return cand

    which_ffmpeg = shutil.which("ffmpeg")
    if which_ffmpeg:
        return which_ffmpeg

    return "ffmpeg"


def get_ffprobe_path() -> str:
    """Returns the absolute path to ffprobe executable, checking bin/ subdirectories first."""
    if getattr(sys, 'frozen', False):
        base = getattr(sys, '_MEIPASS', os.path.dirname(sys.executable))
        app_dir = os.path.dirname(sys.executable)
        for cand in [
            os.path.join(app_dir, "bin", "ffprobe.exe"),
            os.path.join(app_dir, "ffprobe.exe"),
            os.path.join(base, "bin", "ffprobe.exe"),
            os.path.join(base, "ffprobe.exe"),
        ]:
            if os.path.exists(cand):
                return cand

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for cand in [
        os.path.join(root, "bin", "ffprobe.exe"),
        os.path.join(root, "ffprobe.exe"),
        r"C:\ffmpeg\bin\ffprobe.exe"
    ]:
        if os.path.exists(cand):
            return cand

    which_ffprobe = shutil.which("ffprobe")
    if which_ffprobe:
        return which_ffprobe

    return "ffprobe"


def get_log_file_path() -> str:
    """Returns the absolute path to logs/matchtrack.log, creating logs/ folder automatically."""
    if getattr(sys, 'frozen', False):
        app_dir = os.path.dirname(sys.executable)
        log_dir = os.path.join(app_dir, "logs")
        try:
            os.makedirs(log_dir, exist_ok=True)
            test_file = os.path.join(log_dir, "matchtrack.log")
            with open(test_file, "a", encoding="utf-8") as f:
                pass
            return test_file
        except Exception:
            local_app = os.environ.get("LOCALAPPDATA", os.path.expanduser("~"))
            log_dir = os.path.join(local_app, "MatchTrack", "logs")
            os.makedirs(log_dir, exist_ok=True)
            return os.path.join(log_dir, "matchtrack.log")

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    log_dir = os.path.join(root, "logs")
    os.makedirs(log_dir, exist_ok=True)
    return os.path.join(log_dir, "matchtrack.log")


def get_crash_log_file_path() -> str:
    """Returns the absolute path to logs/matchtrack_crash.log."""
    log_path = get_log_file_path()
    log_dir = os.path.dirname(os.path.abspath(log_path))
    return os.path.join(log_dir, "matchtrack_crash.log")


def get_default_settings_path() -> str:
    """Returns the absolute path to configs/user_default_settings.json."""
    if getattr(sys, 'frozen', False):
        app_dir = os.path.dirname(sys.executable)
        cfg_dir = os.path.join(app_dir, "configs")
        try:
            os.makedirs(cfg_dir, exist_ok=True)
            test_file = os.path.join(cfg_dir, "user_default_settings.json")
            # Migrate from app root if old file exists but target doesn't
            old_file = os.path.join(app_dir, "user_default_settings.json")
            if os.path.exists(old_file) and not os.path.exists(test_file):
                try:
                    shutil.copy2(old_file, test_file)
                except Exception:
                    pass
            return test_file
        except Exception:
            local_app = os.environ.get("LOCALAPPDATA", os.path.expanduser("~"))
            cfg_dir = os.path.join(local_app, "MatchTrack", "configs")
            os.makedirs(cfg_dir, exist_ok=True)
            return os.path.join(cfg_dir, "user_default_settings.json")

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg_dir = os.path.join(root, "configs")
    os.makedirs(cfg_dir, exist_ok=True)
    test_file = os.path.join(cfg_dir, "user_default_settings.json")
    old_file = os.path.join(root, "user_default_settings.json")
    if os.path.exists(old_file) and not os.path.exists(test_file):
        try:
            shutil.copy2(old_file, test_file)
        except Exception:
            pass
    return test_file
    return os.path.join(root, "user_default_settings.json")


def get_cloud_credentials_path() -> str:
    """Returns the absolute path to the DPAPI-encrypted cloud_credentials.dat file."""
    local_app = os.environ.get("LOCALAPPDATA", os.path.expanduser("~"))
    cfg_dir = os.path.join(local_app, "MatchTrack")
    try:
        os.makedirs(cfg_dir, exist_ok=True)
        return os.path.join(cfg_dir, "cloud_credentials.dat")
    except Exception:
        settings_dir = os.path.dirname(os.path.abspath(get_default_settings_path()))
        return os.path.join(settings_dir, "cloud_credentials.dat")


def get_autocam_install_dir() -> str:
    """Returns the Once Sport Autocam installation directory if present."""
    candidates = [
        r"D:\Users\SoKo\AppData\Local\Programs\Autocam",
        os.path.expandvars(r"%LOCALAPPDATA%\Programs\Autocam"),
        os.path.expandvars(r"%PROGRAMFILES%\Autocam")
    ]
    for cand in candidates:
        if os.path.isdir(cand):
            return cand
    return ""


def get_autocam_cli_path() -> str:
    """Returns the absolute path to AutocamCLI.exe if available."""
    install_dir = get_autocam_install_dir()
    if install_dir:
        cli = os.path.join(install_dir, "AutocamCLI.exe")
        if os.path.isfile(cli):
            return cli
    return ""


def get_autocam_models_dir() -> str:
    """Returns the directory containing Once Sport Autocam ONNX models (checks local models/ first)."""
    # 1. Check local models directory in dev workspace or app bundle
    if getattr(sys, 'frozen', False):
        app_dir = os.path.dirname(sys.executable)
        local_models = os.path.join(app_dir, "models")
        if os.path.isdir(local_models) and any(f.endswith(".onnx") for f in os.listdir(local_models)):
            return local_models

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    local_models = os.path.join(root, "models")
    if os.path.isdir(local_models) and any(f.endswith(".onnx") for f in os.listdir(local_models)):
        return local_models

    # 2. Fallback to Once Sport Autocam external installation
    install_dir = get_autocam_install_dir()
    if install_dir:
        mdir = os.path.join(install_dir, "onnx_models")
        if os.path.isdir(mdir):
            return mdir
    return ""


def get_available_ai_models() -> list:
    """
    Discovers and returns all available AI tracking and ball detection models:
    - Native MatchTrack PyTorch/YOLO/RT-DETR models
    - Once Sport Autocam ONNX models (soccer_balldet, basketball_balldet, detect_kpts)
    """
    models = [
        {
            "id": "yolo11",
            "name": "🏆 MatchTrack YOLO11n (Native PyTorch • Ultra-Schnell)",
            "path": get_resource_path("yolo11n_football_ball.pt"),
            "category": "native",
            "type": "ball",
            "desc": "MatchTrack Ball- & Spielertracking mit maximaler Bildrate"
        },
        {
            "id": "rtdetr",
            "name": "🤖 MatchTrack RT-DETR Large (Transformer Attention)",
            "path": get_resource_path("rtdetr-l.pt"),
            "category": "native",
            "type": "multi",
            "desc": "Deformable Attention Transformer mit hoher Erkennungspräzision"
        },
        {
            "id": "pose",
            "name": "🦴 MatchTrack YOLO-Pose Skelett (Extremitäten & VAR)",
            "path": get_resource_path("yolov8n-pose.pt"),
            "category": "native",
            "type": "pose",
            "desc": "17-Keypoint Skeletterkennung für Körperausrichtung und Abseitslinien"
        }
    ]

    autocam_dir = get_autocam_models_dir()
    if autocam_dir and os.path.isdir(autocam_dir):
        label_map = {
            "soccer_balldet_fp16.onnx": ("⚽ Once Autocam Soccer Ball (FP16 ONNX)", "ball", "Original Autocam Fußball-Balldetektor (FP16 Präzision)"),
            "soccer_balldet_int8.onnx": ("⚡ Once Autocam Soccer Ball (INT8 Quantized ONNX)", "ball", "Autocam Fußball-Balldetektor (INT8 High-Speed)")
        }
        for fname in sorted(os.listdir(autocam_dir)):
            if fname.lower().endswith(".onnx"):
                full_path = os.path.join(autocam_dir, fname)
                m_id = "autocam_" + os.path.splitext(fname)[0]
                if fname in label_map:
                    title, m_type, m_desc = label_map[fname]
                else:
                    title = f"📦 Once Autocam {fname} (ONNX)"
                    m_type = "custom"
                    m_desc = f"Autocam ONNX Modell ({fname})"
                models.append({
                    "id": m_id,
                    "name": title,
                    "path": full_path,
                    "category": "autocam_onnx",
                    "type": m_type,
                    "desc": m_desc
                })

    return models



