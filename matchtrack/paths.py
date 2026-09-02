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
    """Resolves relative paths to assets, models, or config files in dev and frozen modes."""
    # 1. Check in PyInstaller bundle directory
    if getattr(sys, 'frozen', False):
        base = getattr(sys, '_MEIPASS', os.path.dirname(sys.executable))
        p = os.path.join(base, relative_path)
        if os.path.exists(p):
            return p
        p_exec = os.path.join(os.path.dirname(sys.executable), relative_path)
        if os.path.exists(p_exec):
            return p_exec

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    p_root = os.path.join(root, relative_path)
    if os.path.exists(p_root):
        return p_root
        
    p_pkg = os.path.join(os.path.dirname(os.path.abspath(__file__)), relative_path)
    if os.path.exists(p_pkg):
        return p_pkg

    return p_root


def get_ffmpeg_path() -> str:
    """Returns the absolute path to ffmpeg executable, checking bundled binaries first."""
    # 1. Check PyInstaller _MEIPASS or executable directory
    if getattr(sys, 'frozen', False):
        base = getattr(sys, '_MEIPASS', os.path.dirname(sys.executable))
        for cand in [
            os.path.join(base, "ffmpeg.exe"),
            os.path.join(base, "bin", "ffmpeg.exe"),
            os.path.join(os.path.dirname(sys.executable), "ffmpeg.exe"),
            os.path.join(os.path.dirname(sys.executable), "bin", "ffmpeg.exe")
        ]:
            if os.path.exists(cand):
                return cand

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for cand in [
        os.path.join(root, "ffmpeg.exe"),
        os.path.join(root, "bin", "ffmpeg.exe"),
        r"C:\ffmpeg\bin\ffmpeg.exe"
    ]:
        if os.path.exists(cand):
            return cand

    which_ffmpeg = shutil.which("ffmpeg")
    if which_ffmpeg:
        return which_ffmpeg

    return "ffmpeg"


def get_ffprobe_path() -> str:
    """Returns the absolute path to ffprobe executable."""
    if getattr(sys, 'frozen', False):
        base = getattr(sys, '_MEIPASS', os.path.dirname(sys.executable))
        for cand in [
            os.path.join(base, "ffprobe.exe"),
            os.path.join(base, "bin", "ffprobe.exe"),
            os.path.join(os.path.dirname(sys.executable), "ffprobe.exe")
        ]:
            if os.path.exists(cand):
                return cand

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for cand in [
        os.path.join(root, "ffprobe.exe"),
        os.path.join(root, "bin", "ffprobe.exe"),
        r"C:\ffmpeg\bin\ffprobe.exe"
    ]:
        if os.path.exists(cand):
            return cand

    which_ffprobe = shutil.which("ffprobe")
    if which_ffprobe:
        return which_ffprobe

    return "ffprobe"


def get_log_file_path() -> str:
    """Returns the absolute path to matchtrack.log."""
    if getattr(sys, 'frozen', False):
        app_dir = os.path.dirname(sys.executable)
        test_file = os.path.join(app_dir, "matchtrack.log")
        try:
            with open(test_file, "a", encoding="utf-8") as f:
                pass
            return test_file
        except Exception:
            local_app = os.environ.get("LOCALAPPDATA", os.path.expanduser("~"))
            log_dir = os.path.join(local_app, "MatchTrack")
            os.makedirs(log_dir, exist_ok=True)
            return os.path.join(log_dir, "matchtrack.log")

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(root, "matchtrack.log")


def get_default_settings_path() -> str:
    """Returns the absolute path to user_default_settings.json."""
    if getattr(sys, 'frozen', False):
        app_dir = os.path.dirname(sys.executable)
        test_file = os.path.join(app_dir, "user_default_settings.json")
        try:
            with open(test_file, "a", encoding="utf-8") as f:
                pass
            return test_file
        except Exception:
            local_app = os.environ.get("LOCALAPPDATA", os.path.expanduser("~"))
            cfg_dir = os.path.join(local_app, "MatchTrack")
            os.makedirs(cfg_dir, exist_ok=True)
            return os.path.join(cfg_dir, "user_default_settings.json")

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(root, "user_default_settings.json")

