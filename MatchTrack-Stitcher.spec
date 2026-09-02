# -*- mode: python ; coding: utf-8 -*-
import os
import sys
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

# Collect package data and submodules
datas = [
    ('matchtrack/assets', 'matchtrack/assets'),
    ('default_rig_action4_80deg.json', '.'),
    ('dji_action4_1080p_dewarp_rig.json', '.'),
    ('user_default_settings.json', '.'),
    ('dji_action4_1080p_2.json', '.'),
]

for pt_model in ['yolo11m_football_player.pt', 'yolo11n_football_ball.pt', 'yolo11n.pt', 'yolov8n.pt']:
    if os.path.exists(pt_model):
        datas.append((pt_model, '.'))

if os.path.exists('bin/ffmpeg.exe'):
    datas.append(('bin/ffmpeg.exe', '.'))
if os.path.exists('bin/ffprobe.exe'):
    datas.append(('bin/ffprobe.exe', '.'))

# Add ultralytics default assets if present
try:
    import ultralytics
    datas += collect_data_files('ultralytics')
except Exception:
    pass

hiddenimports = [
    'PySide6.QtCore',
    'PySide6.QtGui',
    'PySide6.QtWidgets',
    'cv2',
    'torch',
    'torchvision',
    'ultralytics',
    'scipy',
    'scipy.signal',
    'scipy.optimize',
    'scipy.spatial',
    'numpy',
    'PIL',
    'PIL.Image',
]
hiddenimports += collect_submodules('matchtrack')

a = Analysis(
    ['main.py'],
    pathex=['.'],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['tkinter'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# 1. Standalone Onedir Distribution (recommended for large PyTorch/CUDA apps to ensure instant startup)
exe_dir = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='MatchTrack-Stitcher',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='matchtrack/assets/icon.ico'
)

coll = COLLECT(
    exe_dir,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='MatchTrack-Stitcher',
)


