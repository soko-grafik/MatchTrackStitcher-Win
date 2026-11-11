# -*- mode: python ; coding: utf-8 -*-
import os
import sys
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

# Collect package data and submodules into structured directories
datas = [
    ('matchtrack/assets', 'matchtrack/assets'),
    ('configs', 'configs'),
    ('models', 'models'),
]

if os.path.exists('bin/ffmpeg.exe'):
    datas.append(('bin/ffmpeg.exe', 'bin'))
if os.path.exists('bin/ffprobe.exe'):
    datas.append(('bin/ffprobe.exe', 'bin'))

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
    'requests',
    'certifi',
    'bleak',
    'qrcode',
    'pygrabber',
    'pygrabber.dshow_graph',
    'comtypes',
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


