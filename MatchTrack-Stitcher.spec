# -*- mode: python ; coding: utf-8 -*-
import os
import sys
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

# Collect package data and submodules into structured directories
datas = [
    ('matchtrack/assets', 'matchtrack/assets'),
    ('configs', 'configs'),
]

# Note: bin/ (ffmpeg.exe, ffprobe.exe) and models/ are placed directly into
# the application root folder (dist/MatchTrack-Stitcher/bin and dist/MatchTrack-Stitcher/models)
# by build_exe.py to prevent duplicate copies inside _internal/

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

excludes = [
    'tkinter',
    'tensorflow',
    'tensorboard',
    'tensorboard_data_server',
    'polars',
    'polars_runtime',
    'keras',
    'tf_keras',
    'matplotlib',
    'seaborn',
    'pandas',
    'grpc',
    'grpcio',
    'h5py',
    'onnx2tf',
    'tflite_support',
    'ai_edge_litert',
    'libclang',
    'PySide6.QtWebEngine',
    'PySide6.QtWebEngineCore',
    'PySide6.QtWebEngineWidgets',
    'PySide6.QtPdf',
    'PySide6.QtPdfWidgets',
    'PySide6.QtQml',
    'PySide6.QtQuick',
    'PySide6.QtQuickWidgets',
    'PySide6.Qt3DCore',
    'PySide6.Qt3DRender',
    'PySide6.QtCharts',
    'PySide6.QtDataVisualization',
    'PySide6.QtBluetooth',
    'PySide6.QtSensors',
    'PySide6.QtPositioning',
    'PySide6.QtNfc',
    'PySide6.QtTest',
]

a = Analysis(
    ['main.py'],
    pathex=['.'],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
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


