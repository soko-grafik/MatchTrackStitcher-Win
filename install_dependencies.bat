@echo off
title MatchTrack-Stitcher Setup
echo ===================================================
echo Setting up MatchTrack-Stitcher Python Environment
echo ===================================================

if not exist ".venv" (
    echo Creating virtual environment with Python 3.11...
    "d:\Users\SoKo\AppData\Local\Programs\Python\Python311\python.exe" -m venv .venv
)

echo Installing dependencies (NumPy, SciPy, OpenCV, PySide6)...
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

echo ===================================================
echo Setup finished! You can now start 'run_stitcher.bat'
echo ===================================================
pause
