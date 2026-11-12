@echo off
title MatchTrack-Stitcher Setup
echo ===================================================
echo Setting up MatchTrack-Stitcher Python Environment
echo ===================================================

if not exist ".venv" (
    echo Creating virtual environment with Python 3.11...
    py -3.11 -m venv .venv 2>nul || python -m venv .venv
)

echo Installing dependencies (NumPy, SciPy, OpenCV, PySide6)...
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

echo ===================================================
echo Setup finished! You can now start 'run_stitcher.bat'
echo ===================================================
pause
