@echo off
title MatchTrack-Stitcher
echo Starting MatchTrack-Stitcher...
cd /d "%~dp0"
setlocal

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" main.py %*
) else (
    py -3.11 main.py %* 2>nul || python main.py %*
)

pause
