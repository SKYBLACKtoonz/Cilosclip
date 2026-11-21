@echo off
rem Command line interface. Examples:
rem   clipforge.bat run video.mp4
rem   clipforge.bat check
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo First run: setting up ClipForge...
    call setup.bat
    if not exist ".venv\Scripts\python.exe" exit /b 1
)

".venv\Scripts\python.exe" -m clipforge %*
