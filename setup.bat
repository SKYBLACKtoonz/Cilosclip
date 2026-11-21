@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

echo.
echo  ============================================
echo   ClipForge setup
echo  ============================================
echo.

where python >nul 2>nul
if errorlevel 1 (
    echo [!] Python was not found on PATH.
    echo.
    echo     Install Python 3.10 or newer from https://www.python.org/downloads/
    echo     IMPORTANT: tick "Add python.exe to PATH" during installation.
    echo.
    pause
    exit /b 1
)

for /f "tokens=2" %%v in ('python --version 2^>^&1') do set PYVER=%%v
echo [1/3] Python %PYVER% found.

if not exist ".venv\Scripts\python.exe" (
    echo [2/3] Creating virtual environment in .venv ...
    python -m venv .venv
    if errorlevel 1 (
        echo [!] Could not create the virtual environment.
        pause
        exit /b 1
    )
) else (
    echo [2/3] Virtual environment already exists.
)

call ".venv\Scripts\activate.bat"

echo       Installing ClipForge. This downloads a few hundred MB the first time.
echo.
python -m pip install --upgrade pip --quiet

echo [2/3] Installing core app + desktop UI ...
python -m pip install -e ".[gui]"
if errorlevel 1 (
    echo.
    echo [!] Core install failed.
    pause
    exit /b 1
)

echo.
echo       Installing local speech-to-text (faster-whisper) ...
python -m pip install -e ".[local]"
if errorlevel 1 (
    echo.
    echo [!] faster-whisper did not install on this Python version.
    echo     Local transcription will be unavailable, but ClipForge still works.
    echo     Options: install Python 3.11 or 3.12 and rerun setup.bat,
    echo     or use the cloud API by setting OPENAI_API_KEY.
    echo.
)

echo.
echo       Installing face tracking (OpenCV) ...
python -m pip install -e ".[vision]"
if errorlevel 1 (
    echo.
    echo [!] OpenCV did not install. Clips will use a centred crop instead.
    echo.
)

echo.
echo [3/3] Checking ffmpeg ...
where ffmpeg >nul 2>nul
if errorlevel 1 (
    echo.
    echo [!] ffmpeg was not found on PATH. Clips cannot be rendered without it.
    echo.
    echo     Install it, then close and reopen this window:
    echo         winget install Gyan.FFmpeg
    echo.
) else (
    echo       ffmpeg found.
)

echo.
python -m clipforge check
echo.
echo  Setup complete. Double-click run.bat to open ClipForge.
echo.
pause
