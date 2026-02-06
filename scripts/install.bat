@echo off
REM Install script for Hola Audio on Windows
setlocal enabledelayedexpansion

echo === Hola Audio Installer ===
echo.

REM Check Python
where python >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Error: Python not found in PATH.
    echo Please install Python 3.10+ from https://www.python.org/
    exit /b 1
)

for /f "tokens=*" %%v in ('python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"') do set PYVER=%%v
echo Using Python %PYVER%

REM Create virtual environment
set VENV_DIR=.venv
if not exist "%VENV_DIR%" (
    echo --- Creating virtual environment ---
    python -m venv %VENV_DIR%
) else (
    echo Virtual environment already exists: %VENV_DIR%
)

REM Activate
call %VENV_DIR%\Scripts\activate.bat

echo.
echo --- Installing hola-audio ---
pip install --upgrade pip setuptools wheel
pip install -e ".[dev]"

echo.
set /p GPU="Install GPU support (requires NVIDIA GPU + CUDA)? [y/N]: "
if /i "%GPU%"=="y" (
    echo Installing NeMo + PyTorch with CUDA support...
    pip install -e ".[gpu]"
)

echo.
set /p TRAY="Install system tray support? [Y/n]: "
if /i not "%TRAY%"=="n" (
    pip install -e ".[tray]"
)

echo.
set /p CORR="Install LLM correction support? [Y/n]: "
if /i not "%CORR%"=="n" (
    pip install -e ".[correction]"
)

echo.
echo === Installation complete! ===
echo.
echo Activate the environment:  %VENV_DIR%\Scripts\activate.bat
echo Start the service:         hola-audio start
echo Transcribe a file:         hola-audio transcribe audio.wav
echo Show configuration:        hola-audio config show
echo List audio devices:        hola-audio devices
echo.

endlocal
