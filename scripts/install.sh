#!/usr/bin/env bash
# Install script for Hola Audio on Linux/macOS
set -euo pipefail

echo "=== Hola Audio Installer ==="
echo ""

# Detect OS
OS="$(uname -s)"
case "$OS" in
    Linux)   PLATFORM="linux"   ;;
    Darwin)  PLATFORM="macos"   ;;
    *)       echo "Unsupported OS: $OS"; exit 1 ;;
esac
echo "Detected platform: $PLATFORM"

# Check Python version
PYTHON=""
for cmd in python3.12 python3.11 python3.10 python3; do
    if command -v "$cmd" &>/dev/null; then
        ver=$("$cmd" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
        major=$(echo "$ver" | cut -d. -f1)
        minor=$(echo "$ver" | cut -d. -f2)
        if [ "$major" -ge 3 ] && [ "$minor" -ge 10 ]; then
            PYTHON="$cmd"
            break
        fi
    fi
done

if [ -z "$PYTHON" ]; then
    echo "Error: Python 3.10+ is required but not found."
    echo "Please install Python 3.10 or later."
    exit 1
fi
echo "Using Python: $PYTHON ($($PYTHON --version))"

# Install system dependencies
echo ""
echo "--- Installing system dependencies ---"
if [ "$PLATFORM" = "linux" ]; then
    if command -v apt-get &>/dev/null; then
        sudo apt-get update -qq
        sudo apt-get install -y -qq portaudio19-dev libsndfile1 xclip
    elif command -v dnf &>/dev/null; then
        sudo dnf install -y portaudio-devel libsndfile xclip
    elif command -v pacman &>/dev/null; then
        sudo pacman -S --noconfirm portaudio libsndfile xclip
    else
        echo "Warning: Could not detect package manager. Please install portaudio and libsndfile manually."
    fi
elif [ "$PLATFORM" = "macos" ]; then
    if command -v brew &>/dev/null; then
        brew install portaudio libsndfile
    else
        echo "Warning: Homebrew not found. Please install portaudio and libsndfile manually."
    fi
fi

# Create virtual environment
echo ""
echo "--- Setting up virtual environment ---"
VENV_DIR=".venv"
if [ ! -d "$VENV_DIR" ]; then
    $PYTHON -m venv "$VENV_DIR"
    echo "Created virtual environment: $VENV_DIR"
else
    echo "Virtual environment already exists: $VENV_DIR"
fi

# Activate and install
source "$VENV_DIR/bin/activate"

echo ""
echo "--- Installing hola-audio ---"
pip install --upgrade pip setuptools wheel
pip install -e ".[dev]"

# Optional: GPU support
echo ""
read -p "Install GPU support (requires NVIDIA GPU + CUDA)? [y/N] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Installing NeMo + PyTorch with CUDA support..."
    pip install -e ".[gpu]"
fi

# Optional: System tray
echo ""
read -p "Install system tray support? [Y/n] " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Nn]$ ]]; then
    pip install -e ".[tray]"
fi

# Optional: Correction
echo ""
read -p "Install LLM correction support? [Y/n] " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Nn]$ ]]; then
    pip install -e ".[correction]"
fi

echo ""
echo "=== Installation complete! ==="
echo ""
echo "Activate the environment:  source $VENV_DIR/bin/activate"
echo "Start the service:         hola-audio start"
echo "Transcribe a file:         hola-audio transcribe audio.wav"
echo "Show configuration:        hola-audio config show"
echo "List audio devices:        hola-audio devices"
echo ""
