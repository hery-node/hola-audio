# Hola Audio

**Cross-platform voice input application with offline speech recognition.**

Hola Audio provides a system-wide voice input tool that runs entirely offline using NVIDIA's [Canary-Qwen-2.5B](https://huggingface.co/nvidia/canary-qwen-2.5b) model. Press a hotkey, speak, and get your speech converted to text — no internet required.

## Features

- **Offline Speech Recognition** — Local, GPU-accelerated ASR using Canary-Qwen-2.5B (state-of-the-art English STT)
- **Global Hotkeys** — Customizable keyboard shortcuts to start/stop recording from any application
- **Cross-Platform** — Works on Windows, Linux, and macOS
- **Fine-Tuning** — Build a personal voice profile with guided recording sessions to improve accuracy
- **LLM Correction** — Optional online text correction via configurable LLM endpoint (OpenAI-compatible)
- **System Tray** — Persistent tray icon with recording status and quick access menu
- **Clipboard Integration** — Transcriptions automatically copied to clipboard

## Architecture

```
┌──────────────────────────────────────────────────┐
│                   Hola Audio                      │
├──────────┬───────────┬───────────┬───────────────┤
│  Hotkey  │   Audio   │    ASR    │  Correction   │
│  Manager │  Capture  │  Engine   │   Client      │
│ (pynput) │(sounddev) │  (NeMo)  │  (LLM API)    │
├──────────┴───────────┴───────────┴───────────────┤
│           Fine-tuning Pipeline                    │
│     (Dataset collection + LoRA training)          │
├──────────────────────────────────────────────────┤
│        UI: System Tray + CLI                      │
└──────────────────────────────────────────────────┘
```

## Requirements

- **Python** 3.10+
- **NVIDIA GPU** with CUDA support (for local ASR inference)
  - Minimum: 6 GB VRAM (e.g., RTX 3060)
  - Recommended: 8+ GB VRAM (e.g., RTX 3070/4070+)
- **System libraries**: PortAudio, libsndfile

## Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/hery-node/hola-audio.git
cd hola-audio

# Linux/macOS — interactive installer
./scripts/install.sh

# Windows
scripts\install.bat

# Or manual install
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

pip install -e ".[gpu,tray,correction]"
```

### System Dependencies

**Linux (Debian/Ubuntu):**

```bash
sudo apt-get install portaudio19-dev libsndfile1 xclip
```

**Linux (Fedora):**

```bash
sudo dnf install portaudio-devel libsndfile xclip
```

**macOS:**

```bash
brew install portaudio libsndfile
```

**Windows:** No extra system packages needed — pip handles everything.

### 2. Start the Service

```bash
hola-audio start
```

This will:

- Load the Canary-Qwen-2.5B model (downloads ~5 GB on first run)
- Register global hotkeys
- Show a system tray icon

### 3. Use It

| Action            | Default Hotkey |
| ----------------- | -------------- |
| Toggle recording  | `Ctrl+Shift+H` |
| Cancel recording  | `Escape`       |
| Correct clipboard | `Ctrl+Shift+J` |

Press the recording hotkey, speak, press again — text appears in your clipboard.

## CLI Reference

```bash
# Start the voice input service
hola-audio start
hola-audio start --no-model    # Start without preloading model
hola-audio start --no-tray     # Start without system tray

# Transcribe audio files
hola-audio transcribe recording.wav
hola-audio transcribe *.wav --correct         # With LLM correction
hola-audio transcribe audio.wav --json        # JSON output
hola-audio transcribe audio.wav --checkpoint model.nemo  # Use fine-tuned model

# Correct text via LLM
hola-audio correct --text "Uncorrected text here"
hola-audio correct --file transcript.txt
echo "text" | hola-audio correct

# Fine-tuning workflow
hola-audio finetune init                       # Initialize sentence bank
hola-audio finetune init --sentences-file my_sentences.txt
hola-audio finetune status                     # Check progress
hola-audio finetune record                     # Record next sentence
hola-audio finetune train                      # Run local fine-tuning
hola-audio finetune train --mode online        # Use cloud fine-tuning
hola-audio finetune export                     # Export NeMo manifest
hola-audio finetune reset                      # Reset recordings

# Audio devices
hola-audio devices

# Configuration
hola-audio config show                         # Show all settings
hola-audio config get audio.sample_rate        # Get a specific value
hola-audio config set hotkey.toggle_recording "<ctrl>+<alt>+r"
hola-audio config path                         # Show config/data directories
hola-audio config init                         # Create user config file
```

## Configuration

Hola Audio uses a layered configuration system:

1. **Defaults** — Built-in `config/default.yaml`
2. **User config** — `~/.config/hola-audio/config.yaml` (Linux), `~/Library/Application Support/hola-audio/config.yaml` (macOS), `%APPDATA%\hola-audio\config.yaml` (Windows)
3. **Environment variables** — `HOLA_AUDIO__SECTION__KEY=value`

### Key Settings

```yaml
# Hotkeys
hotkey:
  toggle_recording: "<ctrl>+<shift>+h"
  cancel_recording: "<escape>"
  correct_clipboard: "<ctrl>+<shift>+j"

# Audio capture
audio:
  sample_rate: 16000
  silence_duration: 3.0 # Auto-stop after 3s of silence (0 = disabled)
  max_duration: 120 # Max recording length in seconds

# ASR engine
asr:
  model_name: "nvidia/canary-qwen-2.5b"
  device: "auto" # auto, cuda, cpu
  compute_type: "float16" # float16, float32, bfloat16

# LLM correction (OpenAI-compatible endpoint)
correction:
  enabled: false
  endpoint: "https://your-server.com/v1/chat/completions"
  api_key: "your-api-key"
  model: "gemini-2.0-flash"

# Clipboard behavior
clipboard:
  auto_copy: true
```

### Environment Variable Overrides

```bash
export HOLA_AUDIO__CORRECTION__ENDPOINT="https://api.example.com/v1/chat/completions"
export HOLA_AUDIO__CORRECTION__API_KEY="sk-..."
export HOLA_AUDIO__CORRECTION__ENABLED="true"
export HOLA_AUDIO__ASR__DEVICE="cuda"
```

## Fine-Tuning Guide

Hola Audio includes a guided fine-tuning workflow to improve recognition accuracy for your voice:

### Step 1: Initialize the Sentence Bank

```bash
hola-audio finetune init
# Or with custom sentences:
hola-audio finetune init --sentences-file my_sentences.txt
```

This creates a bank of 15 sample sentences covering different speech patterns.

### Step 2: Record Your Voice

```bash
hola-audio finetune record
```

The CLI will display each sentence and guide you through recording. Repeat until all sentences are recorded:

```bash
hola-audio finetune status
# Dataset: 15 sentences, 8 recorded
# Progress: 53%
```

### Step 3: Train

**Offline** (requires GPU with 8+ GB VRAM):

```bash
hola-audio finetune train
```

**Online** (cloud-based, configure endpoint first):

```bash
hola-audio finetune train --mode online
```

### Step 4: Use Your Fine-Tuned Model

```bash
hola-audio transcribe audio.wav --checkpoint path/to/finetuned_model.nemo
```

## Project Structure

```
hola-audio/
├── config/
│   └── default.yaml              # Default configuration
├── scripts/
│   ├── install.sh                # Linux/macOS installer
│   └── install.bat               # Windows installer
├── src/hola_audio/
│   ├── __init__.py
│   ├── __main__.py               # python -m hola_audio
│   ├── app.py                    # Main application controller
│   ├── cli.py                    # CLI interface
│   ├── config.py                 # Configuration management
│   ├── audio/
│   │   ├── capture.py            # Cross-platform audio recording
│   │   └── player.py             # Audio playback utilities
│   ├── asr/
│   │   └── engine.py             # Canary-Qwen-2.5B ASR engine
│   ├── hotkey/
│   │   └── manager.py            # Global hotkey management
│   ├── finetune/
│   │   ├── dataset.py            # Sample sentence bank & data collection
│   │   └── trainer.py            # LoRA fine-tuning (offline/online)
│   ├── correction/
│   │   └── client.py             # LLM correction client
│   └── ui/
│       └── tray.py               # System tray icon
├── tests/
│   ├── test_config.py
│   ├── test_audio.py
│   ├── test_asr.py
│   ├── test_finetune.py
│   └── test_correction.py
├── pyproject.toml
├── LICENSE                        # Apache 2.0
└── README.md
```

## Development

```bash
# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest

# Run with coverage
pytest --cov=hola_audio --cov-report=term-missing

# Lint
ruff check src/ tests/

# Format
ruff format src/ tests/
```

## Supported Platforms

| Platform        | Audio | Hotkeys | System Tray | ASR (GPU) | ASR (CPU) |
| --------------- | ----- | ------- | ----------- | --------- | --------- |
| Linux (X11)     | ✅    | ✅      | ✅          | ✅        | ✅\*      |
| Linux (Wayland) | ✅    | ⚠️      | ✅          | ✅        | ✅\*      |
| macOS           | ✅    | ✅      | ✅          | ✅        | ✅\*      |
| Windows         | ✅    | ✅      | ✅          | ✅        | ✅\*      |

\* CPU inference is supported but significantly slower. GPU with 6+ GB VRAM recommended.

⚠️ Wayland: Global hotkeys may require additional permissions or the use of X11 compatibility layer.

## Model Details

**Canary-Qwen-2.5B** is a Speech-Augmented Language Model (SALM) combining:

- **FastConformer** encoder (audio processing)
- **Qwen3-1.7B** LLM (text generation)
- Connected via linear projection + LoRA

Key specs:

- Parameters: 2.5B
- Speed: 418 RTFx (real-time factor)
- Input: 16 kHz mono audio (WAV/FLAC)
- Best WER: 1.60% (LibriSpeech clean)
- License: CC-BY-4.0

## License

Apache License 2.0 — see [LICENSE](LICENSE) for details.

The Canary-Qwen-2.5B model is released under [CC-BY-4.0](https://choosealicense.com/licenses/cc-by-4.0/).
