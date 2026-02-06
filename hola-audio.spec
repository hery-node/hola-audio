# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec file for Hola Audio Windows build."""

import os
import sys
from pathlib import Path

block_cipher = None

# Project root
ROOT = Path(SPECPATH)

a = Analysis(
    [str(ROOT / "src" / "hola_audio" / "cli.py")],
    pathex=[str(ROOT / "src")],
    binaries=[],
    datas=[
        # Bundle default config
        (str(ROOT / "config" / "default.yaml"), "config"),
    ],
    hiddenimports=[
        "hola_audio",
        "hola_audio.app",
        "hola_audio.cli",
        "hola_audio.config",
        "hola_audio.audio",
        "hola_audio.audio.capture",
        "hola_audio.audio.player",
        "hola_audio.asr",
        "hola_audio.asr.engine",
        "hola_audio.hotkey",
        "hola_audio.hotkey.manager",
        "hola_audio.finetune",
        "hola_audio.finetune.dataset",
        "hola_audio.finetune.trainer",
        "hola_audio.correction",
        "hola_audio.correction.client",
        "hola_audio.ui",
        "hola_audio.ui.tray",
        "pynput.keyboard._win32",
        "pynput.mouse._win32",
        "pystray._win32",
        "_sounddevice_data",
        "sounddevice",
        "soundfile",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Exclude heavy ML libs from the exe — user installs them separately
        "torch",
        "torchaudio",
        "nemo",
        "nemo_toolkit",
        "pytorch_lightning",
        "omegaconf",
        "transformers",
        "tensorflow",
        "tensorboard",
    ],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="hola-audio",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,  # Console app for CLI; set False for tray-only mode
    disable_windowed_traceback=False,
    argv_emulation=False,
    icon=None,  # Add .ico path here for custom icon
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="hola-audio",
)
