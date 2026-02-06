"""Audio playback utilities for fine-tuning workflow."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)


def play_audio(filepath: str | Path, blocking: bool = True) -> None:
    """Play a WAV file through the default output device."""
    import sounddevice as sd

    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Audio file not found: {filepath}")

    data, sample_rate = sf.read(str(filepath), dtype="float32")
    logger.info("Playing %s (%.1fs)", filepath.name, len(data) / sample_rate)
    sd.play(data, samplerate=sample_rate)
    if blocking:
        sd.wait()


def load_audio(filepath: str | Path, target_sr: int = 16000) -> np.ndarray:
    """Load an audio file and return as numpy array at the target sample rate."""
    filepath = Path(filepath)
    data, sr = sf.read(str(filepath), dtype="float32")

    # Ensure mono
    if data.ndim > 1:
        data = data.mean(axis=1)

    # Simple resampling if needed (for production, use librosa or torchaudio)
    if sr != target_sr:
        logger.warning(
            "Sample rate mismatch: file=%d, target=%d. Basic resampling applied.",
            sr,
            target_sr,
        )
        ratio = target_sr / sr
        n_samples = int(len(data) * ratio)
        indices = np.linspace(0, len(data) - 1, n_samples)
        data = np.interp(indices, np.arange(len(data)), data).astype(np.float32)

    return data


def get_audio_duration(filepath: str | Path) -> float:
    """Return the duration of an audio file in seconds."""
    info = sf.info(str(filepath))
    return info.duration
