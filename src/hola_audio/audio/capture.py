"""Cross-platform audio capture using sounddevice.

Supports start/stop toggling, silence-based auto-stop, and saving to WAV files
in the 16 kHz mono format required by Canary-Qwen-2.5B.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable

import numpy as np
import sounddevice as sd
import soundfile as sf

logger = logging.getLogger(__name__)


class AudioCapture:
    """Record audio from the default input device."""

    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        dtype: str = "float32",
        max_duration: float = 120.0,
        silence_threshold: float = 0.01,
        silence_duration: float = 3.0,
        output_dir: str | Path = "",
    ) -> None:
        self.sample_rate = sample_rate
        self.channels = channels
        self.dtype = dtype
        self.max_duration = max_duration
        self.silence_threshold = silence_threshold
        self.silence_duration = silence_duration
        self.output_dir = Path(output_dir) if output_dir else Path.cwd()

        self._frames: list[np.ndarray] = []
        self._recording = False
        self._stream: sd.InputStream | None = None
        self._lock = threading.Lock()
        self._last_sound_time: float = 0.0
        self._start_time: float = 0.0

        # Callbacks
        self.on_recording_start: Callable[[], None] | None = None
        self.on_recording_stop: Callable[[Path], None] | None = None
        self.on_level_update: Callable[[float], None] | None = None

    # -- Properties ------------------------------------------------------------

    @property
    def is_recording(self) -> bool:
        return self._recording

    @property
    def elapsed(self) -> float:
        if not self._recording:
            return 0.0
        return time.monotonic() - self._start_time

    # -- Public API ------------------------------------------------------------

    def start(self) -> None:
        """Begin recording."""
        with self._lock:
            if self._recording:
                logger.warning("Already recording – ignoring start()")
                return
            self._frames.clear()
            self._recording = True
            self._start_time = time.monotonic()
            self._last_sound_time = self._start_time

        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype=self.dtype,
            callback=self._audio_callback,
        )
        self._stream.start()
        logger.info("Recording started (sr=%d, ch=%d)", self.sample_rate, self.channels)

        if self.on_recording_start:
            self.on_recording_start()

    def stop(self) -> Path | None:
        """Stop recording and save to a WAV file. Returns the file path."""
        with self._lock:
            if not self._recording:
                logger.warning("Not recording – ignoring stop()")
                return None
            self._recording = False

        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        if not self._frames:
            logger.warning("No audio frames captured")
            return None

        audio = np.concatenate(self._frames, axis=0)
        filepath = self._save_wav(audio)
        logger.info("Recording saved: %s (%.1fs)", filepath, len(audio) / self.sample_rate)

        if self.on_recording_stop:
            self.on_recording_stop(filepath)

        return filepath

    def cancel(self) -> None:
        """Cancel recording without saving."""
        with self._lock:
            self._recording = False
            self._frames.clear()

        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        logger.info("Recording cancelled")

    def toggle(self) -> Path | None:
        """Toggle recording on/off. Returns file path when stopping."""
        if self.is_recording:
            return self.stop()
        self.start()
        return None

    @staticmethod
    def list_devices() -> list[dict]:
        """List available audio input devices."""
        devices = sd.query_devices()
        inputs = []
        for i, dev in enumerate(devices):
            if dev["max_input_channels"] > 0:  # type: ignore[index]
                inputs.append({"index": i, "name": dev["name"], "channels": dev["max_input_channels"]})  # type: ignore[index]
        return inputs

    # -- Internal --------------------------------------------------------------

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info: Any, status: sd.CallbackFlags) -> None:  # noqa: ANN401
        """Called by sounddevice for each audio block."""
        if status:
            logger.debug("Audio callback status: %s", status)

        if not self._recording:
            return

        self._frames.append(indata.copy())

        # Level meter
        rms = float(np.sqrt(np.mean(indata**2)))
        if self.on_level_update:
            self.on_level_update(rms)

        # Silence detection
        now = time.monotonic()
        if rms > self.silence_threshold:
            self._last_sound_time = now

        if self.silence_duration > 0 and (now - self._last_sound_time) > self.silence_duration:
            logger.info("Silence detected – auto-stopping")
            threading.Thread(target=self.stop, daemon=True).start()
            return

        # Max duration guard
        if self.max_duration > 0 and self.elapsed > self.max_duration:
            logger.info("Max duration reached – auto-stopping")
            threading.Thread(target=self.stop, daemon=True).start()

    def _save_wav(self, audio: np.ndarray) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        filename = f"recording_{uuid.uuid4().hex[:8]}.wav"
        filepath = self.output_dir / filename
        sf.write(str(filepath), audio, self.sample_rate)
        return filepath
