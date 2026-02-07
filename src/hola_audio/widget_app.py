"""Standalone widget application for online voice input.

Lightweight controller that bridges the floating widget UI with
audio capture and online ASR — no offline/NeMo dependencies needed.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

import pyperclip

from hola_audio.asr.online_engine import OnlineASREngine
from hola_audio.audio.capture import AudioCapture
from hola_audio.config import Config
from hola_audio.ui.widget import FloatingWidget, WidgetState

logger = logging.getLogger(__name__)


class WidgetApp:
    """Widget application controller for online voice input."""

    def __init__(self, config: Config | None = None) -> None:
        self.config = config or Config()
        self._setup_logging()

        self.capture = AudioCapture(
            sample_rate=self.config.get("audio.sample_rate", 16000),
            channels=self.config.get("audio.channels", 1),
            dtype=self.config.get("audio.dtype", "float32"),
            max_duration=self.config.get("audio.max_duration", 120),
            silence_threshold=self.config.get("audio.silence_threshold", 0.01),
            silence_duration=self.config.get("audio.silence_duration", 3.0),
            output_dir=self.config.recordings_dir,
        )

        self.engine = OnlineASREngine(
            endpoint=self.config.get("asr.online.endpoint", "https://api.groq.com/openai/v1/audio/transcriptions"),
            api_key=self.config.get("asr.online.api_key", ""),
            model=self.config.get("asr.online.model", "whisper-large-v3"),
            language=self.config.get("asr.online.language", "en"),
            timeout=self.config.get("asr.online.timeout", 60),
        )

        self.widget = FloatingWidget(
            on_toggle=self._toggle_recording,
            on_quit=self._cleanup,
        )

        self._recording = False

    def run(self) -> None:
        """Start the widget application."""
        if not self.engine.is_configured:
            logger.warning(
                "Online ASR not configured. Set API key via: "
                "hola-audio config set asr.online.api_key YOUR_KEY"
            )
        logger.info("Starting floating widget...")
        self.widget.run()

    def _toggle_recording(self) -> None:
        """Toggle recording on/off."""
        if self._recording:
            self._stop_recording()
        else:
            self._start_recording()

    def _start_recording(self) -> None:
        """Start audio recording."""
        if not self.engine.is_configured:
            self.widget.show_toast("⚠ Set API key first!", 3.0)
            return

        self._recording = True
        self.widget.set_state(WidgetState.RECORDING)

        self.capture.on_recording_stop = self._on_recording_complete_threadsafe
        self.capture.start()
        logger.info("Recording started")

    def _on_recording_complete_threadsafe(self, audio_path: Path) -> None:
        """Called from audio thread — schedule on main thread."""
        self.widget.schedule(lambda: self._on_recording_complete(audio_path))

    def _stop_recording(self) -> None:
        """Stop recording and start transcription."""
        if not self._recording:
            return
        self._recording = False
        self.widget.set_state(WidgetState.TRANSCRIBING)
        audio_path = self.capture.stop()
        if audio_path:
            self._on_recording_complete(audio_path)
        else:
            self.widget.set_state(WidgetState.IDLE)
            self.widget.show_toast("No audio captured", 2.0)
        logger.info("Recording stopped")

    def _on_recording_complete(self, audio_path: Path) -> None:
        """Handle completed recording — transcribe in background."""
        logger.info("Recording saved: %s", audio_path)
        thread = threading.Thread(target=self._transcribe, args=(audio_path,), daemon=True)
        thread.start()

    def _transcribe(self, audio_path: Path) -> None:
        """Transcribe audio file and copy result to clipboard."""
        try:
            text = self.engine.transcribe(audio_path)
            if text:
                pyperclip.copy(text)
                logger.info("Transcription copied: %s", text[:80])
                self.widget.schedule(lambda: self._show_result(text))
            else:
                logger.warning("Empty transcription result")
                self.widget.schedule(lambda: self._show_result("(no speech detected)"))
        except Exception:
            logger.exception("Transcription failed")
            self.widget.schedule(lambda: self._show_error())

    def _show_result(self, text: str) -> None:
        """Show transcription result and return to idle."""
        self.widget.set_state(WidgetState.IDLE)
        self.widget.show_toast(text, 4.0)

    def _show_error(self) -> None:
        """Show error and return to idle."""
        self.widget.set_state(WidgetState.IDLE)
        self.widget.show_toast("⚠ Transcription failed", 3.0)

    def _cleanup(self) -> None:
        """Clean up resources."""
        if self._recording:
            self.capture.stop()
        logger.info("Widget app stopped")

    def _setup_logging(self) -> None:
        level_name = self.config.get("app.log_level", "INFO")
        level = getattr(logging, level_name.upper(), logging.INFO)
        logging.basicConfig(
            level=level,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
