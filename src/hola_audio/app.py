"""Core application controller.

Orchestrates all components: audio capture, ASR engine, hotkey manager,
correction client, and system tray UI.
"""

from __future__ import annotations

import logging
import sys
import threading
from pathlib import Path
from typing import Any

import pyperclip

from hola_audio.asr.engine import ASREngine
from hola_audio.audio.capture import AudioCapture
from hola_audio.config import Config
from hola_audio.correction.client import CorrectionClient
from hola_audio.hotkey.manager import HotkeyManager

logger = logging.getLogger(__name__)


class Application:
    """Main application controller — wires all components together."""

    def __init__(self, config: Config | None = None) -> None:
        self.config = config or Config()
        self._setup_logging()

        # Components
        self.capture = AudioCapture(
            sample_rate=self.config.get("audio.sample_rate", 16000),
            channels=self.config.get("audio.channels", 1),
            dtype=self.config.get("audio.dtype", "float32"),
            max_duration=self.config.get("audio.max_duration", 120),
            silence_threshold=self.config.get("audio.silence_threshold", 0.01),
            silence_duration=self.config.get("audio.silence_duration", 3.0),
            output_dir=self.config.recordings_dir,
        )

        self.engine = ASREngine(
            model_name=self.config.get("asr.model_name", "nvidia/canary-qwen-2.5b"),
            device=self.config.get("asr.device", "auto"),
            compute_type=self.config.get("asr.compute_type", "float16"),
            max_new_tokens=self.config.get("asr.max_new_tokens", 512),
            beam_size=self.config.get("asr.beam_size", 1),
            prompt=self.config.get("asr.prompt", "Transcribe the following:"),
        )

        self.correction = CorrectionClient(
            endpoint=self.config.get("correction.endpoint", ""),
            api_key=self.config.get("correction.api_key", ""),
            model=self.config.get("correction.model", "gemini-2.0-flash"),
            system_prompt=self.config.get("correction.system_prompt", ""),
            max_tokens=self.config.get("correction.max_tokens", 2048),
            temperature=self.config.get("correction.temperature", 0.1),
        )

        self.hotkeys = HotkeyManager()
        self._tray = None
        self._running = False

        # Wire callbacks
        self.capture.on_recording_start = self._on_recording_start
        self.capture.on_recording_stop = self._on_recording_stop

    # -- Lifecycle -------------------------------------------------------------

    def start(self, load_model: bool = True, enable_tray: bool = True) -> None:
        """Start the application.

        Args:
            load_model: Whether to load the ASR model immediately.
            enable_tray: Whether to show the system tray icon.
        """
        logger.info("Starting Hola Audio")
        self._running = True

        # Register hotkeys
        self.hotkeys.register(
            "toggle_recording",
            self.config.get("hotkey.toggle_recording", "<ctrl>+<shift>+h"),
            self._toggle_recording,
        )
        self.hotkeys.register(
            "cancel_recording",
            self.config.get("hotkey.cancel_recording", "<escape>"),
            self._cancel_recording,
        )
        self.hotkeys.register(
            "correct_clipboard",
            self.config.get("hotkey.correct_clipboard", "<ctrl>+<shift>+j"),
            self._correct_clipboard,
        )
        self.hotkeys.start()

        # System tray
        if enable_tray and self.config.get("ui.show_tray", True):
            try:
                from hola_audio.ui.tray import SystemTray

                self._tray = SystemTray(
                    on_toggle=self._toggle_recording,
                    on_correct=self._correct_clipboard,
                    on_quit=self.stop,
                )
                self._tray.start()
            except Exception:
                logger.warning("System tray unavailable", exc_info=True)

        # Load model in background
        if load_model:
            threading.Thread(target=self.engine.load, daemon=True, name="model-loader").start()

        logger.info("Hola Audio ready. Hotkeys: %s", self.hotkeys.get_bindings())

    def stop(self) -> None:
        """Stop the application and release resources."""
        logger.info("Stopping Hola Audio")
        self._running = False

        if self.capture.is_recording:
            self.capture.cancel()

        self.hotkeys.stop()

        if self._tray:
            self._tray.stop()

        self.engine.unload()
        logger.info("Hola Audio stopped")

    def wait(self) -> None:
        """Block until the application is stopped (e.g. via tray quit)."""
        try:
            while self._running:
                threading.Event().wait(timeout=1.0)
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
            self.stop()

    # -- Actions ---------------------------------------------------------------

    def transcribe_file(self, audio_path: str | Path, correct: bool = False) -> str:
        """Transcribe an audio file, optionally with correction.

        Args:
            audio_path: Path to the audio file.
            correct: Whether to apply LLM correction.

        Returns:
            Transcribed (and optionally corrected) text.
        """
        text = self.engine.transcribe(audio_path)

        if correct and self.correction.is_configured:
            text = self.correction.correct(text)

        if self.config.get("clipboard.auto_copy", True):
            try:
                pyperclip.copy(text)
                logger.info("Transcription copied to clipboard")
            except Exception:
                logger.warning("Failed to copy to clipboard", exc_info=True)

        return text

    # -- Hotkey callbacks ------------------------------------------------------

    def _toggle_recording(self) -> None:
        """Toggle recording on/off."""
        if self.capture.is_recording:
            logger.info("Stopping recording...")
            audio_path = self.capture.stop()
            if audio_path:
                # Transcribe in background
                threading.Thread(
                    target=self._process_recording,
                    args=(audio_path,),
                    daemon=True,
                    name="transcriber",
                ).start()
        else:
            logger.info("Starting recording...")
            self.capture.start()

    def _cancel_recording(self) -> None:
        """Cancel current recording."""
        if self.capture.is_recording:
            self.capture.cancel()
            if self._tray:
                self._tray.set_recording(False)
                self._tray.notify("Hola Audio", "Recording cancelled")

    def _correct_clipboard(self) -> None:
        """Correct text currently in the clipboard."""
        if not self.correction.is_configured:
            logger.warning("Correction endpoint not configured")
            return

        try:
            text = pyperclip.paste()
            if not text.strip():
                logger.warning("Clipboard is empty")
                return

            corrected = self.correction.correct(text)
            pyperclip.copy(corrected)
            logger.info("Clipboard text corrected")

            if self._tray:
                self._tray.notify("Hola Audio", "Clipboard text corrected")
        except Exception:
            logger.exception("Clipboard correction failed")

    # -- Internal --------------------------------------------------------------

    def _on_recording_start(self) -> None:
        if self._tray:
            self._tray.set_recording(True)
            self._tray.notify("Hola Audio", "Recording started...")

    def _on_recording_stop(self, audio_path: Path) -> None:
        if self._tray:
            self._tray.set_recording(False)

    def _process_recording(self, audio_path: Path) -> None:
        """Transcribe a recording and handle the result."""
        try:
            correct = self.config.get("correction.enabled", False) and self.correction.is_configured
            text = self.transcribe_file(audio_path, correct=correct)
            logger.info("Transcription: %s", text)

            if self._tray:
                # Truncate for notification
                preview = text[:100] + "..." if len(text) > 100 else text
                self._tray.notify("Transcription", preview)

        except Exception:
            logger.exception("Processing recording failed")
            if self._tray:
                self._tray.notify("Hola Audio", "Transcription failed. Check logs.")

    def _setup_logging(self) -> None:
        level_name = self.config.get("app.log_level", "INFO")
        level = getattr(logging, level_name.upper(), logging.INFO)
        logging.basicConfig(
            level=level,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
