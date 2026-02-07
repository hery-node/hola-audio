"""Standalone widget application for online voice input.

Lightweight controller that bridges the floating widget UI with
audio capture and online ASR — no offline/NeMo dependencies needed.
"""

from __future__ import annotations

import logging
import threading
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

        self._init_engine()
        self._correction_enabled = self.config.get("correction.enabled", False)

        self.widget = FloatingWidget(
            on_toggle=self._toggle_recording,
            on_settings=self._open_settings,
            on_quit=self._cleanup,
        )

        self._recording = False

    def _init_engine(self) -> None:
        """Initialize or reinitialize the ASR engine from config."""
        self.engine = OnlineASREngine(
            endpoint=self.config.get("asr.online.endpoint", "https://api.groq.com/openai/v1/audio/transcriptions"),
            api_key=self.config.get("asr.online.api_key", ""),
            model=self.config.get("asr.online.model", "whisper-large-v3"),
            language=self.config.get("asr.online.language", "en"),
            timeout=self.config.get("asr.online.timeout", 60),
        )

    def run(self) -> None:
        """Start the widget application."""
        print(f"[widget] API endpoint: {self.engine.endpoint}")
        print(f"[widget] API key set: {'yes' if self.engine.api_key else 'NO'}")
        print(f"[widget] Model: {self.engine.model}")
        print(f"[widget] Correction: {'ON' if self._correction_enabled else 'OFF'}")
        if not self.engine.is_configured:
            print("[widget] ⚠ WARNING: API key not configured! Right-click → Settings to set it.")
        print("[widget] Starting... Click the blue circle to record.")
        self.widget.run()

    # ── Settings ─────────────────────────────────────────────────────────

    def _open_settings(self) -> None:
        """Open the settings dialog."""
        from hola_audio.ui.settings import SettingsDialog

        SettingsDialog(
            parent=self.widget.root,
            api_key=self.config.get("asr.online.api_key", ""),
            correction_enabled=self.config.get("correction.enabled", False),
            correction_api_key=self.config.get("correction.api_key", ""),
            correction_endpoint=self.config.get("correction.endpoint", ""),
            correction_model=self.config.get("correction.model", "gemini-2.0-flash"),
            on_save=self._save_settings,
        )

    def _save_settings(self, settings: dict) -> None:
        """Save settings from dialog to config file."""
        for key, value in settings.items():
            self.config.set(key, value)
        self.config.save()

        # Reload engine with new API key
        self._init_engine()
        self._correction_enabled = self.config.get("correction.enabled", False)

        print(f"[widget] Settings saved. API key: {'set' if self.engine.api_key else 'empty'}, Correction: {'ON' if self._correction_enabled else 'OFF'}")

    # ── Recording ────────────────────────────────────────────────────────

    def _toggle_recording(self) -> None:
        """Toggle recording on/off."""
        print(f"[widget] Toggle clicked. Currently recording: {self._recording}")
        if self._recording:
            self._stop_recording()
        else:
            self._start_recording()

    def _start_recording(self) -> None:
        """Start audio recording."""
        if not self.engine.is_configured:
            print("[widget] ⚠ API key not set! Right-click → Settings to configure.")
            return

        self._recording = True
        self.widget.set_state(WidgetState.RECORDING)
        self.capture.on_recording_stop = self._on_recording_complete_threadsafe
        self.capture.start()
        print("[widget] 🎙 Recording started...")

    def _on_recording_complete_threadsafe(self, audio_path: Path) -> None:
        """Called from audio thread — schedule on main thread."""
        self.widget.schedule(lambda: self._on_recording_complete(audio_path))

    def _stop_recording(self) -> None:
        """Stop recording and start transcription."""
        if not self._recording:
            return
        self._recording = False
        self.widget.set_state(WidgetState.TRANSCRIBING)
        print("[widget] ⏹ Recording stopped. Saving audio...")
        audio_path = self.capture.stop()
        if audio_path:
            print(f"[widget] 📁 Audio saved: {audio_path}")
            self._on_recording_complete(audio_path)
        else:
            print("[widget] ⚠ No audio frames captured!")
            self.widget.set_state(WidgetState.IDLE)

    def _on_recording_complete(self, audio_path: Path) -> None:
        """Handle completed recording — transcribe in background."""
        print("[widget] 🔄 Sending to Groq API for transcription...")
        thread = threading.Thread(target=self._transcribe, args=(audio_path,), daemon=True)
        thread.start()

    # ── Transcription + Correction ───────────────────────────────────────

    def _transcribe(self, audio_path: Path) -> None:
        """Transcribe audio file, optionally correct, and copy to clipboard."""
        try:
            text = self.engine.transcribe(audio_path)
            if not text:
                print("[widget] ⚠ Empty transcription result")
                self.widget.schedule(lambda: self._show_result("(no speech detected)"))
                return

            print(f"[widget] ✅ Transcription: {text}")

            # Apply AI correction if enabled
            if self._correction_enabled:
                text = self._apply_correction(text)

            try:
                pyperclip.copy(text)
                print("[widget] 📋 Copied to clipboard")
            except Exception:
                print("[widget] ⚠ Clipboard not available (install xclip: sudo apt install xclip)")

            self.widget.schedule(lambda: self._show_result(text))

        except Exception as e:
            print(f"[widget] ❌ Transcription failed: {e}")
            import traceback
            traceback.print_exc()
            self.widget.schedule(lambda: self._show_error())

    def _apply_correction(self, text: str) -> str:
        """Apply AI correction to transcribed text."""
        try:
            from hola_audio.correction.client import CorrectionClient

            client = CorrectionClient(
                endpoint=self.config.get("correction.endpoint", ""),
                api_key=self.config.get("correction.api_key", ""),
                model=self.config.get("correction.model", "gemini-2.0-flash"),
            )
            corrected = client.correct(text)
            if corrected and corrected != text:
                print(f"[widget] ✨ Corrected: {corrected}")
                return corrected
            print("[widget] ✨ Correction: no changes needed")
            return text
        except Exception as e:
            print(f"[widget] ⚠ Correction failed: {e}, using original text")
            return text

    # ── UI Updates ───────────────────────────────────────────────────────

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
        print("[widget] Widget app stopped")

    def _setup_logging(self) -> None:
        level_name = self.config.get("app.log_level", "INFO")
        level = getattr(logging, level_name.upper(), logging.INFO)
        logging.basicConfig(
            level=level,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )

