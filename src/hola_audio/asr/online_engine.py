"""Online ASR engine using OpenAI Whisper API (or any compatible endpoint).

Sends recorded audio files to a cloud speech-to-text endpoint for
transcription. Lightweight alternative to the offline NeMo-based engine.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import requests

logger = logging.getLogger(__name__)


class OnlineASREngine:
    """Speech-to-text engine backed by OpenAI Whisper API."""

    def __init__(
        self,
        endpoint: str = "https://api.openai.com/v1/audio/transcriptions",
        api_key: str = "",
        model: str = "whisper-1",
        language: str = "en",
        timeout: int = 60,
    ) -> None:
        self.endpoint = endpoint
        self.api_key = api_key
        self.model = model
        self.language = language
        self.timeout = timeout

    @property
    def is_loaded(self) -> bool:
        """Online engine is always 'loaded' — no model to preload."""
        return True

    @property
    def is_configured(self) -> bool:
        return bool(self.endpoint and self.api_key)

    def load(self, checkpoint_path: str | Path | None = None) -> None:
        """No-op for online engine. Exists for interface compatibility."""
        if not self.is_configured:
            logger.warning("Online ASR not configured — set asr.online.api_key")

    def transcribe(self, audio_path: str | Path) -> str:
        """Transcribe a single audio file via the online API.

        Args:
            audio_path: Path to a WAV/FLAC file.

        Returns:
            Transcribed text string.

        Raises:
            RuntimeError: If the API key is not configured.
            FileNotFoundError: If the audio file does not exist.
        """
        if not self.is_configured:
            raise RuntimeError(
                "Online ASR not configured. Set 'asr.online.api_key' in config "
                "or HOLA_AUDIO__ASR__ONLINE__API_KEY env var."
            )

        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        logger.info("Transcribing online: %s", audio_path.name)
        t0 = time.monotonic()

        headers = {"Authorization": f"Bearer {self.api_key}"}
        data: dict[str, Any] = {"model": self.model}
        if self.language:
            data["language"] = self.language

        try:
            with open(audio_path, "rb") as audio_file:
                response = requests.post(
                    self.endpoint,
                    headers=headers,
                    data=data,
                    files={"file": (audio_path.name, audio_file, "audio/wav")},
                    timeout=self.timeout,
                )
            response.raise_for_status()
        except requests.RequestException:
            logger.exception("Online transcription request failed")
            raise

        result = response.json()
        text = result.get("text", "").strip()

        elapsed = time.monotonic() - t0
        logger.info("Online transcription complete in %.2fs: %s", elapsed, text[:80])
        return text

    def transcribe_batch(self, audio_paths: list[str | Path]) -> list[str]:
        """Transcribe multiple audio files."""
        return [self.transcribe(p) for p in audio_paths]

    def unload(self) -> None:
        """No-op for online engine."""
        pass
