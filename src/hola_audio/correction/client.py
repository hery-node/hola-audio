"""LLM-based correction client for improving ASR transcriptions.

Sends transcribed text to a configurable LLM endpoint (OpenAI-compatible API)
for correction of grammar, punctuation, and recognition errors.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import requests

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = (
    "You are a speech-to-text correction assistant. Fix transcription errors "
    "in the following text. Correct grammar, punctuation, and obvious "
    "misrecognitions while preserving the original meaning. Return only the "
    "corrected text without explanations."
)


class CorrectionClient:
    """Send transcriptions to an LLM endpoint for correction."""

    def __init__(
        self,
        endpoint: str = "",
        api_key: str = "",
        model: str = "gemini-2.0-flash",
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_tokens: int = 2048,
        temperature: float = 0.1,
    ) -> None:
        self.endpoint = endpoint
        self.api_key = api_key
        self.model = model
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens
        self.temperature = temperature

    @property
    def is_configured(self) -> bool:
        return bool(self.endpoint)

    def correct(self, text: str, context: str = "") -> str:
        """Send text to the LLM for correction.

        Args:
            text: The transcribed text to correct.
            context: Optional context to help the LLM (e.g. topic, domain).

        Returns:
            Corrected text from the LLM.

        Raises:
            RuntimeError: If the endpoint is not configured or the request fails.
        """
        if not self.is_configured:
            raise RuntimeError("Correction endpoint not configured")

        user_message = text
        if context:
            user_message = f"Context: {context}\n\nTranscription to correct:\n{text}"

        logger.info("Sending correction request (%d chars)", len(text))
        t0 = time.monotonic()

        try:
            result = self._call_openai_compatible(user_message)
        except Exception:
            logger.exception("Correction request failed")
            raise

        elapsed = time.monotonic() - t0
        logger.info("Correction received in %.2fs (%d chars)", elapsed, len(result))
        return result

    def correct_with_openai_sdk(self, text: str, context: str = "") -> str:
        """Alternative: use the OpenAI Python SDK for correction.

        Requires ``pip install openai``.
        """
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "OpenAI SDK required. Install with: pip install 'hola-audio[correction]'"
            ) from exc

        user_message = text
        if context:
            user_message = f"Context: {context}\n\nTranscription to correct:\n{text}"

        client = OpenAI(
            base_url=self.endpoint.rstrip("/"),
            api_key=self.api_key or "no-key",
        )

        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_message},
            ],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )

        return response.choices[0].message.content or ""

    # -- Internal --------------------------------------------------------------

    def _call_openai_compatible(self, user_message: str) -> str:
        """Call an OpenAI-compatible chat completions endpoint via requests."""
        headers = {
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_message},
            ],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }

        response = requests.post(
            self.endpoint,
            json=payload,
            headers=headers,
            timeout=60,
        )
        response.raise_for_status()
        data: dict[str, Any] = response.json()

        # Standard OpenAI response format
        choices = data.get("choices", [])
        if not choices:
            raise RuntimeError(f"Empty response from correction endpoint: {data}")

        return choices[0].get("message", {}).get("content", "").strip()
