"""ASR engine wrapping NVIDIA Canary-Qwen-2.5B via NeMo SALM.

Provides a unified interface for:
  - Loading the pre-trained model
  - Transcribing audio files (ASR mode)
  - Post-processing via LLM mode
  - Loading fine-tuned checkpoints
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Lazy imports to avoid hard dependency on NeMo at import time
_model_cache: dict[str, Any] = {}


def _get_device(preference: str) -> str:
    """Resolve device string: 'auto' → 'cuda' if available, else 'cpu'."""
    if preference != "auto":
        return preference
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


class ASREngine:
    """Speech-to-text engine backed by Canary-Qwen-2.5B."""

    def __init__(
        self,
        model_name: str = "nvidia/canary-qwen-2.5b",
        device: str = "auto",
        compute_type: str = "float16",
        max_new_tokens: int = 512,
        beam_size: int = 1,
        prompt: str = "Transcribe the following:",
    ) -> None:
        self.model_name = model_name
        self.device = _get_device(device)
        self.compute_type = compute_type
        self.max_new_tokens = max_new_tokens
        self.beam_size = beam_size
        self.prompt = prompt

        self._model: Any = None
        self._loaded = False

    # -- Public API ------------------------------------------------------------

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load(self, checkpoint_path: str | Path | None = None) -> None:
        """Load the ASR model into memory.

        Args:
            checkpoint_path: Optional path to a fine-tuned NeMo checkpoint.
                           If None, downloads/loads the pretrained model.
        """
        if self._loaded:
            logger.info("Model already loaded")
            return

        logger.info("Loading ASR model: %s (device=%s)", self.model_name, self.device)
        t0 = time.monotonic()

        try:
            import torch
            from nemo.collections.speechlm2.models import SALM

            if checkpoint_path:
                ckpt = Path(checkpoint_path)
                if not ckpt.exists():
                    raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
                logger.info("Loading fine-tuned checkpoint: %s", ckpt)
                self._model = SALM.restore_from(str(ckpt))
            else:
                # Use cache to avoid re-downloading
                cache_key = self.model_name
                if cache_key in _model_cache:
                    self._model = _model_cache[cache_key]
                    logger.info("Using cached model instance")
                else:
                    self._model = SALM.from_pretrained(self.model_name)
                    _model_cache[cache_key] = self._model

            # Move to device
            if self.device == "cuda" and torch.cuda.is_available():
                self._model = self._model.cuda()
                if self.compute_type == "float16":
                    self._model = self._model.half()
                elif self.compute_type == "bfloat16":
                    self._model = self._model.to(dtype=torch.bfloat16)

            self._model.eval()
            self._loaded = True

            elapsed = time.monotonic() - t0
            logger.info("Model loaded in %.1fs", elapsed)

        except ImportError as exc:
            raise RuntimeError(
                "NeMo is required for ASR. Install with: pip install 'hola-audio[gpu]'\n"
                "Or: pip install 'nemo_toolkit[asr] @ git+https://github.com/NVIDIA/NeMo.git'"
            ) from exc

    def transcribe(self, audio_path: str | Path) -> str:
        """Transcribe a single audio file to text.

        Args:
            audio_path: Path to a WAV/FLAC file (16 kHz mono recommended).

        Returns:
            Transcribed text string.
        """
        if not self._loaded:
            self.load()

        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        logger.info("Transcribing: %s", audio_path.name)
        t0 = time.monotonic()

        try:
            import torch

            with torch.no_grad():
                answer_ids = self._model.generate(
                    prompts=[
                        [
                            {
                                "role": "user",
                                "content": f"{self.prompt} {self._model.audio_locator_tag}",
                                "audio": [str(audio_path)],
                            }
                        ]
                    ],
                    max_new_tokens=self.max_new_tokens,
                )
                text = self._model.tokenizer.ids_to_text(answer_ids[0].cpu())

        except Exception:
            logger.exception("Transcription failed")
            raise

        elapsed = time.monotonic() - t0
        logger.info("Transcription complete in %.2fs: %s", elapsed, text[:80])
        return text.strip()

    def transcribe_batch(self, audio_paths: list[str | Path]) -> list[str]:
        """Transcribe multiple audio files.

        Args:
            audio_paths: List of paths to audio files.

        Returns:
            List of transcribed text strings.
        """
        return [self.transcribe(p) for p in audio_paths]

    def post_process(self, transcript: str, instruction: str) -> str:
        """Use the model's LLM mode for transcript post-processing.

        The model's LoRA adapter is disabled so the underlying Qwen LLM
        capabilities are used (summarization, Q&A, etc.).

        Args:
            transcript: The transcribed text.
            instruction: The instruction prompt (e.g. "Summarize the following").

        Returns:
            LLM-generated response.
        """
        if not self._loaded:
            self.load()

        try:
            import torch

            with torch.no_grad(), self._model.llm.disable_adapter():
                answer_ids = self._model.generate(
                    prompts=[
                        [{"role": "user", "content": f"{instruction}\n\n{transcript}"}]
                    ],
                    max_new_tokens=self.max_new_tokens,
                )
                text = self._model.tokenizer.ids_to_text(answer_ids[0].cpu())

        except Exception:
            logger.exception("Post-processing failed")
            raise

        return text.strip()

    def unload(self) -> None:
        """Release model from memory."""
        if self._model is not None:
            del self._model
            self._model = None
            self._loaded = False
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
            logger.info("Model unloaded")
