"""Fine-tuning dataset: sample sentence bank and data collection.

Manages the question bank of sample sentences, collects user voice recordings
paired with reference text, and produces NeMo-compatible manifest files.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

from hola_audio.audio.player import get_audio_duration

logger = logging.getLogger(__name__)

# Default sample sentences covering various speech patterns
DEFAULT_SENTENCES: list[str] = [
    "The quick brown fox jumps over the lazy dog.",
    "She sells seashells by the seashore every morning.",
    "How much wood would a woodchuck chuck if a woodchuck could chuck wood?",
    "Peter Piper picked a peck of pickled peppers.",
    "The weather today is partly cloudy with a chance of afternoon showers.",
    "Please schedule a meeting with the engineering team for next Tuesday at three PM.",
    "I need to transfer five hundred dollars to my savings account.",
    "The restaurant on Fifth Avenue serves excellent Italian cuisine.",
    "Can you recommend a good book about machine learning and artificial intelligence?",
    "We should review the quarterly report before the board presentation tomorrow.",
    "The flight from New York to San Francisco departs at seven thirty in the evening.",
    "I'd like to order a large cappuccino with an extra shot of espresso please.",
    "The new software update includes several important security patches.",
    "Please remind me to call Dr. Johnson's office at four o'clock.",
    "The conference will feature keynote speakers from Google, Microsoft, and NVIDIA.",
]


@dataclass
class SampleEntry:
    """A single sample in the fine-tuning dataset."""

    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    text: str = ""
    audio_path: str = ""
    duration: float = 0.0
    recorded: bool = False


class FinetuneDataset:
    """Manage sentence bank and collected audio-text pairs."""

    def __init__(self, data_dir: str | Path) -> None:
        self.data_dir = Path(data_dir)
        self.audio_dir = self.data_dir / "audio"
        self.manifest_path = self.data_dir / "manifest.json"
        self.sentences_path = self.data_dir / "sentences.json"

        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.audio_dir.mkdir(parents=True, exist_ok=True)

        self._samples: list[SampleEntry] = []
        self._load()

    # -- Properties ------------------------------------------------------------

    @property
    def total(self) -> int:
        return len(self._samples)

    @property
    def recorded_count(self) -> int:
        return sum(1 for s in self._samples if s.recorded)

    @property
    def progress(self) -> float:
        if not self._samples:
            return 0.0
        return self.recorded_count / self.total

    @property
    def is_complete(self) -> bool:
        return self.total > 0 and self.recorded_count == self.total

    # -- Public API ------------------------------------------------------------

    def initialize(self, sentences: list[str] | None = None) -> None:
        """Initialize the dataset with sample sentences.

        Args:
            sentences: Custom sentence list. If None, uses defaults.
        """
        texts = sentences or DEFAULT_SENTENCES
        self._samples = [SampleEntry(text=t) for t in texts]
        self._save_sentences()
        logger.info("Initialized dataset with %d sentences", len(self._samples))

    def get_next_unrecorded(self) -> SampleEntry | None:
        """Return the next sentence that hasn't been recorded yet."""
        for sample in self._samples:
            if not sample.recorded:
                return sample
        return None

    def get_sample(self, sample_id: str) -> SampleEntry | None:
        """Get a sample by ID."""
        for sample in self._samples:
            if sample.id == sample_id:
                return sample
        return None

    def get_all_samples(self) -> list[SampleEntry]:
        """Return all samples."""
        return list(self._samples)

    def mark_recorded(self, sample_id: str, audio_path: str | Path) -> None:
        """Mark a sample as recorded with its audio file path.

        Args:
            sample_id: The sample ID.
            audio_path: Path to the recorded WAV file.
        """
        sample = self.get_sample(sample_id)
        if sample is None:
            raise ValueError(f"Sample not found: {sample_id}")

        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        sample.audio_path = str(audio_path)
        sample.duration = get_audio_duration(audio_path)
        sample.recorded = True

        self._save_sentences()
        logger.info("Marked sample %s as recorded: %s (%.1fs)", sample_id, audio_path.name, sample.duration)

    def add_sentence(self, text: str) -> SampleEntry:
        """Add a custom sentence to the dataset."""
        sample = SampleEntry(text=text)
        self._samples.append(sample)
        self._save_sentences()
        return sample

    def remove_sentence(self, sample_id: str) -> bool:
        """Remove a sentence from the dataset."""
        for i, sample in enumerate(self._samples):
            if sample.id == sample_id:
                self._samples.pop(i)
                self._save_sentences()
                return True
        return False

    def export_nemo_manifest(self) -> Path:
        """Export recorded samples as a NeMo-compatible JSONL manifest.

        Each line contains:
          {"audio_filepath": "...", "text": "...", "duration": ...}

        Returns:
            Path to the manifest file.
        """
        recorded = [s for s in self._samples if s.recorded]
        if not recorded:
            raise ValueError("No recorded samples to export")

        with open(self.manifest_path, "w", encoding="utf-8") as fh:
            for sample in recorded:
                entry = {
                    "audio_filepath": sample.audio_path,
                    "text": sample.text,
                    "duration": sample.duration,
                }
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")

        logger.info("Exported %d samples to %s", len(recorded), self.manifest_path)
        return self.manifest_path

    def reset(self) -> None:
        """Reset all recordings (keeps sentences)."""
        for sample in self._samples:
            sample.audio_path = ""
            sample.duration = 0.0
            sample.recorded = False
        self._save_sentences()
        logger.info("Dataset recordings reset")

    # -- Internal --------------------------------------------------------------

    def _save_sentences(self) -> None:
        with open(self.sentences_path, "w", encoding="utf-8") as fh:
            json.dump([asdict(s) for s in self._samples], fh, indent=2, ensure_ascii=False)

    def _load(self) -> None:
        if self.sentences_path.exists():
            with open(self.sentences_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
                self._samples = [SampleEntry(**d) for d in data]
            logger.info("Loaded %d samples (%d recorded)", self.total, self.recorded_count)
