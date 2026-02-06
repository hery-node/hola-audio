"""Tests for the fine-tuning dataset module."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from hola_audio.finetune.dataset import DEFAULT_SENTENCES, FinetuneDataset


class TestFinetuneDataset:
    def _make_dataset(self) -> tuple[FinetuneDataset, Path]:
        tmp_dir = tempfile.mkdtemp()
        ds = FinetuneDataset(tmp_dir)
        return ds, Path(tmp_dir)

    def _make_wav(self, directory: Path, duration_s: float = 1.0) -> Path:
        data = np.random.randn(int(16000 * duration_s)).astype(np.float32)
        path = directory / f"test_{id(data)}.wav"
        sf.write(str(path), data, 16000)
        return path

    def test_initialize_defaults(self):
        ds, _ = self._make_dataset()
        ds.initialize()
        assert ds.total == len(DEFAULT_SENTENCES)
        assert ds.recorded_count == 0
        assert ds.progress == 0.0

    def test_initialize_custom(self):
        ds, _ = self._make_dataset()
        ds.initialize(["Hello world", "Goodbye world"])
        assert ds.total == 2

    def test_get_next_unrecorded(self):
        ds, _ = self._make_dataset()
        ds.initialize(["Sentence one", "Sentence two"])
        sample = ds.get_next_unrecorded()
        assert sample is not None
        assert sample.text == "Sentence one"

    def test_mark_recorded(self):
        ds, tmp_dir = self._make_dataset()
        ds.initialize(["Test sentence"])
        sample = ds.get_next_unrecorded()
        assert sample is not None

        wav_path = self._make_wav(tmp_dir)
        ds.mark_recorded(sample.id, wav_path)

        assert ds.recorded_count == 1
        assert ds.is_complete

    def test_export_manifest(self):
        ds, tmp_dir = self._make_dataset()
        ds.initialize(["Test sentence"])
        sample = ds.get_next_unrecorded()

        wav_path = self._make_wav(tmp_dir)
        ds.mark_recorded(sample.id, wav_path)

        manifest = ds.export_nemo_manifest()
        assert manifest.exists()

        # Check JSONL format
        import json
        with open(manifest) as fh:
            lines = fh.readlines()
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert "audio_filepath" in entry
        assert "text" in entry
        assert "duration" in entry

    def test_add_sentence(self):
        ds, _ = self._make_dataset()
        ds.initialize(["Base sentence"])
        ds.add_sentence("Extra sentence")
        assert ds.total == 2

    def test_remove_sentence(self):
        ds, _ = self._make_dataset()
        ds.initialize(["Sentence A", "Sentence B"])
        samples = ds.get_all_samples()
        ds.remove_sentence(samples[0].id)
        assert ds.total == 1

    def test_reset(self):
        ds, tmp_dir = self._make_dataset()
        ds.initialize(["Test"])
        sample = ds.get_next_unrecorded()
        wav_path = self._make_wav(tmp_dir)
        ds.mark_recorded(sample.id, wav_path)

        ds.reset()
        assert ds.recorded_count == 0
        assert ds.total == 1  # Sentence preserved

    def test_persistence(self):
        tmp_dir = tempfile.mkdtemp()
        ds1 = FinetuneDataset(tmp_dir)
        ds1.initialize(["Persist test"])

        # Reload
        ds2 = FinetuneDataset(tmp_dir)
        assert ds2.total == 1
        assert ds2.get_all_samples()[0].text == "Persist test"
