"""Tests for the audio capture module."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


class TestAudioCapture:
    def test_init_defaults(self):
        from hola_audio.audio.capture import AudioCapture

        cap = AudioCapture()
        assert cap.sample_rate == 16000
        assert cap.channels == 1
        assert not cap.is_recording

    def test_toggle_starts_and_stops(self):
        from hola_audio.audio.capture import AudioCapture

        cap = AudioCapture(output_dir=tempfile.mkdtemp())

        with patch.object(cap, "start") as mock_start:
            cap.toggle()
            mock_start.assert_called_once()

    def test_list_devices(self):
        from hola_audio.audio.capture import AudioCapture

        # Should not raise even without audio hardware
        devices = AudioCapture.list_devices()
        assert isinstance(devices, list)

    def test_save_wav(self):
        from hola_audio.audio.capture import AudioCapture

        cap = AudioCapture(output_dir=tempfile.mkdtemp())
        audio = np.random.randn(16000).astype(np.float32)  # 1 second
        filepath = cap._save_wav(audio)

        assert filepath.exists()
        assert filepath.suffix == ".wav"

    def test_cancel_clears_state(self):
        from hola_audio.audio.capture import AudioCapture

        cap = AudioCapture()
        cap._frames = [np.zeros(1000, dtype=np.float32)]
        cap._recording = True

        cap.cancel()

        assert not cap.is_recording
        assert len(cap._frames) == 0


class TestPlayer:
    def test_load_audio(self):
        import soundfile as sf

        from hola_audio.audio.player import load_audio

        # Create a temp WAV file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            data = np.random.randn(16000).astype(np.float32)
            sf.write(tmp.name, data, 16000)
            tmp_path = tmp.name

        audio = load_audio(tmp_path, target_sr=16000)
        assert audio.shape[0] == 16000
        assert audio.dtype == np.float32

    def test_get_audio_duration(self):
        import soundfile as sf

        from hola_audio.audio.player import get_audio_duration

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            data = np.random.randn(32000).astype(np.float32)
            sf.write(tmp.name, data, 16000)
            tmp_path = tmp.name

        duration = get_audio_duration(tmp_path)
        assert abs(duration - 2.0) < 0.01
