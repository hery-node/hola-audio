"""Tests for the ASR engine module (without GPU/NeMo dependencies)."""

import pytest


class TestASREngine:
    def test_init_defaults(self):
        from hola_audio.asr.engine import ASREngine

        engine = ASREngine()
        assert engine.model_name == "nvidia/canary-qwen-2.5b"
        assert not engine.is_loaded

    def test_device_resolution(self):
        from hola_audio.asr.engine import _get_device

        # CPU should always work
        assert _get_device("cpu") == "cpu"
        # Auto resolves to something valid
        result = _get_device("auto")
        assert result in ("cuda", "cpu")

    def test_load_without_nemo_raises(self):
        from hola_audio.asr.engine import ASREngine

        engine = ASREngine()
        # Without NeMo installed, should give a clear error
        with pytest.raises(RuntimeError, match="NeMo is required"):
            engine.load()

    def test_transcribe_file_not_found(self):
        from hola_audio.asr.engine import ASREngine

        engine = ASREngine()
        engine._loaded = True  # Pretend model is loaded
        engine._model = type("FakeModel", (), {})()  # Dummy model

        with pytest.raises(FileNotFoundError):
            engine.transcribe("/nonexistent/file.wav")

    def test_unload(self):
        from hola_audio.asr.engine import ASREngine

        engine = ASREngine()
        engine._model = "something"
        engine._loaded = True

        engine.unload()
        assert not engine.is_loaded
        assert engine._model is None
