"""Tests for OnlineASREngine."""

from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest

from hola_audio.asr.online_engine import OnlineASREngine


class TestOnlineASREngineInit:
    def test_default_values(self):
        engine = OnlineASREngine()
        assert engine.endpoint == "https://api.openai.com/v1/audio/transcriptions"
        assert engine.api_key == ""
        assert engine.model == "whisper-1"
        assert engine.language == "en"
        assert engine.timeout == 60

    def test_custom_values(self):
        engine = OnlineASREngine(
            endpoint="https://custom.api/v1/audio/transcriptions",
            api_key="sk-test",
            model="whisper-2",
            language="ja",
            timeout=120,
        )
        assert engine.endpoint == "https://custom.api/v1/audio/transcriptions"
        assert engine.api_key == "sk-test"
        assert engine.model == "whisper-2"
        assert engine.language == "ja"
        assert engine.timeout == 120


class TestOnlineASREngineProperties:
    def test_is_loaded_always_true(self):
        engine = OnlineASREngine()
        assert engine.is_loaded is True

    def test_is_configured_without_api_key(self):
        engine = OnlineASREngine(endpoint="https://api.openai.com/v1/audio/transcriptions")
        assert engine.is_configured is False

    def test_is_configured_without_endpoint(self):
        engine = OnlineASREngine(api_key="sk-test", endpoint="")
        assert engine.is_configured is False

    def test_is_configured_with_both(self):
        engine = OnlineASREngine(endpoint="https://api.openai.com/v1/audio/transcriptions", api_key="sk-test")
        assert engine.is_configured is True


class TestOnlineASREngineLoad:
    def test_load_is_noop(self):
        engine = OnlineASREngine(api_key="sk-test")
        engine.load()  # Should not raise

    def test_unload_is_noop(self):
        engine = OnlineASREngine()
        engine.unload()  # Should not raise


class TestOnlineASREngineTranscribe:
    def test_raises_when_not_configured(self, tmp_path):
        engine = OnlineASREngine()
        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"fake audio data")
        with pytest.raises(RuntimeError, match="not configured"):
            engine.transcribe(audio_file)

    def test_raises_when_file_not_found(self):
        engine = OnlineASREngine(endpoint="https://api.openai.com/v1/audio/transcriptions", api_key="sk-test")
        with pytest.raises(FileNotFoundError):
            engine.transcribe("/nonexistent/audio.wav")

    @patch("hola_audio.asr.online_engine.requests.post")
    def test_transcribe_success(self, mock_post, tmp_path):
        mock_response = MagicMock()
        mock_response.json.return_value = {"text": "Hello world"}
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"fake audio data")

        engine = OnlineASREngine(endpoint="https://api.openai.com/v1/audio/transcriptions", api_key="sk-test")
        result = engine.transcribe(audio_file)

        assert result == "Hello world"
        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args
        assert call_kwargs.kwargs["headers"]["Authorization"] == "Bearer sk-test"
        assert call_kwargs.kwargs["data"]["model"] == "whisper-1"
        assert call_kwargs.kwargs["data"]["language"] == "en"

    @patch("hola_audio.asr.online_engine.requests.post")
    def test_transcribe_strips_whitespace(self, mock_post, tmp_path):
        mock_response = MagicMock()
        mock_response.json.return_value = {"text": "  Hello world  "}
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"fake audio data")

        engine = OnlineASREngine(endpoint="https://api.openai.com/v1/audio/transcriptions", api_key="sk-test")
        result = engine.transcribe(audio_file)
        assert result == "Hello world"

    @patch("hola_audio.asr.online_engine.requests.post")
    def test_transcribe_empty_response(self, mock_post, tmp_path):
        mock_response = MagicMock()
        mock_response.json.return_value = {"text": ""}
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"fake audio data")

        engine = OnlineASREngine(endpoint="https://api.openai.com/v1/audio/transcriptions", api_key="sk-test")
        result = engine.transcribe(audio_file)
        assert result == ""

    @patch("hola_audio.asr.online_engine.requests.post")
    def test_transcribe_batch(self, mock_post, tmp_path):
        mock_response = MagicMock()
        mock_response.json.side_effect = [{"text": "First"}, {"text": "Second"}]
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        file1 = tmp_path / "a.wav"
        file2 = tmp_path / "b.wav"
        file1.write_bytes(b"fake1")
        file2.write_bytes(b"fake2")

        engine = OnlineASREngine(endpoint="https://api.openai.com/v1/audio/transcriptions", api_key="sk-test")
        results = engine.transcribe_batch([file1, file2])
        assert len(results) == 2

    @patch("hola_audio.asr.online_engine.requests.post")
    def test_transcribe_http_error(self, mock_post, tmp_path):
        import requests as req

        mock_post.side_effect = req.RequestException("Connection failed")

        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"fake audio data")

        engine = OnlineASREngine(endpoint="https://api.openai.com/v1/audio/transcriptions", api_key="sk-test")
        with pytest.raises(req.RequestException):
            engine.transcribe(audio_file)

    @patch("hola_audio.asr.online_engine.requests.post")
    def test_transcribe_without_language(self, mock_post, tmp_path):
        mock_response = MagicMock()
        mock_response.json.return_value = {"text": "Hello"}
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"fake audio data")

        engine = OnlineASREngine(
            endpoint="https://api.openai.com/v1/audio/transcriptions",
            api_key="sk-test",
            language="",
        )
        engine.transcribe(audio_file)

        call_kwargs = mock_post.call_args
        assert "language" not in call_kwargs.kwargs["data"]
