"""Tests for the correction client."""

import json
from unittest.mock import MagicMock, patch

import pytest

from hola_audio.correction.client import CorrectionClient


class TestCorrectionClient:
    def test_not_configured(self):
        client = CorrectionClient()
        assert not client.is_configured

    def test_configured(self):
        client = CorrectionClient(endpoint="https://example.com/v1/chat/completions")
        assert client.is_configured

    def test_correct_raises_when_not_configured(self):
        client = CorrectionClient()
        with pytest.raises(RuntimeError, match="not configured"):
            client.correct("some text")

    @patch("hola_audio.correction.client.requests.post")
    def test_correct_success(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Corrected text here."}}]
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        client = CorrectionClient(
            endpoint="https://example.com/v1/chat/completions",
            api_key="test-key",
        )

        result = client.correct("Uncorrected text here")
        assert result == "Corrected text here."

        # Verify the request was made correctly
        call_args = mock_post.call_args
        assert call_args[0][0] == "https://example.com/v1/chat/completions"
        payload = call_args[1]["json"]
        assert payload["model"] == "gemini-2.0-flash"
        assert len(payload["messages"]) == 2

    @patch("hola_audio.correction.client.requests.post")
    def test_correct_with_context(self, mock_post):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Fixed."}}]
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        client = CorrectionClient(endpoint="https://example.com/api")
        result = client.correct("text", context="medical domain")

        payload = mock_post.call_args[1]["json"]
        assert "medical domain" in payload["messages"][1]["content"]
