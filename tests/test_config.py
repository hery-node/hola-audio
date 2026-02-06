"""Tests for the configuration module."""

import os
import tempfile
from pathlib import Path

import pytest
import yaml

from hola_audio.config import Config, _deep_merge


class TestDeepMerge:
    def test_flat_merge(self):
        base = {"a": 1, "b": 2}
        override = {"b": 3, "c": 4}
        result = _deep_merge(base, override)
        assert result == {"a": 1, "b": 3, "c": 4}

    def test_nested_merge(self):
        base = {"top": {"a": 1, "b": 2}, "other": 10}
        override = {"top": {"b": 3, "c": 4}}
        result = _deep_merge(base, override)
        assert result == {"top": {"a": 1, "b": 3, "c": 4}, "other": 10}

    def test_override_dict_with_scalar(self):
        base = {"a": {"nested": 1}}
        override = {"a": "replaced"}
        result = _deep_merge(base, override)
        assert result == {"a": "replaced"}


class TestConfig:
    def test_default_config_loads(self):
        config = Config()
        assert config.get("audio.sample_rate") == 16000
        assert config.get("audio.channels") == 1
        assert config.get("asr.model_name") == "nvidia/canary-qwen-2.5b"

    def test_get_with_default(self):
        config = Config()
        assert config.get("nonexistent.key", "fallback") == "fallback"

    def test_set_and_get(self):
        config = Config()
        config.set("custom.nested.value", 42)
        assert config.get("custom.nested.value") == 42

    def test_save_and_reload(self):
        config = Config()
        config.set("test.marker", "hello")

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False, mode="w") as tmp:
            tmp_path = tmp.name

        try:
            config.save(tmp_path)

            # Reload
            config2 = Config(tmp_path)
            assert config2.get("test.marker") == "hello"
            # Defaults should still be present
            assert config2.get("audio.sample_rate") == 16000
        finally:
            os.unlink(tmp_path)

    def test_user_config_overlay(self):
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False, mode="w") as tmp:
            yaml.safe_dump({"audio": {"sample_rate": 44100}}, tmp)
            tmp_path = tmp.name

        try:
            config = Config(tmp_path)
            assert config.get("audio.sample_rate") == 44100
            # Other defaults preserved
            assert config.get("audio.channels") == 1
        finally:
            os.unlink(tmp_path)

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("HOLA_AUDIO__AUDIO__SAMPLE_RATE", "44100")
        config = Config()
        assert config.get("audio.sample_rate") == 44100

    def test_env_bool_parsing(self, monkeypatch):
        monkeypatch.setenv("HOLA_AUDIO__CORRECTION__ENABLED", "true")
        config = Config()
        assert config.get("correction.enabled") is True

    def test_directories_exist(self):
        config = Config()
        assert config.recordings_dir.exists()
        assert config.finetune_dir.exists()
