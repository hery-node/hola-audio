"""Configuration management for Hola Audio."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import yaml
from platformdirs import user_config_dir, user_data_dir

logger = logging.getLogger(__name__)

APP_NAME = "hola-audio"
APP_AUTHOR = "hery-node"

# Resolve paths
_CONFIG_DIR = Path(user_config_dir(APP_NAME, APP_AUTHOR))
_DATA_DIR = Path(user_data_dir(APP_NAME, APP_AUTHOR))
_DEFAULT_CONFIG = Path(__file__).resolve().parent.parent.parent / "config" / "default.yaml"
_USER_CONFIG = _CONFIG_DIR / "config.yaml"


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base*, returning a new dict."""
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


class Config:
    """Hierarchical configuration: defaults → user config → env overrides."""

    def __init__(self, config_path: str | Path | None = None) -> None:
        self._data: dict[str, Any] = {}
        self._load(config_path)

    # -- public helpers --------------------------------------------------------

    @property
    def config_dir(self) -> Path:
        return _CONFIG_DIR

    @property
    def data_dir(self) -> Path:
        return _DATA_DIR

    @property
    def recordings_dir(self) -> Path:
        custom = self.get("audio.output_dir")
        if custom:
            return Path(custom)
        return _DATA_DIR / "recordings"

    @property
    def finetune_dir(self) -> Path:
        custom = self.get("finetune.output_dir")
        if custom:
            return Path(custom)
        return _DATA_DIR / "finetune"

    def get(self, dotted_key: str, default: Any = None) -> Any:
        """Retrieve a value using dot-separated keys, e.g. ``audio.sample_rate``."""
        keys = dotted_key.split(".")
        node: Any = self._data
        for k in keys:
            if isinstance(node, dict) and k in node:
                node = node[k]
            else:
                return default
        return node

    def set(self, dotted_key: str, value: Any) -> None:
        """Set a value using dot-separated keys."""
        keys = dotted_key.split(".")
        node = self._data
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        node[keys[-1]] = value

    def save(self, path: str | Path | None = None) -> Path:
        """Persist current config to disk."""
        dest = Path(path) if path else _USER_CONFIG
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "w", encoding="utf-8") as fh:
            yaml.safe_dump(self._data, fh, default_flow_style=False, sort_keys=False)
        logger.info("Configuration saved to %s", dest)
        return dest

    def as_dict(self) -> dict[str, Any]:
        return self._data.copy()

    # -- internal --------------------------------------------------------------

    def _load(self, config_path: str | Path | None) -> None:
        # 1. Load bundled defaults
        if _DEFAULT_CONFIG.exists():
            self._data = self._read_yaml(_DEFAULT_CONFIG)
        else:
            logger.warning("Default config not found at %s", _DEFAULT_CONFIG)
            self._data = {}

        # 2. Overlay user config
        user_path = Path(config_path) if config_path else _USER_CONFIG
        if user_path.exists():
            user_cfg = self._read_yaml(user_path)
            self._data = _deep_merge(self._data, user_cfg)
            logger.info("Loaded user config from %s", user_path)

        # 3. Apply environment variable overrides (HOLA_AUDIO__section__key)
        self._apply_env_overrides()

        # 4. Ensure dirs exist
        self.recordings_dir.mkdir(parents=True, exist_ok=True)
        self.finetune_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _read_yaml(path: Path) -> dict:
        with open(path, "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}

    def _apply_env_overrides(self) -> None:
        prefix = "HOLA_AUDIO__"
        for key, value in os.environ.items():
            if not key.startswith(prefix):
                continue
            parts = key[len(prefix):].lower().split("__")
            dotted = ".".join(parts)
            # Try to parse booleans / numbers
            parsed = self._parse_env_value(value)
            self.set(dotted, parsed)
            logger.debug("Env override: %s = %r", dotted, parsed)

    @staticmethod
    def _parse_env_value(value: str) -> Any:
        if value.lower() in ("true", "yes", "1"):
            return True
        if value.lower() in ("false", "no", "0"):
            return False
        try:
            return int(value)
        except ValueError:
            pass
        try:
            return float(value)
        except ValueError:
            pass
        return value
