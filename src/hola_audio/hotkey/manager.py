"""Cross-platform global hotkey manager using pynput.

Supports customizable hotkey combinations for:
  - Toggle recording (start/stop)
  - Cancel recording
  - Correct clipboard text
"""

from __future__ import annotations

import logging
import platform
import threading
from typing import Callable

from pynput import keyboard

logger = logging.getLogger(__name__)


def _parse_hotkey(hotkey_str: str) -> set[keyboard.Key | keyboard.KeyCode]:
    """Parse a hotkey string like ``<ctrl>+<shift>+h`` into a set of keys.

    Supports:
      - Modifier names: ``<ctrl>``, ``<shift>``, ``<alt>``, ``<cmd>``
      - Special keys: ``<escape>``, ``<space>``, ``<tab>``, ``<enter>``
      - Character keys: single characters like ``h``, ``j``
    """
    _MODIFIER_MAP: dict[str, keyboard.Key] = {
        "<ctrl>": keyboard.Key.ctrl_l,
        "<shift>": keyboard.Key.shift_l,
        "<alt>": keyboard.Key.alt_l,
        "<cmd>": keyboard.Key.cmd if platform.system() == "Darwin" else keyboard.Key.ctrl_l,
        "<super>": keyboard.Key.cmd if platform.system() == "Darwin" else keyboard.Key.ctrl_l,
    }
    _SPECIAL_MAP: dict[str, keyboard.Key] = {
        "<escape>": keyboard.Key.esc,
        "<esc>": keyboard.Key.esc,
        "<space>": keyboard.Key.space,
        "<tab>": keyboard.Key.tab,
        "<enter>": keyboard.Key.enter,
        "<return>": keyboard.Key.enter,
        "<backspace>": keyboard.Key.backspace,
        "<delete>": keyboard.Key.delete,
    }

    parts = [p.strip().lower() for p in hotkey_str.split("+")]
    keys: set[keyboard.Key | keyboard.KeyCode] = set()

    for part in parts:
        if part in _MODIFIER_MAP:
            keys.add(_MODIFIER_MAP[part])
        elif part in _SPECIAL_MAP:
            keys.add(_SPECIAL_MAP[part])
        elif len(part) == 1:
            keys.add(keyboard.KeyCode.from_char(part))
        else:
            logger.warning("Unknown key in hotkey string: %r", part)

    return keys


class HotkeyManager:
    """Manage global hotkeys across platforms."""

    def __init__(self) -> None:
        self._bindings: dict[str, tuple[set, Callable[[], None]]] = {}
        self._pressed: set[keyboard.Key | keyboard.KeyCode] = set()
        self._listener: keyboard.Listener | None = None
        self._running = False
        self._lock = threading.Lock()

    def register(self, name: str, hotkey_str: str, callback: Callable[[], None]) -> None:
        """Register a hotkey binding.

        Args:
            name: Descriptive name (e.g. "toggle_recording").
            hotkey_str: Hotkey string (e.g. "<ctrl>+<shift>+h").
            callback: Function to call when the hotkey is pressed.
        """
        keys = _parse_hotkey(hotkey_str)
        self._bindings[name] = (keys, callback)
        logger.info("Registered hotkey %r: %s → %s", name, hotkey_str, callback.__name__)

    def unregister(self, name: str) -> None:
        """Remove a hotkey binding."""
        if name in self._bindings:
            del self._bindings[name]
            logger.info("Unregistered hotkey: %s", name)

    def start(self) -> None:
        """Start listening for global hotkeys in a background thread."""
        if self._running:
            logger.warning("Hotkey listener already running")
            return

        self._running = True
        self._listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release,
        )
        self._listener.daemon = True
        self._listener.start()
        logger.info("Hotkey listener started")

    def stop(self) -> None:
        """Stop the hotkey listener."""
        self._running = False
        if self._listener:
            self._listener.stop()
            self._listener = None
        self._pressed.clear()
        logger.info("Hotkey listener stopped")

    @property
    def is_running(self) -> bool:
        return self._running

    def get_bindings(self) -> dict[str, str]:
        """Return current bindings as {name: hotkey_description}."""
        result = {}
        for name, (keys, _) in self._bindings.items():
            parts = []
            for k in keys:
                if isinstance(k, keyboard.Key):
                    parts.append(f"<{k.name}>")
                else:
                    parts.append(str(k.char) if k.char else str(k))
            result[name] = "+".join(sorted(parts))
        return result

    # -- Internal --------------------------------------------------------------

    def _normalize_key(self, key: keyboard.Key | keyboard.KeyCode) -> keyboard.Key | keyboard.KeyCode:
        """Normalize left/right modifier variants to left only."""
        _NORMALIZE: dict[keyboard.Key, keyboard.Key] = {
            keyboard.Key.ctrl_r: keyboard.Key.ctrl_l,
            keyboard.Key.shift_r: keyboard.Key.shift_l,
            keyboard.Key.alt_r: keyboard.Key.alt_l,
            keyboard.Key.alt_gr: keyboard.Key.alt_l,
        }
        if isinstance(key, keyboard.Key):
            return _NORMALIZE.get(key, key)
        return key

    def _on_press(self, key: keyboard.Key | keyboard.KeyCode) -> None:
        normalized = self._normalize_key(key)
        with self._lock:
            self._pressed.add(normalized)

        # Check each binding
        for name, (target_keys, callback) in self._bindings.items():
            if target_keys.issubset(self._pressed):
                logger.debug("Hotkey triggered: %s", name)
                # Run callback in a separate thread to avoid blocking the listener
                threading.Thread(target=callback, daemon=True, name=f"hotkey-{name}").start()

    def _on_release(self, key: keyboard.Key | keyboard.KeyCode) -> None:
        normalized = self._normalize_key(key)
        with self._lock:
            self._pressed.discard(normalized)
