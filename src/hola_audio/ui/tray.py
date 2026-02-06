"""System tray icon for Hola Audio.

Provides a persistent system tray presence with:
  - Recording status indicator (icon color change)
  - Quick access menu (start/stop, settings, quit)
  - Desktop notifications for transcription results
"""

from __future__ import annotations

import logging
import platform
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Callable

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    pass  # forward refs only


def _create_icon_image(color: str = "green", size: int = 64) -> "PIL.Image.Image":
    """Create a simple colored circle icon."""
    from PIL import Image, ImageDraw

    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    color_map = {
        "green": (76, 175, 80, 255),
        "red": (244, 67, 54, 255),
        "yellow": (255, 193, 7, 255),
        "gray": (158, 158, 158, 255),
    }
    fill = color_map.get(color, color_map["green"])

    # Draw circle with padding
    padding = 4
    draw.ellipse(
        [padding, padding, size - padding, size - padding],
        fill=fill,
        outline=(255, 255, 255, 200),
        width=2,
    )

    # Draw microphone shape
    mic_color = (255, 255, 255, 230)
    cx, cy = size // 2, size // 2
    # Mic body
    draw.rounded_rectangle(
        [cx - 6, cy - 14, cx + 6, cy + 4],
        radius=6,
        fill=mic_color,
    )
    # Mic stand
    draw.arc([cx - 10, cy - 6, cx + 10, cy + 10], start=0, end=180, fill=mic_color, width=2)
    draw.line([cx, cy + 10, cx, cy + 16], fill=mic_color, width=2)
    draw.line([cx - 6, cy + 16, cx + 6, cy + 16], fill=mic_color, width=2)

    return img


class SystemTray:
    """Cross-platform system tray icon and menu."""

    def __init__(
        self,
        on_toggle: Callable[[], None] | None = None,
        on_correct: Callable[[], None] | None = None,
        on_settings: Callable[[], None] | None = None,
        on_quit: Callable[[], None] | None = None,
    ) -> None:
        self._on_toggle = on_toggle
        self._on_correct = on_correct
        self._on_settings = on_settings
        self._on_quit = on_quit
        self._icon = None
        self._recording = False

    def start(self) -> None:
        """Start the system tray icon in a background thread."""
        try:
            import pystray
        except ImportError:
            logger.warning("pystray not installed – system tray disabled. Install with: pip install 'hola-audio[tray]'")
            return

        icon_image = _create_icon_image("green")
        menu = pystray.Menu(
            pystray.MenuItem("Toggle Recording", self._handle_toggle, default=True),
            pystray.MenuItem("Correct Clipboard", self._handle_correct),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Settings...", self._handle_settings),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Quit", self._handle_quit),
        )

        self._icon = pystray.Icon(
            name="hola-audio",
            icon=icon_image,
            title="Hola Audio - Ready",
            menu=menu,
        )

        thread = threading.Thread(target=self._icon.run, daemon=True, name="tray-icon")
        thread.start()
        logger.info("System tray icon started")

    def stop(self) -> None:
        """Stop the system tray icon."""
        if self._icon:
            self._icon.stop()
            self._icon = None
        logger.info("System tray icon stopped")

    def set_recording(self, recording: bool) -> None:
        """Update the tray icon to reflect recording state."""
        self._recording = recording
        if self._icon:
            try:
                color = "red" if recording else "green"
                self._icon.icon = _create_icon_image(color)
                self._icon.title = "Hola Audio - Recording..." if recording else "Hola Audio - Ready"
            except Exception:
                logger.debug("Failed to update tray icon", exc_info=True)

    def notify(self, title: str, message: str) -> None:
        """Show a desktop notification."""
        if self._icon:
            try:
                self._icon.notify(message, title)
            except Exception:
                logger.debug("Failed to show notification", exc_info=True)

    # -- Menu handlers ---------------------------------------------------------

    def _handle_toggle(self) -> None:
        if self._on_toggle:
            self._on_toggle()

    def _handle_correct(self) -> None:
        if self._on_correct:
            self._on_correct()

    def _handle_settings(self) -> None:
        if self._on_settings:
            self._on_settings()

    def _handle_quit(self) -> None:
        self.stop()
        if self._on_quit:
            self._on_quit()
