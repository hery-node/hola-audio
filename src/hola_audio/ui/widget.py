"""Floating voice input widget for Linux desktop.

A small, always-on-top, draggable circular button that provides
visual recording and transcription controls. Uses tkinter (no extra deps).
"""

from __future__ import annotations

import math
import threading
import time
import tkinter as tk
from enum import Enum
from typing import Callable


class WidgetState(Enum):
    IDLE = "idle"
    RECORDING = "recording"
    TRANSCRIBING = "transcribing"


# Color definitions per state
COLORS = {
    WidgetState.IDLE: {"bg": "#2563eb", "fg": "#ffffff"},
    WidgetState.RECORDING: {"bg": "#16a34a", "fg": "#ffffff"},
    WidgetState.TRANSCRIBING: {"bg": "#3b82f6", "fg": "#ffffff"},
}


class FloatingWidget:
    """Floating circular widget for voice input."""

    SIZE = 72

    def __init__(
        self,
        on_toggle: Callable[[], None] | None = None,
        on_settings: Callable[[], None] | None = None,
        on_quit: Callable[[], None] | None = None,
    ) -> None:
        self.on_toggle = on_toggle
        self.on_settings = on_settings
        self.on_quit = on_quit
        self.state = WidgetState.IDLE
        self._recording_start: float = 0
        self._pulse_phase: float = 0
        self._spinner_angle: float = 0
        self._animating = False
        self._dragged = False
        self._drag_x = 0
        self._drag_y = 0
        self._toast_text: str = ""
        self._toast_until: float = 0

        self._build_ui()

    def _build_ui(self) -> None:
        self.root = tk.Tk()
        self.root.title("Hola Audio")
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)
        self.root.lift()
        self.root.focus_force()

        # Window size just fits the button + status text
        self._win_w = self.SIZE + 8
        self._win_h = self.SIZE + 24

        # Position: bottom-right of screen
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        x = screen_w - self._win_w - 30
        y = screen_h - self._win_h - 80
        self.root.geometry(f"{self._win_w}x{self._win_h}+{x}+{y}")
        self.root.configure(bg="#1e1e1e")

        # Canvas for drawing
        self.canvas = tk.Canvas(
            self.root,
            width=self._win_w,
            height=self._win_h,
            bg="#1e1e1e",
            highlightthickness=0,
            cursor="hand2",
        )
        self.canvas.pack(fill="both", expand=True)

        # Bind mouse events
        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.canvas.bind("<Button-3>", self._on_right_click)

        # Context menu
        self.menu = tk.Menu(self.root, tearoff=0)
        self.menu.add_command(label="⚙ Settings", command=self._open_settings)
        self.menu.add_separator()
        self.menu.add_command(label="✕ Quit", command=self._quit)

        self._draw()

    def _open_settings(self) -> None:
        if self.on_settings:
            self.on_settings()

    # ── Drawing ──────────────────────────────────────────────────────────

    def _draw(self) -> None:
        """Redraw the widget based on current state."""
        self.canvas.delete("all")
        colors = COLORS[self.state]
        cx = self._win_w // 2
        cy = self.SIZE // 2 + 2
        r = self.SIZE // 2

        # Pulse ring when recording
        if self.state == WidgetState.RECORDING:
            pulse = 2 + 2 * math.sin(self._pulse_phase)
            ring_r = r + int(pulse)
            self.canvas.create_oval(
                cx - ring_r, cy - ring_r, cx + ring_r, cy + ring_r,
                fill="", outline="#22c55e", width=2,
            )

        # Main circle
        self.canvas.create_oval(
            cx - r, cy - r, cx + r, cy + r,
            fill=colors["bg"], outline=colors["bg"], width=0,
        )

        # Inner icon
        if self.state == WidgetState.IDLE:
            self._draw_mic(cx, cy, colors["fg"])
        elif self.state == WidgetState.RECORDING:
            self._draw_stop(cx, cy, colors["fg"])
        elif self.state == WidgetState.TRANSCRIBING:
            self._draw_spinner(cx, cy, colors["fg"])

        # Status label below
        label = ""
        label_color = "#888888"
        if self.state == WidgetState.RECORDING:
            elapsed = time.monotonic() - self._recording_start
            mins, secs = divmod(int(elapsed), 60)
            label = f"● {mins}:{secs:02d}"
            label_color = "#22c55e"
        elif self.state == WidgetState.TRANSCRIBING:
            label = "..."
            label_color = "#60a5fa"

        if label:
            self.canvas.create_text(
                cx, self.SIZE + 10,
                text=label, fill=label_color,
                font=("monospace", 10, "bold"),
            )

    def _draw_mic(self, cx: int, cy: int, color: str) -> None:
        """Draw a microphone icon."""
        # Mic body
        self.canvas.create_rectangle(cx - 5, cy - 10, cx + 5, cy + 2, fill=color, outline=color)
        self.canvas.create_oval(cx - 5, cy - 13, cx + 5, cy - 7, fill=color, outline=color)
        self.canvas.create_oval(cx - 5, cy - 1, cx + 5, cy + 5, fill=color, outline=color)
        # Arc
        self.canvas.create_arc(cx - 9, cy - 7, cx + 9, cy + 9, start=200, extent=140, style="arc", outline=color, width=2)
        # Stand
        self.canvas.create_line(cx, cy + 9, cx, cy + 15, fill=color, width=2)
        self.canvas.create_line(cx - 5, cy + 15, cx + 5, cy + 15, fill=color, width=2)

    def _draw_stop(self, cx: int, cy: int, color: str) -> None:
        """Draw a stop square."""
        s = 10
        self.canvas.create_rectangle(cx - s, cy - s, cx + s, cy + s, fill=color, outline=color)

    def _draw_spinner(self, cx: int, cy: int, color: str) -> None:
        """Draw a rotating arc."""
        r = 12
        self.canvas.create_arc(
            cx - r, cy - r, cx + r, cy + r,
            start=self._spinner_angle, extent=270,
            style="arc", outline=color, width=3,
        )

    # ── Mouse events ─────────────────────────────────────────────────────

    def _on_press(self, event: tk.Event) -> None:
        self._drag_x = event.x_root
        self._drag_y = event.y_root
        self._dragged = False

    def _on_drag(self, event: tk.Event) -> None:
        dx = event.x_root - self._drag_x
        dy = event.y_root - self._drag_y
        if abs(dx) > 3 or abs(dy) > 3:
            self._dragged = True
        x = self.root.winfo_x() + dx
        y = self.root.winfo_y() + dy
        self.root.geometry(f"+{x}+{y}")
        self._drag_x = event.x_root
        self._drag_y = event.y_root

    def _on_release(self, event: tk.Event) -> None:
        if self._dragged:
            return
        if self.on_toggle:
            self.on_toggle()

    def _on_right_click(self, event: tk.Event) -> None:
        self.menu.post(event.x_root, event.y_root)

    # ── State management ─────────────────────────────────────────────────

    def set_state(self, state: WidgetState) -> None:
        """Update widget state and redraw."""
        self.state = state
        if state == WidgetState.RECORDING:
            self._recording_start = time.monotonic()
            self._start_animation()
        elif state == WidgetState.TRANSCRIBING:
            self._start_animation()
        else:
            self._animating = False
        self._draw()

    def show_toast(self, text: str, duration: float = 3.0) -> None:
        """Show a brief toast message."""
        self._toast_text = text
        self._toast_until = time.monotonic() + duration
        self._draw()

    def _start_animation(self) -> None:
        if self._animating:
            return
        self._animating = True
        self._animate()

    def _animate(self) -> None:
        if not self._animating:
            return
        if self.state == WidgetState.RECORDING:
            self._pulse_phase += 0.15
        elif self.state == WidgetState.TRANSCRIBING:
            self._spinner_angle += 10
        self._draw()
        self.root.after(50, self._animate)

    def _quit(self) -> None:
        self._animating = False
        if self.on_quit:
            self.on_quit()
        self.root.quit()
        self.root.destroy()

    def run(self) -> None:
        """Start the tkinter main loop."""
        self.root.mainloop()

    def schedule(self, callback: Callable[[], None]) -> None:
        """Schedule a callback on the main thread."""
        self.root.after(0, callback)
