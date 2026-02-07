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


class FloatingWidget:
    """Floating circular widget for voice input."""

    SIZE = 64
    COLORS = {
        WidgetState.IDLE: {"bg": "#2563eb", "fg": "#ffffff", "ring": "#1d4ed8"},
        WidgetState.RECORDING: {"bg": "#dc2626", "fg": "#ffffff", "ring": "#991b1b"},
        WidgetState.TRANSCRIBING: {"bg": "#f59e0b", "fg": "#ffffff", "ring": "#b45309"},
    }

    def __init__(
        self,
        on_toggle: Callable[[], None] | None = None,
        on_quit: Callable[[], None] | None = None,
    ) -> None:
        self.on_toggle = on_toggle
        self.on_quit = on_quit
        self.state = WidgetState.IDLE
        self._recording_start: float = 0
        self._pulse_phase: float = 0
        self._spinner_angle: float = 0
        self._drag_data = {"x": 0, "y": 0}
        self._animating = False
        self._toast_text: str = ""
        self._toast_until: float = 0

        self._build_ui()

    def _build_ui(self) -> None:
        self.root = tk.Tk()
        self.root.title("Hola Audio")
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)
        self.root.attributes("-type", "dock")

        # Transparent background
        self.root.configure(bg="black")
        try:
            self.root.attributes("-transparentcolor", "black")
        except tk.TclError:
            pass

        # Position: bottom-right of screen
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        x = screen_w - self.SIZE - 40
        y = screen_h - self.SIZE - 100
        self.root.geometry(f"{self.SIZE + 200}x{self.SIZE + 40}+{x}+{y}")

        # Canvas for drawing
        self.canvas = tk.Canvas(
            self.root,
            width=self.SIZE + 200,
            height=self.SIZE + 40,
            bg="black",
            highlightthickness=0,
            cursor="hand2",
        )
        self.canvas.pack()

        # Bind events
        self.canvas.bind("<Button-1>", self._on_click)
        self.canvas.bind("<Button-3>", self._on_right_click)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonPress-1>", self._on_drag_start)

        # Context menu
        self.menu = tk.Menu(self.root, tearoff=0)
        self.menu.add_command(label="Quit", command=self._quit)

        self._draw()

    def _draw(self) -> None:
        """Redraw the widget based on current state."""
        self.canvas.delete("all")
        colors = self.COLORS[self.state]
        cx, cy = self.SIZE // 2 + 4, self.SIZE // 2 + 4
        r = self.SIZE // 2

        # Outer ring / pulse effect
        if self.state == WidgetState.RECORDING:
            pulse = 3 + 3 * math.sin(self._pulse_phase)
            ring_r = r + int(pulse)
            self.canvas.create_oval(
                cx - ring_r, cy - ring_r, cx + ring_r, cy + ring_r,
                fill="", outline="#ef4444", width=2,
            )

        # Main circle with shadow
        self.canvas.create_oval(
            cx - r + 2, cy - r + 2, cx + r + 2, cy + r + 2,
            fill="#333333", outline="",
        )
        self.canvas.create_oval(
            cx - r, cy - r, cx + r, cy + r,
            fill=colors["bg"], outline=colors["ring"], width=2,
        )

        # Icon based on state
        if self.state == WidgetState.IDLE:
            self._draw_mic_icon(cx, cy, colors["fg"])
        elif self.state == WidgetState.RECORDING:
            self._draw_recording_icon(cx, cy, colors["fg"])
        elif self.state == WidgetState.TRANSCRIBING:
            self._draw_spinner(cx, cy, colors["fg"])

        # Status text below circle
        if self.state == WidgetState.RECORDING:
            elapsed = time.monotonic() - self._recording_start
            mins, secs = divmod(int(elapsed), 60)
            time_text = f"{mins}:{secs:02d}"
            self.canvas.create_text(
                cx, cy + r + 14,
                text=f"● REC {time_text}", fill="#dc2626",
                font=("sans-serif", 9, "bold"),
            )
        elif self.state == WidgetState.TRANSCRIBING:
            self.canvas.create_text(
                cx, cy + r + 14,
                text="Transcribing...", fill="#f59e0b",
                font=("sans-serif", 9),
            )

        # Toast notification
        if self._toast_text and time.monotonic() < self._toast_until:
            toast_x = cx + r + 12
            toast_y = cy
            display_text = self._toast_text[:50]
            if len(self._toast_text) > 50:
                display_text += "..."
            bg_width = len(display_text) * 7 + 20
            self.canvas.create_rectangle(
                toast_x, toast_y - 14, toast_x + bg_width, toast_y + 14,
                fill="#1f2937", outline="#374151", width=1,
            )
            self.canvas.create_text(
                toast_x + 10, toast_y,
                text=f"✓ {display_text}", fill="#10b981",
                font=("sans-serif", 9), anchor="w",
            )

    def _draw_mic_icon(self, cx: int, cy: int, color: str) -> None:
        """Draw a microphone icon."""
        # Mic body
        w, h = 6, 10
        self.canvas.create_rectangle(
            cx - w, cy - h, cx + w, cy + 2,
            fill=color, outline=color, width=1,
        )
        self.canvas.create_oval(
            cx - w, cy - h - 3, cx + w, cy - h + 5,
            fill=color, outline=color,
        )
        self.canvas.create_oval(
            cx - w, cy - 2, cx + w, cy + 6,
            fill=color, outline=color,
        )
        # Mic arc
        self.canvas.create_arc(
            cx - 10, cy - 8, cx + 10, cy + 12,
            start=200, extent=140, style="arc",
            outline=color, width=2,
        )
        # Stand
        self.canvas.create_line(cx, cy + 12, cx, cy + 18, fill=color, width=2)
        self.canvas.create_line(cx - 6, cy + 18, cx + 6, cy + 18, fill=color, width=2)

    def _draw_recording_icon(self, cx: int, cy: int, color: str) -> None:
        """Draw a stop square (recording indicator)."""
        s = 10
        self.canvas.create_rectangle(
            cx - s, cy - s, cx + s, cy + s,
            fill=color, outline=color, width=0,
        )

    def _draw_spinner(self, cx: int, cy: int, color: str) -> None:
        """Draw a spinning arc."""
        r = 12
        start = self._spinner_angle
        self.canvas.create_arc(
            cx - r, cy - r, cx + r, cy + r,
            start=start, extent=270, style="arc",
            outline=color, width=3,
        )

    def _on_click(self, event: tk.Event) -> None:
        """Handle click — toggle recording if within the circle."""
        cx, cy = self.SIZE // 2 + 4, self.SIZE // 2 + 4
        dist = math.sqrt((event.x - cx) ** 2 + (event.y - cy) ** 2)
        if dist <= self.SIZE // 2 and self.on_toggle:
            self.on_toggle()

    def _on_right_click(self, event: tk.Event) -> None:
        self.menu.post(event.x_root, event.y_root)

    def _on_drag_start(self, event: tk.Event) -> None:
        self._drag_data["x"] = event.x
        self._drag_data["y"] = event.y

    def _on_drag(self, event: tk.Event) -> None:
        dx = event.x - self._drag_data["x"]
        dy = event.y - self._drag_data["y"]
        x = self.root.winfo_x() + dx
        y = self.root.winfo_y() + dy
        self.root.geometry(f"+{x}+{y}")

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
        """Show a brief toast message next to the widget."""
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
