"""Settings dialog for the floating widget.

Opens from the right-click menu. Allows users to configure
API key and toggle AI correction without editing config files.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Callable


class SettingsDialog:
    """Modal settings dialog for widget configuration."""

    def __init__(
        self,
        parent: tk.Tk,
        api_key: str = "",
        correction_enabled: bool = False,
        correction_api_key: str = "",
        correction_endpoint: str = "",
        correction_model: str = "",
        on_save: Callable[[dict], None] | None = None,
    ) -> None:
        self.on_save = on_save

        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Hola Audio Settings")
        self.dialog.attributes("-topmost", True)
        self.dialog.resizable(False, False)
        self.dialog.configure(bg="#1e1e1e")

        # Center on screen
        w, h = 420, 340
        x = (self.dialog.winfo_screenwidth() - w) // 2
        y = (self.dialog.winfo_screenheight() - h) // 2
        self.dialog.geometry(f"{w}x{h}+{x}+{y}")

        # Style
        style = ttk.Style(self.dialog)
        style.theme_use("clam")
        style.configure("Dark.TLabel", background="#1e1e1e", foreground="#e0e0e0", font=("sans-serif", 10))
        style.configure("Header.TLabel", background="#1e1e1e", foreground="#ffffff", font=("sans-serif", 12, "bold"))
        style.configure("Dark.TEntry", fieldbackground="#2d2d2d", foreground="#e0e0e0")
        style.configure("Dark.TCheckbutton", background="#1e1e1e", foreground="#e0e0e0", font=("sans-serif", 10))
        style.configure("Save.TButton", font=("sans-serif", 10, "bold"))
        style.map("Dark.TCheckbutton", background=[("active", "#2d2d2d")])

        pad = {"padx": 16, "pady": 4}

        # ── ASR Section ──────────────────────────────────────────────────
        ttk.Label(self.dialog, text="🎙 Speech Recognition", style="Header.TLabel").pack(anchor="w", padx=16, pady=(16, 8))

        ttk.Label(self.dialog, text="Groq API Key:", style="Dark.TLabel").pack(anchor="w", **pad)
        self.api_key_var = tk.StringVar(value=api_key)
        api_entry = ttk.Entry(self.dialog, textvariable=self.api_key_var, width=48, show="•")
        api_entry.pack(anchor="w", padx=16, pady=(0, 4))

        # Show/hide toggle
        self._show_key = tk.BooleanVar(value=False)
        show_btn = ttk.Checkbutton(
            self.dialog, text="Show key", variable=self._show_key,
            style="Dark.TCheckbutton",
            command=lambda: api_entry.configure(show="" if self._show_key.get() else "•"),
        )
        show_btn.pack(anchor="w", padx=16, pady=(0, 8))

        # ── Separator ────────────────────────────────────────────────────
        sep = tk.Frame(self.dialog, height=1, bg="#444444")
        sep.pack(fill="x", padx=16, pady=8)

        # ── Correction Section ───────────────────────────────────────────
        ttk.Label(self.dialog, text="✨ AI Correction", style="Header.TLabel").pack(anchor="w", padx=16, pady=(4, 8))

        self.correction_var = tk.BooleanVar(value=correction_enabled)
        ttk.Checkbutton(
            self.dialog, text="Enable AI text correction after transcription",
            variable=self.correction_var, style="Dark.TCheckbutton",
        ).pack(anchor="w", **pad)

        ttk.Label(self.dialog, text="Correction API Key:", style="Dark.TLabel").pack(anchor="w", **pad)
        self.correction_key_var = tk.StringVar(value=correction_api_key)
        ttk.Entry(self.dialog, textvariable=self.correction_key_var, width=48, show="•").pack(anchor="w", padx=16, pady=(0, 4))

        ttk.Label(self.dialog, text="Endpoint:", style="Dark.TLabel").pack(anchor="w", **pad)
        self.correction_endpoint_var = tk.StringVar(value=correction_endpoint)
        ttk.Entry(self.dialog, textvariable=self.correction_endpoint_var, width=48).pack(anchor="w", padx=16, pady=(0, 4))

        ttk.Label(self.dialog, text="Model:", style="Dark.TLabel").pack(anchor="w", **pad)
        self.correction_model_var = tk.StringVar(value=correction_model)
        ttk.Entry(self.dialog, textvariable=self.correction_model_var, width=48).pack(anchor="w", padx=16, pady=(0, 12))

        # ── Buttons ──────────────────────────────────────────────────────
        btn_frame = tk.Frame(self.dialog, bg="#1e1e1e")
        btn_frame.pack(fill="x", padx=16, pady=(0, 16))

        ttk.Button(btn_frame, text="Cancel", command=self.dialog.destroy).pack(side="right", padx=(8, 0))
        ttk.Button(btn_frame, text="Save", style="Save.TButton", command=self._save).pack(side="right")

        # Focus
        self.dialog.grab_set()
        api_entry.focus_set()

    def _save(self) -> None:
        """Save settings and close."""
        settings = {
            "asr.online.api_key": self.api_key_var.get().strip(),
            "correction.enabled": self.correction_var.get(),
            "correction.api_key": self.correction_key_var.get().strip(),
            "correction.endpoint": self.correction_endpoint_var.get().strip(),
            "correction.model": self.correction_model_var.get().strip(),
        }
        if self.on_save:
            self.on_save(settings)
        self.dialog.destroy()
