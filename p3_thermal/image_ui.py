"""Explicit image-export choices keep presentation images distinct from sensor RAW."""

from tkinter import ttk

import tkinter as tk

import cv2

from .export import save_image
from .i18n import filedialog, messagebox
from .widgets import button, caption, checkbox


def image_dialog(parent, frame, rgb, on_saved, legend=None, host=None):
    """Capture frame/RGB when opening the dialog, so a live feed cannot change the export."""
    dialog = host.editor("Save image") if host is not None else ttk.Frame(parent)
    box = ttk.Frame(dialog, padding=18)
    box.pack(fill="both", expand=True)
    kind = tk.StringVar(value="jpeg")
    for label, value in [
        ("JPEG · smooth color image", "jpeg"),
        ("PNG · native 16-bit sensor RAW", "raw_png"),
        ("PNG · current color image (lossless)", "color_png"),
    ]:
        ttk.Radiobutton(box, text=label, value=value, variable=kind).pack(
            anchor="w", pady=5
        )
    include_legend = tk.BooleanVar(value=legend is not None)
    checkbox(
        box, "Include legend (color images only)", include_legend, anchor="w", pady=6
    )
    scale, quality = tk.StringVar(value="3"), tk.StringVar(value="95")
    for label, variable, low, high in [
        ("JPEG enlargement (bicubic)", scale, 1, 8),
        ("JPEG quality", quality, 1, 100),
    ]:
        caption(box, label, anchor="w", pady=(10, 2))
        ttk.Spinbox(box, from_=low, to=high, textvariable=variable, width=10).pack(
            anchor="w"
        )
    caption(
        box,
        "RAW PNG preserves sensor counts and native orientation.\nIt may look dark in ordinary image viewers.\nUse full NPZ data to reopen all channels in this application.",
        wraplength=440,
        anchor="w",
        pady=12,
    )

    def save():
        try:
            factor, q = (
                (int(scale.get()), int(quality.get()))
                if kind.get() == "jpeg"
                else (1, 95)
            )
            if not 1 <= factor <= 8 or not 1 <= q <= 100:
                raise ValueError("JPEG enlargement: 1–8; quality: 1–100")
            ext = ".jpg" if kind.get() == "jpeg" else ".png"
            path = filedialog.asksaveasfilename(
                parent=dialog, defaultextension=ext, filetypes=[("Image", "*" + ext)]
            )
            if not path:
                return
            save_image(
                path,
                frame,
                rgb,
                kind.get(),
                factor,
                q,
                legend if include_legend.get() else None,
            )
            on_saved(path)
            if host is not None:
                host.select("View")
        except (ValueError, OSError, cv2.error) as exc:
            messagebox.showerror("Save failed", str(exc), parent=dialog)

    button(box, "Save…", save)
