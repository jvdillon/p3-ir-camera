"""Two radiometric snapshots with a shared scale and independent pixel inspection."""

from dataclasses import replace
from pathlib import Path
from tkinter import ttk

import copy
import tkinter as tk

import numpy as np

from .acquisition import Frame
from .analysis import AnalysisState
from .canvas import ThermalCanvas
from .export import load_raw_image, load_snapshot
from .i18n import filedialog, messagebox
from .processing import (
    PALETTES,
    DisplaySettings,
    Processor,
    legend_colors,
    orient,
    sensor_coordinates,
    temperature,
)
from .widgets import button, caption, checkbox


class ComparisonPanel(ttk.Frame):
    """Keep comparison data separate from the live/offline application frame."""

    def __init__(self, parent, app):
        super().__init__(parent.viewport, padding=8)
        self.app = app
        self.items: list[tuple[Frame, AnalysisState, str] | None] = [None, None]
        self.active_index = 0
        self.rotation = 0
        self.mirror = False
        self.sources = ["saved", "saved"]
        self.latest_live = None
        self.processors = [Processor(), Processor()]
        self._legend_key = None
        self.canvases = []
        self.image_boxes = []
        self.labels = []
        self.palette = tk.StringVar(value="Inferno")
        self.correct = tk.BooleanVar(value=False)
        self.palette_box = ttk.Combobox(
            self, state="readonly", textvariable=self.palette
        )
        self.palette_box.pack(fill="x")

        def palette_selected(_event):
            self.app.translator.sync_source(self.palette_box)
            self.render()

        self.palette_box.bind("<<ComboboxSelected>>", palette_selected)
        view_controls = ttk.Frame(self)
        view_controls.pack(fill="x", pady=3)
        button(
            view_controls,
            "Rotate 90°",
            self.rotate,
            side="left",
            expand=True,
            fill="x",
            padx=2,
        )
        button(
            view_controls,
            "Mirror",
            self.flip,
            side="left",
            expand=True,
            fill="x",
            padx=2,
        )
        self.bind("<Map>", lambda _: self.refresh_palettes())
        checkbox(
            self,
            "Apply each file's saved correction",
            self.correct,
            self.render,
            anchor="w",
        )
        self.link_view = tk.BooleanVar(value=True)
        checkbox(self, "Link pan/zoom for A and B", self.link_view, anchor="w")
        self.split_view = tk.BooleanVar(value=False)
        checkbox(self, "Split view", self.split_view, self.toggle_split, anchor="w")
        self.split_position = tk.DoubleVar(value=50)
        self.split_slider = ttk.Scale(
            self,
            from_=0,
            to=100,
            variable=self.split_position,
            command=lambda _: self.update_split(),
        )
        self.split_slider.pack(fill="x", pady=(0, 5))
        caption(
            self,
            "Shared min/max scale. RAW coordinates are compared without image registration.",
            wraplength=440,
            fill="x",
            pady=5,
        )
        self.scale_bar = tk.Canvas(self, width=256, height=22, highlightthickness=0)
        self.scale_bar.pack(pady=10)
        self.images = ttk.Frame(app.viewer_stack)
        self.split_canvas = ThermalCanvas(self.images, self.split_pixel, lambda _: None)
        self.split_canvas.markers = False
        self.split_canvas.show_legend = False
        self.split_canvas.overlay_drawer = self.draw_split_boundary
        self.split_dragging = False
        split_controls = ttk.Frame(self.images)
        split_controls.pack(fill="x", pady=2)
        for title, action in (
            ("Fit", self.split_canvas.fit),
            ("+", lambda: self.split_canvas.zoom(1.25)),
            ("−", lambda: self.split_canvas.zoom(0.8)),
        ):
            ttk.Button(split_controls, text=title, command=action, width=5).pack(
                side="left", padx=2
            )
        self.split_controls = split_controls
        self.split_controls.pack_forget()
        split_tag = f"SplitBoundaryDrag{id(self.split_canvas)}"
        self.split_canvas.bindtags((split_tag,) + self.split_canvas.bindtags())
        self.split_canvas.bind_class(
            split_tag, "<ButtonPress-1>", self.start_split_drag
        )
        self.split_canvas.bind_class(split_tag, "<B1-Motion>", self.drag_split)
        self.split_canvas.bind_class(
            split_tag, "<ButtonRelease-1>", self.finish_split_drag
        )
        self.split_canvas.pack_forget()
        self.split_canvas.bind("<Configure>", lambda _: self.update_split(), add="+")
        for index, name in enumerate(("A", "B")):
            image_box = ttk.Frame(self.images, padding=4)
            image_box.pack(side="left", fill="both", expand=True)
            self.image_boxes.append(image_box)
            button(self, f"Open image {name}…", lambda i=index: self.open(i))
            row = ttk.Frame(self)
            row.pack(fill="x", pady=3)
            button(
                row,
                f"Freeze {name}",
                lambda i=index: self.freeze(i),
                side="left",
                expand=True,
                fill="x",
            )
            button(
                row,
                f"Live {name}",
                lambda i=index: self.live(i),
                side="left",
                expand=True,
                fill="x",
            )
            label = tk.StringVar(value=f"{name}: no image")
            self.labels.append(label)
            ttk.Label(image_box, textvariable=label, wraplength=340).pack(fill="x")
            canvas = ThermalCanvas(
                image_box, lambda sample, i=index: self.pixel(i, sample), lambda _: None
            )
            canvas.bind(
                "<ButtonPress-1>",
                lambda _, i=index: setattr(self, "active_index", i),
                add="+",
            )
            canvas.configure(height=185, width=400)
            canvas.pack(fill="both", expand=True, pady=4)
            canvas.markers = False
            canvas.show_legend = False
            controls = ttk.Frame(image_box)
            controls.pack(fill="x")
            for title, action in (
                ("Fit", canvas.fit),
                ("+", lambda c=canvas: c.zoom(1.25)),
                ("−", lambda c=canvas: c.zoom(0.8)),
            ):
                ttk.Button(
                    controls,
                    text=title,
                    command=lambda f=action, c=canvas: (f(), self.sync_canvas(c)),
                    width=5,
                ).pack(side="left", padx=2)
            self.canvases.append(canvas)
            canvas.bind("<B1-Motion>", self.sync_view, add="+")
            canvas.bind("<ButtonRelease-1>", self.sync_view, add="+")
            canvas.bind("<MouseWheel>", self.sync_view, add="+")
            canvas.bind("<Button-4>", self.sync_view, add="+")
            canvas.bind("<Button-5>", self.sync_view, add="+")
        for label in self.labels:
            ttk.Label(self.split_controls, textvariable=label).pack(anchor="w")
        self.summary = tk.StringVar(
            value="Open two NPZ / native RAW PNG / uint16 NPY images"
        )
        ttk.Label(self, textvariable=self.summary, wraplength=440).pack(
            fill="x", pady=5
        )
        self.refresh_palettes()

    def sync_view(self, event):
        self.sync_canvas(event.widget)

    def sync_canvas(self, source):
        if not self.link_view.get() or self.split_view.get():
            return
        if source not in self.canvases:
            return
        for canvas in self.canvases:
            if canvas is source or canvas.raw is None:
                continue
            canvas.auto_fit = False
            canvas.pixel_scale = source.pixel_scale
            canvas.offset = source.offset.copy()
            canvas.redraw()

    def toggle_split(self):
        if self.split_view.get():
            for box in self.image_boxes:
                box.pack_forget()
            self.split_canvas.pack(fill="both", expand=True, pady=4)
            self.split_controls.pack(fill="x", pady=2)
            self.update_split()
        else:
            self.split_canvas.pack_forget()
            self.split_controls.pack_forget()
            for box in self.image_boxes:
                box.pack(side="left", fill="both", expand=True)

    def split_boundary_x(self):
        return (
            self.split_canvas.offset[0]
            + getattr(self, "split_boundary", 0) * self.split_canvas.pixel_scale
        )

    def start_split_drag(self, event):
        if not self.split_view.get() or self.split_canvas.raw is None:
            return
        self.split_dragging = abs(event.x - self.split_boundary_x()) <= 12
        if self.split_dragging:
            return "break"

    def drag_split(self, event):
        if not self.split_dragging or self.split_canvas.raw is None:
            return
        width = self.split_canvas.raw.shape[1] * self.split_canvas.pixel_scale
        position = (event.x - self.split_canvas.offset[0]) / max(width, 1)
        self.split_position.set(max(0, min(100, position * 100)))
        self.update_split()
        return "break"

    def finish_split_drag(self, event):
        if self.split_dragging:
            self.split_dragging = False
            return "break"

    def update_split(self):
        if not self.split_view.get():
            return
        if any(item is None for item in self.items):
            self.split_canvas.clear("Open two images for split view")
            return
        first, second = self.canvases
        if (
            first.rgb is None
            or second.rgb is None
            or first.rgb.shape != second.rgb.shape
        ):
            self.split_canvas.clear("Split view requires equal image dimensions")
            return
        width = first.rgb.shape[1]
        boundary = int(width * self.split_position.get() / 100)
        self.split_boundary = boundary

        def combine(a, b):
            return np.concatenate((a[:, :boundary], b[:, boundary:]), axis=1)

        self.split_canvas.measurements = combine(
            first.measurements, second.measurements
        )
        self.split_canvas.corrected = self.correct.get()
        self.split_canvas.set_frame(
            combine(first.rgb, second.rgb),
            combine(first.raw, second.raw),
            first.coordinates,
        )

    def split_pixel(self, sample):
        """Use the actual A/B side after rotation, rather than always reading A."""
        if sample is None or self.split_canvas.raw is None:
            return
        x, y, _ = sample
        _, column = self.split_canvas.inverse_coordinates[y, x]
        self.pixel(int(column >= self.split_boundary), sample)

    def draw_split_boundary(self):
        if not self.split_view.get() or self.split_canvas.raw is None:
            return
        boundary = getattr(self, "split_boundary", 0)
        self.split_canvas.delete("split_boundary")
        self.split_canvas.create_line(
            self.split_canvas.offset[0] + boundary * self.split_canvas.pixel_scale,
            self.split_canvas.offset[1],
            self.split_canvas.offset[0] + boundary * self.split_canvas.pixel_scale,
            self.split_canvas.offset[1]
            + self.split_canvas.raw.shape[0] * self.split_canvas.pixel_scale,
            fill="#ffffff",
            width=2,
            tags="split_boundary",
        )

    def refresh_palettes(self):
        self.palette_box.configure(
            values=list(PALETTES) + sorted(self.app.library.palettes)
        )

    def rotate(self):
        self.rotation = (self.rotation + 1) % 4
        for canvas in (*self.canvases, self.split_canvas):
            canvas.auto_fit = True
        self.render()

    def flip(self):
        self.mirror = not self.mirror
        self.render()

    def open(self, index):
        path = filedialog.askopenfilename(
            filetypes=[("Radiometric image", "*.npz *.png *.npy")]
        )
        if path:
            try:
                self.load(index, path)
            except (ValueError, OSError, KeyError, TypeError) as exc:
                messagebox.showerror("Compare", str(exc), parent=self)

    def load(self, index, path):
        """Validate the complete input before replacing one comparison slot."""
        frame, metadata = (
            load_snapshot(path)
            if Path(path).suffix.lower() == ".npz"
            else load_raw_image(path)
        )
        state = AnalysisState.from_dict(metadata.get("analysis"), frame.raw.shape)
        self.sources[index] = "saved"
        self.items[index] = (frame, state, Path(path).name)
        self.render()

    @property
    def needs_live(self):
        return "live" in self.sources

    def freeze(self, index):
        """Copy the displayed comparison slot, or the working frame if empty."""
        item = self.items[index]
        frame = item[0] if item else self.app.frame
        if frame is None:
            self.app.status.set("No frame to freeze")
            return
        state = item[1] if item else self.app.analysis
        self.items[index] = (
            Frame(frame.raw.copy(), frame.brightness.copy(), frame.timestamp),
            copy.deepcopy(state),
            "Frozen frame",
        )
        self.sources[index] = "frozen"
        self.render()

    def live(self, index):
        """Subscribe to acquisition without replacing the working offline snapshot."""
        self.sources[index] = "live"
        self.items[index] = None
        if (
            self.latest_live is not None
            and self.app.worker
            and self.app.worker.is_alive()
        ):
            self.receive_live(self.latest_live)
        else:
            self.clear_live()
        self.app.ensure_comparison_stream()

    def receive_live(self, frame):
        self.latest_live = frame
        for index, source in enumerate(self.sources):
            if source == "live":
                # Live material masks may have incompatible dimensions after reconnect.
                state = (
                    self.app.analysis
                    if self.app.analysis_shape == frame.raw.shape
                    else AnalysisState()
                )
                self.items[index] = (frame, state, "Live camera")
        if self.needs_live and self.app.sidebar.selection.get() == "Compare":
            self.render()

    def clear_live(self):
        self.latest_live = None
        for index, source in enumerate(self.sources):
            if source == "live":
                self.items[index] = None
                canvas = self.canvases[index]
                canvas.clear("No camera · waiting for reconnection…")
                self.labels[index].set(f"{'AB'[index]}: waiting for live camera")
        self.render()

    def pixel(self, index, sample):
        if sample is None or self.items[index] is None:
            return
        x, y, raw = sample
        value = self.canvases[index].native_measurements[y, x]
        self.labels[index].set(
            f"{'AB'[index]}: ({x}, {y}) · {value:.6f} °C · RAW {raw}"
        )

    def render(self):
        planes = [
            None
            if item is None
            else (
                item[1].celsius(item[0].raw)
                if self.correct.get()
                else temperature(item[0].raw)
            )
            for item in self.items
        ]
        finite = [
            p[np.isfinite(p)] for p in planes if p is not None and np.isfinite(p).any()
        ]
        if not finite:
            self.summary.set("No valid comparison samples")
            self.update_split()
            return
        low, high = (
            min(float(p.min()) for p in finite),
            max(float(p.max()) for p in finite),
        )
        high = max(high, low + 0.015625)
        settings = DisplaySettings(
            palette=self.palette.get(), range_mode="Fixed", minimum=low, maximum=high
        )
        custom = self.app.library.palettes.get(settings.palette)
        if custom is None and settings.palette not in PALETTES:
            settings = replace(settings, palette="Inferno")
        legend = legend_colors(settings, custom)
        legend_key = (settings.palette, custom)
        if legend_key != self._legend_key:
            self._legend_key = legend_key
            self.scale_bar.delete("all")
            for x, color in enumerate(legend):
                self.scale_bar.create_line(
                    x, 0, x, 22, fill=f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"
                )
        for i, item in enumerate(self.items):
            if item is None:
                continue
            frame, _, name = item
            measured = planes[i]
            rgb, _ = self.processors[i].render(
                frame.raw, frame.brightness, settings, custom, measured
            )
            rgb = orient(rgb, self.rotation, self.mirror)
            oriented_raw = orient(frame.raw, self.rotation, self.mirror)
            oriented_measured = orient(measured, self.rotation, self.mirror)
            oriented_coordinates = sensor_coordinates(
                frame.raw.shape, self.rotation, self.mirror
            )
            rgb[~np.isfinite(oriented_measured)] = (80, 80, 80)
            canvas = self.canvases[i]
            canvas.measurements = oriented_measured
            canvas.native_measurements = measured
            canvas.corrected = self.correct.get()
            canvas.legend_data = (legend, low, high, "°C")
            canvas.set_frame(rgb, oriented_raw, oriented_coordinates)
            self.labels[i].set(f"{'AB'[i]}: {name}")
        text = f"Common range: {low:.3f}–{high:.3f} °C"
        a, b = planes
        if a is not None and b is not None:
            if a.shape == b.shape:
                delta = b - a
                valid = delta[np.isfinite(delta)]
                if valid.size:
                    text += f"\nB − A: mean {valid.mean():+.3f} °C; min {valid.min():+.3f}; max {valid.max():+.3f}"
            else:
                text += "\nDifferent sensor dimensions: no pixelwise difference."
        self.summary.set(text)
        self.update_split()
