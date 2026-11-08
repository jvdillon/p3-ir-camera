"""Thermal canvas with sensor-coordinate picking and temperature pixel labels."""

from collections.abc import Callable

import tkinter as tk

import cv2
import numpy as np

from .i18n import translate
from .processing import pixel_label


class ThermalCanvas(tk.Canvas):
    """Zoom is screen pixels per sensor pixel. Pan and picking share one transform."""

    def __init__(self, parent, on_pixel, on_zoom):
        super().__init__(parent, background="#10151d", highlightthickness=0)
        self.on_pixel, self.on_zoom = on_pixel, on_zoom
        self.rgb = self.raw = self.coordinates = None
        self.pixel_scale = 3.0
        self.tool = "Pan"
        self.on_gesture: Callable | None = None
        self.regions = []
        self.measurements: np.ndarray | None = None
        self.native_measurements: np.ndarray | None = None
        self.corrected = False
        self.gesture_start = None
        self.gesture_last = None
        self.offset = [0.0, 0.0]
        self.auto_fit = True
        self.markers = True
        self.show_legend = True
        self.legend_data: tuple | None = None
        self.legend_rect = None
        self.legend_photo = None
        self.photo = None
        self.overlay_drawer: Callable[[], None] | None = None
        self.empty_message = "Waiting for thermal frames…"
        self.bind(
            "<Configure>", lambda e: self.fit() if self.auto_fit else self.redraw()
        )
        self.bind("<Motion>", self.hover)
        self.bind("<Leave>", lambda e: self.on_pixel(None))
        self.bind(
            "<MouseWheel>", lambda e: self.zoom(1.25 if e.delta > 0 else 0.8, e.x, e.y)
        )
        self.bind("<Button-4>", lambda e: self.zoom(1.25, e.x, e.y))
        self.bind("<Button-5>", lambda e: self.zoom(0.8, e.x, e.y))
        self.bind("<ButtonPress-1>", self.start_pan)
        self.bind("<B1-Motion>", self.pan)
        self.bind("<ButtonRelease-1>", self.finish_gesture)
        self.bind("<Double-Button-1>", lambda e: self.fit())

    def create_text(self, *args, **kwargs):
        if "text" in kwargs:
            kwargs["text"] = translate(kwargs["text"])
        return super().create_text(*args, **kwargs)

    def set_frame(self, rgb, raw, coordinates):
        if coordinates is not self.coordinates:
            height, width = coordinates.max(axis=(0, 1)) + 1
            self.inverse_coordinates = np.empty((height, width, 2), np.int32)
            self.inverse_coordinates[coordinates[..., 0], coordinates[..., 1]] = (
                np.moveaxis(np.indices(raw.shape), 0, -1)
            )
        self.rgb, self.raw, self.coordinates = rgb, raw, coordinates
        if self.auto_fit:
            self.fit()
        else:
            self.redraw()
        x, y = (
            self.winfo_pointerx() - self.winfo_rootx(),
            self.winfo_pointery() - self.winfo_rooty(),
        )
        self.pick(x, y)

    def clear(self, message):
        """Remove stale image and measurement data together on source loss."""
        self.raw = self.rgb = self.coordinates = None
        self.measurements = self.native_measurements = None
        self.gesture_start = self.gesture_last = None
        self.empty_message = message
        self.redraw()

    def fit(self):
        self.auto_fit = True
        if self.raw is None:
            return
        h, w = self.raw.shape
        self.pixel_scale = max(
            0.1, min(self.winfo_width() / w, self.winfo_height() / h) * 0.95
        )
        self.offset = [
            (self.winfo_width() - w * self.pixel_scale) / 2,
            (self.winfo_height() - h * self.pixel_scale) / 2,
        ]
        self.redraw()

    def zoom(self, factor, x=None, y=None):
        self.auto_fit = False
        x = self.winfo_width() / 2 if x is None else x
        y = self.winfo_height() / 2 if y is None else y
        new = min(256.0, max(0.25, self.pixel_scale * factor))
        ratio = new / self.pixel_scale
        self.offset = [
            x - (x - self.offset[0]) * ratio,
            y - (y - self.offset[1]) * ratio,
        ]
        self.pixel_scale = new
        self.redraw()

    def move_image(self, dx, dy):
        """Move the image using screen-space buttons, with the same transform as dragging."""
        self.auto_fit = False
        self.offset = [self.offset[0] + dx, self.offset[1] + dy]
        self.redraw()

    def inspect(self):
        """Ctrl+X selects a legible native-pixel inspection magnification."""
        self.zoom(128 / self.pixel_scale)

    def cell_at(self, x, y):
        """Resolve a displayed pixel once; measurements use its native coordinate map."""
        if self.raw is None or self.coordinates is None:
            return None
        if self.legend_rect is not None:
            left, top, right, bottom = self.legend_rect
            if left <= x <= right and top <= y <= bottom:
                return None
        col = int((x - self.offset[0]) // self.pixel_scale)
        row = int((y - self.offset[1]) // self.pixel_scale)
        return (
            (row, col)
            if 0 <= row < self.raw.shape[0] and 0 <= col < self.raw.shape[1]
            else None
        )

    def sensor_at(self, x, y):
        cell = self.cell_at(x, y)
        if cell is not None:
            assert self.coordinates is not None
            sy, sx = self.coordinates[cell]
            return int(sx), int(sy)
        return None

    def start_pan(self, event):
        if self.tool != "Pan":
            self.gesture_start = self.gesture_last = self.sensor_at(event.x, event.y)
            if self.gesture_start is not None and self.on_gesture:
                self.on_gesture(
                    "begin", self.tool, self.gesture_start, self.gesture_start
                )
            return
        self.drag = (event.x, event.y, *self.offset)

    def pan(self, event):
        if self.tool != "Pan":
            point = self.sensor_at(event.x, event.y)
            if point is not None and self.gesture_start is not None:
                if self.on_gesture:
                    self.on_gesture("move", self.tool, self.gesture_last, point)
                self.gesture_last = point
                self.redraw()
            return
        self.auto_fit = False
        x, y, ox, oy = self.drag
        self.offset = [ox + event.x - x, oy + event.y - y]
        self.redraw()
        self.hover(event)

    def finish_gesture(self, event):
        if self.tool != "Pan" and self.gesture_start is not None and self.on_gesture:
            point = self.sensor_at(event.x, event.y) or self.gesture_last
            start = (
                self.gesture_last
                if self.tool in ("Brush", "Eraser")
                else self.gesture_start
            )
            self.on_gesture("end", self.tool, start, point)
        self.gesture_start = self.gesture_last = None
        self.redraw()

    def hover(self, event):
        self.pick(event.x, event.y)

    def pick(self, x, y):
        cell = self.cell_at(x, y)
        if cell is None:
            self.on_pixel(None)
        else:
            assert self.coordinates is not None and self.raw is not None
            sy, sx = self.coordinates[cell]
            self.on_pixel((int(sx), int(sy), int(self.raw[cell])))

    def redraw(self):
        self.delete("all")
        self.legend_rect = None
        self.on_zoom(self.pixel_scale)
        if self.raw is None or self.rgb is None:
            self.create_text(
                self.winfo_width() / 2,
                self.winfo_height() / 2,
                text=self.empty_message,
                fill="#93a3b8",
                font=("TkDefaultFont", 16),
            )
            return
        width, height = max(1, self.winfo_width()), max(1, self.winfo_height())
        ox, oy = self.offset
        # Render only the viewport, never a 65536 × 49152 full zoom image.
        matrix = np.array(
            [
                [self.pixel_scale, 0, ox + (self.pixel_scale - 1) / 2],
                [0, self.pixel_scale, oy + (self.pixel_scale - 1) / 2],
            ],
            dtype=np.float64,
        )
        view = cv2.warpAffine(
            self.rgb,
            matrix,
            (width, height),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(16, 21, 29),
        )
        self.photo = tk.PhotoImage(
            master=self,
            data=f"P6\n{width} {height}\n255\n".encode() + view.tobytes(),
            format="PPM",
        )
        self.create_image(0, 0, image=self.photo, anchor="nw")
        h, w = self.raw.shape
        if self.pixel_scale >= 28:
            x0, x1 = (
                max(0, int(-ox / self.pixel_scale)),
                min(w, int((width - ox) / self.pixel_scale) + 1),
            )
            y0, y1 = (
                max(0, int(-oy / self.pixel_scale)),
                min(h, int((height - oy) / self.pixel_scale) + 1),
            )
            for col in range(x0, x1 + 1):
                x = ox + col * self.pixel_scale
                self.create_line(
                    x,
                    max(0, oy),
                    x,
                    min(height, oy + h * self.pixel_scale),
                    fill="#677384",
                )
            for row in range(y0, y1 + 1):
                y = oy + row * self.pixel_scale
                self.create_line(
                    max(0, ox),
                    y,
                    min(width, ox + w * self.pixel_scale),
                    y,
                    fill="#677384",
                )
            # Full precision labels become legible once each cell is wide enough.
            if self.pixel_scale >= 96:
                for row in range(y0, y1):
                    for col in range(x0, x1):
                        x, y = (
                            ox + (col + 0.5) * self.pixel_scale,
                            oy + (row + 0.5) * self.pixel_scale,
                        )
                        label = pixel_label(self.raw[row, col]).replace(" °C", "\n°C")
                        if self.corrected and self.measurements is not None:
                            label = f"{self.measurements[row, col]:.3f}\n°C corrected"
                        self.create_text(
                            x + 1,
                            y + 1,
                            text=label,
                            fill="black",
                            font=("TkDefaultFont", 9, "bold"),
                        )
                        self.create_text(
                            x,
                            y,
                            text=label,
                            fill="white",
                            font=("TkDefaultFont", 9, "bold"),
                        )
        spot_data = self.measurements if self.measurements is not None else self.raw
        if self.markers and np.isfinite(spot_data).any():
            for index, color, label in [
                (np.nanargmin(spot_data), "#64c9ff", "MIN"),
                (np.nanargmax(spot_data), "#ff7d7d", "MAX"),
            ]:
                row, col = np.unravel_index(index, self.raw.shape)
                x, y = (
                    ox + (col + 0.5) * self.pixel_scale,
                    oy + (row + 0.5) * self.pixel_scale,
                )
                self.create_line(x - 7, y, x + 7, y, fill=color, width=2)
                self.create_line(x, y - 7, x, y + 7, fill=color, width=2)
                self.create_text(x + 10, y - 10, text=label, fill=color, anchor="w")

        self.draw_regions()
        self.draw_preview()
        self.draw_legend()
        if self.overlay_drawer is not None:
            self.overlay_drawer()

    def draw_legend(self):
        """Overlay a labeled ramp; intensity legends deliberately never claim Celsius."""
        if not self.show_legend or self.legend_data is None:
            return
        colors, low, high, unit = self.legend_data[:4]
        bands = self.legend_data[4] if len(self.legend_data) > 4 else None
        width, height = self.winfo_width(), self.winfo_height()
        if width < 180 or height < 180:
            return
        x, y = width - 115, 18
        bar_height = min(240, height - 85)
        self.legend_rect = (x - 8, y - 8, width - 8, y + bar_height + 45)
        self.create_rectangle(*self.legend_rect, fill="#10151d", outline="#536175")
        self.create_text(
            x + 5,
            y + 4,
            text=unit,
            fill="white",
            anchor="nw",
            font=("TkDefaultFont", 9),
        )
        ramp = cv2.resize(
            colors[::-1].reshape(256, 1, 3),
            (18, bar_height),
            interpolation=cv2.INTER_NEAREST,
        )
        self.legend_photo = tk.PhotoImage(
            master=self,
            data=f"P6\n18 {bar_height}\n255\n".encode() + ramp.tobytes(),
            format="PPM",
        )
        self.create_image(x, y + 25, image=self.legend_photo, anchor="nw")
        for index, fraction in enumerate((0, 0.25, 0.5, 0.75, 1)):
            value = high + (low - high) * fraction
            label = f"{value:.2f}"
            if bands is not None:
                bounds = bands[index]
                label = "—" if bounds is None else f"{bounds[0]:.2f}\n{bounds[1]:.2f}"
            self.create_text(
                x + 25,
                y + 25 + bar_height * fraction,
                text=label,
                anchor="w",
                fill="white",
                font=("TkDefaultFont", 9),
            )

    def screen_point(self, point):
        if self.coordinates is None:
            return None
        x, y = point
        if (
            not 0 <= y < self.inverse_coordinates.shape[0]
            or not 0 <= x < self.inverse_coordinates.shape[1]
        ):
            return None
        row, col = self.inverse_coordinates[y, x]
        return (
            self.offset[0] + (col + 0.5) * self.pixel_scale,
            self.offset[1] + (row + 0.5) * self.pixel_scale,
        )

    def draw_preview(self):
        """Draw provisional geometry using the same sensor transform as saved ROIs."""
        if self.gesture_start is None or self.gesture_last is None:
            return
        a, b = (
            self.screen_point(self.gesture_start),
            self.screen_point(self.gesture_last),
        )
        if a is None or b is None:
            return
        x, y = a
        u, v = b
        if self.tool == "Line":
            self.create_line(
                x, y, u, v, fill="#ffffff", width=3, dash=(6, 4), tags="roi_preview"
            )
        elif self.tool == "Rectangle":
            self.create_rectangle(
                x, y, u, v, outline="#ffffff", width=3, dash=(6, 4), tags="roi_preview"
            )
        elif self.tool == "Circle":
            radius = np.hypot(u - x, v - y)
            self.create_oval(
                x - radius,
                y - radius,
                x + radius,
                y + radius,
                outline="#ffffff",
                width=3,
                dash=(6, 4),
                tags="roi_preview",
            )

    def draw_regions(self):
        if self.coordinates is None:
            return
        for region in self.regions:
            a, b = self.screen_point(region.start), self.screen_point(region.end)
            if a is None or b is None:
                continue
            x, y = a
            u, v = b
            color = "#77ffcc"
            if region.kind == "Spot":
                self.create_line(x - 6, y, x + 6, y, fill=color, width=2)
                self.create_line(x, y - 6, x, y + 6, fill=color, width=2)
            elif region.kind == "Rectangle":
                self.create_rectangle(x, y, u, v, outline=color, width=2)
            elif region.kind == "Circle":
                radius = np.hypot(u - x, v - y)
                self.create_oval(
                    x - radius,
                    y - radius,
                    x + radius,
                    y + radius,
                    outline=color,
                    width=2,
                )
            elif region.kind == "Line":
                self.create_line(x, y, u, v, fill=color, width=2)
            label = region.name
            if region.kind == "Spot" and self.native_measurements is not None:
                sx, sy = region.start
                label += f" {self.native_measurements[sy, sx]:.3f} °C"
            self.create_text(x + 8, y - 12, text=label, fill=color, anchor="w")
            if region.kind != "Spot" and self.native_measurements is not None:
                ys, xs = region.samples(self.native_measurements.shape)
                values = self.native_measurements[ys, xs]
                if np.isfinite(values).any():
                    for index, fill in (
                        (np.nanargmin(values), "#64c9ff"),
                        (np.nanargmax(values), "#ff7d7d"),
                    ):
                        point = self.screen_point((xs[index], ys[index]))
                        if point:
                            px, py = point
                            self.create_oval(
                                px - 3, py - 3, px + 3, py + 3, outline=fill, width=2
                            )
