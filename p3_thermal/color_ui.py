"""Embedded HSV color wheel; the palette's hexadecimal field remains editable."""

from functools import lru_cache
from tkinter import ttk

import colorsys
import math
import re
import tkinter as tk

import cv2
import numpy as np

from .widgets import caption


@lru_cache(maxsize=1)
def wheel_image(size=240):
    """Build a reusable full-brightness hue/saturation disc as a native Tk PPM."""
    y, x = np.mgrid[:size, :size].astype(np.float32)
    radius = (size - 1) / 2
    x, y = (x - radius) / radius, (radius - y) / radius
    saturation = np.hypot(x, y)
    hue = np.degrees(np.arctan2(y, x)) % 360
    hsv = np.stack((hue, np.minimum(saturation, 1), np.ones_like(hue)), axis=-1)
    rgb = (cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB) * 255).round().astype(np.uint8)
    rgb[saturation > 1] = (24, 33, 46)
    return f"P6\n{size} {size}\n255\n".encode() + rgb.tobytes()


class ColorWheel(ttk.Frame):
    """Synchronize wheel drags, value slider and HEX entry without recursive traces."""

    SIZE = 240

    def __init__(self, parent, color):
        super().__init__(parent)
        self.color = color
        self.hue = self.saturation = 0.0
        self.updating = False
        self.value = tk.DoubleVar(self, value=1)
        caption(self, "Hue / saturation", anchor="center")
        self.canvas = tk.Canvas(
            self,
            width=self.SIZE,
            height=self.SIZE,
            background="#18212e",
            highlightthickness=0,
        )
        self.canvas.pack(pady=6)
        self.photo = tk.PhotoImage(
            master=self, data=wheel_image(self.SIZE), format="PPM"
        )
        self.canvas.create_image(0, 0, image=self.photo, anchor="nw")
        self.outer = self.canvas.create_oval(0, 0, 0, 0, outline="black", width=4)
        self.inner = self.canvas.create_oval(0, 0, 0, 0, outline="white", width=2)
        self.canvas.bind("<Button-1>", self.pick)
        self.canvas.bind("<B1-Motion>", self.pick)
        caption(self, "Brightness")
        ttk.Scale(
            self, from_=0, to=1, variable=self.value, command=lambda _: self.publish()
        ).pack(fill="x")
        self.preview = tk.Canvas(
            self, height=28, highlightthickness=1, highlightbackground="#91a8c1"
        )
        self.preview.pack(fill="x", pady=6)
        self.trace = color.trace_add("write", self.read_color)
        self.bind("<Destroy>", self.cleanup, add="+")
        self.read_color()

    def cleanup(self, event):
        if event.widget is self:
            self.color.trace_remove("write", self.trace)

    def read_color(self, *_):
        text = self.color.get()
        if self.updating or not re.fullmatch(r"#[0-9a-fA-F]{6}", text):
            return  # Allow partial manual HEX edits without discarding the last valid color.
        rgb = [int(text[i : i + 2], 16) / 255 for i in (1, 3, 5)]
        hue, saturation, value = colorsys.rgb_to_hsv(*rgb)
        if saturation:
            self.hue = hue
        self.saturation = saturation
        self.value.set(value)
        self.redraw(text)

    def pick(self, event):
        radius = (self.SIZE - 1) / 2
        x, y = (event.x - radius) / radius, (radius - event.y) / radius
        self.hue = math.atan2(y, x) / math.tau % 1
        self.saturation = min(1.0, math.hypot(x, y))
        self.publish()

    def publish(self):
        rgb = colorsys.hsv_to_rgb(self.hue, self.saturation, self.value.get())
        text = "#" + "".join(f"{round(channel * 255):02X}" for channel in rgb)
        self.updating = True
        try:
            self.color.set(text)
        finally:
            self.updating = False
        self.redraw(text)

    def redraw(self, text):
        radius = (self.SIZE - 1) / 2
        angle = self.hue * math.tau
        x = radius + radius * self.saturation * math.cos(angle)
        y = radius - radius * self.saturation * math.sin(angle)
        for marker in (self.outer, self.inner):
            self.canvas.coords(marker, x - 5, y - 5, x + 5, y + 5)
        self.preview.configure(background=text)
