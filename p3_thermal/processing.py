"""Display processing only: measurements always use original uint16 samples."""

from dataclasses import dataclass
from functools import lru_cache

import cv2
import numpy as np


PALETTES = {
    "Inferno": cv2.COLORMAP_INFERNO,
    "Magma": cv2.COLORMAP_MAGMA,
    "Viridis": cv2.COLORMAP_VIRIDIS,
    "Turbo": cv2.COLORMAP_TURBO,
    "Rainbow": cv2.COLORMAP_JET,
    "White hot": None,
    "White hot / red peak": None,
    "Black hot": None,
}


def temperature(raw):
    """Decode 1/64 K samples in float64; do not quantize through float32."""
    return np.asarray(raw, dtype=np.float64) / 64.0 - 273.15


def pixel_label(raw):
    """Six decimals represent every native sample exactly in decimal Celsius."""
    return f"{float(temperature(raw)):.6f} °C"


def orient(array, rotation=0, mirror=False):
    """Apply identical counterclockwise quarter turns and mirror to all planes."""
    result = np.rot90(array, rotation)
    return np.fliplr(result) if mirror else result


@dataclass
class DisplaySettings:
    """Editable display parameters; none mutate the captured frame."""

    mode: str = "Temperature"
    palette: str = "Inferno"
    range_mode: str = "Auto percentile"
    minimum: float = 15.0
    maximum: float = 40.0
    alpha: float = 0.35
    detail: bool = False  # DDE only; legacy snapshots migrate the combined flag
    clahe: bool = False
    dde_strength: float = 1.5
    custom_auto_scale: bool = False
    enhance_custom_palette: bool = False
    enhancements_enabled: bool = True


class Processor:
    """Per-session temporal filter and range state, reset on reconnect/settings."""

    def __init__(self):
        self.clahe = cv2.createCLAHE(2.0, (8, 8))
        self.reset()

    def reset(self):
        self.previous = None
        self.bounds = None
        self.temperature_bands = None

    def render(self, raw, brightness, settings, custom_palette=None, measured=None):
        """Return RGB sensor-sized image and actual display limits (or None)."""
        self.temperature_bands = None
        data = (
            np.asarray(brightness, dtype=np.float64)
            if settings.mode == "Factory brightness"
            else raw.astype(np.float64)
            if settings.mode == "Raw counts"
            else temperature(raw)
            if measured is None
            else measured
        )
        if settings.mode == "Filtered temperature":
            self.previous = (
                data.copy()
                if self.previous is None
                else settings.alpha * data + (1 - settings.alpha) * self.previous
            )
            data = self.previous
        if (
            settings.range_mode == "Fixed"
            and not (custom_palette and settings.custom_auto_scale)
            and (settings.mode != "Raw counts" or custom_palette)
        ):
            low, high = settings.minimum, settings.maximum
            if high <= low:
                raise ValueError("Maximum must be greater than minimum")
        else:
            finite = data[np.isfinite(data)]
            low, high = (
                (
                    (finite.min(), finite.max())
                    if custom_palette and settings.custom_auto_scale
                    else np.percentile(finite, (1, 99))
                )
                if finite.size
                else (0.0, 1.0)
            )
            if custom_palette is None and settings.mode != "Factory brightness":
                if self.bounds is not None:
                    low, high = 0.15 * np.array([low, high]) + 0.85 * self.bounds
                self.bounds = np.array([low, high])
        limits = (float(low), float(high))
        gray = (
            np.nan_to_num(np.clip((data - low) / max(high - low, 1e-9), 0, 1)) * 255
        ).astype(np.uint8)
        if settings.enhancements_enabled:
            if settings.clahe:
                gray = self.clahe.apply(gray)
                limits = None
            if settings.detail and settings.dde_strength > 0:
                values = gray.astype(np.float32)
                gray = (
                    np.clip(
                        values
                        + settings.dde_strength
                        * (values - cv2.GaussianBlur(values, (0, 0), 1.2)),
                        0,
                        255,
                    )
                    .round()
                    .astype(np.uint8)
                )
                limits = None
        # Camera brightness has no global inverse Celsius scale, even without filters.
        if limits is None or settings.mode == "Factory brightness":
            self.temperature_bands = observed_temperature_bands(raw, gray, measured)
        return cv2.applyColorMap(
            gray, palette_colors(settings.palette, custom_palette).reshape(256, 1, 3)
        ), limits


def legend_colors(settings, custom_palette=None):
    """Reuse the exact display LUT for legends; cached arrays are read-only."""
    return palette_colors(settings.palette, custom_palette)


@lru_cache(maxsize=32)
def palette_colors(name, custom_palette=None):
    """Cache small 256-color ramps, rather than rebuilding filters/LUTs per frame."""
    if custom_palette is not None:
        colors = custom_palette.colorize(
            np.linspace(custom_palette.stops[0][0], custom_palette.stops[-1][0], 256)
        )
    elif name == "White hot / red peak":
        colors = red_peak_colors()
    else:
        gray = np.arange(256, dtype=np.uint8).reshape(256, 1)
        if name in ("White hot", "Black hot"):
            colors = np.repeat(255 - gray if name == "Black hot" else gray, 3, axis=1)
        else:
            colors = cv2.cvtColor(
                cv2.applyColorMap(gray, PALETTES[name]), cv2.COLOR_BGR2RGB
            ).reshape(256, 3)
    colors.setflags(write=False)
    return colors


@lru_cache(maxsize=8)
def sensor_coordinates(shape, rotation=0, mirror=False):
    """Bounded, immutable sensor-coordinate maps shared by the live and compare views."""
    coordinates = orient(
        np.moveaxis(np.indices(shape, dtype=np.int32), 0, -1), rotation, mirror
    )
    coordinates.setflags(write=False)
    return coordinates


def observed_temperature_bands(raw, gray, measured=None):
    """Observed Celsius min/max per displayed color band, ordered high intensity first.

    A local contrast operator has no unique inverse temperature curve. These ranges
    report actual pixels in five disjoint intensity bands, not an invented inverse.
    Empty bands are explicitly marked; values need not be monotonic.
    """
    bins = np.digitize(gray, [32, 96, 160, 224])
    temps = temperature(raw) if measured is None else measured
    result = []
    finite = np.isfinite(temps)
    for band in range(4, -1, -1):
        values = temps[(bins == band) & finite]
        result.append(
            (float(values.min()), float(values.max())) if values.size else None
        )
    return tuple(result)


def red_peak_colors():
    """Black-to-white ramp with a red upper tail (top 5% of the display range)."""
    gray = np.arange(256, dtype=np.float64)
    base = np.minimum(gray / 242 * 255, 255)
    ramp = np.repeat(base[:, None], 3, axis=1)
    ramp[243:, 1:] = np.linspace(235, 0, 13)[:, None]
    return ramp.round().astype(np.uint8)


def focus_peaking(rgb, raw, threshold=0.35):
    """Highlight strong native thermal gradients without altering measurement data."""
    values = np.asarray(raw, dtype=np.float32)
    values = cv2.GaussianBlur(values, (0, 0), 0.7)
    dx = cv2.Sobel(values, cv2.CV_32F, 1, 0)
    dy = cv2.Sobel(values, cv2.CV_32F, 0, 1)
    strength = cv2.magnitude(dx, dy)
    peak = float(strength.max())
    result = rgb.copy()
    if peak > 0:
        result[strength > max(1.0, peak * threshold)] = (0, 255, 80)
    return result
