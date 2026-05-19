#!/usr/bin/env python3
"""P3 Thermal Camera Viewer.

Display, colormaps, and image processing (ISP) for the P3 thermal camera.

Controls:
  q - Quit           +/- - Zoom
  r - Rotate 90°     c - Colormap
  s - Shutter/NUC    g - Gain mode
  m - Mirror         h - Help
  space - Screenshot D - Dump raw data
  e - Emissivity     1-9 - Set emissivity (0.1-0.9)
  x - Scale mode     p - Enhanced (CLAHE+DDE)
  a - AGC mode       t - Toggle reticule
  d - Toggle DDE     b - Toggle min/max marker
  v - Toggle colorbar
"""

from __future__ import annotations

from enum import IntEnum
from typing import Any, cast

import logging
import time

from numpy.typing import NDArray

import cv2
import numpy as np
import threading
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

from lockin import LockInController

from thermal_camera_viewer.p3_camera import (
    GainMode,
    Model,
    P3Camera,
    get_model_config,
    raw_to_celsius,
    raw_to_celsius_corrected,
)


# =============================================================================
# Colormaps
# =============================================================================


class ColormapID(IntEnum):
    """Colormap IDs."""

    WHITE_HOT = 0
    BLACK_HOT = 1
    RAINBOW = 2
    IRONBOW = 3
    MILITARY = 4
    SEPIA = 5


class ScaleMode(IntEnum):
    """2x upscaling interpolation modes."""

    OFF = 0  # No upscaling
    NEAREST = 1  # Nearest neighbor (blocky, fast)
    BILINEAR = 2  # Bilinear (smooth, fast)
    BICUBIC = 3  # Bicubic (sharper than bilinear)
    LANCZOS = 4  # Lanczos (sharpest, slowest)

class CVLineType(IntEnum):
    """Draw style for cv2 lines and text."""

    FILLED = cv2.FILLED
    LINE_4 = cv2.LINE_4
    LINE_8 = cv2.LINE_8
    LINE_AA = cv2.LINE_AA


class AGCMode(IntEnum):
    """Auto Gain Control modes."""

    FACTORY = 0  # Use IR brightness from camera (hardware AGC)
    TEMPORAL_1 = 1  # EMA smoothed, 1% percentile
    FIXED_RANGE = 2  # Fixed temperature range (15-40°C)


AGC_PERCENTILES = {
    AGCMode.TEMPORAL_1: 1.0,
}


SCALE_INTERP = {
    ScaleMode.NEAREST: cv2.INTER_NEAREST,
    ScaleMode.BILINEAR: cv2.INTER_LINEAR,
    ScaleMode.BICUBIC: cv2.INTER_CUBIC,
    ScaleMode.LANCZOS: cv2.INTER_LANCZOS4,
}

class HotspotMode(IntEnum):
    """Extreme value marker modes."""

    OFF = 0  # Off
    MAX = 1  # Show maximum
    MIN = 2  # Show minimum
    MINMAX = 3  # Show both


# Colormaps (BGR format for OpenCV)
COLORMAPS: dict[ColormapID, NDArray[np.uint8]] = {}

# Color constants
COLOR_SPOT_MAX =    (0,0,255)
COLOR_SPOT_MIN =    (255,0,0)
COLOR_RETICULE =    (0,255,0)
COLOR_TEXT =        (255,255,255)


def _cv_lut(colormap: int) -> NDArray[np.uint8]:
    """Extract 256x3 LUT from OpenCV colormap."""
    gray = np.arange(256, dtype=np.uint8).reshape(1, 256)
    colored = cv2.applyColorMap(gray, colormap)
    return cast(NDArray[np.uint8], colored.reshape(256, 3))


def _init_colormaps() -> None:
    """Initialize colormap lookup tables."""
    global COLORMAPS
    ramp = np.arange(256, dtype=np.uint8)
    # White hot: grayscale
    lut = np.zeros((256, 3), dtype=np.uint8)
    lut[:, 0] = lut[:, 1] = lut[:, 2] = ramp
    COLORMAPS[ColormapID.WHITE_HOT] = lut
    # Black hot: inverted grayscale
    lut = np.zeros((256, 3), dtype=np.uint8)
    lut[:, 0] = lut[:, 1] = lut[:, 2] = 255 - ramp
    COLORMAPS[ColormapID.BLACK_HOT] = lut
    # Rainbow: OpenCV JET
    COLORMAPS[ColormapID.RAINBOW] = _cv_lut(cv2.COLORMAP_JET)
    # Ironbow: OpenCV INFERNO
    COLORMAPS[ColormapID.IRONBOW] = _cv_lut(cv2.COLORMAP_INFERNO)
    # Military: green-tinted grayscale (BGR: low B, high G, low R)
    lut = np.zeros((256, 3), dtype=np.uint8)
    lut[:, 0] = (ramp * 0.2).astype(np.uint8)  # B
    lut[:, 1] = ramp  # G
    lut[:, 2] = (ramp * 0.3).astype(np.uint8)  # R
    COLORMAPS[ColormapID.MILITARY] = lut
    # Sepia: brown-tinted grayscale
    lut = np.zeros((256, 3), dtype=np.uint8)
    lut[:, 0] = (ramp * 0.4).astype(np.uint8)  # B
    lut[:, 1] = (ramp * 0.7).astype(np.uint8)  # G
    lut[:, 2] = ramp  # R
    COLORMAPS[ColormapID.SEPIA] = lut


_init_colormaps()


def get_colormap(cmap_id: ColormapID | int) -> NDArray[np.uint8]:
    """Get colormap LUT by ID.

    Args:
        cmap_id: Colormap ID.

    Returns:
        256x3 uint8 array (BGR).

    """
    return COLORMAPS[ColormapID(cmap_id)]


def apply_colormap(
    img_u8: NDArray[np.uint8], cmap_id: ColormapID | int
) -> NDArray[np.uint8]:
    """Apply colormap to grayscale image.

    Args:
        img_u8: 8-bit grayscale image (H×W).
        cmap_id: Colormap ID.

    Returns:
        BGR color image (H×W×3).

    """
    lut = get_colormap(cmap_id)
    return lut[img_u8]

def apply_diverging_colormap(data: np.ndarray, cmap_name: str = 'bwr') -> np.ndarray:
  
    data = np.asarray(data, dtype=np.float32)
    lo = float(np.nanpercentile(data, 0.01))
    hi = float(np.nanpercentile(data, 99.99))
    if hi <= lo:
        hi = lo + 1.0
    norm_data = 2 * (data - lo) / (hi - lo) - 1.0 # Normalize to [-1, 1]
    data=np.clip(norm_data, -1.0, 1.0)
    
    # Center at 'center' (usually 0) using TwoSlopeNorm or CenteredNorm
    # Got better results normalizing and clipping above then using TwoSlopeNorm in this
    # fashion, vs. using it directly as intended.  ???.
    norm = mcolors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    
    cmap = cm.get_cmap(cmap_name)
    
    # Map to [0,1] then to RGBA [0,255]
    rgba_float = cmap(norm(data))          # shape (h,w,4), float 0-1
    rgb_float = rgba_float[..., :3]         # drop alpha → (h, w, 3)
    rgb_uint8 = (rgb_float * 255).astype(np.uint8)

    # Convert matplotlib RGB (R,G,B) to OpenCV BGR ordering to match apply_colormap
    bgr_uint8 = rgb_uint8[..., ::-1]
    return bgr_uint8


# =============================================================================
# Image Processing (ISP)
# =============================================================================


# Global state for temporal AGC smoothing
_agc_ema_low: float | None = None
_agc_ema_high: float | None = None


def agc_temporal(
    img: NDArray[np.uint16],
    pct: float = 1.0,
    ema_alpha: float = 0.1,
) -> NDArray[np.uint8]:
    """AGC with EMA-smoothed percentile bounds.

    Args:
        img: 16-bit thermal image.
        pct: Percentile for clipping (uses pct and 100-pct).
        ema_alpha: EMA smoothing factor (higher = faster adaptation).
    """
    global _agc_ema_low, _agc_ema_high
    low = float(np.percentile(img, pct))
    high = float(np.percentile(img, 100.0 - pct))
    if _agc_ema_low is None or _agc_ema_high is None:
        _agc_ema_low, _agc_ema_high = low, high
    else:
        _agc_ema_low = ema_alpha * low + (1 - ema_alpha) * _agc_ema_low
        _agc_ema_high = ema_alpha * high + (1 - ema_alpha) * _agc_ema_high
    if _agc_ema_high <= _agc_ema_low:
        return np.zeros(img.shape, dtype=np.uint8)
    normalized = (img.astype(np.float32) - _agc_ema_low) / (
        _agc_ema_high - _agc_ema_low
    )
    return (np.clip(normalized, 0.0, 1.0) * 255).astype(np.uint8)


def agc_fixed(
    img: NDArray[np.uint16],
    temp_min: float = 18.0,
    temp_max: float = 35.0,
) -> NDArray[np.uint8]:
    """AGC with fixed temperature range (Celsius)."""
    raw_min = (temp_min + 273.15) * 64
    raw_max = (temp_max + 273.15) * 64
    normalized = (img.astype(np.float32) - raw_min) / (raw_max - raw_min)
    return (np.clip(normalized, 0.0, 1.0) * 255).astype(np.uint8)


def dde(
    img_u8: NDArray[np.uint8],
    strength: float = 0.5,
    kernel_size: int = 3,
) -> NDArray[np.uint8]:
    """Apply Digital Detail Enhancement (edge sharpening).

    Uses unsharp masking: enhanced = original + strength * (original - blurred)

    Args:
        img_u8: Input 8-bit image.
        strength: Enhancement strength (0.0-1.0, default 0.5).
        kernel_size: Kernel size for high-pass filter (default 3).

    Returns:
        Enhanced 8-bit image.

    """
    if strength <= 0:
        return img_u8

    # Create blurred version
    ksize = kernel_size | 1  # Ensure odd
    blurred = cv2.GaussianBlur(img_u8, (ksize, ksize), 0)

    # Unsharp mask
    img_f = img_u8.astype(np.float32)
    blurred_f = blurred.astype(np.float32)
    enhanced = img_f + strength * (img_f - blurred_f)

    return np.clip(enhanced, 0, 255).astype(np.uint8)


def tnr(
    img: NDArray[np.uint16],
    prev_img: NDArray[np.uint16] | None,
    alpha: float = 0.3,
) -> NDArray[np.uint16]:
    """Apply Temporal Noise Reduction.

    Blends current frame with previous frame to reduce temporal noise.

    Args:
        img: Current frame.
        prev_img: Previous frame (or None for first frame).
        alpha: Blending factor (0=all previous, 1=all current, default 0.3).

    Returns:
        Filtered frame.

    """
    if prev_img is None:
        return img

    result = alpha * img.astype(np.float32) + (1 - alpha) * prev_img.astype(np.float32)
    return result.astype(np.uint16)


# =============================================================================
# Viewer
# =============================================================================


class P3Viewer:
    """P3 Thermal Camera Viewer."""

    def __init__(self, model: Model | str = Model.P3, serial_port: str = "/dev/ttyACM0",
                 baud_rate: int = 115200, lockin_period: float = 1.0,
                 lockin_integration: float = 60.0, lockin_invert: bool = False) -> None:
        """Initialize viewer.

        Args:
            model: Camera model (P1 or P3).
            serial_port: Serial port for lock-in load controller.
            baud_rate: Baud rate for serial port.
            lockin_period: Lock-in period in seconds.
            lockin_integration: Integration time in seconds.
        """
        config = get_model_config(model)
        self.camera = P3Camera(config=config)
        self.model = config.model
        self.serial_port = serial_port
        self.baud_rate = baud_rate
        self.lockin_period = lockin_period
        self.lockin_integration = lockin_integration
        self.lockin_invert = bool(lockin_invert)
        self.rotation: int = 0
        self.colormap_idx: int = ColormapID.IRONBOW
        self.mirror: bool = False
        self.show_help: bool = False
        self.show_reticule: bool = True
        self.show_colorbar: bool = True
        self.hotspot_mode: int = HotspotMode.OFF
        self.zoom: int = 3
        self.fps: float = 0.0
        self.enhanced: bool = True
        self.use_clahe: bool = True
        self.scale_mode: ScaleMode = ScaleMode.BICUBIC
        self.cv_linetype: CVLineType = CVLineType.LINE_AA
        self.agc_mode: AGCMode = AGCMode.FACTORY
        self.dde_strength: float = 0.3
        self.tnr_alpha: float = 0.5
        self._fps_count: int = 0
        self._fps_time: float = time.time()
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self._last_display: NDArray[np.uint8] | None = None
        self._prev_frame: NDArray[np.uint16] | None = None
        self._ir_brightness: NDArray[np.uint8] | None = None
        # Lock-in controller (created on demand)
        self.lockin_controller: LockInController | None = None
        self.lockin_thread: threading.Thread | None = None
        self.lockin_running: bool = False

    def run(self) -> None:
        """Main viewer loop."""
        model_name = self.model.value.upper()
        print(f"{model_name} Thermal Viewer")
        self.camera.connect()
        name, version = self.camera.init()
        print(f"Device: {name}, Firmware: {version}")
        self.camera.start_streaming()
        print("Press 'h' for help")

        window_name = f"{model_name} Thermal"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 720)

        try:
            while True:

                ir_brightness, thermal = self.camera.read_frame_both()
                if thermal is None:
                    continue
                self._ir_brightness = ir_brightness

                # Feed frames to lock-in controller so only main thread reads USB.
                if self.lockin_controller is not None and self.lockin_running:
                    try:
                        self.lockin_controller.push_frame(thermal.copy(), time.perf_counter())
                    except Exception:
                        pass

                # Apply temporal noise reduction
                thermal = tnr(thermal, self._prev_frame, alpha=self.tnr_alpha)
                self._prev_frame = thermal.copy()

                self._last_display = self._render(thermal)
                window_name = f"{self.model.value.upper()} Thermal"
                cv2.imshow(window_name, self._last_display)
                self._update_fps()

                if not self._handle_key(thermal):
                    break
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break
        finally:
            self.camera.stop_streaming()
            cv2.destroyAllWindows()

    def _update_fps(self) -> None:
        self._fps_count += 1
        now = time.time()
        if now - self._fps_time >= 1.0:
            self.fps = self._fps_count / (now - self._fps_time)
            self._fps_count = 0
            self._fps_time = now

    def _get_spot_coords(self, thermal: NDArray[np.uint16]) -> tuple[int, int]:
        """Get thermal array coords for center spot."""
        th, tw = thermal.shape
        cy, cx = th // 2, tw // 2
        if self.mirror:
            cx = tw - 1 - cx
        return cy, cx

    def _coord_to_image(
        self, 
        coord: tuple[int, int], 
        thermal: NDArray[np.uint16], 
        image: NDArray[np.uint8]
    ) -> tuple[int, int]:
        """Transform thermal coordinate to image coordinate."""
        h, w = image.shape[:2]
        # unflip image size
        if self.rotation == 90 or self.rotation == 270: h,w = w,h
        th, tw = thermal.shape[:2]
        cy, cx = coord

        # Scale to output image
        cx_d = int(round((cx / tw) * w))
        cy_d = int(round((cy / th) * h))

        # Mirror
        if self.mirror:
            cx_d = w - cx_d - 1

        # rotation
        if self.rotation == 90:
            cy_d, cx_d = cx_d, h-cy_d-1
        elif self.rotation == 180:
            cy_d, cx_d = h-cy_d-1, w-cx_d-1
        elif self.rotation == 270:
            cy_d, cx_d = w-cx_d-1, cy_d

        return cy_d, cx_d

    def _draw_box_marker(
        self,
        coord: tuple[int, int],
        image: NDArray[np.uint8],
        annotation: str = "",
        color: tuple[int, int, int] = (255,255,255),
        size: int = 25
    ) -> None:
        """
        Draw a square marker onto the image.
        
        Args:
            coord: tuple (y,x) of rectangle center
            image: image to draw onto
            annotation: text placed next to the rectangle
            color: (b,g,r) color for rectangle and text
            size: size of the rectangle in image space
        """
        size = size // 2 + size % 2
        cy, cx = coord
        h, w = image.shape[:2]
        # prevent text from clipping at image border
        if cy < 20 + size: y_offset = 20 + size
        else: y_offset = -5 - size
        x_offset = min(0, w-cx-55)
        # Draw rectangle and annotation
        cv2.rectangle(image, (cx-size, cy-size), (cx+size, cy+size), color, 1, self.cv_linetype)
        cv2.putText(
            image,
            annotation,
            (cx + x_offset, cy + y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            self.cv_linetype
        )

    def _render(self, thermal: NDArray[np.uint16]) -> NDArray[np.uint8]:
        """Render thermal frame to display image."""

        frame_stats = {}

        # AGC: normalize to 8-bit based on selected mode
        if self.agc_mode == AGCMode.FACTORY:
            # Use hardware AGC'd IR brightness from camera (already 8-bit)
            if self._ir_brightness is not None:
                img = self._ir_brightness.copy()
            else:
                img = agc_temporal(thermal, pct=1.0)
        elif self.agc_mode == AGCMode.FIXED_RANGE:
            img = agc_fixed(thermal)
        else:
            pct = AGC_PERCENTILES.get(self.agc_mode, 1.0)
            img = agc_temporal(thermal, pct=pct)

        # Optional 2x upscaling
        if self.scale_mode != ScaleMode.OFF:
            h, w = img.shape[:2]
            resized: Any = cv2.resize(
                img, (w * 2, h * 2), interpolation=SCALE_INTERP[self.scale_mode]
            )
            # Ensure result is numpy array (cv2.resize may return cv2.UMat on some platforms)
            img = np.asarray(resized, dtype=np.uint8)

        # Optional CLAHE for local contrast enhancement
        if self.use_clahe:
            clahe_result: Any = self._clahe.apply(img)
            # Ensure result is a numpy array (CLAHE may return cv2.UMat on some platforms)
            img = np.asarray(clahe_result, dtype=np.uint8)

        # DDE: edge enhancement
        img = dde(img, strength=self.dde_strength)

        # Temperature values (with emissivity correction)
        cy, cx = self._get_spot_coords(thermal)
        env = self.camera.env_params
        frame_stats['tspot'] = float(raw_to_celsius_corrected(thermal[cy, cx], env))
        frame_stats['cmax'] = thermal.argmax()
        frame_stats['cmin'] = thermal.argmin()
        frame_stats['tmin'] = float(raw_to_celsius_corrected(thermal.ravel()[frame_stats['cmin']], env))
        frame_stats['tmax'] = float(raw_to_celsius_corrected(thermal.ravel()[frame_stats['cmax']], env))
        # value range
        frame_stats['range_min'] = int(np.min(img))
        frame_stats['range_max'] = int(np.max(img))

        # Apply colormap
        img = apply_colormap(img, self.colormap_idx)

        # Mirror
        if self.mirror:
            img = cv2.flip(img, 1)

        # Rotate
        if self.rotation == 90:
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        elif self.rotation == 180:
            img = cv2.rotate(img, cv2.ROTATE_180)
        elif self.rotation == 270:
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # Zoom
        h, w = img.shape[:2]
        result = cast(
            NDArray[np.uint8],
            cv2.resize(
                img, (w * self.zoom, h * self.zoom), interpolation=cv2.INTER_LINEAR
            ),
        )

        if self.show_colorbar:
            self._draw_colorbar(result, frame_stats)

        # Overlays
        self._draw_overlays(result, thermal, frame_stats)

        # If lock-in results exist, composite two panes to the right
        if self.lockin_controller is not None:
            in_phase, quad, amplitude, angle = self.lockin_controller.get_latest()
            if in_phase is not None and quad is not None and amplitude is not None and angle is not None:
                try:
                    result = self._composite_lockin_panes(result, in_phase, quad, amplitude, angle)
                except Exception:
                    # Don't let lock-in display errors break rendering
                    pass

        return result
    
    def _draw_colorbar(
        self, 
        img: NDArray[np.uint8],
        frame_stats: dict,
        height: float =.5, 
        width: int = 20, 
        ticks: int = 5, 
        outline: bool = True
    ) -> None:
        """Draw a colorbar for the current colormap on the center right of the image."""

        # at least ticks for min and max
        ticks = max(2,ticks)

        # resample color map to image resolution
        h, w = img.shape[:2]
        h_cbar = int(height * h)
        val_range = (frame_stats['range_max']-frame_stats['range_min'])
        ind = (np.arange(0.5, h_cbar) / h_cbar) * val_range
        ind = (ind + frame_stats['range_min']+ .5).astype(np.uint8)
        cmap_resamp = get_colormap(self.colormap_idx)[ind[::-1]]

        # draw colorbar
        y_offset = int(.5 * (1-height) * h)
        x_offset = 50
        img[y_offset:y_offset+h_cbar, -x_offset-width:-x_offset] = cmap_resamp.reshape(h_cbar, 1, 3)

        # draw colorbar outline
        if outline:
            cv2.rectangle(
                img, 
                (w-x_offset-width, y_offset), 
                (w-x_offset, y_offset+h_cbar), 
                COLOR_TEXT, 1, self.cv_linetype
            )

        # draw ticks
        tick_pos = (np.linspace(0,1, ticks) * h_cbar + y_offset).astype(np.uint16)[::-1]
        tick_labels = [f'{t:.1f}' for t in np.linspace(frame_stats['tmin'], frame_stats['tmax'], ticks)]
        for pos, label in zip(tick_pos, tick_labels):
            cv2.line(img, (w-x_offset-width, pos), (w-x_offset, pos), COLOR_TEXT, 1, self.cv_linetype)
            cv2.putText(
                img,
                label,
                (w-x_offset+2, pos + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                COLOR_TEXT,
                1,
                self.cv_linetype
            )

    def _normalize(self, arr: NDArray[np.floating]) -> NDArray[np.float32]:
        a = np.asarray(arr, dtype=np.float32)
        # Just a little clip to eliminate ridiculous outliers
        lo = float(np.percentile(a, 0.01))
        hi = float(np.percentile(a, 99.99))
        if hi <= lo:
            hi = lo + 1.0
        norm = (a - lo) / (hi - lo)
        return np.clip(norm, 0, 1.0)

    def _composite_lockin_panes(
        self, main_img: NDArray[np.uint8], in_phase: NDArray[np.floating], quad: NDArray[np.floating],
        amplitude: NDArray[np.floating], angle: NDArray[np.floating]
    ) -> NDArray[np.uint8]:
        """Compose the main display with a 2x2 grid of lock-in results on the right.

        Grid layout:
        [In-Phase    | Amplitude]
        [Quadrature  | Angle    ]
        """
        h, w = main_img.shape[:2]

        # Normalize and colorize all four panes
        # In-Phase
        ip_color = apply_diverging_colormap(in_phase)
        
        # Quadrature
        q_color = apply_diverging_colormap(quad)

        # Amplitude (magnitude) - use current colormap for consistency
        amp_norm = self._normalize(amplitude)
        amp_u8 = (amp_norm * 255.0).astype(np.uint8)
        amp_color = apply_colormap(amp_u8, self.colormap_idx)
        
        # Angle - map [-pi, pi] to color
        masked_angle = np.where(amplitude >= amplitude.mean(), angle, np.nan)
        angle_color = apply_diverging_colormap(masked_angle, cmap_name='twilight')

        # Decide pane dimensions
        pane_w = max(64, w // 3)
        pane_h = h // 2

        # Resize all four panes to consistent size
        ip_resized = cv2.resize(ip_color, (pane_w, pane_h), interpolation=cv2.INTER_AREA)
        q_resized = cv2.resize(q_color, (pane_w, pane_h), interpolation=cv2.INTER_AREA)
        amp_resized = cv2.resize(amp_color, (pane_w, pane_h), interpolation=cv2.INTER_AREA)
        angle_resized = cv2.resize(angle_color, (pane_w, pane_h), interpolation=cv2.INTER_AREA)

        # Create 2x2 grid
        top_row = np.hstack([ip_resized, amp_resized])
        bottom_row = np.hstack([q_resized, angle_resized])
        grid = np.vstack([top_row, bottom_row])

        # If main_img has different number of channels, ensure 3
        if main_img.ndim == 2:
            main_img = cast(NDArray[np.uint8], cv2.cvtColor(main_img, cv2.COLOR_GRAY2BGR))
            main_img = np.asarray(main_img, dtype=np.uint8)

        composite = np.hstack([main_img, grid])
        return np.asarray(composite, dtype=np.uint8)

    def _draw_overlays(
        self, img: NDArray[np.uint8], thermal: NDArray[np.uint16], frame_stats: dict
    ) -> None:
        """Draw temperature overlays and UI elements."""
        h, w = img.shape[:2]
        th, tw = thermal.shape

        # Top status line
        cv2.putText(
            img,
            f"Spot: {frame_stats['tspot']:.1f}C | Range: {frame_stats['tmin']:.1f}-{frame_stats['tmax']:.1f}C",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            COLOR_TEXT,
            1,
            self.cv_linetype
        )

        # Bottom status line
        cmap_name = ColormapID(self.colormap_idx).name
        gain_name = self.camera.gain_mode.name
        emissivity = self.camera.env_params.emissivity
        scale = self.scale_mode.name if self.scale_mode != ScaleMode.OFF else ""
        status = f"{self.fps:.1f} FPS | {cmap_name} | {gain_name} | e={emissivity:.2f} {scale}"
        cv2.putText(
            img, 
            status, 
            (10, h - 10), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.5, 
            COLOR_TEXT, 
            1,
            self.cv_linetype
        )

        # Crosshair/reticule
        if self.show_reticule:
            cx_d, cy_d = w // 2, h // 2
            cv2.line(img, (cx_d - 15, cy_d), (cx_d + 15, cy_d), COLOR_RETICULE, 1, self.cv_linetype)
            cv2.line(img, (cx_d, cy_d - 15), (cx_d, cy_d + 15), COLOR_RETICULE, 1, self.cv_linetype)
            cv2.putText(
                img,
                f"{frame_stats['tspot']:.1f}C",
                (cx_d + 20, cy_d - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                COLOR_RETICULE,
                2,
                self.cv_linetype
            )

        # Maximum spot
        if self.hotspot_mode in [HotspotMode.MAX, HotspotMode.MINMAX]:
            cmax_d = self._coord_to_image(divmod(frame_stats['cmax'], tw), thermal, img)
            self._draw_box_marker(cmax_d, img, f"{frame_stats['tmax']:.1f}C", color = COLOR_SPOT_MAX)

        # Minimum spot
        if self.hotspot_mode in [HotspotMode.MIN, HotspotMode.MINMAX]:
            cmin_d = self._coord_to_image(divmod(frame_stats['cmin'], tw), thermal, img)
            self._draw_box_marker(cmin_d, img, f"{frame_stats['tmin']:.1f}C", color = COLOR_SPOT_MIN)

        # Help overlay
        if self.show_help:
            self._draw_help(img)

    def _draw_help(self, img: NDArray[np.uint8]) -> None:
        """Draw help overlay."""
        lines = [
            "q-Quit  h-help",
            "space-Shot  D-Dump",
            "+/- Zoom  r-Rotate  m-Mirror",
            "s-Shutter  g-High/Low Gain",
            "c-Colormap  v-Colorbar",
            "e-Emissivity  1-9 Set ems",
            "x-Scale  p-Enhanced  d-DDE",
            "t-Reticule  b-Min/max marker"
        ]
        overlay = img.copy()
        cv2.rectangle(overlay, (5, 30), (260, 185), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
        for i, line in enumerate(lines):
            cv2.putText(
                img,
                line,
                (10, 50 + i * 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                COLOR_TEXT,
                1,
                self.cv_linetype
            )

    def _handle_key(self, thermal: NDArray[np.uint16]) -> bool:
        """Handle keyboard input. Returns False to quit."""
        key = cv2.waitKey(1) & 0xFF

        if key == 255:
            return True

        if key == ord("q"):
            return False

        if key == ord("r"):
            self.rotation = (self.rotation + 90) % 360
            print(f"Rotation: {self.rotation} deg")
        elif key == ord("c"):
            self.colormap_idx = (self.colormap_idx + 1) % len(ColormapID)
        elif key == ord("s"):
            self.camera.trigger_shutter()
            print("Shutter triggered")
        elif key == ord("g"):
            # Cycle gain mode: HIGH -> LOW -> HIGH
            new_mode = (
                GainMode.LOW
                if self.camera.gain_mode == GainMode.HIGH
                else GainMode.HIGH
            )
            self.camera.set_gain_mode(new_mode)
            print(f"Gain mode: {new_mode.name}")
        elif key == ord("m"):
            self.mirror = not self.mirror
            print("Mirror:", "ON" if self.mirror else "OFF")
        elif key == ord("h"):
            self.show_help = not self.show_help
        elif key == ord("e"):
            self._cycle_emissivity()
        elif ord("1") <= key <= ord("9"):
            self._set_emissivity((key - ord("0")) / 10.0)
        elif key == ord("d"):
            self._toggle_dde()
        elif key == ord("D"):
            self._dump(thermal)
        elif key == ord(" "):
            self._screenshot()
        elif key in (ord("+"), ord("=")):
            self.zoom = min(5, self.zoom + 1)
        elif key in (ord("-"), ord("_")):
            self.zoom = max(1, self.zoom - 1)
        elif key == ord("x"):
            self.scale_mode = ScaleMode((self.scale_mode + 1) % len(ScaleMode))
            print(f"Scale: {self.scale_mode.name}")
        elif key == ord("p"):
            self._toggle_enhanced()
        elif key == ord("l"):
            # Toggle lock-in capture
            if not self.lockin_running:
                try:
                    # Create and start the lock-in controller with configured parameters
                    self.lockin_controller = LockInController(
                        self.camera, port=self.serial_port, baud_rate=self.baud_rate,
                        period=self.lockin_period, integration=self.lockin_integration,
                        invert=self.lockin_invert,
                    )
                    self.lockin_thread = self.lockin_controller.start_background()
                    self.lockin_running = True
                    print("Lock-in: started")
                except Exception as e:
                    print("Lock-in start failed:", e)
            else:
                # Stop and save results
                try:
                    if self.lockin_controller is not None:
                        self.lockin_controller.stop(wait=True)
                        print("Lock-in: stopped")
                except Exception as e:
                    print("Lock-in stop failed:", e)
                finally:
                    self.lockin_running = False
                    self.lockin_controller = None
                    self.lockin_thread = None
        elif key == ord("a"):
            self.agc_mode = AGCMode((self.agc_mode + 1) % len(AGCMode))
            print(f"AGC: {self.agc_mode.name}")
        elif key == ord("t"):
            self.show_reticule = not self.show_reticule
        elif key == ord("v"):
            self.show_colorbar = not self.show_colorbar
        elif key == ord("b"):
            self.hotspot_mode = HotspotMode((self.hotspot_mode + 1) % len(HotspotMode))
            print(f"Hotspot mode: {self.hotspot_mode.name}")

        return True

    def _toggle_enhanced(self) -> None:
        """Toggle enhanced processing mode (CLAHE + DDE)."""
        self.enhanced = not self.enhanced
        if self.enhanced:
            self.use_clahe = True
            self.dde_strength = 0.3
        else:
            self.use_clahe = False
            self.dde_strength = 0.0
        print(f"Enhanced: {'ON' if self.enhanced else 'OFF'}")

    def _toggle_dde(self) -> None:
        """Toggle DDE (Digital Detail Enhancement)."""
        if self.dde_strength > 0:
            self.dde_strength = 0.0
        else:
            self.dde_strength = 0.3
        print(f"DDE: {'ON' if self.dde_strength > 0 else 'OFF'}")

    def _dump(self, thermal: NDArray[np.uint16]) -> None:
        """Dump raw thermal data to file."""
        ts = time.strftime("%H%M%S")
        cy, cx = self._get_spot_coords(thermal)
        env = self.camera.env_params
        raw_temp = raw_to_celsius(thermal[cy, cx])
        corrected_temp = raw_to_celsius_corrected(thermal[cy, cx], env)
        print(f"\n--- Dump {ts} ---")
        print(f"Shape: {thermal.shape}, Range: {thermal.min()}-{thermal.max()}")
        print(f"Center raw: {thermal[cy, cx]}")
        print(f"Center temp (uncorrected): {raw_temp:.1f}C")
        print(f"Center temp (e={env.emissivity:.2f}): {corrected_temp:.1f}C")
        np.save(f"p3_raw_{ts}.npy", thermal)
        print(f"Saved: p3_raw_{ts}.npy\n")

    def _screenshot(self) -> None:
        """Save screenshot."""
        if self._last_display is None:
            print("No frame to save")
            return
        ts = time.strftime("%Y%m%d_%H%M%S")
        filename = f"p3_{ts}.png"
        cv2.imwrite(filename, self._last_display)
        print(f"Saved: {filename}")

    def _set_emissivity(self, value: float) -> None:
        """Set emissivity value."""
        self.camera.env_params.emissivity = value
        print(f"Emissivity: {value:.2f}")

    def _cycle_emissivity(self) -> None:
        """Cycle through common emissivity values."""
        values = [0.95, 0.90, 0.85, 0.80, 0.70, 0.50, 0.30, 0.10]
        current = self.camera.env_params.emissivity
        idx = 0
        for i, v in enumerate(values):
            if abs(current - v) < 0.01:
                idx = (i + 1) % len(values)
                break
        self._set_emissivity(values[idx])


def main() -> None:
    """Entry point."""
    import argparse

    def _parse_bool(val: str) -> bool:
        if isinstance(val, bool):
            return val
        s = str(val).strip().lower()
        if s in ("1", "true", "t", "yes", "y"):
            return True
        if s in ("0", "false", "f", "no", "n"):
            return False
        raise argparse.ArgumentTypeError(f"invalid boolean value: {val}")

    parser = argparse.ArgumentParser(
        description="Thermal Master P3/P1 USB thermal camera viewer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["p1", "p3"],
        default="p3",
        help="Camera model (default: p3)",
    )
    parser.add_argument(
        "--serial-port",
        type=str,
        default="/dev/ttyACM0",
        help="Serial port for lock-in load controller (default: /dev/ttyACM0)",
    )
    parser.add_argument(
        "--baud-rate",
        type=int,
        default=115200,
        help="Baud rate for serial port (default: 115200)",
    )
    parser.add_argument(
        "--period",
        type=float,
        default=1.0,
        help="Lock-in period in seconds (default: 1.0)",
    )
    parser.add_argument(
        "--integration",
        type=float,
        default=60.0,
        help="Lock-in integration time in seconds (default: 60.0)",
    )
    parser.add_argument(
        "--invert",
        nargs="?",
        const="1",
        type=_parse_bool,
        default=False,
        help="Invert lock-in serial output logic (accepts 0/1 or false/true).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG)

    try:
           P3Viewer(model=args.model, serial_port=args.serial_port, baud_rate=args.baud_rate,
               lockin_period=args.period, lockin_integration=args.integration,
               lockin_invert=args.invert).run()
    except RuntimeError as e:
        print(f"Error: {e}")
    except KeyboardInterrupt:
        print("\nInterrupted")


if __name__ == "__main__":
    main()
