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
  d - Toggle DDE
"""

from __future__ import annotations

import time
from enum import IntEnum
from typing import Any
from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray

from p3_camera import GainMode
from p3_camera import P3Camera
from p3_camera import raw_to_celsius


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


# Colormaps (BGR format for OpenCV)
COLORMAPS: dict[ColormapID, NDArray[np.uint8]] = {}


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

    def __init__(self) -> None:
        self.camera = P3Camera()
        self.rotation: int = 0
        self.colormap_idx: int = ColormapID.IRONBOW
        self.mirror: bool = False
        self.show_help: bool = False
        self.show_reticule: bool = True
        self.zoom: int = 3
        self.fps: float = 0.0
        self.enhanced: bool = True
        self.use_clahe: bool = True
        self.scale_mode: ScaleMode = ScaleMode.BICUBIC
        self.agc_mode: AGCMode = AGCMode.FACTORY
        self.dde_strength: float = 0.3
        self.tnr_alpha: float = 0.5
        self._fps_count: int = 0
        self._fps_time: float = time.time()
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self._last_display: NDArray[np.uint8] | None = None
        self._prev_frame: NDArray[np.uint16] | None = None
        self._ir_brightness: NDArray[np.uint8] | None = None

    def run(self) -> None:
        """Main viewer loop."""
        print("P3 Thermal Viewer")
        self.camera.connect()
        name, version = self.camera.init()
        print(f"Device: {name}, Firmware: {version}")
        self.camera.start_streaming()
        print("Press 'h' for help")

        cv2.namedWindow("P3 Thermal", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("P3 Thermal", 640, 480)

        try:
            while True:
                ir_brightness, thermal = self.camera.read_frame_both()
                if thermal is None:
                    continue
                self._ir_brightness = ir_brightness

                # Apply temporal noise reduction
                thermal = tnr(thermal, self._prev_frame, alpha=self.tnr_alpha)
                self._prev_frame = thermal.copy()

                self._last_display = self._render(thermal)
                cv2.imshow("P3 Thermal", self._last_display)
                self._update_fps()

                if not self._handle_key(thermal):
                    break
                if cv2.getWindowProperty("P3 Thermal", cv2.WND_PROP_VISIBLE) < 1:
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

    def _render(self, thermal: NDArray[np.uint16]) -> NDArray[np.uint8]:
        """Render thermal frame to display image."""

        # AGC: normalize to 8-bit based on selected mode
        if self.agc_mode == AGCMode.FACTORY:
            # Use hardware AGC'd IR brightness from camera (already 8-bit)
            if self._ir_brightness is not None:
                # IR is 192 rows, thermal is 190 rows - crop to match
                img = self._ir_brightness[2:, :].copy()
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
            img = cv2.resize(
                img, (w * 2, h * 2), interpolation=SCALE_INTERP[self.scale_mode]
            )

        # Optional CLAHE for local contrast enhancement
        if self.use_clahe:
            clahe_result: Any = self._clahe.apply(img)
            img = clahe_result

        # DDE: edge enhancement
        img = dde(np.asarray(img, dtype=np.uint8), strength=self.dde_strength)

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

        # Overlays
        self._draw_overlays(result, thermal)

        return result

    def _draw_overlays(
        self, img: NDArray[np.uint8], thermal: NDArray[np.uint16]
    ) -> None:
        """Draw temperature overlays and UI elements."""
        h, w = img.shape[:2]
        cy, cx = self._get_spot_coords(thermal)

        # Temperature values
        spot = raw_to_celsius(thermal[cy, cx])
        tmin = raw_to_celsius(thermal.min())
        tmax = raw_to_celsius(thermal.max())

        # Top status line
        cv2.putText(
            img,
            f"Spot: {spot:.1f}C | Range: {tmin:.1f}-{tmax:.1f}C",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

        # Bottom status line
        cmap_name = ColormapID(self.colormap_idx).name
        gain_name = self.camera.gain_mode.name
        emissivity = self.camera.env_params.emissivity
        scale = self.scale_mode.name if self.scale_mode != ScaleMode.OFF else ""
        status = f"{self.fps:.1f} FPS | {cmap_name} | {gain_name} | e={emissivity:.2f} {scale}"
        cv2.putText(
            img, status, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
        )

        # Crosshair/reticule
        if self.show_reticule:
            cx_d, cy_d = w // 2, h // 2
            cv2.line(img, (cx_d - 15, cy_d), (cx_d + 15, cy_d), (0, 255, 0), 1)
            cv2.line(img, (cx_d, cy_d - 15), (cx_d, cy_d + 15), (0, 255, 0), 1)
            cv2.putText(
                img,
                f"{spot:.1f}C",
                (cx_d + 20, cy_d - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        # Help overlay
        if self.show_help:
            self._draw_help(img)

    def _draw_help(self, img: NDArray[np.uint8]) -> None:
        """Draw help overlay."""
        lines = [
            "q-Quit  +/- Zoom  r-Rotate",
            "c-Color s-Shutter g-Gain",
            "m-Mirror h-Help  space-Shot",
            "e-Emissivity 1-9 Set ems",
            "x-Scale p-Enhanced d-DDE D-Dump",
        ]
        overlay = img.copy()
        cv2.rectangle(overlay, (5, 30), (290, 130), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
        for i, line in enumerate(lines):
            cv2.putText(
                img,
                line,
                (10, 50 + i * 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1,
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
        elif key == ord("a"):
            self.agc_mode = AGCMode((self.agc_mode + 1) % len(AGCMode))
            print(f"AGC: {self.agc_mode.name}")
        elif key == ord("t"):
            self.show_reticule = not self.show_reticule

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
        print(f"\n--- Dump {ts} ---")
        print(f"Shape: {thermal.shape}, Range: {thermal.min()}-{thermal.max()}")
        print(
            f"Center raw: {thermal[cy, cx]}, Center C: {raw_to_celsius(thermal[cy, cx]):.1f}"
        )
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
    try:
        P3Viewer().run()
    except RuntimeError as e:
        print(f"Error: {e}")
    except KeyboardInterrupt:
        print("\nInterrupted")


if __name__ == "__main__":
    main()
