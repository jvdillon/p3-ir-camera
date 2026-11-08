"""Lossless snapshot export, suitable for subsequent radiometric analysis."""

from datetime import datetime, timezone
from pathlib import Path

import json
import math
import zipfile
import zlib

import numpy as np

from .storage import atomic_output


def save_snapshot(path, frame, metadata, corrected_celsius=None):
    """Save native planes and self-describing JSON; never save display-filtered raw."""
    metadata = {
        **metadata,
        "saved_utc": datetime.now(timezone.utc).isoformat(),
        "monotonic_timestamp": frame.timestamp,
        "conversion": "celsius = raw / 64 - 273.15",
        "raw_unit": "1/64 K",
    }
    extra = (
        {} if corrected_celsius is None else {"corrected_celsius": corrected_celsius}
    )
    with atomic_output(path) as stream:
        np.savez_compressed(
            stream,
            raw=frame.raw,
            brightness=frame.brightness,
            metadata=json.dumps(metadata),
            **extra,
        )


def load_snapshot(path):
    """Load validated native planes without pickle; legacy NPZ snapshots are accepted."""
    try:
        return _load_snapshot(path)
    except (
        zipfile.BadZipFile,
        EOFError,
        zlib.error,
        NotImplementedError,
        RuntimeError,
    ) as exc:
        raise ValueError(f"Invalid NPZ snapshot: {exc}") from exc


def _load_snapshot(path):
    from .acquisition import Frame

    if not zipfile.is_zipfile(path):
        raise ValueError("Not a valid NPZ snapshot")
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        if len(entries) > 16 or sum(item.file_size for item in entries) > 32_000_000:
            raise ValueError("Snapshot exceeds supported size")
        names = [item.filename for item in entries]
        if len(names) != len(set(names)) or not {"raw.npy", "brightness.npy"} <= set(
            names
        ):
            raise ValueError("Missing or duplicate snapshot arrays")
        for item in entries:
            with archive.open(item) as stream:
                version = np.lib.format.read_magic(stream)
                readers = {
                    (1, 0): np.lib.format.read_array_header_1_0,
                    (2, 0): np.lib.format.read_array_header_2_0,
                }
                if version not in readers:
                    raise ValueError("Unsupported array header version")
                shape, _, dtype = readers[version](stream)
                size = math.prod(shape) * dtype.itemsize
                if (
                    dtype.hasobject
                    or size > item.file_size - stream.tell()
                    or len(shape) > 2
                ):
                    raise ValueError("Invalid array size or object data")
    with np.load(path, allow_pickle=False) as data:
        raw = data["raw"]
        brightness = data["brightness"]
        if raw.dtype != np.uint16 or raw.ndim != 2 or not 0 < raw.size <= 1_048_576:
            raise ValueError("RAW must be a nonempty 2D uint16 sensor plane")
        if brightness.dtype != np.uint8 or brightness.shape != raw.shape:
            raise ValueError("Brightness must be uint8 with the same dimensions as RAW")
        metadata = json.loads(str(data["metadata"])) if "metadata" in data else {}
        if not isinstance(metadata, dict):
            raise ValueError("Snapshot metadata must be a JSON object")
        if metadata.get("snapshot_version", 1) not in (1, 2):
            raise ValueError("Unsupported snapshot version")
        if metadata.get("raw_unit", "1/64 K") != "1/64 K":
            raise ValueError("Unsupported RAW units")
        stamp = float(metadata.get("monotonic_timestamp", 0.0))
        if not np.isfinite(stamp):
            raise ValueError("Invalid acquisition timestamp")
        return Frame(raw.copy(), brightness.copy(), stamp), metadata


def save_image(path, frame, rgb, kind="jpeg", scale=3, quality=95, legend=None):
    """Export smooth presentation JPEG, native uint16 RAW PNG, or rendered RGB PNG."""
    import cv2

    if kind == "raw_png":
        image = frame.raw  # preserve native orientation and every bit
        parameters = []
    elif kind in ("jpeg", "color_png"):
        image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        parameters = []
        if kind == "jpeg":
            if not 1 <= scale <= 8 or not 1 <= quality <= 100:
                raise ValueError("JPEG scale must be 1–8 and quality 1–100")
            image = cv2.resize(
                image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
            )
            parameters = [cv2.IMWRITE_JPEG_QUALITY, quality]
    else:
        raise ValueError("Unknown image export format")
    if legend is not None and kind != "raw_png":
        image = append_legend(image, legend)
    extension = ".jpg" if kind == "jpeg" else ".png"
    success, encoded = cv2.imencode(extension, image, parameters)
    if not success:
        raise OSError("Image encoder failed")
    with atomic_output(path) as stream:
        stream.write(encoded.tobytes())


def append_legend(image, legend):
    """Append an RGB-palette legend to a BGR presentation image, keeping RAW untouched."""
    import cv2

    colors, low, high, unit = legend[:4]
    bands = legend[4] if len(legend) > 4 else None
    h, w = image.shape[:2]
    height = max(h, 160)
    output = np.full((height, w + 140, 3), (29, 21, 16), np.uint8)
    output[:h, :w] = image
    ramp = cv2.resize(
        colors[::-1].reshape(256, 1, 3),
        (20, height - 50),
        interpolation=cv2.INTER_NEAREST,
    )
    output[30 : height - 20, w + 10 : w + 30] = ramp[..., ::-1]
    cv2.putText(
        output,
        unit.replace("°", ""),
        (w + 10, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    for index, fraction in enumerate((0, 0.25, 0.5, 0.75, 1)):
        value = high + (low - high) * fraction
        y = int(30 + (height - 50) * fraction)
        label = f"{value:.2f}"
        if bands is not None:
            bounds = bands[index]
            label = "--" if bounds is None else f"{bounds[0]:.1f}/{bounds[1]:.1f}"
        cv2.putText(
            output,
            label,
            (w + 37, min(height - 5, y + 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return output


def snapshot_display_settings(metadata):
    """Validate optional presentation metadata, including migration of the old combined enhancement flag."""
    from .processing import DisplaySettings

    data = metadata.get("display", {})
    if not isinstance(data, dict):
        raise ValueError("Display metadata must be an object")
    settings = DisplaySettings()
    for field in settings.__dataclass_fields__:
        if field in data:
            setattr(settings, field, data[field])
    if settings.mode not in (
        "Temperature",
        "Filtered temperature",
        "Raw counts",
        "Factory brightness",
    ) or settings.range_mode not in ("Auto percentile", "Fixed"):
        raise ValueError("Invalid snapshot display mode")
    if not isinstance(settings.palette, str):
        raise ValueError("Invalid snapshot palette name")
    numbers = (
        settings.minimum,
        settings.maximum,
        settings.alpha,
        settings.dde_strength,
    )
    if any(
        isinstance(n, bool) or not isinstance(n, (int, float)) or not np.isfinite(n)
        for n in numbers
    ):
        raise ValueError("Invalid display range or filter weight")
    if (
        settings.maximum <= settings.minimum
        or not 0.05 <= settings.alpha <= 1
        or not 0 <= settings.dde_strength <= 4
    ):
        raise ValueError("Invalid display range or filter weight")
    if not all(
        isinstance(value, bool)
        for value in (
            settings.detail,
            settings.clahe,
            settings.custom_auto_scale,
            settings.enhance_custom_palette,
            settings.enhancements_enabled,
        )
    ):
        raise ValueError("Enhancement flags must be booleans")
    if metadata.get("snapshot_version", 1) == 1 and "clahe" not in data:
        settings.clahe = settings.detail
    return settings


def load_raw_image(path):
    """Load standalone uint16 PNG/NPY; brightness is explicitly reconstructed, not factory data."""
    import cv2

    from .acquisition import Frame

    path = Path(path)
    if path.stat().st_size > 32_000_000:
        raise ValueError("RAW image exceeds 32 MB")
    if path.suffix.lower() == ".npy":
        raw = np.load(path, allow_pickle=False, mmap_mode="r")
    elif path.suffix.lower() == ".png":
        import struct

        content = path.read_bytes()
        if len(content) < 24 or content[:8] != b"\x89PNG\r\n\x1a\n":
            raise ValueError("Invalid PNG header")
        width, height = struct.unpack(">II", content[16:24])
        if not 0 < width * height <= 1_048_576:
            raise ValueError("RAW PNG dimensions exceed the supported limit")
        raw = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_UNCHANGED)
    else:
        raise ValueError("Expected 16-bit PNG or uint16 NPY")
    if (
        raw is None
        or raw.dtype != np.uint16
        or raw.ndim != 2
        or not 0 < raw.size <= 1_048_576
    ):
        raise ValueError(
            "Image must be a nonempty single-channel uint16 RAW sensor plane"
        )
    values = raw.astype(np.float64)
    brightness = (
        (values - values.min()) / max(1.0, values.max() - values.min()) * 255
    ).astype(np.uint8)
    return Frame(raw.copy(), brightness, 0.0), {
        "brightness_origin": "derived_from_raw",
        "raw_unit": "1/64 K",
        "model": "unknown",
        "demo": False,
    }
