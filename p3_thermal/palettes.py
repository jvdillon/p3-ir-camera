"""User temperature palettes: validated portable JSON and atomic local storage."""

from dataclasses import dataclass
from pathlib import Path

import json
import os
import re

import numpy as np

from .storage import atomic_output


@dataclass(frozen=True)
class TemperaturePalette:
    """Absolute Celsius/color stops with linear RGB interpolation or discrete bands."""

    name: str
    stops: tuple[tuple[float, str], ...]
    interpolation: str = "linear"

    @classmethod
    def from_dict(cls, data):
        if not isinstance(data, dict) or data.get("version") != 1:
            raise ValueError("Unsupported palette format (expected version 1)")
        name = data.get("name")
        if not isinstance(name, str) or not name.strip() or len(name) > 80:
            raise ValueError("Palette name must contain 1–80 characters")
        mode = data.get("interpolation", "linear")
        if mode not in ("linear", "steps"):
            raise ValueError("Interpolation must be linear or steps")
        stops = data.get("stops")
        if not isinstance(stops, list) or not 2 <= len(stops) <= 64:
            raise ValueError("Define 2–64 temperature/color stops")
        parsed = []
        for stop in stops:
            if not isinstance(stop, list) or len(stop) != 2:
                raise ValueError("Each stop must be [temperature, '#RRGGBB']")
            temp, color = stop
            if (
                isinstance(temp, bool)
                or not isinstance(temp, (int, float))
                or not np.isfinite(temp)
            ):
                raise ValueError("Temperatures must be finite numbers")
            if not isinstance(color, str) or not re.fullmatch(
                r"#[0-9a-fA-F]{6}", color
            ):
                raise ValueError("Colors must use #RRGGBB")
            parsed.append((float(temp), color.upper()))
        if any(a[0] >= b[0] for a, b in zip(parsed, parsed[1:], strict=False)):
            raise ValueError("Temperatures must be strictly increasing")
        return cls(name.strip(), tuple(parsed), mode)

    def to_dict(self):
        return {
            "version": 1,
            "name": self.name,
            "interpolation": self.interpolation,
            "stops": [list(stop) for stop in self.stops],
        }

    def colorize(self, celsius):
        """Clamp outside endpoints; map absolute temperature, never frame percentiles."""
        points = np.array([stop[0] for stop in self.stops])
        colors = np.array(
            [[int(color[i : i + 2], 16) for i in (1, 3, 5)] for _, color in self.stops]
        )
        if self.interpolation == "steps":
            index = np.clip(
                np.searchsorted(points, celsius, side="right") - 1, 0, len(points) - 1
            )
            return colors[index].astype(np.uint8)
        return (
            np.stack(
                [np.interp(celsius, points, colors[:, i]) for i in range(3)], axis=-1
            )
            .round()
            .astype(np.uint8)
        )


def read_palette(path):
    if Path(path).stat().st_size > 100_000:
        raise ValueError("Palette file is too large")
    return TemperaturePalette.from_dict(
        json.loads(Path(path).read_text(encoding="utf-8"))
    )


def write_json(path, data):
    """Atomic replacement avoids losing an existing library on an interrupted save."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_output(path, "w") as stream:
        json.dump(data, stream, indent=2, ensure_ascii=False)
        stream.write("\n")


def default_library_path():
    base = os.environ.get("XDG_CONFIG_HOME") or os.environ.get("APPDATA")
    return (
        (Path(base) if base else Path.home() / ".config")
        / "p3-thermal-studio"
        / "palettes.json"
    )


class PaletteLibrary:
    """Factory names are reserved; mutations persist before updating in-memory state."""

    def __init__(self, factory_names, path=None):
        self.path = Path(path) if path is not None else default_library_path()
        self.factory_names = set(factory_names)
        self.palettes = {}
        self.load_error = None
        if self.path.exists():
            try:
                if self.path.stat().st_size > 2_000_000:
                    raise ValueError("Palette library is too large")
                data = json.loads(self.path.read_text(encoding="utf-8"))
                if (
                    not isinstance(data, dict)
                    or data.get("version") != 1
                    or not isinstance(data.get("palettes"), list)
                ):
                    raise ValueError("Unsupported palette library")
                for item in data["palettes"]:
                    palette = TemperaturePalette.from_dict(item)
                    if (
                        palette.name in self.factory_names
                        or palette.name in self.palettes
                    ):
                        raise ValueError("Duplicate or reserved palette name")
                    self.palettes[palette.name] = palette
            except (OSError, ValueError, TypeError) as exc:
                self.palettes = {}
                self.load_error = str(exc)

    def _commit(self, palettes):
        if self.load_error:
            raise ValueError(
                f"Existing palette library cannot be read; repair or move {self.path} before saving. {self.load_error}"
            )
        write_json(
            self.path,
            {"version": 1, "palettes": [p.to_dict() for p in palettes.values()]},
        )
        self.palettes = palettes

    def save(self, palette):
        palette = TemperaturePalette.from_dict(palette.to_dict())
        if palette.name in self.factory_names:
            raise ValueError("Factory palettes cannot be overwritten")
        updated = dict(self.palettes)
        updated[palette.name] = palette
        self._commit(updated)

    def delete(self, name):
        if name in self.factory_names:
            raise ValueError("Factory palettes cannot be deleted")
        updated = dict(self.palettes)
        del updated[name]
        self._commit(updated)
