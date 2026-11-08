"""Sensor-coordinate measurements and non-destructive radiometric correction.

The correction is an experimental broad-band T^4 model, not a calibrated P3
spectral inversion. Atmospheric coefficients are generic, not measured for P3.
"""

from dataclasses import asdict, dataclass, field

import cv2
import numpy as np

from .processing import temperature


@dataclass
class Radiometry:
    enabled: bool = False
    emissivity: float = 1.0
    reflected: float = 20.0
    atmosphere: float = 20.0
    distance: float = 0.0
    humidity: float = 50.0
    atmospheric_correction: bool = False

    def validate(self):
        if (
            type(self.enabled) is not bool
            or type(self.atmospheric_correction) is not bool
        ):
            raise ValueError("Correction switches must be booleans")
        values = [
            self.emissivity,
            self.reflected,
            self.atmosphere,
            self.distance,
            self.humidity,
        ]
        if (
            any(type(v) not in (int, float) for v in values)
            or not np.isfinite(values).all()
        ):
            raise ValueError("Radiometry parameters must be finite")
        if not 0.01 <= self.emissivity <= 1 or not -100 <= self.reflected <= 1000:
            raise ValueError("Emissivity: 0.01–1; reflected temperature: −100–1000 °C")
        if (
            not -40 <= self.atmosphere <= 80
            or not 0 <= self.distance <= 1000
            or not 0 <= self.humidity <= 100
        ):
            raise ValueError("Air: −40–80 °C; distance: 0–1000 m; humidity: 0–100%")

    def transmission(self):
        """Generic empirical LWIR model; single object-to-camera path sqrt(distance)."""
        self.validate()
        if not self.atmospheric_correction or self.distance == 0:
            return 1.0
        t = self.atmosphere
        water = (
            self.humidity
            / 100
            * np.exp(1.5587 + 0.06939 * t - 0.00027816 * t * t + 0.00000068455 * t**3)
        )
        d, w = np.sqrt(self.distance), np.sqrt(water)
        tau = 1.9 * np.exp(-d * (0.006569 - 0.002276 * w)) - 0.9 * np.exp(
            -d * (0.01262 - 0.00667 * w)
        )
        if not 0.01 < tau <= 1.000001:
            raise ValueError(
                "Atmospheric model is outside its valid transmission range"
            )
        return min(1.0, float(tau))


@dataclass
class EmissivityLayer:
    name: str
    emissivity: float
    mask: np.ndarray
    enabled: bool = True


@dataclass
class Region:
    name: str
    kind: str
    start: tuple[int, int]  # x, y in original sensor coordinates
    end: tuple[int, int]

    def samples(self, shape):
        """Return unique sensor pixels; line samples remain ordered for the profile."""
        h, w = shape
        x0, y0 = self.start
        x1, y1 = self.end
        x0, x1 = np.clip([x0, x1], 0, w - 1)
        y0, y1 = np.clip([y0, y1], 0, h - 1)
        if self.kind == "Spot":
            return np.array([y0]), np.array([x0])
        if self.kind == "Line":
            n = int(max(abs(x1 - x0), abs(y1 - y0))) + 1
            return np.rint(np.linspace(y0, y1, n)).astype(int), np.rint(
                np.linspace(x0, x1, n)
            ).astype(int)
        if self.kind == "Rectangle":
            y, x = np.mgrid[
                min(y0, y1) : max(y0, y1) + 1, min(x0, x1) : max(x0, x1) + 1
            ]
            return y.ravel(), x.ravel()
        if self.kind == "Circle":
            y, x = np.ogrid[:h, :w]
            radius = np.hypot(x1 - x0, y1 - y0)
            mask = (x - x0) ** 2 + (y - y0) ** 2 <= radius**2
        else:
            raise ValueError("Unknown ROI kind")
        return np.nonzero(mask)

    def statistics(self, celsius):
        ys, xs = self.samples(celsius.shape)
        values = celsius[ys, xs]
        finite = np.isfinite(values)
        if not finite.any():
            return {"count": 0, "min": None, "max": None, "mean": None}
        valid = values[finite]
        return {
            "count": int(finite.sum()),
            "min": float(valid.min()),
            "max": float(valid.max()),
            "mean": float(valid.mean()),
        }

    def profile(self, celsius):
        ys, xs = self.samples(celsius.shape)
        distance = np.concatenate(
            ([0.0], np.cumsum(np.hypot(np.diff(xs), np.diff(ys))))
        )
        return distance, celsius[ys, xs]


@dataclass
class AnalysisState:
    radiometry: Radiometry = field(default_factory=Radiometry)
    regions: list[Region] = field(default_factory=list)
    layers: list[EmissivityLayer] = field(default_factory=list)
    isotherm: bool = False
    iso_min: float = 50.0
    iso_max: float = 1000.0
    iso_color: str = "#FF00FF"
    revision: int = 0

    def touch(self):
        self.revision += 1

    def emissivity_map(self, shape):
        result = np.full(shape, self.radiometry.emissivity, np.float64)
        for layer in self.layers:
            if layer.enabled and layer.mask.shape == shape:
                result[layer.mask] = layer.emissivity
        return result

    def celsius(self, raw):
        apparent = temperature(raw)
        if not self.radiometry.enabled:
            return apparent
        self.radiometry.validate()
        eps = self.emissivity_map(raw.shape)
        tau = self.radiometry.transmission()
        reflected = (self.radiometry.reflected + 273.15) ** 4
        air = (self.radiometry.atmosphere + 273.15) ** 4
        power = (
            (apparent + 273.15) ** 4 - (1 - tau) * air - tau * (1 - eps) * reflected
        ) / (tau * eps)
        # Impossible inversions are NaN, never fabricated zero-K temperatures.
        return np.where(power > 0, np.maximum(power, 0) ** 0.25 - 273.15, np.nan)

    def paint(self, layer_index, start, end, radius, erase=False):
        """Paint a continuous sensor-space stroke; GUI transforms coordinates first."""
        layer = self.layers[layer_index]
        mask = layer.mask.astype(np.uint8)
        cv2.line(
            mask,
            tuple(map(int, start)),
            tuple(map(int, end)),
            0 if erase else 1,
            max(1, int(radius) * 2 + 1),
        )
        layer.mask[:] = mask.astype(bool)
        self.touch()

    def to_dict(self):
        return {
            "version": 1,
            "radiometry": asdict(self.radiometry),
            "regions": [asdict(r) for r in self.regions],
            "isotherm": {
                "enabled": self.isotherm,
                "minimum": self.iso_min,
                "maximum": self.iso_max,
                "color": self.iso_color,
            },
            "layers": [
                {
                    "name": layer.name,
                    "emissivity": layer.emissivity,
                    "enabled": layer.enabled,
                    "shape": list(layer.mask.shape),
                    "mask_hex": np.packbits(layer.mask).tobytes().hex(),
                }
                for layer in self.layers
            ],
        }

    @classmethod
    def from_dict(cls, data, shape):
        if data is None:
            return cls()
        if not isinstance(data, dict) or data.get("version") != 1:
            raise ValueError("Unsupported analysis metadata")
        result = cls(radiometry=Radiometry(**data.get("radiometry", {})))
        result.radiometry.validate()
        if result.radiometry.enabled:
            result.radiometry.transmission()
        for item in data.get("regions", []):
            if len(result.regions) >= 100:
                raise ValueError("Too many ROIs")
            region = Region(**item)
            if region.kind not in (
                "Spot",
                "Line",
                "Rectangle",
                "Circle",
            ) or not isinstance(region.name, str):
                raise ValueError("Invalid ROI")
            for point in (region.start, region.end):
                if (
                    len(point) != 2
                    or any(type(v) is not int for v in point)
                    or not 0 <= point[0] < shape[1]
                    or not 0 <= point[1] < shape[0]
                ):
                    raise ValueError("ROI outside sensor")
            result.regions.append(region)
        for item in data.get("layers", []):
            if (
                not isinstance(item, dict)
                or not isinstance(item.get("name"), str)
                or type(item.get("enabled", True)) is not bool
                or type(item.get("emissivity")) not in (int, float)
                or len(result.layers) >= 32
                or tuple(item["shape"]) != shape
                or not 0.01 <= item["emissivity"] <= 1
            ):
                raise ValueError("Invalid emissivity layer")
            bits = bytes.fromhex(item["mask_hex"])
            if len(bits) != (int(np.prod(shape)) + 7) // 8:
                raise ValueError("Invalid layer mask length")
            mask = (
                np.unpackbits(np.frombuffer(bits, np.uint8))[: int(np.prod(shape))]
                .reshape(shape)
                .astype(bool)
            )
            result.layers.append(
                EmissivityLayer(
                    item["name"],
                    float(item["emissivity"]),
                    mask,
                    bool(item.get("enabled", True)),
                )
            )
        iso = data.get("isotherm", {})
        if not isinstance(iso, dict) or type(iso.get("enabled", False)) is not bool:
            raise ValueError("Invalid isotherm switch")
        result.isotherm = iso.get("enabled", False)
        result.iso_min, result.iso_max = (
            float(iso.get("minimum", 50)),
            float(iso.get("maximum", 1000)),
        )
        result.iso_color = iso.get("color", "#FF00FF")
        import re

        if (
            not np.isfinite([result.iso_min, result.iso_max]).all()
            or result.iso_min > result.iso_max
            or not re.fullmatch(r"#[0-9a-fA-F]{6}", result.iso_color)
        ):
            raise ValueError("Invalid isotherm")
        return result
