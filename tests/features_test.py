"""Round-trip contracts for offline analysis, image formats and temperature presets."""

import json

import cv2
import numpy as np
import pytest

from p3_thermal.acquisition import Frame
from p3_thermal.export import load_snapshot, save_image, save_snapshot
from p3_thermal.palettes import (
    PaletteLibrary,
    TemperaturePalette,
    read_palette,
    write_json,
)
from p3_thermal.processing import PALETTES, DisplaySettings, Processor, legend_colors


def palette(name="Board", interpolation="linear"):
    return TemperaturePalette.from_dict(
        {
            "version": 1,
            "name": name,
            "interpolation": interpolation,
            "stops": [[0, "#000000"], [10, "#FF0000"], [20, "#FFFFFF"]],
        }
    )


def test_absolute_colors_and_legend_agree():
    p = palette()
    np.testing.assert_array_equal(
        p.colorize(np.array([-10, 0, 10, 20, 30])),
        [[0, 0, 0], [0, 0, 0], [255, 0, 0], [255, 255, 255], [255, 255, 255]],
    )
    ramp = legend_colors(DisplaySettings(), p)
    np.testing.assert_array_equal(ramp, p.colorize(np.linspace(0, 20, 256)))
    np.testing.assert_array_equal(
        palette(interpolation="steps").colorize(np.array([9.99, 10, 19.99])),
        [[0, 0, 0], [255, 0, 0], [255, 0, 0]],
    )


def test_palette_library_restart_export_delete_and_factory_protection(tmp_path):
    path = tmp_path / "config" / "palettes.json"
    library = PaletteLibrary(PALETTES, path)
    library.save(palette())
    restarted = PaletteLibrary(PALETTES, path)
    assert restarted.palettes == library.palettes
    exported = tmp_path / "board.json"
    write_json(exported, restarted.palettes["Board"].to_dict())
    assert read_palette(exported) == palette()
    with pytest.raises(ValueError, match="Factory"):
        restarted.save(palette("Inferno"))
    with pytest.raises(ValueError, match="Factory"):
        restarted.delete("Inferno")
    restarted.delete("Board")
    assert not PaletteLibrary(PALETTES, path).palettes


@pytest.mark.parametrize(
    "stops",
    [
        [[0, "red"], [10, "#FFFFFF"]],
        [[1, "#000000"], [1, "#FFFFFF"]],
        [[float("nan"), "#000000"], [10, "#FFFFFF"]],
    ],
)
def test_invalid_palette_rejected(stops):
    with pytest.raises(ValueError):
        TemperaturePalette.from_dict({"version": 1, "name": "Bad", "stops": stops})


def test_corrupt_library_is_not_silently_overwritten(tmp_path):
    path = tmp_path / "palettes.json"
    path.write_text("broken")
    library = PaletteLibrary(PALETTES, path)
    assert library.load_error
    with pytest.raises(ValueError):
        library.save(palette())
    assert path.read_text() == "broken"


def test_full_snapshot_and_native_raw_png_are_lossless(tmp_path):
    raw = np.arange(65536, dtype=np.uint16).reshape(256, 256)
    frame = Frame(raw, (raw % 256).astype(np.uint8), 123.5)
    path = tmp_path / "all.npz"
    save_snapshot(
        path, frame, {"snapshot_version": 2, "palette_spec": palette().to_dict()}
    )
    restored, metadata = load_snapshot(path)
    np.testing.assert_array_equal(restored.raw, frame.raw)
    np.testing.assert_array_equal(restored.brightness, frame.brightness)
    assert TemperaturePalette.from_dict(metadata["palette_spec"]) == palette()
    png = tmp_path / "raw.png"
    save_image(png, frame, None, "raw_png")
    np.testing.assert_array_equal(cv2.imread(str(png), cv2.IMREAD_UNCHANGED), raw)


def test_jpeg_enlargement_color_png_and_legend(tmp_path):
    frame = Frame(np.full((8, 8), 19000, np.uint16), np.zeros((8, 8), np.uint8), 0)
    rgb = np.full((8, 8, 3), (250, 100, 20), np.uint8)
    jpeg = tmp_path / "smooth.jpg"
    save_image(jpeg, frame, rgb, "jpeg", scale=3)
    assert cv2.imread(str(jpeg)).shape == (24, 24, 3)
    png = tmp_path / "color.png"
    save_image(png, frame, rgb, "color_png")
    np.testing.assert_array_equal(cv2.imread(str(png))[..., ::-1], rgb)
    legend = (legend_colors(DisplaySettings()), 15, 40, "°C")
    save_image(png, frame, rgb, "color_png", legend=legend)
    assert cv2.imread(str(png)).shape == (160, 148, 3)
    save_image(png, frame, rgb, "raw_png", legend=legend)
    np.testing.assert_array_equal(cv2.imread(str(png), cv2.IMREAD_UNCHANGED), frame.raw)


def test_clahe_and_dde_are_independent_and_leave_raw_untouched():
    raw = np.arange(256, dtype=np.uint16).reshape(16, 16) + 19000
    original = raw.copy()
    results = []
    for clahe, dde in ((False, False), (True, False), (False, True), (True, True)):
        settings = DisplaySettings(clahe=clahe, detail=dde)
        rgb, limits = Processor().render(raw, None, settings)
        assert (limits is None) == (clahe or dde)
        results.append(rgb)
    assert not np.array_equal(results[0], results[1])
    assert not np.array_equal(results[0], results[2])
    np.testing.assert_array_equal(raw, original)
    enhanced = Processor().render(
        raw, None, DisplaySettings(clahe=True, detail=True), palette()
    )
    plain = Processor().render(raw, None, DisplaySettings(), palette())
    assert not np.array_equal(enhanced[0], plain[0])
    assert enhanced[1] is None


def test_snapshot_rejects_invalid_planes(tmp_path):
    path = tmp_path / "bad.npz"
    np.savez(
        path,
        raw=np.zeros((8, 8), np.float32),
        brightness=np.zeros((8, 8), np.uint8),
        metadata=json.dumps({}),
    )
    with pytest.raises(ValueError, match="uint16"):
        load_snapshot(path)


def test_nonlinear_legend_reports_observed_temperature_ranges():
    from p3_thermal.processing import observed_temperature_bands, temperature

    raw = np.array([[19000, 19064, 19128, 19200, 19300, 19400]], np.uint16)
    gray = np.array([[0, 20, 80, 140, 200, 255]], np.uint8)
    bands = observed_temperature_bands(raw, gray)
    temps = temperature(raw)[0]
    assert bands[0] == (temps[5], temps[5])
    assert bands[-1] == (temps[0], temps[1])
    sparse = observed_temperature_bands(raw, np.zeros_like(gray))
    assert sparse[0] is None
    assert sparse[-1] == (temps.min(), temps.max())


def test_dde_strength_and_snapshot_metadata_roundtrip(tmp_path):
    from dataclasses import asdict

    from p3_thermal.export import snapshot_display_settings

    raw = np.full((32, 32), 19000, np.uint16)
    raw[:, 16:] += 160
    raw[12:20, 12:20] += 40
    settings = DisplaySettings(
        detail=True, dde_strength=0, range_mode="Fixed", minimum=20, maximum=35
    )
    original = Processor().render(raw, None, settings)[0]
    settings.dde_strength = 3
    processor = Processor()
    sharp = processor.render(raw, None, settings)[0]
    assert np.count_nonzero(original != sharp) > 100
    assert processor.temperature_bands is not None
    metadata = {"snapshot_version": 2, "display": asdict(settings)}
    assert snapshot_display_settings(metadata).dde_strength == 3
    frame = Frame(raw, np.zeros_like(raw, dtype=np.uint8), 0)
    save_image(
        tmp_path / "legend.png",
        frame,
        sharp,
        "color_png",
        legend=(
            legend_colors(settings),
            0,
            255,
            "°C min/max",
            processor.temperature_bands,
        ),
    )
