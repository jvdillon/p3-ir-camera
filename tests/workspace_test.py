"""Quantitative contracts for new viewing aids and persisted preferences."""

from dataclasses import replace

import numpy as np

from p3_thermal.export import snapshot_display_settings
from p3_thermal.palettes import TemperaturePalette
from p3_thermal.preferences import Preferences
from p3_thermal.processing import DisplaySettings, Processor, focus_peaking, temperature


def test_user_auto_scale_uses_frame_extrema_without_mutating_stops():
    palette = TemperaturePalette("Test", ((0, "#000000"), (100, "#FFFFFF")))
    raw = np.array([[19000, 19064]], np.uint16)
    settings = DisplaySettings(custom_auto_scale=True)
    rgb, limits = Processor().render(
        raw, np.zeros(raw.shape, np.uint8), settings, palette
    )
    np.testing.assert_array_equal(rgb, [[[0, 0, 0], [255, 255, 255]]])
    np.testing.assert_allclose(limits, temperature(raw)[0])
    assert palette.stops[0][0] == 0
    assert snapshot_display_settings(
        {"display": {"custom_auto_scale": True}}
    ).custom_auto_scale


def test_red_peak_tracks_temperature_even_with_local_filters():
    raw = np.arange(19000, 19256, dtype=np.uint16).reshape(16, 16)
    settings = DisplaySettings(palette="White hot / red peak", clahe=True, detail=True)
    rgb, limits = Processor().render(raw, np.zeros(raw.shape, np.uint8), settings)
    np.testing.assert_array_equal(rgb[-1, -1], [255, 0, 0])
    assert not np.array_equal(rgb[0, 0], [0, 0, 0])
    assert limits is None
    fixed = replace(
        settings,
        range_mode="Fixed",
        minimum=0,
        maximum=100,
        enhancements_enabled=False,
    )
    _, limits = Processor().render(raw, np.zeros(raw.shape, np.uint8), fixed)
    assert limits == (0, 100)


def test_custom_palette_can_opt_into_visual_enhancement():
    raw = np.arange(256, dtype=np.uint16).reshape(16, 16) + 19000
    palette = TemperaturePalette("Custom", ((20, "#000000"), (30, "#FFFFFF")))
    processor = Processor()
    exact, exact_limits = processor.render(
        raw,
        np.zeros(raw.shape, np.uint8),
        DisplaySettings(range_mode="Fixed", minimum=20, maximum=30),
        palette,
    )
    enhanced_settings = DisplaySettings(
        palette="Custom", clahe=True, enhancements_enabled=True
    )
    enhanced, enhanced_limits = processor.render(
        raw, np.zeros(raw.shape, np.uint8), enhanced_settings, palette
    )
    assert exact_limits == (20, 30)
    assert enhanced_limits is None
    assert processor.temperature_bands is not None
    assert not np.array_equal(exact, enhanced)


def test_factory_brightness_accepts_custom_and_red_peak_palettes():
    raw = np.full((8, 8), 19000, np.uint16)
    brightness = np.arange(64, dtype=np.uint8).reshape(8, 8)
    processor = Processor()
    custom = TemperaturePalette("Brightness", ((0, "#000000"), (255, "#FFFFFF")))
    custom_rgb, custom_limits = processor.render(
        raw,
        brightness,
        DisplaySettings(mode="Factory brightness", palette="Brightness"),
        custom,
    )
    red_rgb, red_limits = processor.render(
        raw,
        brightness,
        DisplaySettings(mode="Factory brightness", palette="White hot / red peak"),
    )
    assert custom_rgb.shape == red_rgb.shape == (8, 8, 3)
    assert custom_limits[0] < custom_limits[1]
    assert red_limits[0] < red_limits[1]
    assert not np.array_equal(custom_rgb, red_rgb)


def test_peaking_preserves_flat_fields_and_input_arrays():
    raw = np.full((20, 20), 19000, np.uint16)
    rgb = np.zeros((20, 20, 3), np.uint8)
    np.testing.assert_array_equal(focus_peaking(rgb, raw), rgb)
    raw[:, 10:] += 100
    original = raw.copy()
    output = focus_peaking(rgb, raw)
    assert np.any(output[:, 9:11, 1] == 255)
    assert not rgb.any()
    np.testing.assert_array_equal(raw, original)


def test_mirror_preference_survives_restart_and_can_be_disabled(tmp_path):
    path = tmp_path / "settings.json"
    prefs = Preferences(path)
    prefs.save(True)
    assert Preferences(path).mirror
    prefs.save(True, remember=False)
    restored = Preferences(path)
    assert not restored.mirror and not restored.remember


def test_language_preferences_and_messages_preserve_model_data(tmp_path):
    from types import SimpleNamespace

    from p3_thermal.i18n import Translator

    prefs = Preferences(tmp_path / "settings.json")
    assert prefs.language == "en"
    prefs.save(True, language="pl")
    assert Preferences(prefs.path).language == "pl"
    assert Preferences(prefs.path).mirror
    translator = Translator(SimpleNamespace(), "pl")
    assert translator.text("Raw counts") == "Wartości RAW"
    assert (
        translator.text("Saved image: /tmp/Temperature.npz")
        == "Zapisano obraz: /tmp/Temperature.npz"
    )
    assert translator.text("My Temperature.npz") == "My Temperature.npz"
    assert (
        translator.text("Frames 12 · actual 24.50 fps · dropped 0")
        == "Klatki 12 · rzeczywiste 24.50 kl./s · odrzucone 0"
    )
    assert translator.text("SENSOR TEMPERATURE\nMean  25.500000 °C").startswith(
        "TEMPERATURA SENSORA\nŚr."
    )
    translator.language = "en"
    assert translator.text("Raw counts") == "Raw counts"
