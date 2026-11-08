"""Scientific contracts: geometry, correction, masked materials and recorded RAW."""

import time

import numpy as np
import pytest

from p3_thermal.acquisition import Frame
from p3_thermal.analysis import AnalysisState, EmissivityLayer, Radiometry, Region
from p3_thermal.export import load_snapshot, save_snapshot
from p3_thermal.processing import DisplaySettings, Processor, temperature
from p3_thermal.recording import Recorder, Sequence


def test_roi_stats_and_line_profile_are_native_samples():
    data = np.arange(25, dtype=float).reshape(5, 5)
    rectangle = Region("box", "Rectangle", (1, 1), (3, 3))
    assert rectangle.statistics(data) == {
        "count": 9,
        "min": 6.0,
        "max": 18.0,
        "mean": 12.0,
    }
    circle = Region("circle", "Circle", (2, 2), (3, 2))
    assert circle.statistics(data)["count"] == 5
    line = Region("line", "Line", (0, 0), (4, 4))
    distance, temps = line.profile(data)
    np.testing.assert_array_equal(temps, [0, 6, 12, 18, 24])
    assert distance[-1] == pytest.approx(4 * np.sqrt(2))
    spot = Region("spot", "Spot", (2, 1), (2, 1))
    assert spot.statistics(data)["mean"] == 7


def test_radiometry_identity_and_forward_inverse():
    raw = np.arange(19000, 19024, dtype=np.uint16).reshape(4, 6)
    state = AnalysisState(radiometry=Radiometry(enabled=True))
    np.testing.assert_allclose(state.celsius(raw), temperature(raw), atol=1e-12)
    params = Radiometry(
        enabled=True,
        emissivity=0.6,
        reflected=20,
        atmosphere=25,
        distance=10,
        humidity=50,
        atmospheric_correction=True,
    )
    tau = params.transmission()
    target = 60.0
    incoming = (
        tau
        * (
            params.emissivity * (target + 273.15) ** 4
            + (1 - params.emissivity) * (params.reflected + 273.15) ** 4
        )
        + (1 - tau) * (params.atmosphere + 273.15) ** 4
    )
    raw = np.full((4, 4), round(incoming**0.25 * 64), np.uint16)
    assert AnalysisState(radiometry=params).celsius(raw)[0, 0] == pytest.approx(
        target, abs=0.03
    )
    assert Radiometry(distance=0, atmospheric_correction=True).transmission() == 1
    assert 0 < tau < 1


def test_layers_paint_priority_erase_and_snapshot_roundtrip(tmp_path):
    raw = np.full((12, 12), 20000, np.uint16)
    state = AnalysisState(radiometry=Radiometry(enabled=True, emissivity=0.95))
    state.layers = [
        EmissivityLayer("metal", 0.2, np.zeros(raw.shape, bool)),
        EmissivityLayer("tape", 0.9, np.zeros(raw.shape, bool)),
    ]
    state.paint(0, (2, 2), (8, 2), 1)
    state.paint(1, (4, 2), (4, 2), 1)
    assert state.emissivity_map(raw.shape)[2, 4] == 0.9
    state.paint(1, (4, 2), (4, 2), 1, erase=True)
    assert state.emissivity_map(raw.shape)[2, 4] == 0.2
    state.regions = [Region("spot", "Spot", (2, 2), (2, 2))]
    state.isotherm = True
    path = tmp_path / "corrected.npz"
    save_snapshot(
        path,
        Frame(raw, np.zeros(raw.shape, np.uint8), 0),
        {"analysis": state.to_dict()},
        corrected_celsius=state.celsius(raw),
    )
    frame, metadata = load_snapshot(path)
    restored = AnalysisState.from_dict(metadata["analysis"], raw.shape)
    np.testing.assert_array_equal(frame.raw, raw)
    np.testing.assert_array_equal(
        restored.emissivity_map(raw.shape), state.emissivity_map(raw.shape)
    )
    np.testing.assert_allclose(restored.celsius(raw), state.celsius(raw))
    assert restored.regions[0].name == "spot"
    with np.load(path) as data:
        np.testing.assert_allclose(data["corrected_celsius"], state.celsius(raw))


def test_invalid_correction_stays_invalid_and_renders_without_warning():
    raw = np.full((8, 8), 18000, np.uint16)
    state = AnalysisState(
        radiometry=Radiometry(enabled=True, emissivity=0.01, reflected=1000)
    )
    corrected = state.celsius(raw)
    assert np.isnan(corrected).all()
    rgb, limits = Processor().render(raw, None, DisplaySettings(), measured=corrected)
    assert rgb.shape == (8, 8, 3)
    with pytest.raises(ValueError):
        Radiometry(emissivity=0).validate()


@pytest.mark.parametrize("fps, expected", [(5, 10), (15, 30), (25, 50), (100, 50)])
def test_recording_rates_preserve_timestamps_and_native_samples(
    tmp_path, fps, expected
):
    path = tmp_path / "sequence.p3v"
    recorder = Recorder(
        path, fps, {"model": "p3", "analysis": AnalysisState().to_dict()}
    )
    recorder.start()
    for i in range(50):
        recorder.submit(
            Frame(
                np.full((8, 8), 19000 + i, np.uint16),
                np.full((8, 8), i, np.uint8),
                100 + i / 25,
            )
        )
    recorder.stop()
    recorder.join(5)
    assert not recorder.is_alive() and recorder.error is None
    assert recorder.written == expected and recorder.dropped == 0
    sequence = Sequence(path)
    assert len(sequence.index) == expected
    for index in range(expected):
        frame = sequence.frame(index)
        number = round((frame.timestamp - 100) * 25)
        assert frame.raw[0, 0] == 19000 + number and frame.brightness[0, 0] == number
    assert sequence.times[0] == 0
    assert np.all(np.diff(sequence.times) > 0)
    with pytest.raises(FileExistsError):
        Recorder(path, 25, {})


def test_recording_stopped_ignores_late_frames(tmp_path):
    recorder = Recorder(tmp_path / "stopped.p3v", 25, {})
    recorder.stop()
    recorder.submit(
        Frame(np.zeros((2, 2), np.uint16), np.zeros((2, 2), np.uint8), time.monotonic())
    )
    assert recorder.pending.empty()


def test_standalone_raw_import_preserves_counts_and_marks_missing_brightness(tmp_path):
    from p3_thermal.export import load_raw_image, save_image

    raw = np.arange(256, dtype=np.uint16).reshape(16, 16) + 19000
    frame = Frame(raw, np.zeros(raw.shape, np.uint8), 0)
    png = tmp_path / "sensor.png"
    save_image(png, frame, None, "raw_png")
    loaded, metadata = load_raw_image(png)
    np.testing.assert_array_equal(loaded.raw, raw)
    assert metadata["brightness_origin"] == "derived_from_raw"
    npy = tmp_path / "sensor.npy"
    np.save(npy, raw)
    np.testing.assert_array_equal(load_raw_image(npy)[0].raw, raw)


def test_invalid_layer_metadata_rejected():
    state = AnalysisState()
    state.layers = [EmissivityLayer("metal", 0.5, np.zeros((4, 4), bool))]
    metadata = state.to_dict()
    metadata["layers"][0]["mask_hex"] = "00"
    with pytest.raises(ValueError, match="mask length"):
        AnalysisState.from_dict(metadata, (4, 4))
