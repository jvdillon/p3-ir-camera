"""Regression contracts for bounded imports and shared rendering resources."""

import io
import zipfile
import zlib

import numpy as np
import pytest

from p3_thermal.acquisition import Frame
from p3_thermal.export import load_snapshot, save_snapshot, snapshot_display_settings
from p3_thermal.processing import DisplaySettings, Processor, orient, sensor_coordinates
from p3_thermal.recording import Recorder, Sequence
from p3_thermal.storage import atomic_output


def test_failed_save_preserves_existing_project(tmp_path):
    path = tmp_path / "snapshot.npz"
    frame = Frame(np.full((3, 5), 19000, np.uint16), np.zeros((3, 5), np.uint8), 1)
    save_snapshot(path, frame, {})
    original = path.read_bytes()
    with pytest.raises(TypeError):
        save_snapshot(path, frame, {"invalid": object()})
    assert path.read_bytes() == original
    with pytest.raises(OSError), atomic_output(path) as stream:
        stream.write(b"partial write")
        raise OSError("simulated disk failure")
    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]


def test_npz_checks_advertised_shape_before_allocating(tmp_path, monkeypatch):
    path = tmp_path / "oversized.npz"
    header = io.BytesIO()
    np.lib.format.write_array_header_1_0(
        header, {"descr": "<u2", "fortran_order": False, "shape": (10**9, 10**9)}
    )
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("raw.npy", header.getvalue())
        archive.writestr("brightness.npy", b"")
    monkeypatch.setattr(np, "load", lambda *a, **kw: pytest.fail("Unsafe allocation"))
    with pytest.raises(ValueError, match="array size"):
        load_snapshot(path)


@pytest.mark.parametrize("value", ["false", 1, None])
def test_snapshot_filter_switch_requires_boolean(value):
    with pytest.raises(ValueError, match="booleans"):
        snapshot_display_settings({"display": {"enhancements_enabled": value}})


@pytest.mark.parametrize("rotation", range(4))
@pytest.mark.parametrize("mirror", [False, True])
def test_cached_coordinates_preserve_native_samples(rotation, mirror):
    native = np.arange(15).reshape(3, 5)
    coords = sensor_coordinates(native.shape, rotation, mirror)
    np.testing.assert_array_equal(
        native[coords[..., 0], coords[..., 1]], orient(native, rotation, mirror)
    )
    assert coords is sensor_coordinates(native.shape, rotation, mirror)
    assert not coords.flags.writeable


def test_factory_brightness_legend_uses_observed_temperatures():
    raw = np.array([[18000, 20000], [19000, 21000]], np.uint16)
    brightness = np.array([[255, 0], [0, 255]], np.uint8)
    processor = Processor()
    processor.render(raw, brightness, DisplaySettings(mode="Factory brightness"))
    assert processor.temperature_bands[0] == (18000 / 64 - 273.15, 21000 / 64 - 273.15)
    assert processor.temperature_bands[-1] == (19000 / 64 - 273.15, 20000 / 64 - 273.15)


def test_sequence_stream_matches_random_access_and_reuses_timeline(tmp_path):
    recorder = Recorder(tmp_path / "sequence.p3v", 25, {})
    recorder.start()
    for index in range(3):
        recorder.submit(
            Frame(
                np.full((3, 5), 18000 + index, np.uint16),
                np.zeros((3, 5), np.uint8),
                1 + index / 25,
            )
        )
    recorder.stop()
    recorder.join(3)
    assert not recorder.is_alive() and not recorder.error
    sequence = Sequence(recorder.path)
    assert sequence.times is sequence.times
    for index, frame in enumerate(sequence.frames()):
        expected = sequence.frame(index)
        np.testing.assert_array_equal(frame.raw, expected.raw)
        assert frame.timestamp == expected.timestamp
    with pytest.raises(ValueError, match="Corrupt"):
        Sequence._decode(zlib.compress(b"sample") + b"trailing garbage", 6)
