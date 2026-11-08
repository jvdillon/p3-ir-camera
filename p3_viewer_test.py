"""Regression tests for the replacement viewer (no camera or desktop required)."""

from decimal import Decimal
from unittest.mock import Mock

import json
import time

import numpy as np
import pytest

from p3_camera import P3Camera, raw_to_celsius
from p3_thermal.acquisition import Acquisition, Frame
from p3_thermal.canvas import ThermalCanvas
from p3_thermal.export import save_snapshot
from p3_thermal.processing import (
    DisplaySettings,
    Processor,
    orient,
    pixel_label,
    temperature,
)


def test_every_native_value_has_exact_decimal_readout():
    for raw in range(65536):
        exact = Decimal(raw) / 64 - Decimal("273.15")
        assert pixel_label(raw) == f"{exact:.6f} °C"
    values = np.arange(65536, dtype=np.uint16)
    assert np.max(np.abs(raw_to_celsius(values) - temperature(values))) == 0


@pytest.mark.parametrize("rotation", range(4))
@pytest.mark.parametrize("mirror", [False, True])
def test_hover_uses_native_coordinates_after_all_transforms(rotation, mirror):
    raw = np.arange(12, dtype=np.uint16).reshape(3, 4)
    coords = np.moveaxis(np.indices(raw.shape), 0, -1)
    canvas = Mock()
    canvas.raw = orient(raw, rotation, mirror)
    canvas.coordinates = orient(coords, rotation, mirror)
    canvas.pixel_scale = 128
    canvas.offset = [-40.5, 17.2]
    canvas.legend_rect = None
    for y, x in np.ndindex(canvas.raw.shape):
        canvas.cell_at = lambda px, py: ThermalCanvas.cell_at(canvas, px, py)
        ThermalCanvas.pick(
            canvas,
            canvas.offset[0] + (x + 0.5) * 128,
            canvas.offset[1] + (y + 0.5) * 128,
        )
        sx, sy, sample = canvas.on_pixel.call_args.args[0]
        assert sample == raw[sy, sx]
    ThermalCanvas.pick(canvas, canvas.offset[0] - 0.1, canvas.offset[1])
    canvas.on_pixel.assert_called_with(None)


def test_filter_and_palettes_preserve_raw_samples():
    raw = np.arange(64, dtype=np.uint16).reshape(8, 8) + 19000
    original = raw.copy()
    processor = Processor()
    settings = DisplaySettings(mode="Filtered temperature")
    processor.render(raw, np.zeros((8, 8), np.uint8), settings)
    newer = raw + 64
    processor.render(newer, np.zeros((8, 8), np.uint8), settings)
    assert np.allclose(processor.previous, temperature(raw) + 0.35)
    np.testing.assert_array_equal(raw, original)
    processor.reset()
    assert processor.previous is None and processor.bounds is None


@pytest.mark.parametrize(
    "mode", ["Temperature", "Filtered temperature", "Raw counts", "Factory brightness"]
)
def test_all_sources_render_uniform_frames(mode):
    raw = np.full((8, 8), 19000, np.uint16)
    rgb, limits = Processor().render(
        raw, np.full((8, 8), 100, np.uint8), DisplaySettings(mode=mode)
    )
    assert rgb.shape == (8, 8, 3) and rgb.dtype == np.uint8
    if mode == "Factory brightness":
        assert limits == (100.0, 100.0)
    else:
        assert limits is not None


def test_fixed_range_validation():
    with pytest.raises(ValueError):
        Processor().render(
            np.ones((8, 8), np.uint16),
            None,
            DisplaySettings(range_mode="Fixed", minimum=40, maximum=15),
        )


def test_lossless_snapshot(tmp_path):
    frame = Frame(
        np.arange(12, dtype=np.uint16).reshape(3, 4), np.zeros((3, 4), np.uint8), 12.5
    )
    path = tmp_path / "capture.npz"
    save_snapshot(path, frame, {"model": "p3", "demo": False})
    with np.load(path, allow_pickle=False) as data:
        np.testing.assert_array_equal(data["raw"], frame.raw)
        assert json.loads(str(data["metadata"]))["raw_unit"] == "1/64 K"


def test_disconnect_releases_resources_even_if_stop_fails(monkeypatch):
    camera = P3Camera(dev=Mock(), streaming=True)
    device = camera.dev
    device.set_interface_altsetting.side_effect = RuntimeError("unplugged")
    dispose = Mock()
    monkeypatch.setattr("p3_camera.usb.util.dispose_resources", dispose)
    with pytest.raises(RuntimeError):
        camera.disconnect()
    dispose.assert_called_once_with(device)
    assert camera.dev is None and not camera.streaming


def test_worker_cleanup_on_initialization_failure():
    camera = Mock()
    camera.init.side_effect = RuntimeError("initialization failed")
    worker = Acquisition(camera_factory=lambda **kwargs: camera)
    worker.start()
    worker.join(2)
    assert not worker.is_alive()
    camera.disconnect.assert_called_once()


def test_demo_handoff_is_bounded_and_stops():
    worker = Acquisition(demo=True)
    worker.start()
    frame = worker.frames.get(timeout=2)
    assert frame.raw.shape == (192, 256)
    time.sleep(0.15)
    assert worker.frames.qsize() == 1
    worker.stop_event.set()
    worker.join(2)
    assert not worker.is_alive()


def test_frame_read_can_be_cancelled_before_usb_read():
    import array
    import threading

    camera = P3Camera(dev=Mock(), streaming=True)
    camera._frame_buf = array.array("B", b"\0" * camera.config.frame_buffer_size)
    camera._chunk_buf = array.array("B", b"\0" * 16384)
    camera.cancel_event = threading.Event()
    camera.cancel_event.set()
    with pytest.raises(InterruptedError):
        camera.read_frame()
    camera.dev.read.assert_not_called()


def test_usb_poll_timeout_retries_until_cancelled():
    import array
    import threading

    import usb.core

    camera = P3Camera(dev=Mock(), streaming=True)
    camera._frame_buf = array.array("B", b"\0" * camera.config.frame_buffer_size)
    camera._chunk_buf = array.array("B", b"\0" * 16384)
    camera.cancel_event = threading.Event()

    def timeout(*args):
        camera.cancel_event.set()
        raise usb.core.USBTimeoutError("poll timeout")

    camera.dev.read.side_effect = timeout
    with pytest.raises(InterruptedError):
        camera.read_frame()
    camera.dev.read.assert_called_once()


def test_control_timeout_does_not_discard_live_session():
    import usb.core

    camera = Mock()
    camera.init.return_value = ("P3", "test")
    camera.trigger_shutter.side_effect = usb.core.USBTimeoutError("ACK timeout")
    worker = Acquisition(camera_factory=lambda **kwargs: camera)

    def read_frame():
        worker.stop_event.set()
        return np.zeros((8, 8), np.uint8), np.full((8, 8), 19000, np.uint16)

    camera.read_frame_both.side_effect = read_frame
    worker.commands.put(("shutter", None))
    worker.start()
    worker.join(2)
    assert not worker.is_alive()
    assert worker.frames.get_nowait().raw[0, 0] == 19000
    camera.disconnect.assert_called_once()
    events = []
    while not worker.events.empty():
        events.append(worker.events.get_nowait())
    assert any("Command timed out" in event for event in events)
