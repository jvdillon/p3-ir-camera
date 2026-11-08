"""Single USB owner; bounded latest-frame handoff keeps the GUI responsive."""

from dataclasses import dataclass

import contextlib
import logging
import queue
import threading
import time

import numpy as np
import usb.core

from p3_camera import FrameMarkerMismatchError, GainMode, P3Camera, get_model_config


@dataclass(frozen=True)
class Frame:
    """Owned, unmodified sensor planes and host monotonic acquisition time."""

    raw: np.ndarray
    brightness: np.ndarray
    timestamp: float


class Acquisition(threading.Thread):
    """All camera calls, including cleanup and controls, run on this thread."""

    def __init__(self, model="p3", demo=False, camera_factory=P3Camera):
        super().__init__(name="p3-acquisition", daemon=True)
        self.model, self.demo = model, demo
        self.camera_factory = camera_factory
        self.stop_event = threading.Event()
        self.commands = queue.Queue()
        self.frames = queue.Queue(maxsize=1)
        self.events = queue.Queue()
        self.recorder = None

    def publish(self, frame):
        recorder = self.recorder
        if recorder is not None:
            recorder.submit(frame)
        with contextlib.suppress(queue.Empty):
            self.frames.get_nowait()
        self.frames.put_nowait(frame)

    def run(self):
        camera = None
        try:
            config = get_model_config(self.model)
            if not self.demo:
                self.events.put("Connecting…")
                camera = self.camera_factory(config=config)
                camera.cancel_event = self.stop_event
                camera.connect()
                name, firmware = camera.init()
                camera.start_streaming()
                self.events.put(f"Live · {name} · firmware {firmware}")
            else:
                self.events.put("DEMO · synthetic data · no camera measurements")
            while not self.stop_event.is_set():
                while not self.commands.empty():
                    command, value = self.commands.get_nowait()
                    if camera is not None:
                        try:
                            if command == "shutter":
                                camera.trigger_shutter()
                            elif command == "gain":
                                camera.set_gain_mode(GainMode[value])
                            self.events.put(f"Applied: {command} {value or ''}")
                        except usb.core.USBTimeoutError:
                            self.events.put(
                                f"Command timed out: {command}; awaiting frames"
                            )
                if self.demo:
                    h, w = (120, 160) if self.model == "p1" else (192, 256)
                    y, x = np.mgrid[:h, :w]
                    t = time.monotonic()
                    temp = (
                        20
                        + x / w * 8
                        + 35
                        * np.exp(
                            -((x - w / 2 - 25 * np.sin(t)) ** 2 + (y - h / 2) ** 2)
                            / 180
                        )
                    )
                    raw = np.rint((temp + 273.15) * 64).astype(np.uint16)
                    brightness = np.clip((temp - 20) * 5, 0, 255).astype(np.uint8)
                    self.stop_event.wait(0.04)
                else:
                    assert camera is not None
                    try:
                        brightness, raw = camera.read_frame_both()
                    except FrameMarkerMismatchError:
                        continue
                    if raw is None or brightness is None:
                        self.stop_event.wait(0.01)
                        continue
                self.publish(Frame(raw.copy(), brightness.copy(), time.monotonic()))
        except Exception as exc:
            if not self.stop_event.is_set():
                self.events.put(f"Disconnected / error: {exc}")
                logging.warning(
                    "Acquisition stopped: %s",
                    exc,
                    exc_info=logging.getLogger().isEnabledFor(logging.DEBUG),
                )
        finally:
            if camera is not None:
                try:
                    camera.disconnect()
                except Exception as exc:
                    self.events.put(f"USB cleanup: {exc}")
