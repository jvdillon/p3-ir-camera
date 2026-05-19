
from __future__ import annotations

import math
import threading
import time
from typing import Any, Optional, Tuple
from collections import deque

import numpy as np

try:
    import serial
except Exception as exc:  # pragma: no cover - runtime missing dependency
    serial = None  # type: ignore

from thermal_camera_viewer.p3_camera import P3Camera

class LockInController:
    """Controller to perform lock-in demodulation in a background thread.

    Usage:
      ctrl = LockInController(camera, port="/dev/ttyACM0", period=1.0, integration=60.0)
      t = ctrl.start_background()
      ... ctrl.stop() ...

    Methods are thread-safe for start/stop/get_latest.
    """

    def __init__(
        self,
        camera: P3Camera,
        port: str = "/dev/ttyACM0",
        baud_rate: int = 115200,
        period: float = 1.0,
        integration: float = 60.0,
        invert: bool = False,
    ) -> None:
        if serial is None:
            raise RuntimeError("pyserial is required for lock-in. Install with: pip install pyserial")

        self.camera = camera
        self.port = port
        self.baud_rate = int(baud_rate)
        self.period = float(period)
        self.integration = float(integration)
        self.invert = bool(invert)

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        # protect accumulator / latest-result access
        self._data_lock = threading.Lock()

        # queue for frames pushed from main thread (timestamp, thermal ndarray)
        self._frame_cond = threading.Condition()
        self._frame_queue: deque[tuple[float, np.ndarray]] = deque()

        self._last_in_phase: Optional[np.ndarray] = None
        self._last_quadrature: Optional[np.ndarray] = None
        self._last_amplitude: Optional[np.ndarray] = None
        self._last_angle: Optional[np.ndarray] = None

    def start_background(self) -> threading.Thread:
        """Start a background thread to run a single integration and return the Thread."""
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return self._thread
            # Validate serial port early so UI can report failures immediately.
            SerialClass = getattr(serial, "Serial", None)
            if SerialClass is None:
                # Helpful error: could be missing pyserial or naming conflict
                raise RuntimeError(
                    "pyserial not available: module 'serial' has no attribute 'Serial'. "
                    "Install pyserial (pip install pyserial) or ensure no local module named 'serial' is shadowing the package."
                )
            try:
                ser_test = SerialClass(self.port, self.baud_rate, timeout=1)
                try:
                    ser_test.close()
                except Exception:
                    pass
            except Exception as exc:  # fail early
                raise RuntimeError(f"Failed to open serial port {self.port}: {exc}")

            # Initialize placeholder results so viewer can show panes immediately
            h = self.camera.config.sensor_h
            w = self.camera.config.sensor_w
            with self._data_lock:
                self._last_in_phase = np.zeros((h, w), dtype=np.float32)
                self._last_quadrature = np.zeros((h, w), dtype=np.float32)
                self._last_amplitude = np.zeros((h, w), dtype=np.float32)
                self._last_angle = np.zeros((h, w), dtype=np.float32)

            self._stop_event.clear()
            self._thread = threading.Thread(target=self._background_worker, daemon=True)
            self._thread.start()
            return self._thread

    def push_frame(self, thermal: np.ndarray, timestamp: Optional[float] = None) -> None:
        """Push a thermal frame (uint16 or float) into the controller.

        The background worker will consume frames from this queue and use the
        provided timestamp for phase/weighting. This lets the main UI thread
        be the sole reader of the camera while the lock-in worker only consumes
        frames.
        """
        if timestamp is None:
            timestamp = time.time()
        # make a copy to avoid callers mutating the array
        arr = np.asarray(thermal, dtype=np.float32).copy()
        with self._frame_cond:
            self._frame_queue.append((float(timestamp), arr))
            # keep queue bounded to avoid unbounded memory if UI stalls
            if len(self._frame_queue) > 500:
                self._frame_queue.popleft()
            self._frame_cond.notify()

    def stop(self, wait: bool = True) -> None:
        """Request stop and optionally wait for background thread to finish."""
        self._stop_event.set()
        th = self._thread
        if wait and th is not None:
            th.join()

    def get_latest(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """Return the most recent (in_phase, quadrature, amplitude, angle) tuple, or (None, None, None, None)."""
        with self._data_lock:
            # If final/interim published results exist, return them
            if (self._last_in_phase is not None and self._last_quadrature is not None and
                self._last_amplitude is not None and self._last_angle is not None):
                return (self._last_in_phase.copy(), self._last_quadrature.copy(),
                        self._last_amplitude.copy(), self._last_angle.copy())
            return None, None, None, None

    # --- Internal implementation -------------------------------------------------
    def _background_worker(self) -> None:
        try:
            self.run_once()
        except Exception as exc:
            # Log the error and ensure consumer sees no-results
            print("LockInController error:", exc)
            with self._data_lock:
                self._last_in_phase = None
                self._last_quadrature = None
                self._last_amplitude = None
                self._last_angle = None

    def run_once(self):
        """Blocking run: perform demodulation and return (in_phase, quadrature, amplitude, angle).

        This method will open the serial port, toggle the load for the
        requested integration time, and read frames from `camera`. It will
        skip invalid frames and ensure the load is turned off on exit.
        """
        h = self.camera.config.sensor_h
        w = self.camera.config.sensor_w

        in_phase = np.zeros((h, w), dtype=np.float32)
        quadrature = np.zeros((h, w), dtype=np.float32)
        mag = np.zeros((h, w), dtype=np.float32)
        angle = np.zeros((h, w), dtype=np.float32)

        start_time = time.perf_counter()
        last_toggle = start_time
        # period (seconds) is the timebase input; compute safe value
        period = max(1e-6, float(self.period))
        load_on = True
        total_frames = 0

        tick_rate = 25.0
        tick_interval = 1.0 / tick_rate

        ser = None
        try:
            ser = serial.Serial(self.port, self.baud_rate, timeout=1)
        except Exception as exc:
            raise RuntimeError(f"Failed to open serial port {self.port}: {exc}")

        def _write_state(on: bool) -> None:
            try:
                if ser:
                    val = b"1\n" if on ^ self.invert else b"0\n"
                    ser.write(val)
            except Exception:
                pass

        # Ensure initial state
        _write_state(load_on)

        # Ensure half_period aligns to tick interval: make half_period an integer multiple
        # of the frame tick interval by increasing it if necessary.
        # This avoids fractional alignment issues with toggling.
        half_period = period / 2.0
        n_frames = math.ceil(half_period / tick_interval)
        adjusted_half_period = max(half_period, n_frames * tick_interval)
        adjusted_period = adjusted_half_period * 2.0
        if adjusted_period > period:
            print(f"Adjusted lock-in period from {period:.6f}s to {adjusted_period:.6f}s to align with frame rate {tick_rate}Hz")
            period = adjusted_period
            half_period = period / 2.0

        # Ensure total integration time is an integer multiple of the period
        # by increasing integration if necessary.
        integration = max(1e-6, float(self.integration))
        n_periods = math.ceil(integration / period)
        adjusted_integration = max(integration, n_periods * period)
        if adjusted_integration > integration:
            print(f"Adjusted lock-in integration from {integration:.6f}s to {adjusted_integration:.6f}s to be an integer multiple of period {period:.6f}s")
            integration = adjusted_integration

        try:
            omega = 2.0 * math.pi / period
            while not self._stop_event.is_set() and (time.perf_counter() - start_time) < integration:
                now = time.perf_counter()

                # Handle load toggling
                if now - last_toggle >= half_period:
                    load_on = not load_on
                    phase = omega * (now - start_time)
                    # Debug print, for checking sync between load and integration
                    #print (f'Toggle={load_on}, frame={total_frames}, angle={math.degrees(phase)%360}')

                    _write_state(load_on)
                    last_toggle += half_period
                    # publish interim results in sync with load toggles
                    if load_on and total_frames > 0:
                        mag = np.sqrt(np.square(in_phase) + np.square(quadrature)) / float(total_frames)
                        angle = np.arctan2(quadrature, in_phase)
                        pub_ip = (in_phase / float(total_frames))
                        pub_q = (quadrature / float(total_frames))
                        with self._data_lock:
                            self._last_in_phase = pub_ip
                            self._last_quadrature = pub_q
                            self._last_amplitude = mag
                            self._last_angle = angle

                # Try to consume a pushed frame from the main thread. If none
                # are available within a short timeout, fall back to reading
                # directly from the camera for compatibility.
                thermal = None
                ts = None
                with self._frame_cond:
                    if not self._frame_queue:
                        # wait briefly for main thread to push frames
                        self._frame_cond.wait(timeout=0.1)
                    if self._frame_queue:
                        ts, thermal = self._frame_queue.popleft()

                if thermal is None:
                    time.sleep(0.001)
                    continue

                # Accumulate with sinusoidal weights
                # Use the frame timestamp when available to compute phase
                frame_time = ts if ts is not None else now
                phase = omega * (frame_time - start_time)
                sin_w = math.sin(phase)
                cos_w = math.cos(phase)

                arr = thermal.astype(np.float32)
                in_phase += 2.0 * arr * cos_w
                quadrature += 2.0 * arr * sin_w
                total_frames += 1

                # debug
                if (total_frames % 50 == 0):
                    print(f'Frame: {total_frames}, Time: {now - start_time:.2f}/{integration:.2f}')
            # ensure load off before returning
            _write_state(False)

            # Final normalization and output, *should* be same as last interim publish since we
            # sync the publishes to load-on events and ensure that period and integration align.
            if total_frames > 0:
                mag = np.sqrt(np.square(in_phase) + np.square(quadrature)) / float(total_frames)
                angle = np.arctan2(quadrature, in_phase)
                in_phase /= float(total_frames)
                quadrature /= float(total_frames)

            with self._data_lock:
                self._last_in_phase = in_phase
                self._last_quadrature = quadrature
                self._last_amplitude = mag
                self._last_angle = angle
            
            print(f'Frame: {total_frames}, lock-in integration complete')
            return

        finally:
            try:
                if ser:
                    try:
                        ser.write(b"0\n")
                    except Exception:
                        pass
                    try:
                        ser.close()
                    except Exception:
                        pass
            except Exception:
                pass
