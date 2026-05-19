"""P3 Thermal Camera Driver.

USB protocol, frame parsing, temperature conversion, and calibration for
P3-series USB thermal cameras.

Supports:
- P3: VID=0x3474, PID=0x45A2, 256×192 resolution
- P1: VID=0x3474, PID=0x45C2, 160×120 resolution

Protocol details based on USB traffic analysis by @aeternium.
See: https://github.com/jvdillon/p3-ir-camera/issues/2

Contains portions derived from p3-ir-camera (Apache-2.0):
https://github.com/jvdillon/p3-ir-camera — see NOTICE in the repo root.

SPDX-License-Identifier: Apache-2.0
"""

from __future__ import annotations

from enum import Enum, IntEnum
from typing import TYPE_CHECKING, Any

import array
import contextlib
import dataclasses
import logging
import struct
import sys
import time

import numpy as np
import usb.core
import usb.util


if TYPE_CHECKING:
    from numpy.typing import NDArray

LOGGER = logging.getLogger(__name__)

# Device constants
VID = 0x3474
MARKER_SIZE = 12
# Temperature conversion constants
TEMP_SCALE = 64  # Raw values are in 1/64 Kelvin units
KELVIN_OFFSET = 273.15
FRAME_READ_CHUNK = 16384


class FrameMarkerMismatchError(Exception):
    """Raised when start and end frame markers don't match (cnt1 mismatch)."""

    pass


# Frame marker dtype for parsing start/end markers
MARKER_DTYPE = np.dtype(
    [
        ("length", "<u1"),  # Always 12
        ("sync", "<u1"),  # 0x8c/0x8d for start, 0x8e/0x8f for end
        ("cnt1", "<u4"),  # Frame counter (same in start/end)
        ("cnt2", "<u4"),  # Secondary counter
        ("cnt3", "<u2"),  # Wrapping counter (mod 2048, increments ~40/frame)
    ]
)

# Marker sync byte values
SYNC_START_EVEN = 0x8C  # Start marker, even frame
SYNC_START_ODD = 0x8D  # Start marker, odd frame
SYNC_END_EVEN = 0x8E  # End marker, even frame
SYNC_END_ODD = 0x8F  # End marker, odd frame

# Expected cnt3 increment per frame (~40, wraps at 2048)
CNT3_INCREMENT = 40
CNT3_WRAP = 2048


class Model(str, Enum):
    """Camera model."""

    P1 = "p1"  # 160×120 resolution, PID=0x45C2
    P3 = "p3"  # 256×192 resolution, PID=0x45A2


@dataclasses.dataclass(kw_only=True, frozen=True, slots=True)
class ModelConfig:
    """Model-specific configuration."""

    model: Model
    pid: int
    sensor_w: int  # Sensor width (columns)
    sensor_h: int  # Sensor height (rows) - same for IR and thermal
    shutter_seg_1_lines: int
    shutter_seg_2_lines: int

    @property
    def frame_rows(self) -> int:
        """Total rows in frame: IR + 2 metadata + thermal."""
        return 2 * self.sensor_h + 2

    @property
    def frame_size(self) -> int:
        """Frame data size in bytes (excluding markers)."""
        return 2 * self.frame_rows * self.sensor_w

    @property
    def frame_read_size(self) -> int:
        """Full frame data read size from USB in bytes (including markers)."""
        return self.frame_size + 2 * MARKER_SIZE
    
    @property
    def frame_buffer_size(self) -> int:
        """Frame buffer size ."""
        return self.frame_read_size + self.shutter_seg_1

    @property
    def ir_row_end(self) -> int:
        """End row for IR data (exclusive)."""
        return self.sensor_h

    @property
    def thermal_row_start(self) -> int:
        """Start row for thermal data."""
        return self.sensor_h + 2

    @property
    def thermal_row_end(self) -> int:
        """End row for thermal data (exclusive)."""
        return 2 * self.sensor_h + 2
    
    @property
    def shutter_seg_1(self) -> int:
        """After manual shutter activation: Start of first frame segment."""
        return self.shutter_seg_1_lines * self.sensor_w
    
    @property
    def shutter_seg_2(self) -> int:
        """After manual shutter activation: Start of second frame segment."""
        return self.shutter_seg_2_lines * self.sensor_w + MARKER_SIZE


def get_model_config(model: Model | str = Model.P3) -> ModelConfig:
    """Get configuration for a camera model.

    Args:
        model: Camera model (P1 or P3).

    Returns:
        Model configuration.
    """
    model = Model(model.lower())
    if model == Model.P1:
        return ModelConfig(
            model=Model.P1,
            pid=0x45C2,
            sensor_w=160,
            sensor_h=120,  # 120 IR + 2 metadata + 120 thermal = 242 rows
            shutter_seg_1_lines=36,
            shutter_seg_2_lines=800
        )
    else:  # P3
        return ModelConfig(
            model=Model.P3,
            pid=0x45A2,
            sensor_w=256,
            sensor_h=192,  # 192 IR + 2 metadata + 192 thermal = 386 rows
            shutter_seg_1_lines=36,
            shutter_seg_2_lines=800
        )


# Default model config
_DEFAULT_CONFIG = get_model_config(Model.P3)


class GainMode(IntEnum):
    """Sensor gain mode."""

    LOW = 0  # Extended range: 0°C to 550°C, lower sensitivity
    HIGH = 1  # Limited range: -20°C to 150°C, higher sensitivity
    AUTO = 2  # Auto-switching between HIGH and LOW


@dataclasses.dataclass(kw_only=True, slots=True)
class EnvParams:
    """Environmental parameters for temperature correction."""

    emissivity: float = 0.95  # Surface emissivity (0.0-1.0)
    ambient_temp: float = 25.0  # Ambient temperature (°C)
    reflected_temp: float = 25.0  # Reflected/target temperature (°C)
    distance: float = 1.0  # Distance to target (meters, 0.25-49.99)
    humidity: float = 0.5  # Relative humidity (0.0-1.0)


@dataclasses.dataclass(kw_only=True, slots=True)
class FrameStats:
    """Frame statistics for tracking and validation."""

    frames_read: int = 0  # Total frames successfully read
    frames_dropped: int = 0  # Frames dropped due to cnt3 gaps
    marker_mismatches: int = 0  # Frames with cnt1 mismatch (still returned)
    last_cnt1: int = 0  # Last frame's cnt1 value
    last_cnt3: int = 0  # Last frame's cnt3 value (for drop detection)


# Pre-computed USB commands with CRC
# Note: Camera does not verify CRCs, but we include correct ones anyway.
# CRC values from USB traffic analysis by @aeternium.
COMMANDS: dict[str, bytes] = {
    # Register reads (0x0101 command type)
    # reg 0x01, 30 bytes
    "read_name": bytes.fromhex(
        "0101810001000000000000001e0000004f90",
    ),
    # reg 0x02, 12 bytes
    "read_version": bytes.fromhex(
        "0101810002000000000000000c0000001f63",
    ),
    # reg 0x06, 64 bytes
    "read_part_number": bytes.fromhex(
        "01018100060000000000000040000000654f",
    ),
    # reg 0x07, 64 bytes
    "read_serial": bytes.fromhex(
        "01018100070000000000000040000000104c",
    ),
    # reg 0x0a, 64 bytes
    "read_hw_version": bytes.fromhex(
        "010181000a00000000000000400000001959",
    ),
    # reg 0x0f, 64 bytes
    "read_model_long": bytes.fromhex(
        "010181000f0000000000000040000000b857",
    ),
    # Status (0x1021 command type)
    "status": bytes.fromhex(
        "1021810000000000000000000200000095d1",
    ),
    # Stream control (0x012f command type)
    "start_stream": bytes.fromhex(
        "012f81000000000000000000010000004930",
    ),
    "gain_low": bytes.fromhex(
        "012f41000000000000000000000000003c3a",
    ),
    "gain_high": bytes.fromhex(
        "012f41000100000000000000000000004939",
    ),
    # Shutter (0x0136 command type)
    "shutter": bytes.fromhex(
        "01364300000000000000000000000000cd0b",
    ),
}


# =============================================================================
# Temperature Conversion (Pure Functions)
# =============================================================================


def raw_to_kelvin(raw: float | NDArray[np.uint16]) -> float | NDArray[np.float32]:
    """Convert raw sensor value to Kelvin.

    Raw values are in 1/64 Kelvin units (centikelvin).

    Args:
        raw: Raw 16-bit sensor value(s).

    Returns:
        Temperature in Kelvin.

    """
    return np.float32(raw) / TEMP_SCALE


def kelvin_to_celsius(
    kelvin: float | NDArray[np.float32],
) -> float | NDArray[np.float32]:
    """Convert Kelvin to Celsius.

    Args:
        kelvin: Temperature in Kelvin.

    Returns:
        Temperature in Celsius.

    """
    return kelvin - KELVIN_OFFSET


def celsius_to_kelvin(celsius: float) -> float:
    """Convert Celsius to Kelvin.

    Args:
        celsius: Temperature in Celsius.

    Returns:
        Temperature in Kelvin.

    """
    return celsius + KELVIN_OFFSET


def raw_to_celsius(raw: float | NDArray[np.uint16]) -> float | NDArray[np.float32]:
    """Convert raw sensor value directly to Celsius.

    Formula: (raw / 64) - 273.15

    Args:
        raw: Raw 16-bit sensor value(s).

    Returns:
        Temperature in Celsius.

    """
    return kelvin_to_celsius(raw_to_kelvin(raw))


def celsius_to_raw(celsius: float) -> int:
    """Convert Celsius to raw sensor value.

    Args:
        celsius: Temperature in Celsius.

    Returns:
        Raw 16-bit sensor value.

    """
    return int((celsius + KELVIN_OFFSET) * TEMP_SCALE)


def apply_emissivity_correction(
    apparent_temp_k: float | NDArray[np.float32],
    emissivity: float,
    reflected_temp_c: float = 25.0,
) -> float | NDArray[np.float32]:
    """Apply emissivity correction to apparent temperature.

    Uses the Stefan-Boltzmann radiometric correction formula:
    T_object = ((T_apparent^4 - (1-ε) * T_reflected^4) / ε)^0.25

    This accounts for the fact that real objects are not perfect blackbodies
    and reflect some ambient radiation.

    Args:
        apparent_temp_k: Apparent temperature in Kelvin (what camera measures).
        emissivity: Object emissivity (0.0-1.0, where 1.0 = perfect blackbody).
        reflected_temp_c: Reflected/ambient temperature in Celsius (default 25°C).

    Returns:
        Corrected object temperature in Kelvin.

    Note:
        Common emissivity values:
        - Human skin: 0.98
        - Matte black paint: 0.97
        - Oxidized steel: 0.79
        - Polished aluminum: 0.05
        - Water: 0.96

    Warning:
        This correction becomes less accurate when:
        - Emissivity < 0.6 (highly reflective surfaces)
        - Reflected temperature differs greatly from object temperature
        For emissivity < 0.5, accurate temperature measurement is unlikely.

    See Also:
        https://en.wikipedia.org/wiki/Emissivity
        https://en.wikipedia.org/wiki/Stefan-Boltzmann_law
        https://flir.custhelp.com/app/answers/detail/a_id/3321/~/flir-cameras---temperature-measurement-formula
        https://www.reliableplant.com/Read/14134/emissivity-underst-difference-between-apparent,-actual-ir-temps

    """
    if emissivity >= 1.0:
        return apparent_temp_k  # Perfect blackbody, no correction needed
    if emissivity <= 0.0:
        return apparent_temp_k  # Invalid, return unchanged

    reflected_temp_k = np.float32(reflected_temp_c + KELVIN_OFFSET)

    # Stefan-Boltzmann radiometric correction
    # T_object^4 = (T_apparent^4 - (1-ε) * T_reflected^4) / ε
    t_apparent_4 = np.float32(apparent_temp_k) ** 4
    t_reflected_4 = reflected_temp_k**4
    t_object_4 = (t_apparent_4 - (1.0 - emissivity) * t_reflected_4) / emissivity

    # Handle edge case where correction yields negative value
    if isinstance(t_object_4, np.ndarray):
        t_object_4 = np.maximum(t_object_4, np.float32(0.0))
        return (t_object_4**0.25).astype(np.float32)
    else:
        t_object_4 = max(float(t_object_4), 0.0)
        return float(t_object_4**0.25)


def raw_to_celsius_corrected(
    raw: float | NDArray[np.uint16],
    env: EnvParams,
) -> float | NDArray[np.float32]:
    """Convert raw sensor value to Celsius with emissivity correction.

    Args:
        raw: Raw 16-bit sensor value(s).
        env: Environmental parameters (emissivity, reflected_temp, etc.).

    Returns:
        Corrected temperature in Celsius.

    """
    apparent_k = raw_to_kelvin(raw)
    corrected_k = apply_emissivity_correction(
        apparent_k,
        emissivity=env.emissivity,
        reflected_temp_c=env.reflected_temp,
    )
    return kelvin_to_celsius(corrected_k)


# =============================================================================
# Frame Parsing (Pure Functions)
# =============================================================================


def parse_marker(data: bytes | array.array[int] | memoryview) -> np.ndarray:
    """Parse a 12-byte frame marker.

    Args:
        data: 12-byte marker data.

    Returns:
        Structured numpy array with marker fields.
    """
    return np.frombuffer(data, dtype=MARKER_DTYPE)


def extract_thermal_data(
    frame_data: bytes,
    config: ModelConfig | None = None,
) -> NDArray[np.uint16] | None:
    """Extract temperature image from raw frame data.

    Args:
        frame_data: Raw USB frame data (with start marker).
        config: Model configuration (defaults to P3).

    Returns:
        Temperature image as uint16 array, or None if invalid.
    """
    if config is None:
        config = _DEFAULT_CONFIG
    expected_size = MARKER_SIZE + config.frame_size
    if len(frame_data) < expected_size:
        return None

    pixels = np.frombuffer(
        frame_data[MARKER_SIZE : MARKER_SIZE + config.frame_size],
        dtype="<u2",
    )
    full_frame = pixels.reshape((config.frame_rows, config.sensor_w))

    # Extract thermal region
    thermal = full_frame[config.thermal_row_start : config.thermal_row_end, :].copy()
    return thermal


def extract_ir_brightness(
    frame_data: bytes,
    config: ModelConfig | None = None,
) -> NDArray[np.uint8] | None:
    """Extract IR brightness image from raw frame data.

    Rows 0-(ir_row_end-1) contain display data, taking the low byte of each 16-bit
    value. This data is hardware AGC'd by the camera.

    Args:
        frame_data: Raw USB frame data (with start marker).
        config: Model configuration (defaults to P3).

    Returns:
        IR brightness image as uint8 array, or None if invalid.
    """
    if config is None:
        config = _DEFAULT_CONFIG
    expected_size = MARKER_SIZE + config.frame_size
    if len(frame_data) < expected_size:
        return None

    pixels = np.frombuffer(
        frame_data[MARKER_SIZE : MARKER_SIZE + config.frame_size],
        dtype="<u2",
    )
    full_frame = pixels.reshape((config.frame_rows, config.sensor_w))

    # Extract IR brightness region (low byte contains 8-bit brightness)
    ir_16bit = full_frame[: config.ir_row_end, :].copy()
    ir_8bit = (ir_16bit & 0xFF).astype(np.uint8)
    return ir_8bit


def extract_both(
    frame_data: bytes,
    config: ModelConfig | None = None,
) -> tuple[NDArray[np.uint8] | None, NDArray[np.uint16] | None]:
    """Extract both IR brightness and temperature data from frame.

    Args:
        frame_data: Raw USB frame data (with start marker).
        config: Model configuration (defaults to P3).

    Returns:
        Tuple of (ir_brightness, temperature), either can be None on error.
    """
    if config is None:
        config = _DEFAULT_CONFIG
    expected_size = MARKER_SIZE + config.frame_size
    if len(frame_data) < expected_size:
        return None, None

    pixels = np.frombuffer(
        frame_data[MARKER_SIZE : MARKER_SIZE + config.frame_size],
        dtype="<u2",
    )
    full_frame = pixels.reshape((config.frame_rows, config.sensor_w))

    # IR brightness (low byte)
    ir_16bit = full_frame[: config.ir_row_end, :].copy()
    ir_8bit = (ir_16bit & 0xFF).astype(np.uint8)

    # Temperature
    thermal = full_frame[config.thermal_row_start : config.thermal_row_end, :].copy()

    return ir_8bit, thermal


# =============================================================================
# USB Protocol (Pure Functions)
# =============================================================================


def crc16_ccitt(data: bytes, poly: int = 0x1021, init: int = 0x0000) -> int:
    """Compute CRC16-CCITT checksum.

    Args:
        data: Input bytes.
        poly: Polynomial (default: 0x1021 for CCITT).
        init: Initial value.

    Returns:
        16-bit CRC value.

    """
    crc = init
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = (crc << 1) ^ poly
            else:
                crc <<= 1
            crc &= 0xFFFF
    return crc


def build_command(
    cmd_type: int,
    param: int,
    register: int,
    resp_len: int,
) -> bytes:
    """Build an 18-byte USB command with CRC.

    Command format:
    - Bytes 0-1: Command type (LE)
    - Bytes 2-3: Parameter (LE)
    - Bytes 4-5: Register ID (LE)
    - Bytes 6-13: Reserved (zeros)
    - Bytes 14-15: Response length (LE)
    - Bytes 16-17: CRC16 (LE)

    Args:
        cmd_type: Command type (e.g., 0x0101 for read register).
        param: Parameter value (usually 0x0081).
        register: Register ID.
        resp_len: Expected response length.

    Returns:
        18-byte command with CRC.

    """
    payload = struct.pack(
        "<HHHQH",
        cmd_type,
        param,
        register,
        0,  # 8 bytes reserved
        resp_len,
    )
    crc = crc16_ccitt(payload)
    return payload + struct.pack("<H", crc)


# =============================================================================
# Camera Class (Stateful)
# =============================================================================


@dataclasses.dataclass(kw_only=True, slots=True)
class P3Camera:
    """P3 Thermal Camera interface.

    Handles USB communication, streaming, and device state.
    """

    dev: Any = None  # usb.core.Device
    streaming: bool = False
    gain_mode: GainMode = GainMode.HIGH
    env_params: EnvParams = dataclasses.field(default_factory=EnvParams)
    config: ModelConfig = dataclasses.field(default_factory=lambda: _DEFAULT_CONFIG)
    stats: FrameStats = dataclasses.field(default_factory=FrameStats)
    validate_markers: bool = True  # Enable marker validation (cnt1 matching)
    _frame_buf: array.array[int] | None = dataclasses.field(
        default=None,
        repr=False,
    )  # array.array for frame reads
    _chunk_buf: array.array[int] | None = dataclasses.field(
        default=None,
        repr=False,
    )  # array.array for chunk reads - array slicing creates a copy :(

    def connect(self) -> None:
        """Connect to the camera."""

        self.dev = usb.core.find(idVendor=VID, idProduct=self.config.pid)
        if self.dev is None:
            model_name = self.config.model.value.upper()
            raise RuntimeError(
                f"{model_name} camera not found (PID=0x{self.config.pid:04X})"
            )
        self._detach_kernel_drivers()
        self._claim_interfaces()

        # Pre-allocate frame buffer for efficient reads
        self._frame_buf = array.array("B", b"\x00" * self.config.frame_buffer_size)
        self._chunk_buf = array.array("B", b"\x00" * FRAME_READ_CHUNK)

    def disconnect(self) -> None:
        """Disconnect from the camera."""
        if self.streaming:
            self.stop_streaming()
        self.dev = None

    def init(self) -> tuple[str, str]:
        """Initialize camera and read device info.

        Returns:
            Tuple of (device_name, firmware_version).
        """
        if self.dev is None:
            raise RuntimeError("Not connected")

        self._send_command(COMMANDS["read_name"])
        self._read_status()  # ACK
        name = bytes(self._read_response(30))
        self._read_status()  # ACK
        name_str = name.rstrip(b"\x00").decode(errors="replace")

        self._send_command(COMMANDS["read_version"])
        self._read_status()  # ACK
        version = bytes(self._read_response(12))
        self._read_status()  # ACK
        version_str = version.rstrip(b"\x00").decode(errors="replace")

        return name_str, version_str

    def read_register(self, cmd_name: str, length: int) -> str:
        """Read a register and return decoded string.

        Args:
            cmd_name: Command name from COMMANDS dict.
            length: Expected response length.

        Returns:
            Register value as string (null-terminated).
        """
        if self.dev is None:
            raise RuntimeError("Not connected")
        self._send_command(COMMANDS[cmd_name])
        self._read_status()  # ACK after write
        data = bytes(self._read_response(length))
        self._read_status()  # ACK after read
        return data.rstrip(b"\x00").decode(errors="replace")

    def read_device_info(self) -> dict[str, str]:
        """Read all device information registers.

        Returns:
            Dictionary with device info:
            - model: Model name (e.g., "P3")
            - fw_version: Firmware version (e.g., "00.00.02.17")
            - part_number: Part number (e.g., "P30-1Axxxxxxxx")
            - serial: Serial number
            - hw_version: Hardware revision (e.g., "P3-00.04")
            - model_long: Model name (64-byte version)
        """
        if self.dev is None:
            raise RuntimeError("Not connected")
        return {
            "model": self.read_register("read_name", 30),
            "fw_version": self.read_register("read_version", 12),
            "part_number": self.read_register("read_part_number", 64),
            "serial": self.read_register("read_serial", 64),
            "hw_version": self.read_register("read_hw_version", 64),
            "model_long": self.read_register("read_model_long", 64),
        }

    def read_status_command(self) -> str:
        """Read the status command register (0x00).

        This returns a 2-byte status that typically contains "11" (0x31 0x31).

        Returns:
            Status value as string.
        """
        if self.dev is None:
            raise RuntimeError("Not connected")
        self._send_command(COMMANDS["status"])
        self._read_status()  # ACK after write
        data = bytes(self._read_response(2))
        self._read_status()  # ACK after read
        return data.decode(errors="replace")

    def start_streaming(self) -> None:
        """Start video streaming.

        Follows the sequence observed from the Windows Temp Master tool:
        1. Send start_stream command and check status
        2. Wait 1 second
        3. Set interface alternate setting
        4. Send 0xEE control transfer
        5. Wait 2 seconds for camera to be ready
        6. Issue async bulk read (Windows tool does this)
        7. Send start_stream command again
        """
        if self.dev is None:
            raise RuntimeError("Not connected")

        # Reset frame statistics
        self.stats = FrameStats()

        # Initial start_stream with status checks
        self._send_command(COMMANDS["start_stream"])
        self._read_status()  # reads 0x02
        resp = self._read_response(1)  # reads 0x01 (or 0x35 if restarting)
        self._read_status()  # reads 0x03

        # Check for restart response (0x35 = '5' when stream was already active)
        if resp and resp[0] == 0x35:
            # Camera was already streaming, this is a restart
            pass  # Continue with sequence

        # Wait before configuring interface (per Windows tool timing)
        time.sleep(1.0)

        # Configure streaming interface
        self.dev.set_interface_altsetting(interface=1, alternate_setting=1)
        self.dev.ctrl_transfer(0x40, 0xEE, 0, 1, None, 1000)

        # Wait for camera to be ready (Windows tool waits ~2 seconds)
        time.sleep(2.0)

        # Issue async bulk read before final start_stream (per Windows tool)
        # This read happens asynchronously in the Windows tool
        with contextlib.suppress(Exception):
            self.dev.read(0x81, self.config.frame_size, 100)

        # Final start_stream with status checks
        self._send_command(COMMANDS["start_stream"])
        self._read_status()
        resp = self._read_response(1)
        self._read_status()

        # Handle restart response on final start_stream too
        if resp and resp[0] == 0x35:
            pass  # Camera acknowledges restart

        self.streaming = True

    def stop_streaming(self) -> None:
        """Stop video streaming."""
        if self.streaming and self.dev is not None:
            self.dev.set_interface_altsetting(interface=1, alternate_setting=0)
            self.streaming = False

    def read_frame(self) -> bytes:
        """Read a complete frame from the camera.

        As per https://github.com/jvdillon/p3-ir-camera/issues/2 and P3_PROTOCOL.md,
        official app transmits frames as 3 USB bulk transfers:
        1. Main frame data (frame_size bytes): start marker (12) + pixel data (frame_size - 12)
        2. Remaining pixel data (12 bytes)
        3. End marker (12 bytes)

        However, this only works reliably on Linux:
        - On Windows, we get "device is not working" error that is
          not recoverable until the USB device is closed.
        - On Mac, we get "overflow" exceptions often

        So instead, we do 16KiB bulk reads until we read frame_size + 12 + 12 bytes.
        While this does not match the official app, it works reliably on all platforms.

        The first transfer includes the 12-byte start marker at the beginning.

        Frame markers are validated:
        - cnt1 must match between start and end markers (same frame)
        - cnt3 is tracked to detect dropped frames (~40 increment per frame)

        Returns:
            Complete frame: start marker (12) + pixel data (frame_size).

        Raises:
            FrameMarkerMismatchError: If cnt1 doesn't match (when validate_markers=True).
        """
        if self.dev is None or not self.streaming:
            return b""

        assert self._frame_buf is not None
        assert self._chunk_buf is not None

        frame_read_size = self.config.frame_read_size
        chunk_buf = self._chunk_buf

        # Use memoryviews to prevent copies
        frame_buf_view = memoryview(self._frame_buf)
        chunk_buf_view = memoryview(chunk_buf)

        # Read frame data into pre-allocated buffer
        # This includes: start marker (12) + pixel data (frame_size) + end marker (12)
        pos = 0
        while pos < frame_read_size:
            n = self.dev.read(0x81, chunk_buf, 10000)

            next_pos = pos + n

            # Frame always ends with 12 byte read with the end marker, so if
            # * got 12 bytes, but not at the end - we found end of frame,
            #   so reset to 0 and read from start OR,
            # * got full frame of data, but did not finish with 12b read,
            #   so reset to 0 and try again to find the end.
            if (n == MARKER_SIZE and next_pos < frame_read_size) or (
                next_pos >= frame_read_size and n != MARKER_SIZE
            ):
                pos = 0
                LOGGER.debug("Frame reading out of sync, dropping frame")
                continue

            frame_buf_view[pos:next_pos] = chunk_buf_view[:n]
            pos = next_pos

        start_marker = parse_marker(frame_buf_view[:MARKER_SIZE])
        end_marker = parse_marker(frame_buf_view[frame_read_size-MARKER_SIZE:frame_read_size])

        start_cnt1 = int(start_marker["cnt1"][0])
        end_cnt1 = int(end_marker["cnt1"][0])
        frame_cnt3 = int(end_marker["cnt3"][0])

        # Validate cnt1 matches between start and end markers
        if start_cnt1 != end_cnt1:
            self.stats.marker_mismatches += 1
            if self.validate_markers:
                raise FrameMarkerMismatchError(
                    f"cnt1 mismatch: start={start_cnt1}, end={end_cnt1}"
                )

        # Track dropped frames via cnt3 gaps
        # cnt3 increments by ~40 per frame and wraps at 2048
        if self.stats.frames_read > 0:
            expected_cnt3 = (self.stats.last_cnt3 + CNT3_INCREMENT) % CNT3_WRAP
            # Allow some tolerance (~10) for timing variations
            cnt3_diff = (frame_cnt3 - expected_cnt3) % CNT3_WRAP
            if (
                cnt3_diff > CNT3_INCREMENT // 2
                and cnt3_diff < CNT3_WRAP - CNT3_INCREMENT
            ):
                # Gap detected - estimate dropped frames
                dropped = cnt3_diff // CNT3_INCREMENT
                self.stats.frames_dropped += dropped

        # Update statistics
        self.stats.frames_read += 1
        self.stats.last_cnt1 = start_cnt1
        self.stats.last_cnt3 = frame_cnt3

        # Return complete frame: start marker + all pixel data
        return bytes(frame_buf_view[:frame_read_size-MARKER_SIZE])

    def read_frame_both(
        self,
    ) -> tuple[NDArray[np.uint8] | None, NDArray[np.uint16] | None]:
        """Read both IR brightness and temperature data.

        Returns:
            Tuple of (ir_brightness uint8, temperature uint16).
            Either can be None on error.
        """
        if self.dev is None or not self.streaming:
            return None, None

        raw_data = self.read_frame()
        return extract_both(raw_data, config=self.config)

    def trigger_shutter(self, return_partial: bool = False) -> None:
        """Trigger shutter/NUC calibration und read first frame.

        The shutter command uses param 0x43 and has no response data.
        The camera automatically triggers shutter approximately every 90 seconds.

        The camera will provide a (potentially mistimed) frame after shutter,
        including a partial duplicate.

        Args:
            return_partial: whether to return the incomplete frame as well.

        Returns:
            Complete frame: start marker (12) + pixel data (frame_size)
            (Optional) Additional incomplete frame.
        """
        self._send_command(COMMANDS["shutter"])
        self._read_status()  # ACK after write

        assert self._frame_buf is not None
        assert self._chunk_buf is not None

        shutter_seg_1_start = self.config.shutter_seg_1
        shutter_seg_1_end = self.config.shutter_seg_2 - MARKER_SIZE
        shutter_seg_2_start = self.config.shutter_seg_2
        # larger read size to accomodate for partial frame data
        frame_read_size = self.config.frame_read_size + shutter_seg_1_start
        chunk_buf = self._chunk_buf

        frame_buf_view = memoryview(self._frame_buf)

        # Read frame data into pre-allocated buffer.
        # Shutter response has a different layout than normal frames, so
        # we use direct array assignment instead of memoryview slicing to
        # avoid potential structure mismatches across PyUSB/Python versions.
        pos = 0
        while pos < frame_read_size:
            raw = self.dev.read(0x81, chunk_buf, 10000)
            n = len(raw)
            next_pos = pos + n
            self._frame_buf[pos:next_pos] = raw if isinstance(raw, array.array) else array.array("B", bytes(raw))
            pos = next_pos

        start_marker = parse_marker(frame_buf_view[:MARKER_SIZE])
        end_marker = parse_marker(
            frame_buf_view[frame_read_size-MARKER_SIZE:frame_read_size]
            )

        start_cnt1 = int(start_marker["cnt1"][0])
        end_cnt1 = int(end_marker["cnt1"][0])
        frame_cnt3 = int(end_marker["cnt3"][0])

        # Validate cnt1 matches between start and end markers
        if start_cnt1 != end_cnt1:
            self.stats.marker_mismatches += 1
            if self.validate_markers:
                raise FrameMarkerMismatchError(
                    f"cnt1 mismatch: start={start_cnt1}, end={end_cnt1}"
                )

        # assemble segmented frame data
        frame_data = bytes(frame_buf_view[:MARKER_SIZE]) \
            + bytes(frame_buf_view[shutter_seg_1_start:shutter_seg_1_end]) \
            + bytes(frame_buf_view[shutter_seg_2_start:frame_read_size - MARKER_SIZE])

        if return_partial:
            return frame_data, bytes(frame_buf_view[MARKER_SIZE:shutter_seg_1_start])
        else:
            return frame_data

    def set_gain_mode(self, mode: GainMode) -> None:
        """Set sensor gain mode.

        Gain commands use param 0x41 instead of 0x81 and have no response data.
        We still follow the status ACK pattern for consistency.

        Args:
            mode: Gain mode (LOW, HIGH, or AUTO).
        """
        if mode == GainMode.LOW:
            self._send_command(COMMANDS["gain_low"])
            self._read_status()  # ACK after write
        elif mode == GainMode.HIGH:
            self._send_command(COMMANDS["gain_high"])
            self._read_status()  # ACK after write
        # AUTO mode requires firmware support (not implemented in protocol)
        self.gain_mode = mode

    # Private methods

    def _send_command(self, cmd: bytes) -> None:
        """Send a control command to the device."""
        if self.dev is None:
            raise RuntimeError("Not connected")
        self.dev.ctrl_transfer(0x41, 0x20, 0, 0, cmd, 1000)

    def _read_response(self, length: int) -> bytes:
        """Read response data from the device."""
        if self.dev is None:
            raise RuntimeError("Not connected")
        return bytes(self.dev.ctrl_transfer(0xC1, 0x21, 0, 0, length, 1000))

    def _read_status(self) -> int:
        """Read status byte from the device.

        Status values:
        - 0x02: After write command
        - 0x03: After read command
        """
        if self.dev is None:
            raise RuntimeError("Not connected")
        return self.dev.ctrl_transfer(0xC1, 0x22, 0, 0, 1, 1000)[0]

    def read_debug_log(self, length: int = 128) -> str:
        """Read debug/log messages from the extended status register.

        The camera exposes debug messages when reading more than 1 byte
        from the status register. Messages appear starting around byte 64.

        Example messages:
        - "[162] I/default.conf: default configuration already load"
        - "[34064] I/std cmd: in : 1 1 81"
        - "[91403] I/shutter: === Shutter close ==="

        Args:
            length: Number of bytes to read (default 128).

        Returns:
            Debug message string (may be empty or contain partial messages).
        """
        if self.dev is None:
            raise RuntimeError("Not connected")
        data = bytes(self.dev.ctrl_transfer(0xC1, 0x22, 0, 0, length, 1000))
        # Debug messages typically start around byte 64
        msg_data = data[64:] if len(data) > 64 else data
        # Extract printable ASCII, filtering control characters
        result = []
        for b in msg_data:
            if 32 <= b <= 126 or b in (10, 13):  # Printable or newline
                result.append(chr(b))
        return "".join(result).strip()

    def _detach_kernel_drivers(self) -> None:
        """Detach kernel drivers from USB interfaces (Linux only; no-op on Windows/macOS)."""
        if self.dev is None:
            return
        if sys.platform == "win32":
            return
        for cfg in self.dev:
            for intf in cfg:
                try:
                    if self.dev.is_kernel_driver_active(intf.bInterfaceNumber):
                        self.dev.detach_kernel_driver(intf.bInterfaceNumber)
                except Exception:
                    pass

    def _claim_interfaces(self) -> None:
        """Claim USB interfaces."""
        if self.dev is None:
            return
        self.dev.set_configuration()
        usb.util.claim_interface(self.dev, 0)
        usb.util.claim_interface(self.dev, 1)
