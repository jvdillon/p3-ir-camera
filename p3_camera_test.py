"""Unit tests for p3_camera.py."""

from __future__ import annotations

import sys

import numpy as np
import pytest

from thermal_camera_viewer.p3_camera import (
    CNT3_INCREMENT,
    CNT3_WRAP,
    COMMANDS,
    KELVIN_OFFSET,
    MARKER_DTYPE,
    MARKER_SIZE,
    SYNC_END_EVEN,
    SYNC_END_ODD,
    SYNC_START_EVEN,
    SYNC_START_ODD,
    TEMP_SCALE,
    EnvParams,
    FrameMarkerMismatchError,
    FrameStats,
    GainMode,
    Model,
    apply_emissivity_correction,
    build_command,
    celsius_to_kelvin,
    celsius_to_raw,
    crc16_ccitt,
    extract_both,
    extract_ir_brightness,
    extract_thermal_data,
    get_model_config,
    kelvin_to_celsius,
    parse_marker,
    raw_to_celsius,
    raw_to_celsius_corrected,
    raw_to_kelvin,
)


class TestConstants:
    """Test that constants have expected values."""

    def test_temp_scale(self):
        assert TEMP_SCALE == 64

    def test_kelvin_offset(self):
        assert KELVIN_OFFSET == 273.15

    def test_marker_size(self):
        assert MARKER_SIZE == 12

    def test_marker_dtype_size(self):
        assert MARKER_DTYPE.itemsize == 12

    def test_marker_sync_constants(self):
        """Test marker sync byte constants."""
        assert SYNC_START_EVEN == 0x8C
        assert SYNC_START_ODD == 0x8D
        assert SYNC_END_EVEN == 0x8E
        assert SYNC_END_ODD == 0x8F

    def test_cnt3_constants(self):
        """Test cnt3 frame counter constants."""
        assert CNT3_INCREMENT == 40
        assert CNT3_WRAP == 2048

    def test_p1_model_config(self):
        """Test P1 model configuration."""
        config = get_model_config(Model.P1)
        assert config.model == Model.P1
        assert config.pid == 0x45C2
        assert config.sensor_w == 160
        assert config.sensor_h == 120
        # Derived properties
        assert config.frame_rows == 242  # 120 + 2 + 120
        assert config.frame_size == 2 * 242 * 160  # 77,440 bytes
        assert config.ir_row_end == 120
        assert config.thermal_row_start == 122
        assert config.thermal_row_end == 242

    def test_p3_model_config(self):
        """Test P3 model configuration."""
        config = get_model_config(Model.P3)
        assert config.model == Model.P3
        assert config.pid == 0x45A2
        assert config.sensor_w == 256
        assert config.sensor_h == 192
        # Derived properties
        assert config.frame_rows == 386  # 192 + 2 + 192
        assert config.frame_size == 2 * 386 * 256  # 197,632 bytes
        assert config.ir_row_end == 192
        assert config.thermal_row_start == 194
        assert config.thermal_row_end == 386

    def test_model_config_from_string(self):
        """Test model config can be created from string."""
        config = get_model_config("p3")
        assert config.model == Model.P3
        config = get_model_config("P1")
        assert config.model == Model.P1


class TestTemperatureConversion:
    """Tests for temperature conversion functions."""

    def test_raw_to_kelvin_zero(self):
        # 0 raw = 0 Kelvin (absolute zero)
        assert raw_to_kelvin(0) == pytest.approx(0.0)

    def test_raw_to_kelvin_room_temp(self):
        # Room temp ~25°C = 298.15K
        # raw = 298.15 * 64 = 19081.6
        raw = int(298.15 * TEMP_SCALE)
        assert raw_to_kelvin(raw) == pytest.approx(298.15, rel=1e-3)

    def test_raw_to_kelvin_array(self):
        raw = np.array([0, 17481, 19082], dtype=np.uint16)
        kelvin = raw_to_kelvin(raw)
        assert isinstance(kelvin, np.ndarray)
        assert kelvin.shape == (3,)
        assert kelvin[0] == pytest.approx(0.0)
        assert kelvin[1] == pytest.approx(273.14, rel=1e-3)  # ~0°C
        assert kelvin[2] == pytest.approx(298.16, rel=1e-3)  # ~25°C

    def test_kelvin_to_celsius_freezing(self):
        assert kelvin_to_celsius(273.15) == pytest.approx(0.0)

    def test_kelvin_to_celsius_boiling(self):
        assert kelvin_to_celsius(373.15) == pytest.approx(100.0)

    def test_kelvin_to_celsius_array(self):
        kelvin = np.array([273.15, 298.15, 373.15], dtype=np.float32)
        celsius = kelvin_to_celsius(kelvin)
        assert isinstance(celsius, np.ndarray)
        assert celsius[0] == pytest.approx(0.0)
        assert celsius[1] == pytest.approx(25.0)
        assert celsius[2] == pytest.approx(100.0)

    def test_raw_to_celsius_freezing(self):
        # 0°C = 273.15K, raw = 273.15 * 64 = 17481.6
        raw = int(273.15 * TEMP_SCALE)
        assert raw_to_celsius(raw) == pytest.approx(0.0, abs=0.02)

    def test_raw_to_celsius_body_temp(self):
        # 37°C = 310.15K, raw = 310.15 * 64 = 19849.6
        raw = int(310.15 * TEMP_SCALE)
        assert raw_to_celsius(raw) == pytest.approx(37.0, abs=0.02)

    def test_raw_to_celsius_boiling(self):
        # 100°C = 373.15K, raw = 373.15 * 64 = 23881.6
        raw = int(373.15 * TEMP_SCALE)
        assert raw_to_celsius(raw) == pytest.approx(100.0, abs=0.02)

    def test_celsius_to_raw_roundtrip(self):
        for temp in [-40.0, 0.0, 25.0, 37.0, 100.0, 200.0]:
            raw = celsius_to_raw(temp)
            recovered = raw_to_celsius(raw)
            assert recovered == pytest.approx(temp, abs=0.02)

    def test_celsius_to_kelvin(self):
        assert celsius_to_kelvin(0.0) == pytest.approx(273.15)
        assert celsius_to_kelvin(100.0) == pytest.approx(373.15)
        assert celsius_to_kelvin(-273.15) == pytest.approx(0.0)


class TestEmissivityCorrection:
    """Tests for emissivity correction functions."""

    def test_emissivity_1_no_correction(self):
        """Perfect blackbody (e=1.0) should return unchanged temperature."""
        temp_k = 300.0  # 26.85°C
        result = apply_emissivity_correction(temp_k, emissivity=1.0)
        assert result == pytest.approx(temp_k)

    def test_emissivity_correction_higher_than_reflected(self):
        """When apparent > reflected, lower emissivity gives higher corrected temp."""
        # Object at 50°C (323.15K), reflected at 25°C (298.15K)
        temp_k = 323.15
        result_95 = apply_emissivity_correction(
            temp_k, emissivity=0.95, reflected_temp_c=25.0
        )
        result_80 = apply_emissivity_correction(
            temp_k, emissivity=0.80, reflected_temp_c=25.0
        )
        # Lower emissivity = object is hotter than it appears
        assert result_80 > result_95 > temp_k

    def test_emissivity_correction_array(self):
        """Test emissivity correction works with arrays."""
        # All temps above reflected (25°C = 298.15K)
        temps_k = np.array([310.0, 320.0, 330.0], dtype=np.float32)
        result = apply_emissivity_correction(
            temps_k, emissivity=0.95, reflected_temp_c=25.0
        )
        assert isinstance(result, np.ndarray)
        assert result.shape == (3,)
        assert result.dtype == np.float32

    def test_emissivity_correction_equal_to_reflected(self):
        """When apparent = reflected, correction has no effect."""
        reflected_c = 25.0
        temp_k = reflected_c + 273.15  # Same as reflected
        result = apply_emissivity_correction(
            temp_k,
            emissivity=0.95,
            reflected_temp_c=reflected_c,
        )
        assert result == pytest.approx(temp_k, rel=1e-3)

    def test_raw_to_celsius_corrected_with_env(self):
        """Test end-to-end corrected temperature conversion."""
        # 25°C = 298.15K, raw = 298.15 * 64 = 19081.6
        raw = int(298.15 * TEMP_SCALE)
        env = EnvParams(emissivity=1.0)  # No correction
        result = raw_to_celsius_corrected(raw, env)
        assert result == pytest.approx(25.0, abs=0.1)

    def test_raw_to_celsius_corrected_hot_object(self):
        """Test correction on object hotter than ambient."""
        # 50°C object, reflected at 25°C
        raw = int((50.0 + 273.15) * TEMP_SCALE)
        env_100 = EnvParams(emissivity=1.0, reflected_temp=25.0)
        env_95 = EnvParams(emissivity=0.95, reflected_temp=25.0)
        result_100 = raw_to_celsius_corrected(raw, env_100)
        result_95 = raw_to_celsius_corrected(raw, env_95)
        # Lower emissivity = higher corrected temperature
        assert result_95 > result_100


class TestFrameParsing:
    """Tests for frame parsing functions."""

    def test_parse_marker(self):
        """Test marker parsing."""
        # Create a synthetic marker
        marker_bytes = bytes(
            [
                0x0C,  # length = 12
                0x8C,  # sync byte (start, even frame)
                0x01,
                0x00,
                0x00,
                0x00,  # cnt1 = 1
                0x02,
                0x00,
                0x00,
                0x00,  # cnt2 = 2
                0x28,
                0x00,  # cnt3 = 40
            ]
        )
        marker = parse_marker(marker_bytes)
        assert marker["length"][0] == 12
        assert marker["sync"][0] == SYNC_START_EVEN
        assert marker["cnt1"][0] == 1
        assert marker["cnt2"][0] == 2
        assert marker["cnt3"][0] == 40

    def test_parse_marker_sync_values(self):
        """Test all marker sync byte values."""
        # Start marker, even frame
        marker = parse_marker(bytes([0x0C, SYNC_START_EVEN] + [0] * 10))
        assert marker["sync"][0] == SYNC_START_EVEN

        # Start marker, odd frame
        marker = parse_marker(bytes([0x0C, SYNC_START_ODD] + [0] * 10))
        assert marker["sync"][0] == SYNC_START_ODD

        # End marker, even frame
        marker = parse_marker(bytes([0x0C, SYNC_END_EVEN] + [0] * 10))
        assert marker["sync"][0] == SYNC_END_EVEN

        # End marker, odd frame
        marker = parse_marker(bytes([0x0C, SYNC_END_ODD] + [0] * 10))
        assert marker["sync"][0] == SYNC_END_ODD

    def test_parse_marker_cnt3_wrap(self):
        """Test cnt3 wraps at 2048."""
        # cnt3 at maximum value before wrap
        marker_bytes = bytes(
            [
                0x0C,
                0x8C,
                0x00,
                0x00,
                0x00,
                0x00,  # cnt1
                0x00,
                0x00,
                0x00,
                0x00,  # cnt2
                0x00,
                0x08,  # cnt3 = 2048 (at wrap point)
            ]
        )
        marker = parse_marker(marker_bytes)
        assert marker["cnt3"][0] == CNT3_WRAP

    def test_extract_thermal_data_valid_p3(self):
        """Test thermal data extraction for P3 model."""
        config = get_model_config(Model.P3)
        # Frame data includes marker + pixel data
        frame_size = MARKER_SIZE + config.frame_size
        data = bytearray(frame_size)

        # Fill thermal region with recognizable pattern
        pixels = np.zeros((config.frame_rows, config.sensor_w), dtype=np.uint16)
        pixels[config.thermal_row_start : config.thermal_row_end, :] = 20000

        # Pack into bytes (after marker)
        data[MARKER_SIZE:] = pixels.tobytes()
        result = extract_thermal_data(bytes(data), config=config)

        assert result is not None
        assert result.shape == (config.sensor_h, config.sensor_w)  # 192x256
        assert result[0, 0] == 20000

    def test_extract_thermal_data_valid_p1(self):
        """Test thermal data extraction for P1 model."""
        config = get_model_config(Model.P1)
        frame_size = MARKER_SIZE + config.frame_size
        data = bytearray(frame_size)

        pixels = np.zeros((config.frame_rows, config.sensor_w), dtype=np.uint16)
        pixels[config.thermal_row_start : config.thermal_row_end, :] = 20000

        data[MARKER_SIZE:] = pixels.tobytes()
        result = extract_thermal_data(bytes(data), config=config)

        assert result is not None
        assert result.shape == (config.sensor_h, config.sensor_w)  # 120x160
        assert result[0, 0] == 20000

    def test_extract_thermal_data_too_short(self):
        data = b"\x00" * 100  # Way too short
        result = extract_thermal_data(data)
        assert result is None

    def test_extract_ir_brightness_valid(self):
        """Test IR brightness extraction."""
        config = get_model_config(Model.P3)
        frame_size = MARKER_SIZE + config.frame_size
        data = bytearray(frame_size)

        # Fill IR region with pattern (low byte = brightness)
        pixels = np.zeros((config.frame_rows, config.sensor_w), dtype=np.uint16)
        pixels[: config.ir_row_end, :] = 0x80FF  # Low byte = 0xFF (255)

        data[MARKER_SIZE:] = pixels.tobytes()
        result = extract_ir_brightness(bytes(data), config=config)

        assert result is not None
        assert result.shape == (config.sensor_h, config.sensor_w)
        assert result.dtype == np.uint8
        assert result[0, 0] == 0xFF

    def test_extract_both(self):
        """Test extracting both IR and thermal data."""
        config = get_model_config(Model.P3)
        frame_size = MARKER_SIZE + config.frame_size
        data = bytearray(frame_size)

        pixels = np.zeros((config.frame_rows, config.sensor_w), dtype=np.uint16)
        # IR brightness (low byte)
        pixels[: config.ir_row_end, :] = 0x0080  # Low byte = 0x80 (128)
        # Thermal data
        pixels[config.thermal_row_start : config.thermal_row_end, :] = 19000

        data[MARKER_SIZE:] = pixels.tobytes()
        ir, thermal = extract_both(bytes(data), config=config)

        assert ir is not None
        assert thermal is not None
        assert ir.shape == (config.sensor_h, config.sensor_w)
        assert thermal.shape == (config.sensor_h, config.sensor_w)
        assert ir[0, 0] == 0x80
        assert thermal[0, 0] == 19000


class TestProtocol:
    """Tests for USB protocol functions."""

    def test_crc16_known_values(self):
        payload = bytes.fromhex("0101810001000000000000001e000000")
        crc = crc16_ccitt(payload)
        assert crc == 0x904F

    def test_build_command_length(self):
        cmd = build_command(cmd_type=0x0101, param=0x0081, register=0x0001, resp_len=30)
        assert len(cmd) == 18

    def test_build_command_structure(self):
        cmd = build_command(cmd_type=0x0101, param=0x0081, register=0x0006, resp_len=64)
        # Check command type (first 2 bytes, little-endian)
        assert cmd[0:2] == b"\x01\x01"
        # Check param (bytes 2-3)
        assert cmd[2:4] == b"\x81\x00"
        # Check register (bytes 4-5)
        assert cmd[4:6] == b"\x06\x00"
        # Check response length (bytes 14-15)
        assert cmd[14:16] == b"\x40\x00"  # 64 in little-endian

    def test_precomputed_commands_length(self):
        for name, cmd in COMMANDS.items():
            assert len(cmd) == 18, f"Command {name} has wrong length"


class TestEnums:
    """Tests for enum values."""

    def test_gain_mode_values(self):
        assert GainMode.LOW == 0
        assert GainMode.HIGH == 1
        assert GainMode.AUTO == 2


class TestDataClasses:
    """Tests for dataclass defaults."""

    def test_env_params_defaults(self):
        env = EnvParams()
        assert env.emissivity == 0.95
        assert env.ambient_temp == 25.0
        assert env.distance == 1.0
        assert env.humidity == 0.5

    def test_frame_stats_defaults(self):
        """Test FrameStats dataclass defaults."""
        stats = FrameStats()
        assert stats.frames_read == 0
        assert stats.frames_dropped == 0
        assert stats.marker_mismatches == 0
        assert stats.last_cnt1 == 0
        assert stats.last_cnt3 == 0

    def test_frame_stats_mutable(self):
        """Test FrameStats is mutable for tracking."""
        stats = FrameStats()
        stats.frames_read = 10
        stats.frames_dropped = 2
        stats.marker_mismatches = 1
        stats.last_cnt1 = 12345
        stats.last_cnt3 = 80
        assert stats.frames_read == 10
        assert stats.frames_dropped == 2
        assert stats.marker_mismatches == 1
        assert stats.last_cnt1 == 12345
        assert stats.last_cnt3 == 80


class TestExceptions:
    """Tests for custom exceptions."""

    def test_frame_marker_mismatch_error(self):
        """Test FrameMarkerMismatchError exception."""
        with pytest.raises(FrameMarkerMismatchError):
            raise FrameMarkerMismatchError("cnt1 mismatch: start=1, end=2")


def _run_tests(test_file: str) -> None:
    """Run pytest on this file."""
    sys.exit(
        pytest.main(
            [
                test_file,
                "-v",
                "-s",
                "-W",
                "ignore::pytest.PytestAssertRewriteWarning",
                *sys.argv[1:],
            ]
        )
    )


if __name__ == "__main__":
    _run_tests(__file__)
