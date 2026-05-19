"""Unit tests for p3_viewer.py."""

from __future__ import annotations

import sys

import numpy as np
import pytest

from p3_viewer import (
    COLORMAPS,
    ColormapID,
    agc_fixed,
    agc_temporal,
    apply_colormap,
    dde,
    get_colormap,
    tnr,
)


class TestColormaps:
    """Tests for colormap functions."""

    def test_all_colormaps_exist(self):
        for cmap_id in ColormapID:
            assert cmap_id in COLORMAPS
            lut = COLORMAPS[cmap_id]
            assert lut.shape == (256, 3)
            assert lut.dtype == np.uint8

    def test_get_colormap_by_id(self):
        lut = get_colormap(ColormapID.WHITE_HOT)
        assert lut.shape == (256, 3)

    def test_get_colormap_by_int(self):
        lut = get_colormap(0)  # WHITE_HOT
        assert lut.shape == (256, 3)

    def test_white_hot_is_grayscale(self):
        lut = get_colormap(ColormapID.WHITE_HOT)
        # White hot should go from black (0,0,0) to white (255,255,255)
        assert lut[0, 0] == 0  # Black at start
        assert lut[0, 1] == 0
        assert lut[0, 2] == 0
        assert lut[255, 0] == 255  # White at end
        assert lut[255, 1] == 255
        assert lut[255, 2] == 255

    def test_black_hot_is_inverse_grayscale(self):
        lut = get_colormap(ColormapID.BLACK_HOT)
        # Black hot should go from white to black
        assert lut[0, 0] == 255  # White at start
        assert lut[255, 0] == 0  # Black at end

    def test_apply_colormap_shape(self):
        img = np.zeros((100, 100), dtype=np.uint8)
        result = apply_colormap(img, ColormapID.IRONBOW)
        assert result.shape == (100, 100, 3)
        assert result.dtype == np.uint8

    def test_apply_colormap_values(self):
        # Create image with known values
        img = np.array([[0, 127, 255]], dtype=np.uint8)
        result = apply_colormap(img, ColormapID.WHITE_HOT)

        # WHITE_HOT: 0 -> black, 255 -> white
        assert np.all(result[0, 0] == [0, 0, 0])  # Black
        assert np.all(result[0, 2] == [255, 255, 255])  # White


class TestAGC:
    """Tests for Auto Gain Control."""

    def test_agc_temporal_basic(self):
        # Reset EMA state
        import p3_viewer

        p3_viewer._agc_ema_low = None
        p3_viewer._agc_ema_high = None

        # Create image with values 1000-2000
        img = np.linspace(1000, 2000, 100).reshape(10, 10).astype(np.uint16)
        result = agc_temporal(img, pct=1.0)
        assert result.shape == img.shape
        assert result.dtype == np.uint8
        assert result.min() >= 0
        assert result.max() <= 255

    def test_agc_temporal_percentile_clipping(self):
        # Reset EMA state
        import p3_viewer

        p3_viewer._agc_ema_low = None
        p3_viewer._agc_ema_high = None

        # Create image with outliers
        img = np.ones((10, 10), dtype=np.uint16) * 1000
        img[0, 0] = 0  # Low outlier
        img[9, 9] = 10000  # High outlier

        # With percentile clipping, outliers should be clipped
        result = agc_temporal(img, pct=10.0)
        assert result.dtype == np.uint8

    def test_agc_temporal_uniform_image(self):
        # Reset EMA state
        import p3_viewer

        p3_viewer._agc_ema_low = None
        p3_viewer._agc_ema_high = None

        # Uniform image should not crash
        img = np.ones((10, 10), dtype=np.uint16) * 5000
        result = agc_temporal(img)
        assert result.shape == img.shape

    def test_agc_fixed_basic(self):
        # Create image with temperature values around room temp
        # Room temp ~25°C = (25 + 273.15) * 64 ≈ 19082 raw
        img = np.linspace(18000, 20000, 100).reshape(10, 10).astype(np.uint16)
        result = agc_fixed(img)
        assert result.shape == img.shape
        assert result.dtype == np.uint8

    def test_agc_fixed_clips_extremes(self):
        # Values outside 15-40°C range should be clipped
        low_temp = int((15 + 273.15) * 64)  # ~18442
        high_temp = int((40 + 273.15) * 64)  # ~20042
        img = np.array([[low_temp - 1000, high_temp + 1000]], dtype=np.uint16)
        result = agc_fixed(img)
        # Low temp should map to 0, high temp should map to 255
        assert result[0, 0] == 0
        assert result[0, 1] == 255


class TestDDE:
    """Tests for Digital Detail Enhancement."""

    def test_dde_basic(self):
        img = np.random.randint(0, 256, (50, 50), dtype=np.uint8)
        result = dde(img)
        assert result.shape == img.shape
        assert result.dtype == np.uint8

    def test_dde_zero_strength(self):
        img = np.random.randint(0, 256, (50, 50), dtype=np.uint8)
        result = dde(img, strength=0)
        np.testing.assert_array_equal(result, img)

    def test_dde_enhances_edges(self):
        # Create image with sharp edge
        img = np.zeros((20, 20), dtype=np.uint8)
        img[:, 10:] = 200

        result = dde(img, strength=1.0, kernel_size=3)

        # Edge should be more pronounced (higher gradient)
        edge_original = np.abs(img[:, 9].astype(int) - img[:, 10].astype(int)).mean()
        edge_enhanced = np.abs(
            result[:, 9].astype(int) - result[:, 10].astype(int)
        ).mean()
        assert edge_enhanced >= edge_original


class TestTNR:
    """Tests for Temporal Noise Reduction."""

    def test_tnr_first_frame(self):
        img = np.ones((10, 10), dtype=np.uint16) * 1000
        result = tnr(img, None)
        np.testing.assert_array_equal(result, img)

    def test_tnr_blending(self):
        prev = np.zeros((10, 10), dtype=np.uint16)
        curr = np.ones((10, 10), dtype=np.uint16) * 1000

        # 50% blend
        result = tnr(curr, prev, alpha=0.5)
        expected = 500
        assert result[0, 0] == pytest.approx(expected, abs=1)

    def test_tnr_alpha_zero(self):
        prev = np.ones((10, 10), dtype=np.uint16) * 100
        curr = np.ones((10, 10), dtype=np.uint16) * 200

        # alpha=0 means all previous frame
        result = tnr(curr, prev, alpha=0)
        np.testing.assert_array_almost_equal(result, prev)

    def test_tnr_alpha_one(self):
        prev = np.ones((10, 10), dtype=np.uint16) * 100
        curr = np.ones((10, 10), dtype=np.uint16) * 200

        # alpha=1 means all current frame
        result = tnr(curr, prev, alpha=1.0)
        np.testing.assert_array_almost_equal(result, curr)


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
