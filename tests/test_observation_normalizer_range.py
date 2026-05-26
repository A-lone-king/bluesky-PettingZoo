"""Tests for observation normalizer output range (spec4 F1).

Verify that all normalized features are in [-1, 1] range.
"""

from __future__ import annotations

import pytest

from bluesky_pettingzoo.observations.normalizer import Normalizer


@pytest.fixture
def config() -> dict:
    return {
        "normalization": {
            "heading": {"mid": 180, "range": 180},
            "altitude": {"mid": 33000, "range": 10000},
            "speed": {"mid": 450, "range": 100},
            "distance": {"max": 20},
            "latitude": {"mid": 39.25, "range": 0.25},
            "longitude": {"mid": 116.5, "range": 0.5},
            "vertical_speed": {"max": 6000},
        }
    }


class TestHeadingNormalization:
    """Heading normalization should output [-1, 1]."""

    def test_heading_mid_range(self, config: dict) -> None:
        norm = Normalizer(config)
        assert norm.normalize_heading(180.0) == pytest.approx(0.0)

    def test_heading_min(self, config: dict) -> None:
        norm = Normalizer(config)
        assert norm.normalize_heading(0.0) == pytest.approx(-1.0)

    def test_heading_max(self, config: dict) -> None:
        norm = Normalizer(config)
        assert norm.normalize_heading(360.0) == pytest.approx(1.0)

    def test_heading_clips_overflow(self, config: dict) -> None:
        norm = Normalizer(config)
        assert norm.normalize_heading(400.0) == pytest.approx(1.0)

    def test_heading_clips_underflow(self, config: dict) -> None:
        norm = Normalizer(config)
        assert norm.normalize_heading(-10.0) == pytest.approx(-1.0)


class TestAltitudeNormalization:
    """Altitude normalization should output [-1, 1]."""

    def test_altitude_mid_range(self, config: dict) -> None:
        norm = Normalizer(config)
        assert norm.normalize_altitude(33000.0) == pytest.approx(0.0)

    def test_altitude_min(self, config: dict) -> None:
        norm = Normalizer(config)
        assert norm.normalize_altitude(23000.0) == pytest.approx(-1.0)

    def test_altitude_max(self, config: dict) -> None:
        norm = Normalizer(config)
        assert norm.normalize_altitude(43000.0) == pytest.approx(1.0)


class TestSpeedNormalization:
    """Speed normalization should output [-1, 1]."""

    def test_speed_mid_range(self, config: dict) -> None:
        norm = Normalizer(config)
        assert norm.normalize_speed(450.0) == pytest.approx(0.0)

    def test_speed_min(self, config: dict) -> None:
        norm = Normalizer(config)
        assert norm.normalize_speed(350.0) == pytest.approx(-1.0)

    def test_speed_max(self, config: dict) -> None:
        norm = Normalizer(config)
        assert norm.normalize_speed(550.0) == pytest.approx(1.0)


class TestLatitudeNormalization:
    """Latitude normalization should output [-1, 1]."""

    def test_lat_mid_range(self, config: dict) -> None:
        norm = Normalizer(config)
        assert norm.normalize_lat(39.25) == pytest.approx(0.0)

    def test_lat_min(self, config: dict) -> None:
        norm = Normalizer(config)
        assert norm.normalize_lat(39.0) == pytest.approx(-1.0)

    def test_lat_max(self, config: dict) -> None:
        norm = Normalizer(config)
        assert norm.normalize_lat(39.5) == pytest.approx(1.0)


class TestLongitudeNormalization:
    """Longitude normalization should output [-1, 1]."""

    def test_lon_mid_range(self, config: dict) -> None:
        norm = Normalizer(config)
        assert norm.normalize_lon(116.5) == pytest.approx(0.0)

    def test_lon_min(self, config: dict) -> None:
        norm = Normalizer(config)
        assert norm.normalize_lon(116.0) == pytest.approx(-1.0)

    def test_lon_max(self, config: dict) -> None:
        norm = Normalizer(config)
        assert norm.normalize_lon(117.0) == pytest.approx(1.0)


class TestVerticalSpeedNormalization:
    """Vertical speed normalization should output [-1, 1]."""

    def test_vs_zero(self, config: dict) -> None:
        norm = Normalizer(config)
        assert norm.normalize_vs(0.0) == pytest.approx(0.0)

    def test_vs_max_positive(self, config: dict) -> None:
        norm = Normalizer(config)
        assert norm.normalize_vs(6000.0) == pytest.approx(1.0)

    def test_vs_max_negative(self, config: dict) -> None:
        norm = Normalizer(config)
        assert norm.normalize_vs(-6000.0) == pytest.approx(-1.0)


class TestDistanceNormalization:
    """Distance normalization should output [0, 1]."""

    def test_distance_zero(self, config: dict) -> None:
        norm = Normalizer(config)
        assert norm.normalize_distance(0.0) == pytest.approx(0.0)

    def test_distance_max(self, config: dict) -> None:
        norm = Normalizer(config)
        assert norm.normalize_distance(20.0) == pytest.approx(1.0)

    def test_distance_clips_overflow(self, config: dict) -> None:
        norm = Normalizer(config)
        assert norm.normalize_distance(30.0) == pytest.approx(1.0)


class TestCircularNormalization:
    """Circular normalization (cos/sin) should output [-1, 1]."""

    def test_heading_cos_range(self, config: dict) -> None:
        norm = Normalizer(config)
        for hdg in [0, 90, 180, 270, 360]:
            result = norm.normalize_heading_cos(hdg)
            assert -1.0 <= result <= 1.0

    def test_heading_sin_range(self, config: dict) -> None:
        norm = Normalizer(config)
        for hdg in [0, 90, 180, 270, 360]:
            result = norm.normalize_heading_sin(hdg)
            assert -1.0 <= result <= 1.0

    def test_bearing_cos_range(self, config: dict) -> None:
        norm = Normalizer(config)
        for brg in [0, 90, 180, 270, 360]:
            result = norm.normalize_bearing_cos(brg)
            assert -1.0 <= result <= 1.0

    def test_bearing_sin_range(self, config: dict) -> None:
        norm = Normalizer(config)
        for brg in [0, 90, 180, 270, 360]:
            result = norm.normalize_bearing_sin(brg)
            assert -1.0 <= result <= 1.0
