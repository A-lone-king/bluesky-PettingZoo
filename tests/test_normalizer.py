"""Tests for observation normalizer module."""

from __future__ import annotations

import pytest

from bluesky_pettingzoo.observations.normalizer import Normalizer


class TestNormalizeHeading:
    """Test heading normalization."""

    def test_normalize_heading_0(self, default_config: dict) -> None:
        """Heading 0° should normalize to -1.0."""
        norm = Normalizer(default_config)
        result = norm.normalize_heading(0.0)
        assert result == pytest.approx(-1.0)

    def test_normalize_heading_180(self, default_config: dict) -> None:
        """Heading 180° should normalize to 0.0."""
        norm = Normalizer(default_config)
        result = norm.normalize_heading(180.0)
        assert result == pytest.approx(0.0)

    def test_normalize_heading_360(self, default_config: dict) -> None:
        """Heading 360° should normalize to 1.0."""
        norm = Normalizer(default_config)
        result = norm.normalize_heading(360.0)
        assert result == pytest.approx(1.0)

    def test_normalize_heading_90(self, default_config: dict) -> None:
        """Heading 90° should normalize to -0.5."""
        norm = Normalizer(default_config)
        result = norm.normalize_heading(90.0)
        assert result == pytest.approx(-0.5)

    def test_normalize_heading_270(self, default_config: dict) -> None:
        """Heading 270° should normalize to 0.5."""
        norm = Normalizer(default_config)
        result = norm.normalize_heading(270.0)
        assert result == pytest.approx(0.5)


class TestNormalizeAltitude:
    """Test altitude normalization."""

    def test_normalize_altitude_low(self, default_config: dict) -> None:
        """Low altitude (23000ft) should normalize to negative value."""
        norm = Normalizer(default_config)
        result = norm.normalize_altitude(23000.0)
        assert result == pytest.approx(-1.0)

    def test_normalize_altitude_mid(self, default_config: dict) -> None:
        """Mid altitude (33000ft) should normalize to 0.0."""
        norm = Normalizer(default_config)
        result = norm.normalize_altitude(33000.0)
        assert result == pytest.approx(0.0)

    def test_normalize_altitude_high(self, default_config: dict) -> None:
        """High altitude (43000ft) should normalize to 1.0."""
        norm = Normalizer(default_config)
        result = norm.normalize_altitude(43000.0)
        assert result == pytest.approx(1.0)

    def test_normalize_altitude_29000(self, default_config: dict) -> None:
        """Altitude 29000ft should normalize to -0.4."""
        norm = Normalizer(default_config)
        result = norm.normalize_altitude(29000.0)
        assert result == pytest.approx(-0.4)

    def test_normalize_altitude_37000(self, default_config: dict) -> None:
        """Altitude 37000ft should normalize to 0.4."""
        norm = Normalizer(default_config)
        result = norm.normalize_altitude(37000.0)
        assert result == pytest.approx(0.4)


class TestNormalizeSpeed:
    """Test speed normalization."""

    def test_normalize_speed_low(self, default_config: dict) -> None:
        """Low speed (350kt) should normalize to -1.0."""
        norm = Normalizer(default_config)
        result = norm.normalize_speed(350.0)
        assert result == pytest.approx(-1.0)

    def test_normalize_speed_mid(self, default_config: dict) -> None:
        """Mid speed (450kt) should normalize to 0.0."""
        norm = Normalizer(default_config)
        result = norm.normalize_speed(450.0)
        assert result == pytest.approx(0.0)

    def test_normalize_speed_high(self, default_config: dict) -> None:
        """High speed (550kt) should normalize to 1.0."""
        norm = Normalizer(default_config)
        result = norm.normalize_speed(550.0)
        assert result == pytest.approx(1.0)

    def test_normalize_speed_range(self, default_config: dict) -> None:
        """Speed at range boundaries should be clamped."""
        norm = Normalizer(default_config)
        assert norm.normalize_speed(400.0) == pytest.approx(-0.5)
        assert norm.normalize_speed(500.0) == pytest.approx(0.5)


class TestNormalizeDistance:
    """Test distance normalization."""

    def test_normalize_distance_zero(self, default_config: dict) -> None:
        """Distance 0 should normalize to 0.0."""
        norm = Normalizer(default_config)
        result = norm.normalize_distance(0.0)
        assert result == pytest.approx(0.0)

    def test_normalize_distance_max(self, default_config: dict) -> None:
        """Distance at max (20NM) should normalize to 1.0."""
        norm = Normalizer(default_config)
        result = norm.normalize_distance(20.0)
        assert result == pytest.approx(1.0)

    def test_normalize_distance_half(self, default_config: dict) -> None:
        """Distance at half max (10NM) should normalize to 0.5."""
        norm = Normalizer(default_config)
        result = norm.normalize_distance(10.0)
        assert result == pytest.approx(0.5)


class TestNormalizeBearing:
    """Test bearing normalization."""

    def test_normalize_bearing_north(self, default_config: dict) -> None:
        """Bearing 0° (north) should normalize to 0.0."""
        norm = Normalizer(default_config)
        result = norm.normalize_bearing(0.0)
        assert result == pytest.approx(0.0)

    def test_normalize_bearing_east(self, default_config: dict) -> None:
        """Bearing 90° (east) should normalize to 0.25."""
        norm = Normalizer(default_config)
        result = norm.normalize_bearing(90.0)
        assert result == pytest.approx(0.25)

    def test_normalize_bearing_south(self, default_config: dict) -> None:
        """Bearing 180° (south) should normalize to 0.5."""
        norm = Normalizer(default_config)
        result = norm.normalize_bearing(180.0)
        assert result == pytest.approx(0.5)

    def test_normalize_bearing_west(self, default_config: dict) -> None:
        """Bearing 270° (west) should normalize to 0.75."""
        norm = Normalizer(default_config)
        result = norm.normalize_bearing(270.0)
        assert result == pytest.approx(0.75)

    def test_normalize_bearing_360(self, default_config: dict) -> None:
        """Bearing 360° should normalize to 1.0."""
        norm = Normalizer(default_config)
        result = norm.normalize_bearing(360.0)
        assert result == pytest.approx(1.0)


class TestOutputClipping:
    """Test output clipping to [-1, 1]."""

    def test_clipping_above_max(self, default_config: dict) -> None:
        """Values above max should be clipped to 1.0."""
        norm = Normalizer(default_config)
        result = norm.normalize_altitude(50000.0)
        assert result == pytest.approx(1.0)

    def test_clipping_below_min(self, default_config: dict) -> None:
        """Values below min should be clipped to -1.0."""
        norm = Normalizer(default_config)
        result = norm.normalize_altitude(10000.0)
        assert result == pytest.approx(-1.0)

    def test_clipping_heading_above(self, default_config: dict) -> None:
        """Heading above 360 should be clipped to 1.0."""
        norm = Normalizer(default_config)
        result = norm.normalize_heading(400.0)
        assert result == pytest.approx(1.0)

    def test_clipping_heading_below(self, default_config: dict) -> None:
        """Heading below 0 should be clipped to -1.0."""
        norm = Normalizer(default_config)
        result = norm.normalize_heading(-10.0)
        assert result == pytest.approx(-1.0)


class TestNormalizeAircraftState:
    """Test full aircraft state normalization."""

    def test_normalize_aircraft_state(
        self,
        default_config: dict,
        sample_aircraft_state: dict,
    ) -> None:
        """Test normalizing complete aircraft state."""
        norm = Normalizer(default_config)
        result = norm.normalize_aircraft_state(sample_aircraft_state)

        assert "heading" in result
        assert "altitude" in result
        assert "speed" in result
        assert result["heading"] == pytest.approx(-0.5)  # 90° -> -0.5
        assert result["altitude"] == pytest.approx(0.2)  # 35000ft -> 0.2
        assert result["speed"] == pytest.approx(0.0)  # 450kt -> 0.0

    def test_normalize_aircraft_state_fields(
        self,
        default_config: dict,
        sample_aircraft_state: dict,
    ) -> None:
        """Test that all required fields are present in normalized state."""
        norm = Normalizer(default_config)
        result = norm.normalize_aircraft_state(sample_aircraft_state)

        assert "heading" in result
        assert "altitude" in result
        assert "speed" in result
        assert "lat" in result
        assert "lon" in result
        assert "vs" in result


class TestNormalizeRelativePosition:
    """Test relative position normalization."""

    def test_normalize_relative_position(self, default_config: dict) -> None:
        """Test normalizing relative position (distance, bearing)."""
        norm = Normalizer(default_config)
        result = norm.normalize_relative_position(
            distance_nm=10.0,
            bearing_deg=90.0,
        )

        assert "distance" in result
        assert "bearing" in result
        assert result["distance"] == pytest.approx(0.5)  # 10NM / 20NM
        assert result["bearing"] == pytest.approx(0.25)  # 90° / 360°

    def test_normalize_relative_position_zero(self, default_config: dict) -> None:
        """Test normalizing zero relative position."""
        norm = Normalizer(default_config)
        result = norm.normalize_relative_position(
            distance_nm=0.0,
            bearing_deg=0.0,
        )

        assert result["distance"] == pytest.approx(0.0)
        assert result["bearing"] == pytest.approx(0.0)
