"""Tests for enhanced observation space — bearing decomposition (T-V03)."""

from __future__ import annotations

import math

import pytest

from bluesky_pettingzoo.observations.normalizer import Normalizer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config() -> dict:
    return {
        "normalization": {
            "heading": {"mid": 180, "range": 180},
            "altitude": {"mid": 33000, "range": 10000},
            "speed": {"mid": 450, "range": 100},
            "distance": {"max": 20},
        },
        "observation": {
            "perception_radius_nm": 20,
            "perception_alt_diff_ft": 3000,
            "max_observable_aircraft": 10,
        },
    }


# ===========================================================================
# T-V03: Heading cos/sin decomposition
# ===========================================================================


class TestHeadingCosSin:
    """Heading should be decomposed into cos/sin components."""

    def test_heading_cos_sin_0(self) -> None:
        """Heading 0° (north) → cos=1, sin=0."""
        norm = Normalizer(_make_config())
        cos_val = norm.normalize_heading_cos(0.0)
        sin_val = norm.normalize_heading_sin(0.0)
        assert cos_val == pytest.approx(1.0, abs=1e-6)
        assert sin_val == pytest.approx(0.0, abs=1e-6)

    def test_heading_cos_sin_90(self) -> None:
        """Heading 90° (east) → cos=0, sin=1."""
        norm = Normalizer(_make_config())
        cos_val = norm.normalize_heading_cos(90.0)
        sin_val = norm.normalize_heading_sin(90.0)
        assert cos_val == pytest.approx(0.0, abs=1e-6)
        assert sin_val == pytest.approx(1.0, abs=1e-6)

    def test_heading_cos_sin_180(self) -> None:
        """Heading 180° (south) → cos=-1, sin=0."""
        norm = Normalizer(_make_config())
        cos_val = norm.normalize_heading_cos(180.0)
        sin_val = norm.normalize_heading_sin(180.0)
        assert cos_val == pytest.approx(-1.0, abs=1e-6)
        assert sin_val == pytest.approx(0.0, abs=1e-6)

    def test_heading_cos_sin_270(self) -> None:
        """Heading 270° (west) → cos=0, sin=-1."""
        norm = Normalizer(_make_config())
        cos_val = norm.normalize_heading_cos(270.0)
        sin_val = norm.normalize_heading_sin(270.0)
        assert cos_val == pytest.approx(0.0, abs=1e-6)
        assert sin_val == pytest.approx(-1.0, abs=1e-6)


# ===========================================================================
# T-V03: Bearing cos/sin decomposition
# ===========================================================================


class TestBearingCosSin:
    """Bearing should be decomposed into cos/sin components."""

    def test_bearing_cos_sin_north(self) -> None:
        """North bearing (0°) → cos=1, sin=0."""
        norm = Normalizer(_make_config())
        cos_val = norm.normalize_bearing_cos(0.0)
        sin_val = norm.normalize_bearing_sin(0.0)
        assert cos_val == pytest.approx(1.0, abs=1e-6)
        assert sin_val == pytest.approx(0.0, abs=1e-6)

    def test_bearing_cos_sin_east(self) -> None:
        """East bearing (90°) → cos=0, sin=1."""
        norm = Normalizer(_make_config())
        cos_val = norm.normalize_bearing_cos(90.0)
        sin_val = norm.normalize_bearing_sin(90.0)
        assert cos_val == pytest.approx(0.0, abs=1e-6)
        assert sin_val == pytest.approx(1.0, abs=1e-6)

    def test_bearing_no_discontinuity(self) -> None:
        """359° and 1° should be continuous (close cos/sin values)."""
        norm = Normalizer(_make_config())
        cos_359 = norm.normalize_bearing_cos(359.0)
        sin_359 = norm.normalize_bearing_sin(359.0)
        cos_1 = norm.normalize_bearing_cos(1.0)
        sin_1 = norm.normalize_bearing_sin(1.0)
        # 359° and 1° are only 2° apart, cos/sin should be very close
        # sin(359°) ≈ -0.0175, sin(1°) ≈ 0.0175, diff ≈ 0.035
        assert abs(cos_359 - cos_1) < 0.05
        assert abs(sin_359 - sin_1) < 0.05


# ===========================================================================
# T-V03: Observation space shape
# ===========================================================================


class TestObservationShape:
    """Observation space dimensions should reflect enhanced layout."""

    def test_self_state_shape(self) -> None:
        """self_state should have 8 dimensions after enhancement."""
        from bluesky_pettingzoo.observations.manager import ObservationManager

        config = _make_config()
        mgr = ObservationManager(config)
        space = mgr.observation_space()
        assert space["self_state"].shape == (9,)

    def test_goal_bearing_cos_sin(self) -> None:
        """Goal should contain bearing_cos and bearing_sin (not raw bearing)."""
        from bluesky_pettingzoo.observations.manager import ObservationManager
        from bluesky_pettingzoo.utils.types import AircraftState

        config = _make_config()
        mgr = ObservationManager(config)

        own = AircraftState(id="AC000", lat=39.0, lon=116.0, alt=35000, hdg=90.0, tas=450, vs=0)
        goal = {"lat": 39.1, "lon": 116.1, "alt": 35000, "hdg": 90.0}
        result = mgr.generate(own_state=own, other_states=[], goal=goal)
        goal_vec = result["observation"]["goal"]

        # goal should be [distance, bearing_cos, bearing_sin, alt_diff]
        assert goal_vec.shape == (4,)
        # bearing to (39.1, 116.1) from (39.0, 116.0) is roughly northeast (45°)
        # cos(45°) ≈ 0.707, sin(45°) ≈ 0.707
        assert goal_vec[1] == pytest.approx(math.cos(math.radians(45)), abs=0.2)
        assert goal_vec[2] == pytest.approx(math.sin(math.radians(45)), abs=0.2)


# ===========================================================================
# T-V04: Relative speed components
# ===========================================================================


class TestRelativeSpeed:
    """Relative speed x/y components should be computed correctly."""

    def test_relative_speed_head_on(self) -> None:
        """Head-on flight: relative speed ≈ sum of both speeds."""
        from bluesky_pettingzoo.observations.manager import ObservationManager
        from bluesky_pettingzoo.utils.types import AircraftState

        config = _make_config()
        mgr = ObservationManager(config)

        # AC000 heading north (0°) at 450kt, AC001 heading south (180°) at 450kt
        own = AircraftState(id="AC000", lat=39.0, lon=116.0, alt=35000, hdg=0.0, tas=450, vs=0)
        other = AircraftState(id="AC001", lat=39.05, lon=116.0, alt=35000, hdg=180.0, tas=450, vs=0)
        goal = {"lat": 39.5, "lon": 116.0, "alt": 35000, "hdg": 0.0}

        result = mgr.generate(own, [other], goal)
        obs = result["observation"]
        other_row = obs["other_aircraft"][0]

        # relative_speed_x (index 7) and relative_speed_y (index 8) should be large
        # Head-on: relative speed y component should be close to -(450+450)/normalization
        rel_speed_x = other_row[7]
        rel_speed_y = other_row[8]
        # At least one component should be significantly non-zero
        assert abs(rel_speed_x) > 0.1 or abs(rel_speed_y) > 0.1

    def test_relative_speed_parallel(self) -> None:
        """Parallel flight at same speed: relative speed ≈ 0."""
        from bluesky_pettingzoo.observations.manager import ObservationManager
        from bluesky_pettingzoo.utils.types import AircraftState

        config = _make_config()
        mgr = ObservationManager(config)

        # Both heading north at same speed
        own = AircraftState(id="AC000", lat=39.0, lon=116.0, alt=35000, hdg=0.0, tas=450, vs=0)
        other = AircraftState(id="AC001", lat=39.05, lon=116.0, alt=35000, hdg=0.0, tas=450, vs=0)
        goal = {"lat": 39.5, "lon": 116.0, "alt": 35000, "hdg": 0.0}

        result = mgr.generate(own, [other], goal)
        obs = result["observation"]
        other_row = obs["other_aircraft"][0]

        # Relative speed should be near zero
        rel_speed_x = other_row[7]
        rel_speed_y = other_row[8]
        assert abs(rel_speed_x) < 0.1
        assert abs(rel_speed_y) < 0.1

    def test_relative_speed_crossing(self) -> None:
        """Crossing flight: x and y components both non-zero."""
        from bluesky_pettingzoo.observations.manager import ObservationManager
        from bluesky_pettingzoo.utils.types import AircraftState

        config = _make_config()
        mgr = ObservationManager(config)

        # AC000 heading north, AC001 heading east
        own = AircraftState(id="AC000", lat=39.0, lon=116.0, alt=35000, hdg=0.0, tas=450, vs=0)
        other = AircraftState(id="AC001", lat=39.05, lon=116.0, alt=35000, hdg=90.0, tas=450, vs=0)
        goal = {"lat": 39.5, "lon": 116.0, "alt": 35000, "hdg": 0.0}

        result = mgr.generate(own, [other], goal)
        obs = result["observation"]
        other_row = obs["other_aircraft"][0]

        rel_speed_x = other_row[7]
        rel_speed_y = other_row[8]
        # Both components should be non-zero for crossing scenario
        assert abs(rel_speed_x) > 0.01
        assert abs(rel_speed_y) > 0.01

    def test_other_aircraft_shape(self) -> None:
        """other_aircraft should have 9 columns (including relative speed)."""
        from bluesky_pettingzoo.observations.manager import ObservationManager

        config = _make_config()
        mgr = ObservationManager(config)
        space = mgr.observation_space()
        assert space["other_aircraft"].shape == (10, 10)

    def test_relative_speed_normalized(self) -> None:
        """Relative speed values should be in [-1, 1]."""
        from bluesky_pettingzoo.observations.manager import ObservationManager
        from bluesky_pettingzoo.utils.types import AircraftState

        config = _make_config()
        mgr = ObservationManager(config)

        own = AircraftState(id="AC000", lat=39.0, lon=116.0, alt=35000, hdg=0.0, tas=450, vs=0)
        other = AircraftState(id="AC001", lat=39.05, lon=116.0, alt=35000, hdg=180.0, tas=450, vs=0)
        goal = {"lat": 39.5, "lon": 116.0, "alt": 35000, "hdg": 0.0}

        result = mgr.generate(own, [other], goal)
        obs = result["observation"]
        other_row = obs["other_aircraft"][0]

        # All values should be in [-1, 1]
        for i in range(9):
            assert -1.0 <= other_row[i] <= 1.0, f"other_aircraft[{i}] = {other_row[i]} out of range"


# ===========================================================================
# Observation normalization: lat/lon/vs should be in [-1, 1]
# ===========================================================================


class TestLatLonVsNormalization:
    """lat, lon, vs should be normalized to [-1, 1] like other features."""

    def _make_config_with_airspace(self) -> dict:
        config = _make_config()
        config["normalization"]["latitude"] = {"mid": 39.25, "range": 0.25}
        config["normalization"]["longitude"] = {"mid": 116.5, "range": 0.5}
        config["normalization"]["vertical_speed"] = {"max": 6000}
        return config

    def test_lat_center_maps_to_zero(self) -> None:
        """lat at airspace center should normalize to ~0."""
        from bluesky_pettingzoo.observations.manager import ObservationManager
        from bluesky_pettingzoo.utils.types import AircraftState

        config = self._make_config_with_airspace()
        mgr = ObservationManager(config)
        own = AircraftState(id="AC000", lat=39.25, lon=116.5, alt=35000, hdg=0.0, tas=450, vs=0)
        goal = {"lat": 39.25, "lon": 116.5, "alt": 35000, "hdg": 0.0}
        obs = mgr.generate(own, [], goal)["observation"]["self_state"]
        # lat is index 4
        assert -1.0 <= obs[4] <= 1.0, f"lat={obs[4]} out of [-1, 1]"
        assert abs(obs[4]) < 0.01, f"lat at center should be ~0, got {obs[4]}"

    def test_lon_center_maps_to_zero(self) -> None:
        """lon at airspace center should normalize to ~0."""
        from bluesky_pettingzoo.observations.manager import ObservationManager
        from bluesky_pettingzoo.utils.types import AircraftState

        config = self._make_config_with_airspace()
        mgr = ObservationManager(config)
        own = AircraftState(id="AC000", lat=39.25, lon=116.5, alt=35000, hdg=0.0, tas=450, vs=0)
        goal = {"lat": 39.25, "lon": 116.5, "alt": 35000, "hdg": 0.0}
        obs = mgr.generate(own, [], goal)["observation"]["self_state"]
        # lon is index 5
        assert -1.0 <= obs[5] <= 1.0, f"lon={obs[5]} out of [-1, 1]"
        assert abs(obs[5]) < 0.01, f"lon at center should be ~0, got {obs[5]}"

    def test_vs_zero_maps_to_zero(self) -> None:
        """vs=0 should normalize to ~0."""
        from bluesky_pettingzoo.observations.manager import ObservationManager
        from bluesky_pettingzoo.utils.types import AircraftState

        config = self._make_config_with_airspace()
        mgr = ObservationManager(config)
        own = AircraftState(id="AC000", lat=39.25, lon=116.5, alt=35000, hdg=0.0, tas=450, vs=0)
        goal = {"lat": 39.25, "lon": 116.5, "alt": 35000, "hdg": 0.0}
        obs = mgr.generate(own, [], goal)["observation"]["self_state"]
        # vs is index 6
        assert abs(obs[6]) < 0.01, f"vs=0 should normalize to ~0, got {obs[6]}"

    def test_vs_positive_normalized(self) -> None:
        """vs=3000 should normalize to ~0.5."""
        norm = Normalizer(_make_config())
        result = norm.normalize_vs(3000.0)
        assert -1.0 <= result <= 1.0
        assert abs(result - 0.5) < 0.01

    def test_vs_negative_normalized(self) -> None:
        """vs=-6000 should normalize to ~-1.0."""
        norm = Normalizer(_make_config())
        result = norm.normalize_vs(-6000.0)
        assert abs(result - (-1.0)) < 0.01

    def test_all_self_state_in_minus1_1(self) -> None:
        """All 9 self_state features should be in [-1, 1]."""
        from bluesky_pettingzoo.observations.manager import ObservationManager
        from bluesky_pettingzoo.utils.types import AircraftState

        config = self._make_config_with_airspace()
        mgr = ObservationManager(config)
        own = AircraftState(id="AC000", lat=39.0, lon=116.0, alt=29000, hdg=0.0, tas=400, vs=-3000)
        goal = {"lat": 39.5, "lon": 117.0, "alt": 37000, "hdg": 180.0}
        obs = mgr.generate(own, [], goal)["observation"]["self_state"]
        for i in range(9):
            assert -1.0 <= obs[i] <= 1.0, f"self_state[{i}] = {obs[i]} out of [-1, 1]"
