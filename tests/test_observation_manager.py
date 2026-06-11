"""Tests for observation manager module."""

from __future__ import annotations

import numpy as np
from gymnasium import spaces

from bluesky_pettingzoo.observations.manager import ObservationManager
from bluesky_pettingzoo.utils.types import AircraftState


def make_state(
    acid: str,
    lat: float,
    lon: float,
    alt: float,
    hdg: float = 90.0,
    tas: float = 450.0,
    vs: float = 0.0,
) -> AircraftState:
    return AircraftState(
        id=acid,
        lat=lat,
        lon=lon,
        alt=alt,
        hdg=hdg,
        tas=tas,
        vs=vs,
    )


class TestObservationSpaceShape:
    """Test observation space definition."""

    def test_observation_space_type(self, default_config: dict) -> None:
        """Observation space should be a gymnasium Dict."""
        mgr = ObservationManager(default_config)
        space = mgr.observation_space()
        assert isinstance(space, spaces.Dict)

    def test_observation_space_keys(self, default_config: dict) -> None:
        """Observation space should have all required keys including conflict_state."""
        mgr = ObservationManager(default_config)
        space = mgr.observation_space()
        assert "self_state" in space.spaces
        assert "other_aircraft" in space.spaces
        assert "other_aircraft_mask" in space.spaces
        assert "goal" in space.spaces
        assert "conflict_state" in space.spaces

    def test_self_state_shape(self, default_config: dict) -> None:
        """self_state should be shape (9,)."""
        mgr = ObservationManager(default_config)
        space = mgr.observation_space()
        assert space["self_state"].shape == (9,)

    def test_other_aircraft_shape(self, default_config: dict) -> None:
        """other_aircraft should be shape (MAX_OBS, 12) with conflict prediction."""
        mgr = ObservationManager(default_config)
        space = mgr.observation_space()
        assert space["other_aircraft"].shape == (10, 12)

    def test_other_aircraft_mask_shape(self, default_config: dict) -> None:
        """other_aircraft_mask should be shape (MAX_OBS,)."""
        mgr = ObservationManager(default_config)
        space = mgr.observation_space()
        assert space["other_aircraft_mask"].shape == (10,)

    def test_goal_shape(self, default_config: dict) -> None:
        """goal should be shape (4,)."""
        mgr = ObservationManager(default_config)
        space = mgr.observation_space()
        assert space["goal"].shape == (4,)

    def test_conflict_state_shape(self, default_config: dict) -> None:
        """conflict_state should be shape (3,) with float32 dtype."""
        mgr = ObservationManager(default_config)
        space = mgr.observation_space()
        assert space["conflict_state"].shape == (3,)
        assert space["conflict_state"].dtype == np.float32


class TestSelfStateFields:
    """Test self_state observation fields."""

    def test_self_state_fields(self, default_config: dict) -> None:
        """self_state should contain 8 normalized values."""
        mgr = ObservationManager(default_config)
        own = make_state("OWN", 39.25, 116.25, 35000.0, hdg=90.0, tas=450.0)
        goal = {"lat": 39.50, "lon": 116.50, "alt": 35000.0, "hdg": 90.0}

        result = mgr.generate(own, [], goal)
        obs = result["observation"]

        assert obs["self_state"].shape == (9,)
        assert obs["self_state"].dtype == np.float32


class TestOtherAircraftFields:
    """Test other_aircraft observation fields."""

    def test_other_aircraft_fields(self, default_config: dict) -> None:
        """Each row in other_aircraft should have 9 values."""
        mgr = ObservationManager(default_config)
        own = make_state("OWN", 39.25, 116.25, 35000.0)
        other = make_state("AC001", 39.26, 116.25, 35000.0)
        goal = {"lat": 39.50, "lon": 116.50, "alt": 35000.0, "hdg": 90.0}

        result = mgr.generate(own, [other], goal)
        obs = result["observation"]

        assert obs["other_aircraft"].shape == (10, 12)
        assert obs["other_aircraft"].dtype == np.float32


class TestOtherAircraftMask:
    """Test other_aircraft_mask."""

    def test_mask_valid_position(self, default_config: dict) -> None:
        """Mask should be 1 for valid aircraft positions."""
        mgr = ObservationManager(default_config)
        own = make_state("OWN", 39.25, 116.25, 35000.0)
        other = make_state("AC001", 39.26, 116.25, 35000.0)
        goal = {"lat": 39.50, "lon": 116.50, "alt": 35000.0, "hdg": 90.0}

        result = mgr.generate(own, [other], goal)
        mask = result["observation"]["other_aircraft_mask"]

        assert mask[0] == 1
        assert mask.dtype == np.int8

    def test_mask_empty_position(self, default_config: dict) -> None:
        """Mask should be 0 for empty (padded) positions."""
        mgr = ObservationManager(default_config)
        own = make_state("OWN", 39.25, 116.25, 35000.0)
        other = make_state("AC001", 39.26, 116.25, 35000.0)
        goal = {"lat": 39.50, "lon": 116.50, "alt": 35000.0, "hdg": 90.0}

        result = mgr.generate(own, [other], goal)
        mask = result["observation"]["other_aircraft_mask"]

        assert mask[1] == 0


class TestGoalFields:
    """Test goal observation fields."""

    def test_goal_fields(self, default_config: dict) -> None:
        """goal should contain 4 normalized values."""
        mgr = ObservationManager(default_config)
        own = make_state("OWN", 39.25, 116.25, 35000.0)
        goal = {"lat": 39.50, "lon": 116.50, "alt": 35000.0, "hdg": 90.0}

        result = mgr.generate(own, [], goal)
        obs = result["observation"]

        assert obs["goal"].shape == (4,)
        assert obs["goal"].dtype == np.float32


class TestPadding:
    """Test padding behavior."""

    def test_padding_with_mask(self, default_config: dict) -> None:
        """When fewer than MAX_OBS aircraft, padded rows should have mask=0 and zero values."""
        mgr = ObservationManager(default_config)
        own = make_state("OWN", 39.25, 116.25, 35000.0)
        other = make_state("AC001", 39.26, 116.25, 35000.0)
        goal = {"lat": 39.50, "lon": 116.50, "alt": 35000.0, "hdg": 90.0}

        result = mgr.generate(own, [other], goal)
        obs = result["observation"]

        # Padded row should be all zeros
        assert np.all(obs["other_aircraft"][1] == 0.0)
        assert obs["other_aircraft_mask"][1] == 0

    def test_full_observable(self, default_config: dict) -> None:
        """When MAX_OBS aircraft are observable, mask should be all 1s."""
        mgr = ObservationManager(default_config)
        own = make_state("OWN", 39.25, 116.25, 35000.0)
        goal = {"lat": 39.50, "lon": 116.50, "alt": 35000.0, "hdg": 90.0}

        others = []
        for i in range(10):
            others.append(make_state(f"AC{i:03d}", 39.25 + (i + 1) * 0.01, 116.25, 35000.0))

        result = mgr.generate(own, others, goal)
        mask = result["observation"]["other_aircraft_mask"]

        assert np.all(mask == 1)


class TestTextualState:
    """Test textual state generation."""

    def test_textual_state_structure(self, default_config: dict) -> None:
        """textual_state should contain all required fields."""
        mgr = ObservationManager(default_config)
        own = make_state("OWN", 39.25, 116.25, 35000.0, hdg=90.0, tas=450.0)
        other = make_state("AC001", 39.26, 116.25, 35000.0)
        goal = {"lat": 39.50, "lon": 116.50, "alt": 35000.0, "hdg": 90.0}

        result = mgr.generate(own, [other], goal)
        ts = result["textual_state"]

        assert "agent_id" in ts
        assert "position" in ts
        assert "heading" in ts
        assert "altitude" in ts
        assert "speed" in ts
        assert "observable_aircraft" in ts
        assert "conflict_status" in ts
        assert "text" in ts

    def test_textual_state_text_content(self, default_config: dict) -> None:
        """Generated text should contain key information."""
        mgr = ObservationManager(default_config)
        own = make_state("OWN", 39.25, 116.25, 35000.0, hdg=90.0, tas=450.0)
        goal = {"lat": 39.50, "lon": 116.50, "alt": 35000.0, "hdg": 90.0}

        result = mgr.generate(own, [], goal)
        text = result["textual_state"]["text"]

        assert "OWN" in text
        assert "35000" in text

    def test_textual_state_conflict_status(self, default_config: dict) -> None:
        """Conflict status should be correctly marked."""
        mgr = ObservationManager(default_config)
        own = make_state("OWN", 39.25, 116.25, 35000.0)
        goal = {"lat": 39.50, "lon": 116.50, "alt": 35000.0, "hdg": 90.0}

        result = mgr.generate(own, [], goal, conflict_status="warning")

        assert result["textual_state"]["conflict_status"] == "warning"


class TestAirspaceSnapshot:
    """Test airspace snapshot generation."""

    def test_airspace_snapshot_structure(self, default_config: dict) -> None:
        """airspace_snapshot should have correct structure."""
        mgr = ObservationManager(default_config)
        own = make_state("OWN", 39.25, 116.25, 35000.0)
        other = make_state("AC001", 39.26, 116.25, 35000.0)
        goal = {"lat": 39.50, "lon": 116.50, "alt": 35000.0, "hdg": 90.0}
        airspace = {
            "sectors": [{"id": "s1", "bounds": [[39.0, 116.0], [39.5, 116.5]]}],
            "waypoints": [],
        }

        result = mgr.generate(own, [other], goal, airspace=airspace)
        snap = result["airspace_snapshot"]

        assert "sectors" in snap
        assert "waypoints" in snap
        assert "aircraft_positions" in snap
        assert "OWN" in snap["aircraft_positions"]
        assert "AC001" in snap["aircraft_positions"]


class TestObservationConsistency:
    """Test determinism."""

    def test_observation_consistency(self, default_config: dict) -> None:
        """Two consecutive calls with same input should produce identical results."""
        mgr = ObservationManager(default_config)
        own = make_state("OWN", 39.25, 116.25, 35000.0)
        other = make_state("AC001", 39.26, 116.25, 35000.0)
        goal = {"lat": 39.50, "lon": 116.50, "alt": 35000.0, "hdg": 90.0}

        r1 = mgr.generate(own, [other], goal)
        r2 = mgr.generate(own, [other], goal)

        np.testing.assert_array_equal(
            r1["observation"]["self_state"],
            r2["observation"]["self_state"],
        )
        np.testing.assert_array_equal(
            r1["observation"]["other_aircraft"],
            r2["observation"]["other_aircraft"],
        )
        assert r1["textual_state"]["text"] == r2["textual_state"]["text"]


class TestConflictState:
    """Test conflict_state one-hot encoding in observations."""

    def test_conflict_state_safe(self, default_config: dict) -> None:
        """Safe status should encode as [0, 0, 1]."""
        mgr = ObservationManager(default_config)
        own = make_state("OWN", 39.25, 116.25, 35000.0)
        goal = {"lat": 39.50, "lon": 116.50, "alt": 35000.0, "hdg": 90.0}

        result = mgr.generate(own, [], goal, conflict_status="safe")
        cs = result["observation"]["conflict_state"]

        assert cs.shape == (3,)
        assert cs.dtype == np.float32
        np.testing.assert_array_equal(cs, [0.0, 0.0, 1.0])

    def test_conflict_state_warning(self, default_config: dict) -> None:
        """Warning status should encode as [0, 1, 0]."""
        mgr = ObservationManager(default_config)
        own = make_state("OWN", 39.25, 116.25, 35000.0)
        goal = {"lat": 39.50, "lon": 116.50, "alt": 35000.0, "hdg": 90.0}

        result = mgr.generate(own, [], goal, conflict_status="warning")
        cs = result["observation"]["conflict_state"]

        np.testing.assert_array_equal(cs, [0.0, 1.0, 0.0])

    def test_conflict_state_nmac(self, default_config: dict) -> None:
        """NMAC status should encode as [1, 0, 0]."""
        mgr = ObservationManager(default_config)
        own = make_state("OWN", 39.25, 116.25, 35000.0)
        goal = {"lat": 39.50, "lon": 116.50, "alt": 35000.0, "hdg": 90.0}

        result = mgr.generate(own, [], goal, conflict_status="nmac")
        cs = result["observation"]["conflict_state"]

        np.testing.assert_array_equal(cs, [1.0, 0.0, 0.0])

    def test_conflict_state_default_safe(self, default_config: dict) -> None:
        """Default conflict_status (safe) should be used when not specified."""
        mgr = ObservationManager(default_config)
        own = make_state("OWN", 39.25, 116.25, 35000.0)
        goal = {"lat": 39.50, "lon": 116.50, "alt": 35000.0, "hdg": 90.0}

        result = mgr.generate(own, [], goal)
        cs = result["observation"]["conflict_state"]

        np.testing.assert_array_equal(cs, [0.0, 0.0, 1.0])

    def test_conflict_state_in_observation_space(self, default_config: dict) -> None:
        """conflict_state should be within observation space bounds."""
        mgr = ObservationManager(default_config)
        space = mgr.observation_space()
        own = make_state("OWN", 39.25, 116.25, 35000.0)
        goal = {"lat": 39.50, "lon": 116.50, "alt": 35000.0, "hdg": 90.0}

        for status in ["safe", "warning", "nmac"]:
            result = mgr.generate(own, [], goal, conflict_status=status)
            cs = result["observation"]["conflict_state"]
            assert space["conflict_state"].contains(cs), f"conflict_state not in space for {status}"
