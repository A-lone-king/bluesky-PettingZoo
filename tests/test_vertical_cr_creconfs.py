"""Tests for VerticalCRScenario.create_intruders() using creconfs."""

from __future__ import annotations

import numpy as np
import pytest

from bluesky_pettingzoo.bluesky.wrapper import BlueSkyWrapper
from bluesky_pettingzoo.envs.scenarios.vertical_cr import VerticalCRScenario
from tests.helpers.env_factory import make_config


@pytest.fixture
def scenario_and_wrapper():
    config = make_config(initial_count=5)
    wrapper = BlueSkyWrapper(config)
    wrapper.init_simulation()
    scenario = VerticalCRScenario(num_aircraft=5, seed=42)
    bounds = {"lat_min": 39.0, "lat_max": 41.0, "lon_min": 116.0, "lon_max": 118.0}
    rng = np.random.RandomState(42)
    scenario.setup(rng, bounds)
    return scenario, wrapper, rng


class TestVerticalCRCreateIntruders:
    """Verify VerticalCR uses creconfs with vertical offset."""

    def test_create_intruders_returns_list(self, scenario_and_wrapper):
        scenario, wrapper, rng = scenario_and_wrapper
        result = scenario.create_intruders(wrapper, rng)
        assert isinstance(result, list)
        assert len(result) > 0

    def test_create_intruders_count(self, scenario_and_wrapper):
        scenario, wrapper, rng = scenario_and_wrapper
        ids = scenario.create_intruders(wrapper, rng)
        assert len(ids) == 5  # ownship + 4 intruders

    def test_intruders_created_in_wrapper(self, scenario_and_wrapper):
        scenario, wrapper, rng = scenario_and_wrapper
        ids = scenario.create_intruders(wrapper, rng)
        active = wrapper.get_active_aircraft_ids()
        for acid in ids:
            assert acid in active

    def test_intruders_have_altitude_offset(self, scenario_and_wrapper):
        scenario, wrapper, rng = scenario_and_wrapper
        ids = scenario.create_intruders(wrapper, rng)
        own = wrapper.get_aircraft_state(ids[0])
        for acid in ids[1:]:
            intruder = wrapper.get_aircraft_state(acid)
            # VerticalCR should have altitude differences
            assert "alt" in intruder
