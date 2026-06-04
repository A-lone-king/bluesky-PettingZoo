"""Tests for HorizontalCRScenario.create_intruders() using creconfs."""

from __future__ import annotations

import numpy as np
import pytest

from bluesky_pettingzoo.bluesky.wrapper import BlueSkyWrapper
from bluesky_pettingzoo.envs.scenarios.horizontal_cr import HorizontalCRScenario
from tests.helpers.env_factory import make_config


@pytest.fixture
def scenario_and_wrapper():
    config = make_config(initial_count=5)
    wrapper = BlueSkyWrapper(config)
    wrapper.init_simulation()
    scenario = HorizontalCRScenario(num_aircraft=5, seed=42)
    bounds = {"lat_min": 39.0, "lat_max": 41.0, "lon_min": 116.0, "lon_max": 118.0}
    rng = np.random.RandomState(42)
    scenario.setup(rng, bounds)
    return scenario, wrapper, rng


class TestHorizontalCRCreateIntruders:
    """Verify HorizontalCR uses creconfs to generate intruders."""

    def test_create_intruders_returns_id_list(self, scenario_and_wrapper):
        scenario, wrapper, rng = scenario_and_wrapper
        result = scenario.create_intruders(wrapper, rng)
        assert isinstance(result, list)
        assert len(result) > 0

    def test_create_intruders_creates_aircraft(self, scenario_and_wrapper):
        scenario, wrapper, rng = scenario_and_wrapper
        ids = scenario.create_intruders(wrapper, rng)
        active = wrapper.get_active_aircraft_ids()
        for acid in ids:
            assert acid in active

    def test_create_intruders_count(self, scenario_and_wrapper):
        scenario, wrapper, rng = scenario_and_wrapper
        ids = scenario.create_intruders(wrapper, rng)
        # 1 ownship + (num_aircraft - 1) intruders = num_aircraft total
        assert len(ids) == 5

    def test_intruders_have_valid_state(self, scenario_and_wrapper):
        scenario, wrapper, rng = scenario_and_wrapper
        ids = scenario.create_intruders(wrapper, rng)
        for acid in ids:
            st = wrapper.get_aircraft_state(acid)
            assert "lat" in st
            assert "lon" in st
            assert "alt" in st
