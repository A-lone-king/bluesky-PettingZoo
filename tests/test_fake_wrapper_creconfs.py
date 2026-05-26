"""Tests for BlueSkyWrapper.create_conflict_aircraft()."""

from __future__ import annotations

from pathlib import Path

import pytest

from bluesky_pettingzoo.bluesky.wrapper import BlueSkyWrapper
from tests.helpers.env_factory import make_config


@pytest.fixture
def wrapper():
    config = make_config()
    w = BlueSkyWrapper(config)
    w.init_simulation()
    w.reset()
    yield w
    w.close()


class TestFakeWrapperCreconfs:
    """Verify BlueSkyWrapper simulates creconfs behavior."""

    def test_creconfs_returns_id_list(self, wrapper):
        result = wrapper.create_conflict_aircraft(
            ownship_lat=40.0, ownship_lon=117.0, ownship_alt=35000.0,
            ownship_hdg=90.0, ownship_spd=450.0,
            count=3, dpsi=45.0, dcpa=5.0,
        )
        assert isinstance(result, list)
        assert len(result) == 4  # ownship + 3 intruders

    def test_creconfs_creates_aircraft_in_wrapper(self, wrapper):
        wrapper.create_conflict_aircraft(
            ownship_lat=40.0, ownship_lon=117.0, ownship_alt=35000.0,
            ownship_hdg=90.0, ownship_spd=450.0,
            count=2, dpsi=45.0, dcpa=5.0,
        )
        ids = wrapper.get_active_aircraft_ids()
        assert len(ids) == 3  # ownship + 2 intruders

    def test_creconfs_ownship_at_correct_position(self, wrapper):
        result = wrapper.create_conflict_aircraft(
            ownship_lat=40.0, ownship_lon=117.0, ownship_alt=35000.0,
            ownship_hdg=90.0, ownship_spd=450.0,
            count=1, dpsi=30.0, dcpa=5.0,
        )
        own_id = result[0]
        st = wrapper.get_aircraft_state(own_id)
        assert abs(st["lat"] - 40.0) < 0.01
        assert abs(st["lon"] - 117.0) < 0.01
        assert abs(st["alt"] - 35000.0) < 1.0

    def test_creconfs_intruders_exist_in_state(self, wrapper):
        result = wrapper.create_conflict_aircraft(
            ownship_lat=40.0, ownship_lon=117.0, ownship_alt=35000.0,
            ownship_hdg=90.0, ownship_spd=450.0,
            count=2, dpsi=45.0, dcpa=5.0,
        )
        for acid in result:
            st = wrapper.get_aircraft_state(acid)
            assert st["id"] == acid
