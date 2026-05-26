"""Tests for BlueSkyWrapper.create_conflict_aircraft() — creconfs封装."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from bluesky_pettingzoo.bluesky.wrapper import BlueSkyWrapper


@pytest.fixture
def wrapper():
    config = {
        "simulation": {"dt": 5.0},
        "airspace": {"sectors": []},
    }
    return BlueSkyWrapper(config)


class TestCreateConflictAircraft:
    """Verify create_conflict_aircraft wraps bs.traf.creconfs."""

    @patch("bluesky_pettingzoo.bluesky.wrapper.bs")
    def test_returns_ownship_plus_intruders(self, mock_bs, wrapper):
        mock_bs.traf.creconfs = MagicMock()
        result = wrapper.create_conflict_aircraft(
            ownship_lat=40.0, ownship_lon=117.0, ownship_alt=35000.0,
            ownship_hdg=90.0, ownship_spd=450.0,
            count=3, dpsi=45.0, dcpa=5.0,
        )
        assert isinstance(result, list)
        assert len(result) == 4  # 1 ownship + 3 intruders
        assert result[0] == "CR000"  # ownship
        assert result[1:] == ["CR001", "CR002", "CR003"]  # intruders

    @patch("bluesky_pettingzoo.bluesky.wrapper.bs")
    def test_calls_creconfs_for_each_intruder(self, mock_bs, wrapper):
        mock_bs.traf.creconfs = MagicMock()
        wrapper.create_conflict_aircraft(
            ownship_lat=40.0, ownship_lon=117.0, ownship_alt=35000.0,
            ownship_hdg=90.0, ownship_spd=450.0,
            count=3, dpsi=30.0, dcpa=5.0,
        )
        assert mock_bs.traf.creconfs.call_count == 3

    @patch("bluesky_pettingzoo.bluesky.wrapper.bs")
    def test_creates_ownship_via_cre(self, mock_bs, wrapper):
        mock_bs.traf.cre = MagicMock()
        mock_bs.traf.creconfs = MagicMock()
        wrapper.create_conflict_aircraft(
            ownship_lat=40.0, ownship_lon=117.0, ownship_alt=35000.0,
            ownship_hdg=90.0, ownship_spd=450.0,
            count=2, dpsi=45.0, dcpa=5.0,
        )
        mock_bs.traf.cre.assert_called_once()

    @patch("bluesky_pettingzoo.bluesky.wrapper.bs")
    def test_passes_dH_and_tlosv(self, mock_bs, wrapper):
        mock_bs.traf.creconfs = MagicMock()
        result = wrapper.create_conflict_aircraft(
            ownship_lat=40.0, ownship_lon=117.0, ownship_alt=35000.0,
            ownship_hdg=90.0, ownship_spd=450.0,
            count=1, dpsi=30.0, dcpa=5.0, dH=1000.0, tlosv=60.0,
        )
        assert len(result) == 2  # ownship + 1 intruder
        call_kwargs = mock_bs.traf.creconfs.call_args[1]
        assert call_kwargs["dH"] == 1000.0
        assert call_kwargs["tlosv"] == 60.0

    @patch("bluesky_pettingzoo.bluesky.wrapper.bs")
    def test_default_params(self, mock_bs, wrapper):
        mock_bs.traf.creconfs = MagicMock()
        result = wrapper.create_conflict_aircraft(
            ownship_lat=40.0, ownship_lon=117.0, ownship_alt=35000.0,
            ownship_hdg=90.0, ownship_spd=450.0,
        )
        assert isinstance(result, list)
        assert len(result) == 6  # 1 ownship + 5 default intruders

    @patch("bluesky_pettingzoo.bluesky.wrapper.bs")
    def test_per_intruder_dpsi_list(self, mock_bs, wrapper):
        """Per-intruder dpsi_list overrides scalar dpsi."""
        mock_bs.traf.creconfs = MagicMock()
        wrapper.create_conflict_aircraft(
            ownship_lat=40.0, ownship_lon=117.0, ownship_alt=35000.0,
            ownship_hdg=90.0, ownship_spd=450.0,
            count=3, dpsi_list=[30.0, 60.0, 90.0], dcpa=5.0,
        )
        calls = mock_bs.traf.creconfs.call_args_list
        assert calls[0][0][3] == 30.0  # dpsi for intruder 0
        assert calls[1][0][3] == 60.0  # dpsi for intruder 1
        assert calls[2][0][3] == 90.0  # dpsi for intruder 2

    @patch("bluesky_pettingzoo.bluesky.wrapper.bs")
    def test_per_intruder_dcpa_list(self, mock_bs, wrapper):
        """Per-intruder dcpa_list overrides scalar dcpa."""
        mock_bs.traf.creconfs = MagicMock()
        wrapper.create_conflict_aircraft(
            ownship_lat=40.0, ownship_lon=117.0, ownship_alt=35000.0,
            ownship_hdg=90.0, ownship_spd=450.0,
            count=2, dpsi=45.0, dcpa_list=[2.0, 8.0],
        )
        calls = mock_bs.traf.creconfs.call_args_list
        assert calls[0][0][4] == 2.0  # dcpa for intruder 0
        assert calls[1][0][4] == 8.0  # dcpa for intruder 1

    @patch("bluesky_pettingzoo.bluesky.wrapper.bs")
    def test_per_intruder_tlosh_list(self, mock_bs, wrapper):
        """Per-intruder tlosh_list overrides scalar tlosh."""
        mock_bs.traf.creconfs = MagicMock()
        wrapper.create_conflict_aircraft(
            ownship_lat=40.0, ownship_lon=117.0, ownship_alt=35000.0,
            ownship_hdg=90.0, ownship_spd=450.0,
            count=2, dpsi=45.0, dcpa=5.0, tlosh_list=[100.0, 500.0],
        )
        calls = mock_bs.traf.creconfs.call_args_list
        assert calls[0][0][5] == 100.0  # tlosh for intruder 0
        assert calls[1][0][5] == 500.0  # tlosh for intruder 1

    @patch("bluesky_pettingzoo.bluesky.wrapper.bs")
    def test_per_intruder_backward_compatible(self, mock_bs, wrapper):
        """Scalar params still work when lists are not provided."""
        mock_bs.traf.creconfs = MagicMock()
        wrapper.create_conflict_aircraft(
            ownship_lat=40.0, ownship_lon=117.0, ownship_alt=35000.0,
            ownship_hdg=90.0, ownship_spd=450.0,
            count=2, dpsi=45.0, dcpa=5.0, tlosh=120.0,
        )
        calls = mock_bs.traf.creconfs.call_args_list
        # Both intruders should get the same dcpa and tlosh
        assert calls[0][0][4] == 5.0
        assert calls[1][0][4] == 5.0
        assert calls[0][0][5] == 120.0
        assert calls[1][0][5] == 120.0
