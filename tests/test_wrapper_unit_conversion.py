"""Tests for BlueSkyWrapper unit conversion (feet↔meters, knots↔m/s).

BlueSky's Python API (bs.traf.cre) expects altitude in meters and speed
in m/s, but our wrapper's public interface uses feet and knots. These
tests verify the conversions happen correctly in both directions.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from bluesky_pettingzoo.bluesky.wrapper import BlueSkyWrapper

# Conversion constants matching those used in the wrapper
FT_TO_M = 0.3048
KTS_TO_MS = 1852.0 / 3600.0

try:
    from bluesky.tools.aero import vtas2cas
except ImportError:
    vtas2cas = None


class TestCreateAircraftUnitConversion:
    """create_aircraft must convert feet→meters and knots→m/s."""

    @patch("bluesky_pettingzoo.bluesky.wrapper.bs")
    def test_altitude_feet_converted_to_meters(
        self, mock_bs: MagicMock, default_config: dict
    ) -> None:
        """Altitude in feet is converted to meters before calling bs.traf.cre."""
        wrapper = BlueSkyWrapper(default_config)
        wrapper.init_simulation()

        wrapper.create_aircraft("AC001", "B737", 39.0, 116.0, 35000.0, 90.0, 450.0)

        call_args = mock_bs.traf.cre.call_args
        # bs.traf.cre(acid, actype, lat, lon, hdg, acalt, acspd)
        acalt = call_args[0][5]
        assert acalt == pytest.approx(35000.0 * FT_TO_M)

    @patch("bluesky_pettingzoo.bluesky.wrapper.bs")
    def test_speed_knots_converted_to_mps(self, mock_bs: MagicMock, default_config: dict) -> None:
        """Speed in knots is converted to m/s (then TAS→CAS) before calling bs.traf.cre."""
        wrapper = BlueSkyWrapper(default_config)
        wrapper.init_simulation()

        wrapper.create_aircraft("AC001", "B737", 39.0, 116.0, 35000.0, 90.0, 450.0)

        call_args = mock_bs.traf.cre.call_args
        acspd = call_args[0][6]
        # Wrapper converts knots→m/s then TAS→CAS at altitude
        tas_mps = 450.0 * KTS_TO_MS
        alt_m = 35000.0 * FT_TO_M
        expected_cas = vtas2cas(tas_mps, alt_m) if vtas2cas is not None else tas_mps
        assert acspd == pytest.approx(expected_cas)

    @patch("bluesky_pettingzoo.bluesky.wrapper.bs")
    def test_lat_lon_heading_unchanged(self, mock_bs: MagicMock, default_config: dict) -> None:
        """Lat, lon, and heading are passed through without conversion."""
        wrapper = BlueSkyWrapper(default_config)
        wrapper.init_simulation()

        wrapper.create_aircraft("AC001", "B737", 39.25, 116.75, 35000.0, 270.0, 450.0)

        call_args = mock_bs.traf.cre.call_args
        assert call_args[0][0] == "AC001"  # acid
        assert call_args[0][1] == "B737"  # actype
        assert call_args[0][2] == pytest.approx(39.25)  # lat
        assert call_args[0][3] == pytest.approx(116.75)  # lon
        assert call_args[0][4] == pytest.approx(270.0)  # hdg

    @patch("bluesky_pettingzoo.bluesky.wrapper.bs")
    def test_sector_cr_altitudes_converted(self, mock_bs: MagicMock, default_config: dict) -> None:
        """SectorCR staggered altitudes (31000-39000 ft) are correctly converted."""
        wrapper = BlueSkyWrapper(default_config)
        wrapper.init_simulation()

        altitudes_ft = [31000.0, 33000.0, 35000.0, 37000.0, 39000.0]
        for i, alt in enumerate(altitudes_ft):
            wrapper.create_aircraft(f"AC{i:03d}", "B737", 39.0, 116.0, alt, 90.0, 450.0)

        for i, alt_ft in enumerate(altitudes_ft):
            call_args = mock_bs.traf.cre.call_args_list[i]
            acalt = call_args[0][5]
            assert acalt == pytest.approx(alt_ft * FT_TO_M)


class TestGetAircraftStateUnitConversion:
    """get_aircraft_state must convert meters→feet and m/s→knots."""

    @patch("bluesky_pettingzoo.bluesky.wrapper.bs")
    def test_altitude_meters_converted_to_feet(
        self, mock_bs: MagicMock, default_config: dict
    ) -> None:
        """Altitude returned by BlueSky (meters) is converted to feet."""
        # BlueSky stores altitude in meters internally
        mock_bs.traf.id = np.array(["AC001"], dtype="U10")
        mock_bs.traf.lat = np.array([39.25])
        mock_bs.traf.lon = np.array([116.25])
        mock_bs.traf.alt = np.array([10668.0])  # 35000 ft in meters
        mock_bs.traf.hdg = np.array([90.0])
        mock_bs.traf.tas = np.array([231.5])  # 450 kts in m/s
        mock_bs.traf.vs = np.array([0.0])
        mock_bs.traf.id2idx.return_value = 0

        wrapper = BlueSkyWrapper(default_config)
        wrapper.init_simulation()

        state = wrapper.get_aircraft_state("AC001")

        assert state["alt"] == pytest.approx(10668.0 / FT_TO_M, rel=1e-3)

    @patch("bluesky_pettingzoo.bluesky.wrapper.bs")
    def test_speed_mps_converted_to_knots(self, mock_bs: MagicMock, default_config: dict) -> None:
        """TAS returned by BlueSky (m/s) is converted to knots."""
        mock_bs.traf.id = np.array(["AC001"], dtype="U10")
        mock_bs.traf.lat = np.array([39.25])
        mock_bs.traf.lon = np.array([116.25])
        mock_bs.traf.alt = np.array([10668.0])
        mock_bs.traf.hdg = np.array([90.0])
        mock_bs.traf.tas = np.array([231.5])  # m/s
        mock_bs.traf.vs = np.array([0.0])
        mock_bs.traf.id2idx.return_value = 0

        wrapper = BlueSkyWrapper(default_config)
        wrapper.init_simulation()

        state = wrapper.get_aircraft_state("AC001")

        assert state["tas"] == pytest.approx(231.5 / KTS_TO_MS, rel=1e-3)

    @patch("bluesky_pettingzoo.bluesky.wrapper.bs")
    def test_vs_fps_converted_to_fpm(self, mock_bs: MagicMock, default_config: dict) -> None:
        """Vertical speed returned by BlueSky (m/s) is converted to ft/min."""
        mock_bs.traf.id = np.array(["AC001"], dtype="U10")
        mock_bs.traf.lat = np.array([39.25])
        mock_bs.traf.lon = np.array([116.25])
        mock_bs.traf.alt = np.array([10668.0])
        mock_bs.traf.hdg = np.array([90.0])
        mock_bs.traf.tas = np.array([231.5])
        mock_bs.traf.vs = np.array([2.54])  # m/s
        mock_bs.traf.id2idx.return_value = 0

        wrapper = BlueSkyWrapper(default_config)
        wrapper.init_simulation()

        state = wrapper.get_aircraft_state("AC001")

        # vs: m/s → ft/min = m/s * 60 / 0.3048
        expected_vs_fpm = 2.54 * 60.0 / FT_TO_M
        assert state["vs"] == pytest.approx(expected_vs_fpm, rel=1e-3)

    @patch("bluesky_pettingzoo.bluesky.wrapper.bs")
    def test_lat_lon_heading_unchanged_in_state(
        self, mock_bs: MagicMock, default_config: dict
    ) -> None:
        """Lat, lon, hdg returned without conversion."""
        mock_bs.traf.id = np.array(["AC001"], dtype="U10")
        mock_bs.traf.lat = np.array([39.25])
        mock_bs.traf.lon = np.array([116.25])
        mock_bs.traf.alt = np.array([10668.0])
        mock_bs.traf.hdg = np.array([270.0])
        mock_bs.traf.tas = np.array([231.5])
        mock_bs.traf.vs = np.array([0.0])
        mock_bs.traf.id2idx.return_value = 0

        wrapper = BlueSkyWrapper(default_config)
        wrapper.init_simulation()

        state = wrapper.get_aircraft_state("AC001")

        assert state["lat"] == pytest.approx(39.25)
        assert state["lon"] == pytest.approx(116.25)
        assert state["hdg"] == pytest.approx(270.0)


class TestGetAllAircraftStatesUnitConversion:
    """get_all_aircraft_states must apply the same unit conversions."""

    @patch("bluesky_pettingzoo.bluesky.wrapper.bs")
    def test_all_states_converted(self, mock_bs: MagicMock, default_config: dict) -> None:
        """All aircraft states have alt in feet and tas in knots."""
        mock_bs.traf.id = np.array(["AC001", "AC002"], dtype="U10")
        mock_bs.traf.lat = np.array([39.25, 39.30])
        mock_bs.traf.lon = np.array([116.25, 116.30])
        mock_bs.traf.alt = np.array([10668.0, 10363.2])  # 35000 ft, 34000 ft in meters
        mock_bs.traf.hdg = np.array([90.0, 270.0])
        mock_bs.traf.tas = np.array([231.5, 226.3])  # m/s
        mock_bs.traf.vs = np.array([0.0, 0.0])

        wrapper = BlueSkyWrapper(default_config)
        wrapper.init_simulation()

        states = wrapper.get_all_aircraft_states()

        assert states["AC001"]["alt"] == pytest.approx(10668.0 / FT_TO_M, rel=1e-3)
        assert states["AC002"]["alt"] == pytest.approx(10363.2 / FT_TO_M, rel=1e-3)
        assert states["AC001"]["tas"] == pytest.approx(231.5 / KTS_TO_MS, rel=1e-3)
        assert states["AC002"]["tas"] == pytest.approx(226.3 / KTS_TO_MS, rel=1e-3)


class TestRoundTripConsistency:
    """Verify that create → state → NMAC check is consistent."""

    @patch("bluesky_pettingzoo.bluesky.wrapper.bs")
    def test_staggered_altitudes_preserved_through_roundtrip(
        self, mock_bs: MagicMock, default_config: dict
    ) -> None:
        """Altitudes created in feet return as feet after roundtrip."""
        wrapper = BlueSkyWrapper(default_config)
        wrapper.init_simulation()

        # Simulate what SectorCR does: staggered altitudes
        altitudes_ft = [31000.0, 33000.0, 35000.0]

        # After create_aircraft, BlueSky stores in meters
        stored_alts_m = [alt * FT_TO_M for alt in altitudes_ft]

        # Mock get_aircraft_state to return what BlueSky would store
        mock_bs.traf.id = np.array(["AC000", "AC001", "AC002"], dtype="U10")
        mock_bs.traf.lat = np.array([39.0, 39.01, 39.02])
        mock_bs.traf.lon = np.array([116.0, 116.01, 116.02])
        mock_bs.traf.alt = np.array(stored_alts_m)
        mock_bs.traf.hdg = np.array([90.0, 90.0, 90.0])
        mock_bs.traf.tas = np.array([231.5, 231.5, 231.5])
        mock_bs.traf.vs = np.array([0.0, 0.0, 0.0])
        mock_bs.traf.id2idx.side_effect = lambda acid: int(acid[-1])

        states = wrapper.get_all_aircraft_states()

        # After roundtrip, altitudes should match original feet values
        assert states["AC000"]["alt"] == pytest.approx(31000.0, rel=1e-3)
        assert states["AC001"]["alt"] == pytest.approx(33000.0, rel=1e-3)
        assert states["AC002"]["alt"] == pytest.approx(35000.0, rel=1e-3)

        # Vertical separations should be preserved
        v_dist_01 = abs(states["AC000"]["alt"] - states["AC001"]["alt"])
        v_dist_12 = abs(states["AC001"]["alt"] - states["AC002"]["alt"])
        assert v_dist_01 == pytest.approx(2000.0, rel=1e-3)
        assert v_dist_12 == pytest.approx(2000.0, rel=1e-3)
