"""Tests for BlueSky wrapper module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from bluesky_pettingzoo.bluesky.wrapper import BlueSkyWrapper


class TestLazyImport:
    """Test that BlueSkyWrapper can be imported without bluesky installed."""

    def test_init_simulation_raises_if_bluesky_missing(self, default_config: dict) -> None:
        """init_simulation() raises ImportError when bluesky is not installed."""
        with patch("bluesky_pettingzoo.bluesky.wrapper.bs", None):
            wrapper = BlueSkyWrapper(default_config)
            with pytest.raises(ImportError, match="bluesky"):
                wrapper.init_simulation()


class TestBlueSkyWrapperInit:
    """Test BlueSkyWrapper initialization."""

    def test_init_with_config(self, default_config: dict) -> None:
        """Test initialization with configuration."""
        wrapper = BlueSkyWrapper(default_config)
        assert wrapper.config == default_config
        assert wrapper._initialized is False

    def test_init_default_dt(self, default_config: dict) -> None:
        """Test default dt from config."""
        wrapper = BlueSkyWrapper(default_config)
        assert wrapper.dt == default_config["simulation"]["dt"]

    def test_init_custom_dt(self, default_config: dict) -> None:
        """Test custom dt override."""
        default_config["simulation"]["dt"] = 10.0
        wrapper = BlueSkyWrapper(default_config)
        assert wrapper.dt == 10.0


class TestBlueSkyWrapperSimulation:
    """Test BlueSkyWrapper simulation lifecycle."""

    @patch("bluesky_pettingzoo.bluesky.wrapper.bs")
    def test_init_simulation(self, mock_bs: MagicMock, default_config: dict) -> None:
        """Test headless simulation initialization."""
        wrapper = BlueSkyWrapper(default_config)
        wrapper.init_simulation()

        mock_bs.init.assert_called_once_with(mode="sim", detached=True)
        assert wrapper._initialized is True

    @patch("bluesky_pettingzoo.bluesky.wrapper.bs")
    def test_init_simulation_called_once(self, mock_bs: MagicMock, default_config: dict) -> None:
        """Test that init_simulation is idempotent."""
        wrapper = BlueSkyWrapper(default_config)
        wrapper.init_simulation()
        wrapper.init_simulation()

        mock_bs.init.assert_called_once()

    @patch("bluesky_pettingzoo.bluesky.wrapper.bs")
    def test_step_advances_time(self, mock_bs: MagicMock, default_config: dict) -> None:
        """Test that step advances simulation time."""
        mock_bs.sim.simt = 0.0
        wrapper = BlueSkyWrapper(default_config)
        wrapper.init_simulation()

        wrapper.step()

        mock_bs.sim.step.assert_called_once()

    @patch("bluesky_pettingzoo.bluesky.wrapper.bs")
    def test_step_returns_sim_time(self, mock_bs: MagicMock, default_config: dict) -> None:
        """Test that step returns current simulation time."""
        mock_bs.sim.simt = 5.0
        wrapper = BlueSkyWrapper(default_config)
        wrapper.init_simulation()

        result = wrapper.step()
        assert result == 5.0

    @patch("bluesky_pettingzoo.bluesky.wrapper.bs")
    def test_reset_clears_state(self, mock_bs: MagicMock, default_config: dict) -> None:
        """Test that reset clears all aircraft."""
        mock_bs.traf.id = ["AC000", "AC001"]
        wrapper = BlueSkyWrapper(default_config)
        wrapper.init_simulation()

        wrapper.reset()

        # Should delete each aircraft individually
        calls = mock_bs.stack.stack.call_args_list
        delete_calls = [c for c in calls if "DELETE" in str(c)]
        assert len(delete_calls) == 2


class TestBlueSkyWrapperAircraft:
    """Test BlueSkyWrapper aircraft operations."""

    @patch("bluesky_pettingzoo.bluesky.wrapper.bs")
    def test_create_aircraft(self, mock_bs: MagicMock, default_config: dict) -> None:
        """Test aircraft creation."""
        mock_bs.traf.id = np.array(["AC001"], dtype="U10")
        mock_bs.traf.lat = np.array([39.25])
        mock_bs.traf.lon = np.array([116.25])
        mock_bs.traf.alt = np.array([35000.0])
        mock_bs.traf.hdg = np.array([90.0])
        mock_bs.traf.tas = np.array([450.0])
        mock_bs.traf.vs = np.array([0.0])

        wrapper = BlueSkyWrapper(default_config)
        wrapper.init_simulation()

        wrapper.create_aircraft("AC001", "B737", 39.25, 116.25, 35000.0, 90.0, 450.0)

        mock_bs.traf.cre.assert_called_once()
        call_args = mock_bs.traf.cre.call_args
        assert call_args[0][0] == "AC001"  # acid
        assert call_args[0][1] == "B737"  # actype

    @patch("bluesky_pettingzoo.bluesky.wrapper.bs")
    def test_remove_aircraft(self, mock_bs: MagicMock, default_config: dict) -> None:
        """Test aircraft removal."""
        mock_bs.traf.id = np.array(["AC001"], dtype="U10")

        wrapper = BlueSkyWrapper(default_config)
        wrapper.init_simulation()

        wrapper.remove_aircraft("AC001")

        mock_bs.stack.stack.assert_called_with("DELETE AC001")

    @patch("bluesky_pettingzoo.bluesky.wrapper.bs")
    def test_get_aircraft_state(self, mock_bs: MagicMock, default_config: dict) -> None:
        """Test getting aircraft state."""
        mock_bs.traf.id = np.array(["AC001"], dtype="U10")
        mock_bs.traf.lat = np.array([39.25])
        mock_bs.traf.lon = np.array([116.25])
        mock_bs.traf.alt = np.array([35000.0])
        mock_bs.traf.hdg = np.array([90.0])
        mock_bs.traf.tas = np.array([450.0])
        mock_bs.traf.vs = np.array([0.0])
        mock_bs.traf.id2idx.return_value = 0

        wrapper = BlueSkyWrapper(default_config)
        wrapper.init_simulation()

        state = wrapper.get_aircraft_state("AC001")

        assert state["id"] == "AC001"
        assert state["lat"] == pytest.approx(39.25)
        assert state["lon"] == pytest.approx(116.25)
        assert state["alt"] == pytest.approx(35000.0)
        assert state["hdg"] == pytest.approx(90.0)
        assert state["tas"] == pytest.approx(450.0)
        assert state["vs"] == pytest.approx(0.0)

    @patch("bluesky_pettingzoo.bluesky.wrapper.bs")
    def test_get_aircraft_state_not_found(self, mock_bs: MagicMock, default_config: dict) -> None:
        """Test getting state of non-existent aircraft."""
        mock_bs.traf.id = np.array([], dtype="U10")
        mock_bs.traf.id2idx.return_value = np.array([])

        wrapper = BlueSkyWrapper(default_config)
        wrapper.init_simulation()

        with pytest.raises(ValueError, match="Aircraft .* not found"):
            wrapper.get_aircraft_state("AC999")

    @patch("bluesky_pettingzoo.bluesky.wrapper.bs")
    def test_get_all_aircraft_states(self, mock_bs: MagicMock, default_config: dict) -> None:
        """Test getting all aircraft states."""
        mock_bs.traf.id = np.array(["AC001", "AC002"], dtype="U10")
        mock_bs.traf.lat = np.array([39.25, 39.30])
        mock_bs.traf.lon = np.array([116.25, 116.30])
        mock_bs.traf.alt = np.array([35000.0, 34000.0])
        mock_bs.traf.hdg = np.array([90.0, 270.0])
        mock_bs.traf.tas = np.array([450.0, 440.0])
        mock_bs.traf.vs = np.array([0.0, 0.0])

        wrapper = BlueSkyWrapper(default_config)
        wrapper.init_simulation()

        states = wrapper.get_all_aircraft_states()

        assert len(states) == 2
        assert "AC001" in states
        assert "AC002" in states
        assert states["AC001"]["lat"] == pytest.approx(39.25)
        assert states["AC002"]["lat"] == pytest.approx(39.30)

    @patch("bluesky_pettingzoo.bluesky.wrapper.bs")
    def test_get_active_aircraft_ids(self, mock_bs: MagicMock, default_config: dict) -> None:
        """Test getting active aircraft IDs."""
        mock_bs.traf.id = np.array(["AC001", "AC002", "AC003"], dtype="U10")

        wrapper = BlueSkyWrapper(default_config)
        wrapper.init_simulation()

        ids = wrapper.get_active_aircraft_ids()

        assert ids == ["AC001", "AC002", "AC003"]

    @patch("bluesky_pettingzoo.bluesky.wrapper.bs")
    def test_get_active_aircraft_ids_empty(self, mock_bs: MagicMock, default_config: dict) -> None:
        """Test getting active aircraft IDs when none exist."""
        mock_bs.traf.id = np.array([], dtype="U10")

        wrapper = BlueSkyWrapper(default_config)
        wrapper.init_simulation()

        ids = wrapper.get_active_aircraft_ids()

        assert ids == []


class TestBlueSkyWrapperCommands:
    """Test BlueSkyWrapper command operations."""

    @patch("bluesky_pettingzoo.bluesky.wrapper.bs")
    def test_send_command(self, mock_bs: MagicMock, default_config: dict) -> None:
        """Test sending a single command."""
        wrapper = BlueSkyWrapper(default_config)
        wrapper.init_simulation()

        wrapper.send_command("HDG AC001 90")

        mock_bs.stack.stack.assert_called_with("HDG AC001 90")

    @patch("bluesky_pettingzoo.bluesky.wrapper.bs")
    def test_send_commands_batch(self, mock_bs: MagicMock, default_config: dict) -> None:
        """Test sending batch commands."""
        wrapper = BlueSkyWrapper(default_config)
        wrapper.init_simulation()
        mock_bs.stack.stack.reset_mock()

        commands = ["HDG AC001 90", "ALT AC001 35000", "SPD AC001 450"]
        wrapper.send_commands_batch(commands)

        assert mock_bs.stack.stack.call_count == 3
        calls = [call[0][0] for call in mock_bs.stack.stack.call_args_list]
        assert calls == commands

    @patch("bluesky_pettingzoo.bluesky.wrapper.bs")
    def test_send_commands_batch_empty(self, mock_bs: MagicMock, default_config: dict) -> None:
        """Test sending empty batch commands."""
        wrapper = BlueSkyWrapper(default_config)
        wrapper.init_simulation()
        mock_bs.stack.stack.reset_mock()

        wrapper.send_commands_batch([])

        mock_bs.stack.stack.assert_not_called()


class TestBlueSkyWrapperAirspace:
    """Test BlueSkyWrapper airspace operations."""

    @patch("bluesky_pettingzoo.bluesky.wrapper.bs")
    def test_is_aircraft_in_airspace_true(self, mock_bs: MagicMock, default_config: dict) -> None:
        """Test aircraft within airspace bounds."""
        mock_bs.traf.id = np.array(["AC001"], dtype="U10")
        mock_bs.traf.lat = np.array([39.25])
        mock_bs.traf.lon = np.array([116.25])
        mock_bs.traf.id2idx.return_value = 0

        wrapper = BlueSkyWrapper(default_config)
        wrapper.init_simulation()

        assert wrapper.is_aircraft_in_airspace("AC001") is True

    @patch("bluesky_pettingzoo.bluesky.wrapper.bs")
    def test_is_aircraft_in_airspace_false(self, mock_bs: MagicMock, default_config: dict) -> None:
        """Test aircraft outside airspace bounds."""
        mock_bs.traf.id = np.array(["AC001"], dtype="U10")
        mock_bs.traf.lat = np.array([40.0])  # Outside bounds
        mock_bs.traf.lon = np.array([116.25])
        mock_bs.traf.id2idx.return_value = 0

        wrapper = BlueSkyWrapper(default_config)
        wrapper.init_simulation()

        assert wrapper.is_aircraft_in_airspace("AC001") is False

    @patch("bluesky_pettingzoo.bluesky.wrapper.bs")
    def test_is_aircraft_in_airspace_not_found(self, mock_bs: MagicMock, default_config: dict) -> None:
        """Test non-existent aircraft not in airspace."""
        mock_bs.traf.id = np.array([], dtype="U10")
        mock_bs.traf.id2idx.return_value = np.array([])

        wrapper = BlueSkyWrapper(default_config)
        wrapper.init_simulation()

        assert wrapper.is_aircraft_in_airspace("AC999") is False

    @patch("bluesky_pettingzoo.bluesky.wrapper.bs")
    def test_close(self, mock_bs: MagicMock, default_config: dict) -> None:
        """Test closing the wrapper deletes managed aircraft."""
        wrapper = BlueSkyWrapper(default_config)
        wrapper.init_simulation()

        wrapper.create_aircraft("AC000", "B737", 39.0, 116.0, 35000, 90, 450)
        mock_bs.stack.stack.reset_mock()

        wrapper.close()

        assert wrapper._initialized is False
        calls = mock_bs.stack.stack.call_args_list
        delete_calls = [c for c in calls if "DELETE" in str(c)]
        assert len(delete_calls) >= 1


class TestCloseScoping:
    """Test that close() only deletes managed aircraft."""

    @patch("bluesky_pettingzoo.bluesky.wrapper.bs")
    def test_close_only_deletes_managed_aircraft(self, mock_bs: MagicMock, default_config: dict) -> None:
        """close() only deletes aircraft created by this wrapper instance."""
        mock_bs.traf.id = np.array(["AC001", "EXTERNAL001"], dtype="U10")
        mock_bs.traf.cre = MagicMock()
        mock_bs.sim.simt = 0.0

        wrapper = BlueSkyWrapper(default_config)
        wrapper.init_simulation()

        # Only create AC001 through the wrapper
        wrapper.create_aircraft("AC001", "B737", 39.0, 116.0, 35000, 90, 450)
        mock_bs.stack.stack.reset_mock()

        wrapper.close()

        # Should only delete AC001, not EXTERNAL001
        calls = mock_bs.stack.stack.call_args_list
        delete_calls = [c for c in calls if "DELETE" in str(c)]
        assert len(delete_calls) == 1
        assert "AC001" in str(delete_calls[0])

    @patch("bluesky_pettingzoo.bluesky.wrapper.bs")
    def test_create_aircraft_tracks_managed(self, mock_bs: MagicMock, default_config: dict) -> None:
        """create_aircraft adds to managed set."""
        wrapper = BlueSkyWrapper(default_config)
        wrapper.init_simulation()

        wrapper.create_aircraft("AC001", "B737", 39.0, 116.0, 35000, 90, 450)
        assert "AC001" in wrapper._managed_aircraft

    @patch("bluesky_pettingzoo.bluesky.wrapper.bs")
    def test_remove_aircraft_untracks_managed(self, mock_bs: MagicMock, default_config: dict) -> None:
        """remove_aircraft removes from managed set."""
        wrapper = BlueSkyWrapper(default_config)
        wrapper.init_simulation()

        wrapper.create_aircraft("AC001", "B737", 39.0, 116.0, 35000, 90, 450)
        wrapper.remove_aircraft("AC001")
        assert "AC001" not in wrapper._managed_aircraft


class TestResolveIdx:
    """Test _resolve_idx handles different BlueSky return types."""

    @patch("bluesky_pettingzoo.bluesky.wrapper.bs")
    def test_resolve_idx_returns_int(self, mock_bs: MagicMock, default_config: dict) -> None:
        """id2idx returning int is passed through."""
        mock_bs.traf.id = np.array(["AC001"], dtype="U10")
        mock_bs.traf.id2idx.return_value = 3

        wrapper = BlueSkyWrapper(default_config)
        wrapper.init_simulation()

        assert wrapper._resolve_idx("AC001") == 3

    @patch("bluesky_pettingzoo.bluesky.wrapper.bs")
    def test_resolve_idx_handles_ndarray(self, mock_bs: MagicMock, default_config: dict) -> None:
        """id2idx returning ndarray extracts first element."""
        mock_bs.traf.id = np.array(["AC001"], dtype="U10")
        mock_bs.traf.id2idx.return_value = np.array([5])

        wrapper = BlueSkyWrapper(default_config)
        wrapper.init_simulation()

        assert wrapper._resolve_idx("AC001") == 5

    @patch("bluesky_pettingzoo.bluesky.wrapper.bs")
    def test_resolve_idx_empty_ndarray_returns_neg1(self, mock_bs: MagicMock, default_config: dict) -> None:
        """id2idx returning empty ndarray gives -1."""
        mock_bs.traf.id = np.array([], dtype="U10")
        mock_bs.traf.id2idx.return_value = np.array([])

        wrapper = BlueSkyWrapper(default_config)
        wrapper.init_simulation()

        assert wrapper._resolve_idx("AC999") == -1

    @patch("bluesky_pettingzoo.bluesky.wrapper.bs")
    def test_resolve_idx_handles_list(self, mock_bs: MagicMock, default_config: dict) -> None:
        """id2idx returning list extracts first element."""
        mock_bs.traf.id = np.array(["AC001"], dtype="U10")
        mock_bs.traf.id2idx.return_value = [2]

        wrapper = BlueSkyWrapper(default_config)
        wrapper.init_simulation()

        assert wrapper._resolve_idx("AC001") == 2
