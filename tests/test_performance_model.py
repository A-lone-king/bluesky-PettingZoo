"""Tests for performance model integration (OpenAP/BADA)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from bluesky_pettingzoo.bluesky.wrapper import BlueSkyWrapper


@pytest.fixture(autouse=True)
def _reset_bs_global_flag():
    """Reset the module-level bs init flag before each test."""
    import bluesky_pettingzoo.bluesky.wrapper as w

    w._bs_global_initialized = False
    yield


class TestPerformanceModelConfig:
    """Test performance model configuration loading."""

    def test_default_config_has_performance_model(self, default_config: dict) -> None:
        """default.yaml includes performance_model key."""
        assert "performance_model" in default_config["simulation"]

    def test_default_performance_model_is_openap(self, default_config: dict) -> None:
        """Default performance model is openap."""
        assert default_config["simulation"]["performance_model"] == "openap"


class TestPerformanceModelActivation:
    """Test performance model activation via PERF command."""

    @patch("bluesky_pettingzoo.bluesky.wrapper.bs")
    def test_init_simulation_sends_perf_command(
        self, mock_bs: MagicMock, default_config: dict
    ) -> None:
        """init_simulation() sends PERF openap command."""
        wrapper = BlueSkyWrapper(default_config)
        wrapper.init_simulation()

        # Check that PERF openap was stacked
        perf_calls = [call for call in mock_bs.stack.stack.call_args_list if "PERF" in str(call)]
        assert len(perf_calls) >= 1

    @patch("bluesky_pettingzoo.bluesky.wrapper.bs")
    def test_init_simulation_no_perf_when_off(
        self, mock_bs: MagicMock, default_config: dict
    ) -> None:
        """init_simulation() skips PERF command when performance_model is off."""
        default_config["simulation"]["performance_model"] = "off"
        wrapper = BlueSkyWrapper(default_config)
        wrapper.init_simulation()

        perf_calls = [call for call in mock_bs.stack.stack.call_args_list if "PERF" in str(call)]
        assert len(perf_calls) == 0

    @patch("bluesky_pettingzoo.bluesky.wrapper.bs")
    def test_init_simulation_bada_model(self, mock_bs: MagicMock, default_config: dict) -> None:
        """init_simulation() sends PERF bada when configured."""
        default_config["simulation"]["performance_model"] = "bada"
        wrapper = BlueSkyWrapper(default_config)
        wrapper.init_simulation()

        perf_calls = [
            call for call in mock_bs.stack.stack.call_args_list if "PERF bada" in str(call)
        ]
        assert len(perf_calls) >= 1

    @patch("bluesky_pettingzoo.bluesky.wrapper.bs")
    def test_init_simulation_missing_config_defaults_to_openap(
        self, mock_bs: MagicMock, default_config: dict
    ) -> None:
        """init_simulation() defaults to openap when performance_model is not set."""
        del default_config["simulation"]["performance_model"]
        wrapper = BlueSkyWrapper(default_config)
        wrapper.init_simulation()

        perf_calls = [
            call for call in mock_bs.stack.stack.call_args_list if "PERF openap" in str(call)
        ]
        assert len(perf_calls) >= 1


class TestSetPerformanceModel:
    """Test runtime performance model switching."""

    @patch("bluesky_pettingzoo.bluesky.wrapper.bs")
    def test_set_performance_model_openap(self, mock_bs: MagicMock, default_config: dict) -> None:
        """set_performance_model('openap') sends PERF openap."""
        wrapper = BlueSkyWrapper(default_config)
        wrapper.init_simulation()
        mock_bs.stack.stack.reset_mock()

        wrapper.set_performance_model("openap")

        mock_bs.stack.stack.assert_called_with("PERF openap")

    @patch("bluesky_pettingzoo.bluesky.wrapper.bs")
    def test_set_performance_model_off(self, mock_bs: MagicMock, default_config: dict) -> None:
        """set_performance_model('off') sends PERF off."""
        wrapper = BlueSkyWrapper(default_config)
        wrapper.init_simulation()
        mock_bs.stack.stack.reset_mock()

        wrapper.set_performance_model("off")

        mock_bs.stack.stack.assert_called_with("PERF off")

    @patch("bluesky_pettingzoo.bluesky.wrapper.bs")
    def test_set_performance_model_not_initialized(
        self, mock_bs: MagicMock, default_config: dict
    ) -> None:
        """set_performance_model() raises RuntimeError when not initialized."""
        wrapper = BlueSkyWrapper(default_config)
        with pytest.raises(RuntimeError, match="not initialized"):
            wrapper.set_performance_model("openap")

    @patch("bluesky_pettingzoo.bluesky.wrapper.bs")
    def test_set_performance_model_invalid(self, mock_bs: MagicMock, default_config: dict) -> None:
        """set_performance_model() raises ValueError for invalid model."""
        wrapper = BlueSkyWrapper(default_config)
        wrapper.init_simulation()
        with pytest.raises(ValueError, match="Invalid performance model"):
            wrapper.set_performance_model("turbofan")


class TestPerformanceModelActivationWarning:
    """Test warning when performance model fails to activate."""

    @patch("bluesky_pettingzoo.bluesky.wrapper.bs")
    def test_warns_when_settings_none(self, mock_bs: MagicMock, default_config: dict) -> None:
        """init_simulation() warns when settings.performance_model is None after PERF command."""
        mock_bs.settings.performance_model = None
        with pytest.warns(UserWarning, match="failed to activate"):
            wrapper = BlueSkyWrapper(default_config)
            wrapper.init_simulation()

    @patch("bluesky_pettingzoo.bluesky.wrapper.bs")
    def test_no_warning_when_settings_matches(
        self, mock_bs: MagicMock, default_config: dict
    ) -> None:
        """init_simulation() does not warn when settings.performance_model matches config."""
        mock_bs.settings.performance_model = "openap"
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            wrapper = BlueSkyWrapper(default_config)
            wrapper.init_simulation()

    @patch("bluesky_pettingzoo.bluesky.wrapper.bs")
    def test_no_warning_when_off(self, mock_bs: MagicMock, default_config: dict) -> None:
        """init_simulation() does not warn when performance_model is off."""
        default_config["simulation"]["performance_model"] = "off"
        mock_bs.settings.performance_model = None
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            wrapper = BlueSkyWrapper(default_config)
            wrapper.init_simulation()
