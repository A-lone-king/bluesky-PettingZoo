"""Tests for BlueSky wrapper switching (B3)."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from bluesky_pettingzoo.bluesky.wrapper import BlueSkyWrapper
from bluesky_pettingzoo.envs.scenarios.horizontal_cr import HorizontalCRScenario

# Import the factory function under test
from scripts.train_ppo_scenarios import make_scenario_env_factory


class TestWrapperSwitch:
    """make_scenario_env_factory should support switching between wrappers."""

    def test_factory_with_fake_wrapper(self, tmp_path: Path) -> None:
        """Default factory creates env with BlueSkyWrapper."""
        scenario = HorizontalCRScenario(num_aircraft=2, seed=42)
        factory = make_scenario_env_factory(
            tmp_path,
            scenario,
            num_aircraft=2,
            max_steps=10,
        )
        env = factory()
        assert isinstance(env._env._wrapper, BlueSkyWrapper)
        env.close()

    def test_factory_with_custom_wrapper_cls(self, tmp_path: Path) -> None:
        """Factory accepts a custom wrapper_cls parameter."""
        scenario = HorizontalCRScenario(num_aircraft=2, seed=42)
        mock_cls = MagicMock()
        mock_instance = MagicMock()
        mock_cls.return_value = mock_instance

        factory = make_scenario_env_factory(
            tmp_path,
            scenario,
            num_aircraft=2,
            max_steps=10,
            wrapper_cls=mock_cls,
        )
        env = factory()
        mock_cls.assert_called_once()
        env.close()

    def test_bluesky_wrapper_init_called(self, tmp_path: Path) -> None:
        """When BlueSkyWrapper is used, init_simulation should be callable."""
        scenario = HorizontalCRScenario(num_aircraft=2, seed=42)
        mock_cls = MagicMock()
        mock_instance = MagicMock()
        mock_cls.return_value = mock_instance

        factory = make_scenario_env_factory(
            tmp_path,
            scenario,
            num_aircraft=2,
            max_steps=10,
            wrapper_cls=mock_cls,
        )
        env = factory()
        # The wrapper instance should be used
        assert env._env._wrapper is mock_instance
        env.close()

    def test_factory_default_uses_fake_wrapper(self, tmp_path: Path) -> None:
        """When wrapper_cls=None, BlueSkyWrapper is used."""
        scenario = HorizontalCRScenario(num_aircraft=2, seed=42)
        factory = make_scenario_env_factory(
            tmp_path,
            scenario,
            num_aircraft=2,
            max_steps=10,
            wrapper_cls=None,
        )
        env = factory()
        assert isinstance(env._env._wrapper, BlueSkyWrapper)
        env.close()
