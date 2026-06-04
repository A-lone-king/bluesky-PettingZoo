"""Tests for parallel_env.py reset with delay component (A15)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from bluesky_pettingzoo.envs.scenarios.horizontal_cr import HorizontalCRScenario
from bluesky_pettingzoo.rewards.components.delay import DelayPenalty
from tests.helpers.env_factory import make_env


class TestParallelEnvResetWithDelay:
    """Reset should not crash when a DelayPenalty component is registered."""

    def test_reset_with_delay_component_no_error(self, tmp_path: Path) -> None:
        """Reset must succeed when DelayPenalty is registered and scenario has waypoints."""
        scenario = HorizontalCRScenario(num_aircraft=2, seed=42)

        rewards = {
            "components": {
                "conflict": {
                    "enabled": True,
                    "weight": 1.0,
                    "nmac_penalty": -100,
                    "warning_penalty": -10,
                    "separation_penalty": -5,
                    "thresholds": {
                        "nmac_horizontal_nm": 5,
                        "nmac_vertical_ft": 1000,
                        "warning_horizontal_nm": 10,
                        "warning_vertical_ft": 2000,
                    },
                },
                "smoothness": {"enabled": True, "weight": 0.5, "action_penalty": -0.1},
                "efficiency": {
                    "enabled": True,
                    "weight": 0.3,
                    "max_deviation_nm": 200,
                    "deviation_penalty_scale": 5,
                    "arrival_reward": 10,
                    "step_penalty": -0.01,
                    "arrival_threshold_nm": 2,
                },
                "delay": {
                    "enabled": True,
                    "weight": 0.2,
                    "delay_penalty_per_step": -0.05,
                },
            }
        }

        env = make_env(tmp_path=tmp_path, scenario=scenario, rewards=rewards)
        env._reward_calculator.register(DelayPenalty(env.config), weight=0.2)

        # This should NOT raise NameError for 'new_states'
        observations, infos = env.reset(seed=42)
        assert isinstance(observations, dict)
        assert len(observations) > 0
        env.close()

    def test_reset_returns_valid_observations(self, tmp_path: Path) -> None:
        """Reset must return observations with correct agent keys and space shapes."""
        scenario = HorizontalCRScenario(num_aircraft=2, seed=42)

        rewards = {
            "components": {
                "conflict": {
                    "enabled": True,
                    "weight": 1.0,
                    "nmac_penalty": -100,
                    "warning_penalty": -10,
                    "separation_penalty": -5,
                    "thresholds": {
                        "nmac_horizontal_nm": 5,
                        "nmac_vertical_ft": 1000,
                        "warning_horizontal_nm": 10,
                        "warning_vertical_ft": 2000,
                    },
                },
                "smoothness": {"enabled": True, "weight": 0.5, "action_penalty": -0.1},
                "efficiency": {
                    "enabled": True,
                    "weight": 0.3,
                    "max_deviation_nm": 200,
                    "deviation_penalty_scale": 5,
                    "arrival_reward": 10,
                    "step_penalty": -0.01,
                    "arrival_threshold_nm": 2,
                },
                "delay": {
                    "enabled": True,
                    "weight": 0.2,
                    "delay_penalty_per_step": -0.05,
                },
            }
        }

        env = make_env(tmp_path=tmp_path, scenario=scenario, rewards=rewards)
        env._reward_calculator.register(DelayPenalty(env.config), weight=0.2)

        observations, infos = env.reset(seed=42)

        # All agents should have observations
        for agent in env.agents:
            assert agent in observations, f"Missing observation for {agent}"
            obs = observations[agent]
            # Observation should be a dict (Dict space)
            assert isinstance(obs, dict)
            assert len(obs) > 0

        env.close()
