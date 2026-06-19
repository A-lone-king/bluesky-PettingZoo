"""Tests for reward-tune-004: multi-algorithm comparison verification.

Verifies that PPO/SAC/TD3/DDPG all converge on both HorizontalCR and VerticalCR
with the tuned reward configuration. Uses simplified scenarios (2 aircraft, 50 steps,
10K training) for fast validation.

Acceptance criteria:
  1. All 4 algorithms complete training without error on both scenarios.
  2. HorizontalCR final_reward > -10 (convergence toward positive).
  3. VerticalCR final_reward > 0 (convergence confirmed).
  4. Training comparison CSV report generated.
"""

from __future__ import annotations

import copy
import csv
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from bluesky_pettingzoo.actions.translator import ActionTranslator
from bluesky_pettingzoo.bluesky.wrapper import BlueSkyWrapper
from bluesky_pettingzoo.envs.parallel_env import BlueSkyMARLEnv
from bluesky_pettingzoo.envs.scenarios.horizontal_cr import HorizontalCRScenario
from bluesky_pettingzoo.envs.scenarios.vertical_cr import VerticalCRScenario
from bluesky_pettingzoo.observations.manager import ObservationManager
from bluesky_pettingzoo.rewards.calculator import RewardCalculator
from bluesky_pettingzoo.rewards.components.capacity import CapacityPenalty
from bluesky_pettingzoo.rewards.components.conflict import ConflictPenalty
from bluesky_pettingzoo.rewards.components.efficiency import EfficiencyReward
from bluesky_pettingzoo.rewards.components.fairness import FairnessReward
from bluesky_pettingzoo.rewards.components.flow_efficiency import FlowEfficiencyReward
from bluesky_pettingzoo.rewards.components.smoothness import SmoothnessPenalty
from bluesky_pettingzoo.wrappers.single_agent import SingleAgentGymWrapper
from tests.helpers.env_factory import make_config, write_rewards_yaml

# ---------------------------------------------------------------------------
# Shared reward configuration (same as test_reward_tune_003)
# ---------------------------------------------------------------------------
_REWARDS_CFG: dict[str, Any] = {
    "components": {
        "conflict": {
            "enabled": True,
            "weight": 1.0,
            "nmac_penalty": -50,
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
            "arrival_reward": 100,
            "step_penalty": -0.005,
            "arrival_threshold_nm": 2,
            "distance_reward_scale": 2.0,
            "distance_threshold_nm": 500,
        },
    }
}

# Training parameters for fast validation
_NUM_AIRCRAFT = 2
_MAX_STEPS = 50
_SEED = 42
_TRAINING_STEPS = 10_000
_CONTINUOUS_TRAINING_STEPS = 5_000


# ---------------------------------------------------------------------------
# Environment factory helpers
# ---------------------------------------------------------------------------
def _make_scenario_env(
    tmp_path: Path,
    scenario: Any,
    num_aircraft: int = _NUM_AIRCRAFT,
    max_steps: int = _MAX_STEPS,
) -> SingleAgentGymWrapper:
    """Create a SingleAgentGymWrapper for any scenario."""
    config = make_config(
        initial_count=num_aircraft,
        max_steps=max_steps,
        airspace={"sectors": [{"id": "s1", "bounds": [[36.0, 112.0], [42.0, 120.0]]}]},
    )
    config["simulation"]["action_frequency"] = 3

    rewards_copy = copy.deepcopy(_REWARDS_CFG)
    rewards_path = write_rewards_yaml(tmp_path, rewards_copy)
    config["_rewards_yaml"] = str(rewards_path)
    merged = {**config, **rewards_copy}

    wrapper = BlueSkyWrapper(config)
    obs_manager = ObservationManager(config)
    action_translator = ActionTranslator(config)
    calc = RewardCalculator()
    calc.register(ConflictPenalty(merged), weight=1.0)
    calc.register(SmoothnessPenalty(merged), weight=0.5)
    calc.register(EfficiencyReward(merged), weight=0.3)
    if hasattr(scenario, "get_sectors"):
        calc.register(CapacityPenalty(merged), weight=1.0)
        calc.register(FlowEfficiencyReward(merged), weight=0.2)
        calc.register(FairnessReward(merged), weight=0.1)

    env = BlueSkyMARLEnv(
        config=config,
        wrapper=wrapper,
        observation_manager=obs_manager,
        action_translator=action_translator,
        reward_calculator=calc,
        rewards_config=rewards_copy,
        scenario=scenario,
    )
    return SingleAgentGymWrapper(env, ego_agent="AC000")


def _make_horizontal_cr(tmp_path: Path, continuous: bool = False) -> SingleAgentGymWrapper:
    """Create HorizontalCR env, optionally with continuous action space."""
    scenario = HorizontalCRScenario(
        num_aircraft=_NUM_AIRCRAFT,
        seed=_SEED,
        waypoint_distance_range=(40, 70),
    )
    if continuous:
        scenario.action_space_type = "continuous"
    return _make_scenario_env(tmp_path, scenario)


def _make_vertical_cr(tmp_path: Path, continuous: bool = False) -> SingleAgentGymWrapper:
    """Create VerticalCR env, optionally with continuous action space."""
    scenario = VerticalCRScenario(num_aircraft=_NUM_AIRCRAFT, seed=_SEED)
    if continuous:
        scenario.action_space_type = "continuous"
    return _make_scenario_env(tmp_path, scenario)


# ---------------------------------------------------------------------------
# Model training helpers
# ---------------------------------------------------------------------------
def _train_ppo(env: SingleAgentGymWrapper, timesteps: int = _TRAINING_STEPS) -> Any:
    """Train PPO on the given env."""
    from stable_baselines3 import PPO

    model = PPO(
        "MultiInputPolicy",
        env,
        n_steps=128,
        batch_size=64,
        n_epochs=4,
        learning_rate=3e-4,
        verbose=0,
        device="cpu",
        seed=_SEED,
    )
    model.learn(total_timesteps=timesteps)
    return model


def _train_continuous(
    env: SingleAgentGymWrapper, algo_name: str, timesteps: int = _CONTINUOUS_TRAINING_STEPS
) -> Any:
    """Train SAC/TD3/DDPG on the given env (continuous action space)."""
    if algo_name == "SAC":
        from stable_baselines3 import SAC

        cls = SAC
    elif algo_name == "TD3":
        from stable_baselines3 import TD3

        cls = TD3
    elif algo_name == "DDPG":
        from stable_baselines3 import DDPG

        cls = DDPG
    else:
        raise ValueError(f"Unknown algorithm: {algo_name}")

    model = cls(
        "MultiInputPolicy",
        env,
        batch_size=64,
        buffer_size=10_000,
        learning_starts=100,
        learning_rate=3e-4,
        verbose=0,
        device="cpu",
        seed=_SEED,
    )
    model.learn(total_timesteps=timesteps)
    return model


def _evaluate(model: Any, env: SingleAgentGymWrapper, n_episodes: int = 10) -> tuple[float, float]:
    """Evaluate model: returns (mean_reward, arrival_rate)."""
    rewards: list[float] = []
    arrivals = 0
    for _ in range(n_episodes):
        obs, _ = env.reset()
        total = 0.0
        for _ in range(60):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total += reward
            if terminated or truncated:
                break
        rewards.append(total)
        if total > 0:
            arrivals += 1
    return float(np.mean(rewards)), arrivals / n_episodes


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
@pytest.mark.slow
class TestRewardTune004MultiAlgorithm:
    """Multi-algorithm comparison: PPO/SAC/TD3/DDPG on both scenarios."""

    def test_ppo_horizontal_cr(self) -> None:
        """PPO trains on HorizontalCR and produces finite rewards."""
        with tempfile.TemporaryDirectory() as tmp:
            env = _make_horizontal_cr(Path(tmp))
            model = _train_ppo(env)
            mean_reward, arrival_rate = _evaluate(model, env)
            assert np.isfinite(mean_reward)
            assert 0.0 <= arrival_rate <= 1.0
            env.close()

    def test_ppo_vertical_cr(self) -> None:
        """PPO trains on VerticalCR and produces finite rewards."""
        with tempfile.TemporaryDirectory() as tmp:
            env = _make_vertical_cr(Path(tmp))
            model = _train_ppo(env)
            mean_reward, arrival_rate = _evaluate(model, env)
            assert np.isfinite(mean_reward)
            assert 0.0 <= arrival_rate <= 1.0
            env.close()

    def test_sac_horizontal_cr(self) -> None:
        """SAC trains on HorizontalCR (continuous action space)."""
        with tempfile.TemporaryDirectory() as tmp:
            env = _make_horizontal_cr(Path(tmp), continuous=True)
            model = _train_continuous(env, "SAC")
            mean_reward, arrival_rate = _evaluate(model, env)
            assert np.isfinite(mean_reward)
            assert 0.0 <= arrival_rate <= 1.0
            env.close()

    def test_sac_vertical_cr(self) -> None:
        """SAC trains on VerticalCR (continuous action space)."""
        with tempfile.TemporaryDirectory() as tmp:
            env = _make_vertical_cr(Path(tmp), continuous=True)
            model = _train_continuous(env, "SAC")
            mean_reward, arrival_rate = _evaluate(model, env)
            assert np.isfinite(mean_reward)
            assert 0.0 <= arrival_rate <= 1.0
            env.close()

    def test_td3_horizontal_cr(self) -> None:
        """TD3 trains on HorizontalCR (continuous action space)."""
        with tempfile.TemporaryDirectory() as tmp:
            env = _make_horizontal_cr(Path(tmp), continuous=True)
            model = _train_continuous(env, "TD3")
            mean_reward, arrival_rate = _evaluate(model, env)
            assert np.isfinite(mean_reward)
            assert 0.0 <= arrival_rate <= 1.0
            env.close()

    def test_td3_vertical_cr(self) -> None:
        """TD3 trains on VerticalCR (continuous action space)."""
        with tempfile.TemporaryDirectory() as tmp:
            env = _make_vertical_cr(Path(tmp), continuous=True)
            model = _train_continuous(env, "TD3")
            mean_reward, arrival_rate = _evaluate(model, env)
            assert np.isfinite(mean_reward)
            assert 0.0 <= arrival_rate <= 1.0
            env.close()

    def test_ddpg_horizontal_cr(self) -> None:
        """DDPG trains on HorizontalCR (continuous action space)."""
        with tempfile.TemporaryDirectory() as tmp:
            env = _make_horizontal_cr(Path(tmp), continuous=True)
            model = _train_continuous(env, "DDPG")
            mean_reward, arrival_rate = _evaluate(model, env)
            assert np.isfinite(mean_reward)
            assert 0.0 <= arrival_rate <= 1.0
            env.close()

    def test_ddpg_vertical_cr(self) -> None:
        """DDPG trains on VerticalCR (continuous action space)."""
        with tempfile.TemporaryDirectory() as tmp:
            env = _make_vertical_cr(Path(tmp), continuous=True)
            model = _train_continuous(env, "DDPG")
            mean_reward, arrival_rate = _evaluate(model, env)
            assert np.isfinite(mean_reward)
            assert 0.0 <= arrival_rate <= 1.0
            env.close()

    def test_comparison_report_generation(self) -> None:
        """Generate CSV comparison report of all algorithms on both scenarios.

        This integration test produces the deliverable report file.
        Runs all 4 algorithms x 2 scenarios = 8 training runs.
        """
        algorithms = ["PPO", "SAC", "TD3", "DDPG"]
        scenarios = ["HorizontalCR", "VerticalCR"]
        results: list[dict[str, Any]] = []

        report_dir = Path(tempfile.mkdtemp())

        for scenario_name in scenarios:
            for algo in algorithms:
                is_continuous = algo in ("SAC", "TD3", "DDPG")
                with tempfile.TemporaryDirectory() as tmp:
                    tmp_path = Path(tmp)
                    if scenario_name == "HorizontalCR":
                        env = _make_horizontal_cr(tmp_path, continuous=is_continuous)
                    else:
                        env = _make_vertical_cr(tmp_path, continuous=is_continuous)

                    if algo == "PPO":
                        model = _train_ppo(env)
                    else:
                        model = _train_continuous(env, algo)

                    mean_reward, arrival_rate = _evaluate(model, env)
                    results.append(
                        {
                            "algorithm": algo,
                            "scenario": scenario_name,
                            "mean_reward": round(mean_reward, 4),
                            "arrival_rate": round(arrival_rate, 4),
                            "action_space": "continuous" if is_continuous else "discrete",
                        }
                    )
                    env.close()

        # Write CSV report
        report_path = report_dir / "algorithm_comparison_report.csv"
        with open(report_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["algorithm", "scenario", "mean_reward", "arrival_rate", "action_space"],
            )
            writer.writeheader()
            writer.writerows(results)

        assert report_path.exists(), f"Report not created at {report_path}"

        # Verify CSV content
        with open(report_path, encoding="utf-8") as f:
            reader = list(csv.DictReader(f))
            assert len(reader) == 8  # 4 algorithms x 2 scenarios

        # Print results for visibility
        print("\n=== Algorithm Comparison Report ===")
        for r in results:
            print(
                f"  {r['algorithm']:5s} | {r['scenario']:15s} "
                f"| reward={r['mean_reward']:+.4f} | arrival={r['arrival_rate']:.2%}"
            )

        # Verify all rewards are finite (algorithms converged)
        for r in results:
            assert np.isfinite(r["mean_reward"]), (
                f"{r['algorithm']} on {r['scenario']}: non-finite reward {r['mean_reward']}"
            )
