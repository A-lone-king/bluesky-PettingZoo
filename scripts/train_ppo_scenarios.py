"""PPO multi-scenario training and baseline comparison.

Train PPO on all scenarios and compare against Random and RuleBased baselines.

Usage:
    python scripts/train_ppo_scenarios.py
"""

from __future__ import annotations

import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from bluesky_pettingzoo.actions.translator import ActionTranslator
from bluesky_pettingzoo.envs.parallel_env import BlueSkyMARLEnv
from bluesky_pettingzoo.envs.scenarios.base import BaseScenario
from bluesky_pettingzoo.observations.manager import ObservationManager
from bluesky_pettingzoo.rewards.calculator import RewardCalculator
from bluesky_pettingzoo.rewards.components.conflict import ConflictPenalty
from bluesky_pettingzoo.rewards.components.efficiency import EfficiencyReward
from bluesky_pettingzoo.rewards.components.smoothness import SmoothnessPenalty
from bluesky_pettingzoo.wrappers.single_agent import SingleAgentGymWrapper

from scripts.evaluate_baselines import BaselineMetrics

from tests.helpers.fake_wrapper import FakeBlueSkyWrapper
from tests.helpers.env_factory import make_config, write_rewards_yaml


# Scenario configurations for training
SCENARIO_CONFIGS: list[dict[str, Any]] = [
    {"name": "HorizontalCR", "scenario": None, "num_aircraft": 3, "scenario_cls": "HorizontalCRScenario"},
    {"name": "VerticalCR", "scenario": None, "num_aircraft": 3, "scenario_cls": "VerticalCRScenario"},
    {"name": "SectorCR", "scenario": None, "num_aircraft": 3, "scenario_cls": "SectorCRScenario"},
    {"name": "WaypointNav", "scenario": None, "num_aircraft": 3, "scenario_cls": "WaypointNavScenario"},
    {"name": "Merge", "scenario": None, "num_aircraft": 5, "scenario_cls": "MergeScenario"},
    {"name": "Descent", "scenario": None, "num_aircraft": 3, "scenario_cls": "DescentScenario"},
]


def make_scenario_env_factory(
    tmp_path: Path,
    scenario: BaseScenario,
    num_aircraft: int,
    max_steps: int,
) -> Callable[[], SingleAgentGymWrapper]:
    """Return a callable that creates a SingleAgentGymWrapper env."""

    def factory() -> SingleAgentGymWrapper:
        config = make_config(initial_count=num_aircraft, max_steps=max_steps)
        rewards_path = write_rewards_yaml(tmp_path)
        config["_rewards_yaml"] = str(rewards_path)

        with open(rewards_path, encoding="utf-8") as f:
            rewards_cfg = yaml.safe_load(f)
        merged = {**config, **rewards_cfg}

        wrapper = FakeBlueSkyWrapper(config)
        obs_manager = ObservationManager(config)
        action_translator = ActionTranslator(config)
        calc = RewardCalculator()
        calc.register(ConflictPenalty(merged), weight=1.0)
        calc.register(SmoothnessPenalty(merged), weight=0.5)
        calc.register(EfficiencyReward(merged), weight=0.3)

        env = BlueSkyMARLEnv(
            config=config,
            wrapper=wrapper,
            observation_manager=obs_manager,
            action_translator=action_translator,
            reward_calculator=calc,
            rewards_config=rewards_cfg,
            scenario=scenario,
        )
        return SingleAgentGymWrapper(env, ego_agent="AC000")

    return factory


class PPOTrainer:
    """Train and evaluate PPO on a single scenario."""

    def __init__(
        self,
        tmp_path: Path,
        scenario_name: str,
        scenario: BaseScenario,
        num_aircraft: int,
        max_steps: int = 50,
        total_timesteps: int = 50_000,
    ) -> None:
        from stable_baselines3 import PPO

        self.scenario_name = scenario_name
        self.total_timesteps = total_timesteps
        self._tmp_path = tmp_path
        self._factory = make_scenario_env_factory(tmp_path, scenario, num_aircraft, max_steps)
        self.env = self._factory()

        self.model = PPO(
            "MultiInputPolicy",
            self.env,
            n_steps=128,
            batch_size=64,
            n_epochs=4,
            learning_rate=3e-4,
            verbose=0,
            device="cpu",
            seed=42,
        )

    def train(self) -> dict[str, float]:
        """Train the model and return metrics."""
        initial_reward = self._evaluate_model(n_episodes=3)
        self.model.learn(total_timesteps=self.total_timesteps)
        final_reward = self._evaluate_model(n_episodes=3)
        return {
            "initial_reward": initial_reward,
            "final_reward": final_reward,
            "improvement": final_reward - initial_reward,
        }

    def evaluate(self, n_episodes: int = 20) -> BaselineMetrics:
        """Evaluate the trained model."""
        from scripts.evaluate_baselines import EpisodeResult

        results: list[Any] = []
        for _ in range(n_episodes):
            env = self._factory()
            try:
                obs, _ = env.reset(seed=None)
                total_reward = 0.0
                arrived = False
                nmac = False
                truncated = False

                for step in range(60):
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated_flag, info = env.step(action)
                    total_reward += reward
                    if terminated:
                        if reward > 0:
                            arrived = True
                        else:
                            nmac = True
                        break
                    if truncated_flag:
                        truncated = True
                        break

                results.append(EpisodeResult(
                    total_reward=total_reward,
                    steps=min(step + 1, 60),
                    arrived=arrived,
                    nmac=nmac,
                    truncated=truncated,
                ))
            finally:
                env.close()

        return BaselineMetrics.from_results(results)

    def _evaluate_model(self, n_episodes: int = 3) -> float:
        """Quick evaluation during training."""
        rewards = []
        for _ in range(n_episodes):
            obs, _ = self.env.reset(seed=None)
            total = 0.0
            for _ in range(60):
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = self.env.step(action)
                total += reward
                if terminated or truncated:
                    break
            rewards.append(total)
        return float(np.mean(rewards))

    def close(self) -> None:
        self.env.close()


def make_baseline_env_factory(
    tmp_path: Path,
    scenario: BaseScenario,
    num_aircraft: int,
    max_steps: int,
) -> Callable[[], BlueSkyMARLEnv]:
    """Return a callable that creates a raw BlueSkyMARLEnv for baseline evaluation."""

    def factory() -> BlueSkyMARLEnv:
        config = make_config(initial_count=num_aircraft, max_steps=max_steps)
        rewards_path = write_rewards_yaml(tmp_path)
        config["_rewards_yaml"] = str(rewards_path)

        with open(rewards_path, encoding="utf-8") as f:
            rewards_cfg = yaml.safe_load(f)
        merged = {**config, **rewards_cfg}

        wrapper = FakeBlueSkyWrapper(config)
        obs_manager = ObservationManager(config)
        action_translator = ActionTranslator(config)
        calc = RewardCalculator()
        calc.register(ConflictPenalty(merged), weight=1.0)
        calc.register(SmoothnessPenalty(merged), weight=0.5)
        calc.register(EfficiencyReward(merged), weight=0.3)

        return BlueSkyMARLEnv(
            config=config,
            wrapper=wrapper,
            observation_manager=obs_manager,
            action_translator=action_translator,
            reward_calculator=calc,
            rewards_config=rewards_cfg,
            scenario=scenario,
        )

    return factory


def main() -> None:
    """Train PPO on all scenarios and compare against baselines."""
    from bluesky_pettingzoo.agents.random_agent import RandomAgent
    from bluesky_pettingzoo.agents.rule_based_agent import RuleBasedAgent
    from bluesky_pettingzoo.envs.scenarios.horizontal_cr import HorizontalCRScenario
    from bluesky_pettingzoo.envs.scenarios.vertical_cr import VerticalCRScenario
    from bluesky_pettingzoo.envs.scenarios.sector_cr import SectorCRScenario
    from bluesky_pettingzoo.envs.scenarios.waypoint_nav import WaypointNavScenario
    from bluesky_pettingzoo.envs.scenarios.merge import MergeScenario
    from bluesky_pettingzoo.envs.scenarios.descent import DescentScenario
    from scripts.evaluate_baselines import evaluate_agent

    scenarios = [
        {"name": "HorizontalCR", "scenario": HorizontalCRScenario(num_aircraft=3, seed=42), "num_aircraft": 3},
        {"name": "VerticalCR", "scenario": VerticalCRScenario(num_aircraft=3, seed=42), "num_aircraft": 3},
        {"name": "SectorCR", "scenario": SectorCRScenario(num_aircraft=3, seed=42), "num_aircraft": 3},
        {"name": "WaypointNav", "scenario": WaypointNavScenario(num_aircraft=3, seed=42), "num_aircraft": 3},
        {"name": "Merge", "scenario": MergeScenario(num_aircraft=5, seed=42), "num_aircraft": 5},
        {"name": "Descent", "scenario": DescentScenario(num_aircraft=3, seed=42), "num_aircraft": 3},
    ]

    total_timesteps = 50_000
    max_steps = 50
    eval_episodes = 20

    print("=" * 90)
    print("PPO Multi-Scenario Training & Baseline Comparison")
    print(f"Timesteps per scenario: {total_timesteps}, Max steps/episode: {max_steps}")
    print(f"Evaluation episodes: {eval_episodes}")
    print("=" * 90)

    header = f"{'Scenario':<15} {'Agent':<12} {'MeanReward':>12} {'StdReward':>10} {'Arrival%':>10} {'NMAC%':>8} {'MeanSteps':>10}"
    print(header)
    print("-" * 90)

    for cfg in scenarios:
        name = cfg["name"]
        scenario = cfg["scenario"]
        num_aircraft = cfg["num_aircraft"]

        # Train PPO
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            trainer = PPOTrainer(
                tmp_path=tmp_path,
                scenario_name=name,
                scenario=scenario,
                num_aircraft=num_aircraft,
                max_steps=max_steps,
                total_timesteps=total_timesteps,
            )
            train_metrics = trainer.train()
            ppo_metrics = trainer.evaluate(n_episodes=eval_episodes)
            trainer.close()

        print(
            f"{name:<15} {'PPO':<12} "
            f"{ppo_metrics.mean_reward:>12.2f} {ppo_metrics.std_reward:>10.2f} "
            f"{ppo_metrics.arrival_rate:>9.1%} {ppo_metrics.nmac_rate:>7.1%} "
            f"{ppo_metrics.mean_steps:>10.1f}"
        )

        # Evaluate baselines using raw BlueSkyMARLEnv
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)

            def baseline_factory(s=scenario, n=num_aircraft, m=max_steps, t=tmp_path):
                return make_baseline_env_factory(t, s, n, m)()

            random_metrics = evaluate_agent(baseline_factory, RandomAgent(), num_episodes=eval_episodes)
            rule_metrics = evaluate_agent(baseline_factory, RuleBasedAgent(), num_episodes=eval_episodes)

        print(
            f"{name:<15} {'Random':<12} "
            f"{random_metrics.mean_reward:>12.2f} {random_metrics.std_reward:>10.2f} "
            f"{random_metrics.arrival_rate:>9.1%} {random_metrics.nmac_rate:>7.1%} "
            f"{random_metrics.mean_steps:>10.1f}"
        )
        print(
            f"{name:<15} {'RuleBased':<12} "
            f"{rule_metrics.mean_reward:>12.2f} {rule_metrics.std_reward:>10.2f} "
            f"{rule_metrics.arrival_rate:>9.1%} {rule_metrics.nmac_rate:>7.1%} "
            f"{rule_metrics.mean_steps:>10.1f}"
        )
        print("-" * 90)

    print("=" * 90)
    print("Training complete.")


if __name__ == "__main__":
    main()
