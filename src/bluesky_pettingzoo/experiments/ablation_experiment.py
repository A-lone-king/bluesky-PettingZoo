"""Ablation experiment framework.

Evaluates the impact of different action space dimensions and reward components
to understand what drives the multi-agent system's performance.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class AblationResult:
    """Result from a single ablation configuration."""

    scenario: str
    ablation_type: str
    config_name: str
    mean_reward: float
    std_reward: float
    mean_arrival_rate: float
    mean_nmac_rate: float
    mean_steps: float
    num_episodes: int
    seed: int


@dataclass
class AblationSummary:
    """Aggregated summary of ablation results."""

    scenario: str
    ablation_type: str
    configs: List[str]
    seeds: List[int]
    results: List[AblationResult]
    timestamp: str


class AblationExperiment:
    """Ablation experiment runner.

    Supports two types of ablation:
    1. Action space dimension ablation (with/without speed control, heading control)
    2. Reward component ablation (conflict penalty, efficiency reward, smoothness penalty)
    """

    DEFAULT_SEEDS: List[int] = [42, 123, 456]

    ACTION_CONFIGS: Dict[str, Dict[str, bool]] = {
        "full": {"heading": True, "speed": True, "altitude": True},
        "heading_only": {"heading": True, "speed": False, "altitude": False},
        "speed_only": {"heading": False, "speed": True, "altitude": False},
        "heading_speed": {"heading": True, "speed": True, "altitude": False},
    }

    REWARD_CONFIGS: Dict[str, Dict[str, bool]] = {
        "full": {"conflict": True, "efficiency": True, "smoothness": True},
        "no_conflict": {"conflict": False, "efficiency": True, "smoothness": True},
        "no_efficiency": {"conflict": True, "efficiency": False, "smoothness": True},
        "no_smoothness": {"conflict": True, "efficiency": True, "smoothness": False},
        "conflict_only": {"conflict": True, "efficiency": False, "smoothness": False},
    }

    def __init__(
        self,
        scenario_name: str,
        ablation_type: str = "action_space",
        configs: Optional[List[str]] = None,
        seeds: Optional[List[int]] = None,
    ) -> None:
        """Initialize ablation experiment.

        Args:
            scenario_name: Name of the scenario to test
            ablation_type: "action_space" or "reward"
            configs: List of configuration names to test
            seeds: List of random seeds
        """
        self.scenario_name = scenario_name
        self.ablation_type = ablation_type
        self.seeds = seeds or self.DEFAULT_SEEDS

        if ablation_type == "action_space":
            self.configs = configs or list(self.ACTION_CONFIGS.keys())
        elif ablation_type == "reward":
            self.configs = configs or list(self.REWARD_CONFIGS.keys())
        else:
            raise ValueError(f"Unknown ablation type: {ablation_type}")

    def run(
        self,
        num_episodes: int = 20,
        save_results: bool = True,
        results_dir: str = "results/ablation",
    ) -> AblationSummary:
        """Run ablation experiment across all configurations.

        Args:
            num_episodes: Number of episodes per configuration
            save_results: Whether to save results to disk
            results_dir: Directory to save results

        Returns:
            AblationSummary with aggregated results
        """
        all_results: List[AblationResult] = []

        for config_name in self.configs:
            for seed in self.seeds:
                result = self._evaluate_configuration(
                    config_name, seed, num_episodes
                )
                all_results.append(result)

        summary = AblationSummary(
            scenario=self.scenario_name,
            ablation_type=self.ablation_type,
            configs=self.configs,
            seeds=self.seeds,
            results=all_results,
            timestamp=np.datetime64("now").astype(str),
        )

        if save_results:
            self._save_results(summary, results_dir)

        return summary

    def _evaluate_configuration(
        self,
        config_name: str,
        seed: int,
        num_episodes: int,
    ) -> AblationResult:
        """Evaluate a single ablation configuration.

        Args:
            config_name: Name of the configuration
            seed: Random seed
            num_episodes: Number of evaluation episodes

        Returns:
            AblationResult for this configuration
        """
        from bluesky_pettingzoo import make

        if self.ablation_type == "action_space":
            action_config = self.ACTION_CONFIGS.get(config_name, {})
            env = make(
                self.scenario_name,
                enable_heading=action_config.get("heading", True),
                enable_speed=action_config.get("speed", True),
                enable_altitude=action_config.get("altitude", True),
            )
        else:
            reward_config = self.REWARD_CONFIGS.get(config_name, {})
            env = make(
                self.scenario_name,
                use_conflict_penalty=reward_config.get("conflict", True),
                use_efficiency_reward=reward_config.get("efficiency", True),
                use_smoothness_penalty=reward_config.get("smoothness", True),
            )

        return self._evaluate_ppo(env, config_name, seed, num_episodes)

    def _evaluate_ppo(
        self,
        env: Any,
        config_name: str,
        seed: int,
        num_episodes: int,
    ) -> AblationResult:
        """Evaluate PPO on the configuration."""
        from stable_baselines3 import PPO

        model = PPO(
            "MlpPolicy",
            env,
            verbose=0,
            seed=seed,
        )

        model.learn(total_timesteps=100_000)

        rewards: List[float] = []
        steps_list: List[int] = []
        arrivals = 0
        nmacs = 0

        for _ in range(num_episodes):
            obs, _ = env.reset(seed=seed)
            total_reward = 0.0
            episode_steps = 0
            terminated = {agent: False for agent in env.possible_agents}
            truncated = {agent: False for agent in env.possible_agents}

            while not all(terminated.values()) and not all(truncated.values()):
                actions = {}
                for agent_id in env.possible_agents:
                    if not terminated.get(agent_id, False) and not truncated.get(agent_id, False):
                        action, _ = model.predict(obs[agent_id], deterministic=True)
                        actions[agent_id] = action
                obs, reward, terminated, truncated, info = env.step(actions)
                total_reward += sum(float(r) for r in reward.values())
                episode_steps += 1

            rewards.append(total_reward)
            steps_list.append(episode_steps)

            if total_reward > 0:
                arrivals += 1
            else:
                nmacs += 1

        env.close()

        return AblationResult(
            scenario=self.scenario_name,
            ablation_type=self.ablation_type,
            config_name=config_name,
            mean_reward=float(np.mean(rewards)) if rewards else 0.0,
            std_reward=float(np.std(rewards)) if rewards else 0.0,
            mean_arrival_rate=arrivals / len(rewards) if rewards else 0.0,
            mean_nmac_rate=nmacs / len(rewards) if rewards else 0.0,
            mean_steps=float(np.mean(steps_list)) if steps_list else 0.0,
            num_episodes=num_episodes,
            seed=seed,
        )

    def _save_results(self, summary: AblationSummary, results_dir: str) -> None:
        """Save results to JSON file.

        Args:
            summary: Ablation summary to save
            results_dir: Directory to save results
        """
        import json

        path = Path(results_dir)
        path.mkdir(parents=True, exist_ok=True)

        file_path = path / f"{summary.scenario}_{summary.ablation_type}_ablation.json"

        data = {
            "scenario": summary.scenario,
            "ablation_type": summary.ablation_type,
            "configs": summary.configs,
            "seeds": summary.seeds,
            "timestamp": summary.timestamp,
            "results": [
                {
                    "config_name": r.config_name,
                    "seed": r.seed,
                    "mean_reward": r.mean_reward,
                    "std_reward": r.std_reward,
                    "mean_arrival_rate": r.mean_arrival_rate,
                    "mean_nmac_rate": r.mean_nmac_rate,
                    "mean_steps": r.mean_steps,
                    "num_episodes": r.num_episodes,
                }
                for r in summary.results
            ],
        }

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def get_aggregated_results(self, summary: AblationSummary) -> Dict[str, Any]:
        """Get aggregated results by configuration.

        Args:
            summary: Ablation summary

        Returns:
            Dictionary with aggregated metrics per configuration
        """
        aggregated: Dict[str, Any] = {}

        for config_name in self.configs:
            config_results = [r for r in summary.results if r.config_name == config_name]

            rewards = [r.mean_reward for r in config_results]
            arrival_rates = [r.mean_arrival_rate for r in config_results]
            nmac_rates = [r.mean_nmac_rate for r in config_results]
            steps = [r.mean_steps for r in config_results]

            aggregated[config_name] = {
                "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
                "std_reward": float(np.std(rewards)) if rewards else 0.0,
                "mean_arrival_rate": float(np.mean(arrival_rates)) if arrival_rates else 0.0,
                "std_arrival_rate": float(np.std(arrival_rates)) if arrival_rates else 0.0,
                "mean_nmac_rate": float(np.mean(nmac_rates)) if nmac_rates else 0.0,
                "std_nmac_rate": float(np.std(nmac_rates)) if nmac_rates else 0.0,
                "mean_steps": float(np.mean(steps)) if steps else 0.0,
                "std_steps": float(np.std(steps)) if steps else 0.0,
                "num_seeds": len(config_results),
            }

        return aggregated