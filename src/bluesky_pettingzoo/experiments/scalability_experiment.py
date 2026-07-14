"""Scalability experiment framework.

Evaluates performance across different numbers of aircraft (3/5/10/15/20)
to assess how the multi-agent system scales with increasing complexity.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from bluesky_pettingzoo.training.evaluator import EvalResult


@dataclass
class ScalabilityResult:
    """Result from a scalability experiment."""

    scenario: str
    algorithm: str
    num_aircraft: int
    mean_reward: float
    std_reward: float
    mean_arrival_rate: float
    mean_nmac_rate: float
    mean_steps: float
    num_episodes: int
    seed: int


@dataclass
class ScalabilitySummary:
    """Aggregated summary across all aircraft counts."""

    scenario: str
    algorithm: str
    seeds: List[int]
    aircraft_counts: List[int]
    results: List[ScalabilityResult]
    timestamp: str


class ScalabilityExperiment:
    """Scalability experiment runner.

    Evaluates PPO and baseline strategies across different aircraft counts.
    """

    DEFAULT_AIRCRAFT_COUNTS: List[int] = [3, 5, 10, 15, 20]
    DEFAULT_SEEDS: List[int] = [42, 123, 456]

    def __init__(
        self,
        scenario_name: str,
        algorithm: str = "PPO",
        aircraft_counts: Optional[List[int]] = None,
        seeds: Optional[List[int]] = None,
    ) -> None:
        """Initialize scalability experiment.

        Args:
            scenario_name: Name of the scenario to test
            algorithm: Algorithm to use (PPO, Random, RuleBased)
            aircraft_counts: List of aircraft counts to test
            seeds: List of random seeds
        """
        self.scenario_name = scenario_name
        self.algorithm = algorithm
        self.aircraft_counts = aircraft_counts or self.DEFAULT_AIRCRAFT_COUNTS
        self.seeds = seeds or self.DEFAULT_SEEDS

    def run(
        self,
        num_episodes: int = 20,
        save_results: bool = True,
        results_dir: str = "results/scalability",
    ) -> ScalabilitySummary:
        """Run scalability experiment across all aircraft counts.

        Args:
            num_episodes: Number of episodes per configuration
            save_results: Whether to save results to disk
            results_dir: Directory to save results

        Returns:
            ScalabilitySummary with aggregated results
        """
        all_results: List[ScalabilityResult] = []

        for num_ac in self.aircraft_counts:
            for seed in self.seeds:
                result = self._evaluate_configuration(
                    num_ac, seed, num_episodes
                )
                all_results.append(result)

        summary = ScalabilitySummary(
            scenario=self.scenario_name,
            algorithm=self.algorithm,
            seeds=self.seeds,
            aircraft_counts=self.aircraft_counts,
            results=all_results,
            timestamp=np.datetime64("now").astype(str),
        )

        if save_results:
            self._save_results(summary, results_dir)

        return summary

    def _evaluate_configuration(
        self,
        num_aircraft: int,
        seed: int,
        num_episodes: int,
    ) -> ScalabilityResult:
        """Evaluate a single configuration (num_aircraft, seed).

        Args:
            num_aircraft: Number of aircraft
            seed: Random seed
            num_episodes: Number of evaluation episodes

        Returns:
            ScalabilityResult for this configuration
        """
        from bluesky_pettingzoo import make

        env = make(self.scenario_name, num_aircraft=num_aircraft)

        if self.algorithm == "Random":
            return self._evaluate_random(env, num_aircraft, seed, num_episodes)
        elif self.algorithm == "RuleBased":
            return self._evaluate_rulebased(env, num_aircraft, seed, num_episodes)
        else:
            return self._evaluate_ppo(env, num_aircraft, seed, num_episodes)

    def _evaluate_random(
        self,
        env: Any,
        num_aircraft: int,
        seed: int,
        num_episodes: int,
    ) -> ScalabilityResult:
        """Evaluate random policy."""
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
                actions = {agent: env.action_space(agent).sample() for agent in env.possible_agents}
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

        return ScalabilityResult(
            scenario=self.scenario_name,
            algorithm=self.algorithm,
            num_aircraft=num_aircraft,
            mean_reward=float(np.mean(rewards)) if rewards else 0.0,
            std_reward=float(np.std(rewards)) if rewards else 0.0,
            mean_arrival_rate=arrivals / len(rewards) if rewards else 0.0,
            mean_nmac_rate=nmacs / len(rewards) if rewards else 0.0,
            mean_steps=float(np.mean(steps_list)) if steps_list else 0.0,
            num_episodes=num_episodes,
            seed=seed,
        )

    def _evaluate_rulebased(
        self,
        env: Any,
        num_aircraft: int,
        seed: int,
        num_episodes: int,
    ) -> ScalabilityResult:
        """Evaluate rule-based policy."""
        from bluesky_pettingzoo.baselines import RuleBasedStrategy

        strategy = RuleBasedStrategy()

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
                        actions[agent_id] = strategy.act(obs[agent_id])
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

        return ScalabilityResult(
            scenario=self.scenario_name,
            algorithm=self.algorithm,
            num_aircraft=num_aircraft,
            mean_reward=float(np.mean(rewards)) if rewards else 0.0,
            std_reward=float(np.std(rewards)) if rewards else 0.0,
            mean_arrival_rate=arrivals / len(rewards) if rewards else 0.0,
            mean_nmac_rate=nmacs / len(rewards) if rewards else 0.0,
            mean_steps=float(np.mean(steps_list)) if steps_list else 0.0,
            num_episodes=num_episodes,
            seed=seed,
        )

    def _evaluate_ppo(
        self,
        env: Any,
        num_aircraft: int,
        seed: int,
        num_episodes: int,
    ) -> ScalabilityResult:
        """Evaluate PPO policy."""
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

        return ScalabilityResult(
            scenario=self.scenario_name,
            algorithm=self.algorithm,
            num_aircraft=num_aircraft,
            mean_reward=float(np.mean(rewards)) if rewards else 0.0,
            std_reward=float(np.std(rewards)) if rewards else 0.0,
            mean_arrival_rate=arrivals / len(rewards) if rewards else 0.0,
            mean_nmac_rate=nmacs / len(rewards) if rewards else 0.0,
            mean_steps=float(np.mean(steps_list)) if steps_list else 0.0,
            num_episodes=num_episodes,
            seed=seed,
        )

    def _save_results(self, summary: ScalabilitySummary, results_dir: str) -> None:
        """Save results to JSON file.

        Args:
            summary: Scalability summary to save
            results_dir: Directory to save results
        """
        import json

        path = Path(results_dir)
        path.mkdir(parents=True, exist_ok=True)

        file_path = path / f"{summary.scenario}_{summary.algorithm}_scalability.json"

        data = {
            "scenario": summary.scenario,
            "algorithm": summary.algorithm,
            "seeds": summary.seeds,
            "aircraft_counts": summary.aircraft_counts,
            "timestamp": summary.timestamp,
            "results": [
                {
                    "num_aircraft": r.num_aircraft,
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

    def get_aggregated_results(self, summary: ScalabilitySummary) -> Dict[str, Any]:
        """Get aggregated results by aircraft count.

        Args:
            summary: Scalability summary

        Returns:
            Dictionary with aggregated metrics per aircraft count
        """
        aggregated: Dict[str, Any] = {}

        for num_ac in self.aircraft_counts:
            ac_results = [r for r in summary.results if r.num_aircraft == num_ac]

            rewards = [r.mean_reward for r in ac_results]
            arrival_rates = [r.mean_arrival_rate for r in ac_results]
            nmac_rates = [r.mean_nmac_rate for r in ac_results]
            steps = [r.mean_steps for r in ac_results]

            aggregated[num_ac] = {
                "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
                "std_reward": float(np.std(rewards)) if rewards else 0.0,
                "mean_arrival_rate": float(np.mean(arrival_rates)) if arrival_rates else 0.0,
                "std_arrival_rate": float(np.std(arrival_rates)) if arrival_rates else 0.0,
                "mean_nmac_rate": float(np.mean(nmac_rates)) if nmac_rates else 0.0,
                "std_nmac_rate": float(np.std(nmac_rates)) if nmac_rates else 0.0,
                "mean_steps": float(np.mean(steps)) if steps else 0.0,
                "std_steps": float(np.std(steps)) if steps else 0.0,
                "num_seeds": len(ac_results),
            }

        return aggregated