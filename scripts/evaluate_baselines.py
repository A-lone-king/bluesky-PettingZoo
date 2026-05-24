"""Baseline evaluation — run agents on scenarios and collect metrics."""

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
from bluesky_pettingzoo.agents.base import BaseAgent
from bluesky_pettingzoo.envs.parallel_env import BlueSkyMARLEnv
from bluesky_pettingzoo.observations.manager import ObservationManager
from bluesky_pettingzoo.rewards.calculator import RewardCalculator
from bluesky_pettingzoo.rewards.components.capacity import CapacityPenalty
from bluesky_pettingzoo.rewards.components.conflict import ConflictPenalty
from bluesky_pettingzoo.rewards.components.efficiency import EfficiencyReward
from bluesky_pettingzoo.rewards.components.obstacle_intrusion import ObstacleIntrusion
from bluesky_pettingzoo.rewards.components.smoothness import SmoothnessPenalty
from bluesky_pettingzoo.utils.types import AgentID

from tests.helpers.fake_wrapper import FakeBlueSkyWrapper
from tests.helpers.env_factory import make_config, write_rewards_yaml


@dataclass
class EpisodeResult:
    """Result of a single episode."""
    total_reward: float
    steps: int
    arrived: bool
    nmac: bool
    truncated: bool


@dataclass
class BaselineMetrics:
    """Aggregated metrics over multiple episodes."""
    mean_reward: float
    std_reward: float
    arrival_rate: float
    nmac_rate: float
    mean_steps: float
    num_episodes: int

    @classmethod
    def from_results(cls, results: list[EpisodeResult]) -> BaselineMetrics:
        if not results:
            return cls(0.0, 0.0, 0.0, 0.0, 0.0, 0)
        rewards = [r.total_reward for r in results]
        return cls(
            mean_reward=float(np.mean(rewards)),
            std_reward=float(np.std(rewards)),
            arrival_rate=sum(1 for r in results if r.arrived) / len(results),
            nmac_rate=sum(1 for r in results if r.nmac) / len(results),
            mean_steps=float(np.mean([r.steps for r in results])),
            num_episodes=len(results),
        )


def run_episode(env: BlueSkyMARLEnv, agent: BaseAgent, max_steps: int = 100) -> EpisodeResult:
    """Run a single episode and return results."""
    observations, infos = env.reset(seed=None)
    agent.reset()

    total_reward = 0.0
    arrived = False
    nmac = False
    truncated = False

    for step in range(max_steps):
        if not env.agents:
            break

        # Build action spaces for active agents
        action_spaces = {aid: env.action_space(aid) for aid in env.agents}
        actions = agent.act(observations, action_spaces)

        observations, rewards, terminations, truncations, infos = env.step(actions)
        total_reward += sum(rewards.values())

        for aid, t in terminations.items():
            if t:
                # Check if it was arrival (efficiency reward > 0) or NMAC
                if rewards.get(aid, 0) > 0:
                    arrived = True
                else:
                    nmac = True

        for aid, t in truncations.items():
            if t:
                truncated = True

        if not env.agents:
            break

    return EpisodeResult(
        total_reward=total_reward,
        steps=min(step + 1, max_steps),
        arrived=arrived,
        nmac=nmac,
        truncated=truncated,
    )


def evaluate_agent(
    env_factory: Callable[[], BlueSkyMARLEnv],
    agent: BaseAgent,
    num_episodes: int = 20,
) -> BaselineMetrics:
    """Run multiple episodes and compute aggregated metrics."""
    results = []
    for _ in range(num_episodes):
        env = env_factory()
        try:
            result = run_episode(env, agent)
            results.append(result)
        finally:
            env.close()
    return BaselineMetrics.from_results(results)


def make_env_factory(
    tmp_path: Path,
    num_aircraft: int = 5,
    max_steps: int = 50,
    seed: int = 42,
    scenario=None,
) -> Callable[[], BlueSkyMARLEnv]:
    """Return a callable that creates a BlueSkyMARLEnv."""

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
        if hasattr(scenario, "get_obstacles"):
            obs_comp = ObstacleIntrusion()
            obs_comp.set_obstacles(scenario.get_obstacles())
            calc.register(obs_comp, weight=1.0)
        if hasattr(scenario, "get_sectors"):
            cap_comp = CapacityPenalty(merged)
            calc.register(cap_comp, weight=1.0)

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
    """Run baseline agents on all scenarios and print results."""
    from bluesky_pettingzoo.agents.random_agent import RandomAgent
    from bluesky_pettingzoo.agents.rule_based_agent import RuleBasedAgent
    from bluesky_pettingzoo.envs.scenarios.horizontal_cr import HorizontalCRScenario
    from bluesky_pettingzoo.envs.scenarios.vertical_cr import VerticalCRScenario
    from bluesky_pettingzoo.envs.scenarios.sector_cr import SectorCRScenario
    from bluesky_pettingzoo.envs.scenarios.waypoint_nav import WaypointNavScenario
    from bluesky_pettingzoo.envs.scenarios.merge import MergeScenario
    from bluesky_pettingzoo.envs.scenarios.descent import DescentScenario
    from bluesky_pettingzoo.envs.scenarios.sector_capacity import SectorCapacityScenario
    from bluesky_pettingzoo.envs.scenarios.static_obstacle import StaticObstacleScenario

    scenarios = {
        "HorizontalCR": HorizontalCRScenario(num_aircraft=3, seed=42),
        "VerticalCR": VerticalCRScenario(num_aircraft=3, seed=42),
        "SectorCR": SectorCRScenario(num_aircraft=3, seed=42),
        "WaypointNav": WaypointNavScenario(num_aircraft=3, seed=42),
        "Merge": MergeScenario(num_aircraft=5, seed=42),
        "Descent": DescentScenario(num_aircraft=3, seed=42),
        "StaticObstacle": StaticObstacleScenario(num_aircraft=1, seed=42),
        "SectorCapacity": SectorCapacityScenario(num_aircraft=6, num_sectors=2, sector_capacity=4, seed=42),
    }

    agents = {
        "Random": RandomAgent(),
        "RuleBased": RuleBasedAgent(),
    }

    num_episodes = 20
    max_steps = 50

    print("=" * 80)
    print("Baseline Evaluation Results")
    print(f"Episodes per scenario: {num_episodes}, Max steps: {max_steps}")
    print("=" * 80)

    header = f"{'Scenario':<15} {'Agent':<10} {'MeanReward':>12} {'StdReward':>10} {'Arrival%':>10} {'NMAC%':>8} {'MeanSteps':>10}"
    print(header)
    print("-" * 80)

    _num_aircraft = {"Merge": 5, "StaticObstacle": 1, "SectorCapacity": 6}

    for scenario_name, scenario in scenarios.items():
        n_ac = _num_aircraft.get(scenario_name, 3)
        for agent_name, agent in agents.items():
            with tempfile.TemporaryDirectory() as tmp:
                tmp_path = Path(tmp)
                factory = make_env_factory(
                    tmp_path=tmp_path,
                    num_aircraft=n_ac,
                    max_steps=max_steps,
                    seed=42,
                    scenario=scenario,
                )
                metrics = evaluate_agent(factory, agent, num_episodes=num_episodes)

            print(
                f"{scenario_name:<15} {agent_name:<10} "
                f"{metrics.mean_reward:>12.2f} {metrics.std_reward:>10.2f} "
                f"{metrics.arrival_rate:>9.1%} {metrics.nmac_rate:>7.1%} "
                f"{metrics.mean_steps:>10.1f}"
            )

    print("=" * 80)


if __name__ == "__main__":
    main()
