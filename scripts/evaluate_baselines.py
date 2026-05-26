"""Baseline evaluation — run agents on scenarios and collect metrics.

Usage:
    python scripts/evaluate_baselines.py --scenario HorizontalCR --episodes 20
    python scripts/evaluate_baselines.py --scenario HorizontalCR --model models/HorizontalCR/checkpoint_final.zip
"""

from __future__ import annotations

import argparse
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
from bluesky_pettingzoo.training.evaluator import EvalResult, ModelEvaluator
from bluesky_pettingzoo.utils.types import AgentID

from bluesky_pettingzoo.bluesky.wrapper import BlueSkyWrapper
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

        action_spaces = {aid: env.action_space(aid) for aid in env.agents}
        actions = agent.act(observations, action_spaces)

        observations, rewards, terminations, truncations, infos = env.step(actions)
        total_reward += sum(rewards.values())

        for aid, t in terminations.items():
            if t:
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

        wrapper = BlueSkyWrapper(config)
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Evaluate baselines and trained models")
    parser.add_argument("--scenario", type=str, default="HorizontalCR",
                        help="Scenario to evaluate on")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to trained PPO model checkpoint")
    parser.add_argument("--episodes", type=int, default=20,
                        help="Number of evaluation episodes")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--max-steps", type=int, default=50,
                        help="Max steps per episode")
    parser.add_argument("--num-aircraft", type=int, default=3,
                        help="Number of aircraft in scenario")
    return parser.parse_args(argv)


def run_evaluation(args: argparse.Namespace) -> list[EvalResult]:
    """Run evaluation and print comparison table."""
    from scripts.train_ppo_scenarios import _resolve_scenario, make_scenario_env_factory

    scenario = _resolve_scenario(args.scenario, args.num_aircraft, args.seed)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        def env_factory():
            return make_scenario_env_factory(
                tmp_path, scenario, args.num_aircraft, args.max_steps,
            )()

        evaluator = ModelEvaluator(
            env_factory=env_factory,
            num_episodes=args.episodes,
            max_steps=args.max_steps,
            seed=args.seed,
        )

        from bluesky_pettingzoo.agents.random_agent import RandomAgent
        from bluesky_pettingzoo.agents.rule_based_agent import RuleBasedAgent

        results: list[EvalResult] = []

        # Random baseline
        random_result = evaluator._run_episodes("Random", agent=RandomAgent())
        random_result.scenario = args.scenario
        results.append(random_result)

        # RuleBased baseline
        rule_result = evaluator._run_episodes("RuleBased", agent=RuleBasedAgent())
        rule_result.scenario = args.scenario
        results.append(rule_result)

        # PPO (trained or untrained)
        if args.model:
            ppo_result = evaluator.evaluate_ppo(Path(args.model))
        else:
            ppo_result = evaluator._run_episodes("PPO", agent=None)
        ppo_result.scenario = args.scenario
        results.append(ppo_result)

    print(f"\nScenario: {args.scenario} ({args.episodes} episodes, seed={args.seed})")
    print(ModelEvaluator.format_table(results))
    return results


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    run_evaluation(args)


if __name__ == "__main__":
    main()
