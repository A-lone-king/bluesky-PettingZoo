"""Evaluate all algorithms across scenarios.

Usage:
    python scripts/evaluate_all.py --scenario HorizontalCR
    python scripts/evaluate_all.py --scenario HorizontalCR --model-dir models
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from bluesky_pettingzoo.actions.translator import ActionTranslator
from bluesky_pettingzoo.envs.parallel_env import BlueSkyMARLEnv
from bluesky_pettingzoo.envs.scenarios.base import BaseScenario
from bluesky_pettingzoo.observations.manager import ObservationManager
from bluesky_pettingzoo.rewards.calculator import RewardCalculator
from bluesky_pettingzoo.rewards.components.capacity import CapacityPenalty
from bluesky_pettingzoo.rewards.components.conflict import ConflictPenalty
from bluesky_pettingzoo.rewards.components.efficiency import EfficiencyReward
from bluesky_pettingzoo.rewards.components.obstacle_intrusion import ObstacleIntrusion
from bluesky_pettingzoo.rewards.components.smoothness import SmoothnessPenalty
from bluesky_pettingzoo.training.evaluator import EvalResult, ModelEvaluator

from bluesky_pettingzoo.bluesky.wrapper import BlueSkyWrapper
from tests.helpers.env_factory import make_config, write_rewards_yaml


def _make_env_factory(
    tmp_path: Path,
    scenario: BaseScenario,
    num_aircraft: int,
    max_steps: int,
) -> callable:
    """Create environment factory for evaluation."""

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
            calc.register(CapacityPenalty(merged), weight=1.0)

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


def evaluate_all_scenarios(
    scenario_name: str,
    num_aircraft: int,
    max_steps: int,
    num_episodes: int,
    model_dir: Path,
    seed: int = 42,
) -> list[EvalResult]:
    """Evaluate all algorithms for a single scenario.

    Returns results for Random, RuleBased, PPO, SAC, TD3, DDPG.
    Missing models are skipped gracefully.
    """
    from scripts.train_ppo_scenarios import _resolve_scenario

    scenario = _resolve_scenario(scenario_name, num_aircraft, seed)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        env_factory = _make_env_factory(tmp_path, scenario, num_aircraft, max_steps)
        evaluator = ModelEvaluator(
            env_factory=env_factory,
            num_episodes=num_episodes,
            max_steps=max_steps,
            seed=seed,
        )

        results: list[EvalResult] = []

        # Random baseline
        random_result = evaluator.evaluate_random()
        random_result.scenario = scenario_name
        results.append(random_result)

        # RuleBased baseline
        rule_result = evaluator.evaluate_rule_based()
        rule_result.scenario = scenario_name
        results.append(rule_result)

        # Trained models
        algo_map = {
            "PPO": ("PPO", evaluator.evaluate_ppo),
            "SAC": ("SAC", evaluator.evaluate_sac),
            "TD3": ("TD3", evaluator.evaluate_td3),
            "DDPG": ("DDPG", evaluator.evaluate_ddpg),
        }

        for algo_name, (dir_name, eval_fn) in algo_map.items():
            model_path = model_dir / scenario_name / dir_name / "checkpoint_final.zip"
            if not model_path.exists():
                model_path = model_dir / scenario_name / "checkpoint_final.zip"
            try:
                result = eval_fn(model_path)
                result.scenario = scenario_name
                results.append(result)
            except (FileNotFoundError, Exception):
                pass

    return results


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Evaluate all algorithms")
    parser.add_argument("--scenario", type=str, default="HorizontalCR")
    parser.add_argument("--num-aircraft", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--model-dir", type=str, default="models")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    results = evaluate_all_scenarios(
        scenario_name=args.scenario,
        num_aircraft=args.num_aircraft,
        max_steps=args.max_steps,
        num_episodes=args.episodes,
        model_dir=Path(args.model_dir),
        seed=args.seed,
    )

    print(f"\nScenario: {args.scenario} ({args.episodes} episodes, seed={args.seed})")
    print(ModelEvaluator.format_table(results))


if __name__ == "__main__":
    main()
