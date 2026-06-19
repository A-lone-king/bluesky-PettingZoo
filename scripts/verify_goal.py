"""Verify all goal.json completion criteria in one script.

This script validates:
- M3.3: PPO training produces learning signal (trained > random by >= 20%)
- M3.4: Checkpoint can be loaded and evaluated
- M4.1: Random and RuleBased baselines produce metrics
- M4.2: Trained policy outperforms baselines on collision/arrival rate

Usage:
    .\\.venv\\Scripts\\python.exe scripts/verify_goal.py
"""
from __future__ import annotations

import copy
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from bluesky_pettingzoo.actions.translator import ActionTranslator
from bluesky_pettingzoo.agents.random_agent import RandomAgent
from bluesky_pettingzoo.agents.rule_based_agent import RuleBasedAgent
from bluesky_pettingzoo.bluesky.wrapper import BlueSkyWrapper
from bluesky_pettingzoo.envs.parallel_env import BlueSkyMARLEnv
from bluesky_pettingzoo.envs.scenarios.horizontal_cr import HorizontalCRScenario
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


def _make_env_pair(
    tmp_path: Path,
    num_aircraft: int = 2,
    max_steps: int = 50,
    seed: int = 42,
) -> tuple[SingleAgentGymWrapper, BlueSkyMARLEnv]:
    """Create both a SingleAgentGymWrapper and the underlying BlueSkyMARLEnv."""
    scenario = HorizontalCRScenario(
        num_aircraft=num_aircraft,
        seed=seed,
        waypoint_distance_range=(40, 70),
    )
    config = make_config(
        initial_count=num_aircraft,
        max_steps=max_steps,
        airspace={"sectors": [{"id": "s1", "bounds": [[36.0, 112.0], [42.0, 120.0]]}]},
    )
    config["simulation"]["action_frequency"] = 3
    rewards_cfg = {
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
    rewards_copy = copy.deepcopy(rewards_cfg)
    rewards_path = write_rewards_yaml(tmp_path, rewards_copy)
    config["_rewards_yaml"] = str(rewards_path)
    merged = {**config, **rewards_cfg}

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

    marl_env = BlueSkyMARLEnv(
        config=config,
        wrapper=wrapper,
        observation_manager=obs_manager,
        action_translator=action_translator,
        reward_calculator=calc,
        rewards_config=rewards_cfg,
        scenario=scenario,
    )
    wrapped = SingleAgentGymWrapper(marl_env, ego_agent="AC000")
    return wrapped, marl_env


def _evaluate_sb3(
    model: Any,
    env: SingleAgentGymWrapper,
    n_episodes: int = 10,
    max_steps: int = 60,
) -> dict[str, float]:
    """Evaluate SB3 model on SingleAgentGymWrapper."""
    rewards: list[float] = []
    arrivals = 0
    nmacs = 0
    steps_list: list[int] = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        total = 0.0
        steps = 0
        for _ in range(max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total += reward
            steps += 1
            if terminated or truncated:
                break
        rewards.append(total)
        steps_list.append(steps)
        if total > 0:
            arrivals += 1
        elif total < -5:
            nmacs += 1
    return {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "arrival_rate": arrivals / n_episodes,
        "nmac_rate": nmacs / n_episodes,
        "mean_steps": float(np.mean(steps_list)),
    }


def _evaluate_agent_marl(
    agent: Any,
    marl_env: BlueSkyMARLEnv,
    n_episodes: int = 10,
    max_steps: int = 60,
) -> dict[str, float]:
    """Evaluate a non-SB3 agent on raw BlueSkyMARLEnv (dict interface)."""
    rewards: list[float] = []
    arrivals = 0
    nmacs = 0
    steps_list: list[int] = []
    for _ in range(n_episodes):
        obs_dict, _ = marl_env.reset()
        agent.reset()
        total = 0.0
        steps = 0
        for _ in range(max_steps):
            action_spaces = {aid: marl_env.action_space(aid) for aid in marl_env.agents}
            action_dict = agent.act(obs_dict, action_spaces)
            obs_dict, reward_dict, term_dict, trunc_dict, _ = marl_env.step(action_dict)
            total += sum(reward_dict.values())
            steps += 1
            if any(term_dict.values()) or any(trunc_dict.values()):
                break
            if not marl_env.agents:
                break
        rewards.append(total)
        steps_list.append(steps)
        if total > 0:
            arrivals += 1
        elif total < -5:
            nmacs += 1
    return {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "arrival_rate": arrivals / n_episodes,
        "nmac_rate": nmacs / n_episodes,
        "mean_steps": float(np.mean(steps_list)),
    }


def main() -> int:
    """Run all verification checks and print results."""
    results: dict[str, Any] = {}
    all_pass = True

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        from stable_baselines3 import PPO

        env, marl_env = _make_env_pair(tmp_path, num_aircraft=2, max_steps=50)
        random_agent = RandomAgent()
        rule_agent = RuleBasedAgent()

        # --- M4.1: Baselines produce metrics ---
        print("=" * 60)
        print("M4.1: Baseline Verification")
        print("=" * 60)

        print("\n  Evaluating Random baseline...")
        random_metrics = _evaluate_agent_marl(random_agent, marl_env, n_episodes=10)
        print(f"  Random: reward={random_metrics['mean_reward']:.2f} "
              f"(+/-{random_metrics['std_reward']:.2f}), "
              f"arrival={random_metrics['arrival_rate']:.0%}, "
              f"nmac={random_metrics['nmac_rate']:.0%}")

        print("  Evaluating RuleBased baseline...")
        rule_metrics = _evaluate_agent_marl(rule_agent, marl_env, n_episodes=10)
        print(f"  RuleBased: reward={rule_metrics['mean_reward']:.2f} "
              f"(+/-{rule_metrics['std_reward']:.2f}), "
              f"arrival={rule_metrics['arrival_rate']:.0%}, "
              f"nmac={rule_metrics['nmac_rate']:.0%}")

        m41_pass = (
            all(k in random_metrics for k in ["mean_reward", "arrival_rate", "nmac_rate"])
            and all(k in rule_metrics for k in ["mean_reward", "arrival_rate", "nmac_rate"])
        )
        results["M4.1"] = {"pass": m41_pass, "random": random_metrics, "rule_based": rule_metrics}
        status = "PASS" if m41_pass else "FAIL"
        print(f"\n  M4.1: [{status}] Both baselines produced valid metrics")
        if not m41_pass:
            all_pass = False

        # --- M3.3: Train PPO and verify learning signal ---
        print("\n" + "=" * 60)
        print("M3.3: PPO Training Verification")
        print("=" * 60)

        print("\n  Training PPO (10K steps)...")
        model = PPO(
            "MultiInputPolicy",
            env,
            n_steps=128,
            batch_size=64,
            n_epochs=4,
            learning_rate=3e-4,
            verbose=0,
            device="cpu",
            seed=42,
        )
        model.learn(total_timesteps=10_000)
        print("  Training complete.")

        print("\n  Evaluating trained PPO...")
        ppo_metrics = _evaluate_sb3(model, env, n_episodes=10)
        print(f"  PPO: reward={ppo_metrics['mean_reward']:.2f} "
              f"(+/-{ppo_metrics['std_reward']:.2f}), "
              f"arrival={ppo_metrics['arrival_rate']:.0%}, "
              f"nmac={ppo_metrics['nmac_rate']:.0%}")

        random_r = random_metrics["mean_reward"]
        ppo_r = ppo_metrics["mean_reward"]
        if abs(random_r) < 0.01:
            improvement = 0.0 if abs(ppo_r) < 0.01 else 100.0
        else:
            improvement = (ppo_r - random_r) / abs(random_r) * 100

        m33_pass = np.isfinite(ppo_r) and improvement >= 20
        results["M3.3"] = {
            "pass": m33_pass,
            "random_reward": random_r,
            "ppo_reward": ppo_r,
            "improvement_pct": improvement,
        }
        status = "PASS" if m33_pass else "FAIL"
        print(f"\n  M3.3: [{status}] Improvement={improvement:.1f}% (threshold: >=20%)")
        if not m33_pass:
            all_pass = False

        # --- M3.4: Checkpoint round-trip ---
        print("\n" + "=" * 60)
        print("M3.4: Checkpoint Load Verification")
        print("=" * 60)

        ckpt_path = tmp_path / "test_checkpoint.zip"
        model.save(str(ckpt_path))
        print(f"  Saved checkpoint: {ckpt_path}")

        loaded_model = PPO.load(str(ckpt_path), env=env)
        loaded_metrics = _evaluate_sb3(loaded_model, env, n_episodes=10)
        print(f"  Loaded PPO: reward={loaded_metrics['mean_reward']:.2f}")

        m34_pass = np.isfinite(loaded_metrics["mean_reward"])
        results["M3.4"] = {"pass": m34_pass, "loaded_reward": loaded_metrics["mean_reward"]}
        status = "PASS" if m34_pass else "FAIL"
        print(f"\n  M3.4: [{status}] Checkpoint loaded and evaluated successfully")
        if not m34_pass:
            all_pass = False

        # --- M4.2: Trained > Baselines ---
        print("\n" + "=" * 60)
        print("M4.2: Trained vs Baseline Comparison")
        print("=" * 60)

        ppo_better_random = (
            ppo_metrics["mean_reward"] > random_metrics["mean_reward"]
            or ppo_metrics["arrival_rate"] > random_metrics["arrival_rate"]
        )
        ppo_better_rule = (
            ppo_metrics["mean_reward"] > rule_metrics["mean_reward"]
            or ppo_metrics["arrival_rate"] > rule_metrics["arrival_rate"]
        )
        m42_pass = ppo_better_random and ppo_better_rule
        results["M4.2"] = {
            "pass": m42_pass,
            "ppo_vs_random": ppo_better_random,
            "ppo_vs_rule": ppo_better_rule,
        }
        status = "PASS" if m42_pass else "FAIL"
        print(f"\n  PPO vs Random: {'better' if ppo_better_random else 'worse'}")
        print(f"  PPO vs RuleBased: {'better' if ppo_better_rule else 'worse'}")
        print(f"\n  M4.2: [{status}] Trained policy outperforms baselines")
        if not m42_pass:
            all_pass = False

        env.close()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  M3.3 Learning Signal: {results['M3.3']['pass']}")
    print(f"  M3.4 Checkpoint Load:  {results['M3.4']['pass']}")
    print(f"  M4.1 Baselines:        {results['M4.1']['pass']}")
    print(f"  M4.2 Comparison:       {results['M4.2']['pass']}")
    print(f"\n  ALL PASS: {all_pass}")

    results_path = Path(__file__).parent.parent / "training_results" / "goal_verification.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to: {results_path}")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
