"""Model evaluator for comparing PPO, Random, and RuleBased strategies."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from bluesky_pettingzoo.training.metrics import ExtendedMetrics


@dataclass
class EvalResult:
    """Evaluation result for a single strategy."""

    strategy: str
    scenario: str
    mean_reward: float
    std_reward: float
    min_reward: float
    max_reward: float
    mean_steps: float
    arrival_rate: float
    nmac_rate: float
    num_episodes: int
    seed: int
    extended_metrics: "ExtendedMetrics | None" = None


class ModelEvaluator:
    """Evaluates trained PPO models and baseline strategies."""

    def __init__(
        self,
        env_factory: Callable[[], Any],
        num_episodes: int = 20,
        max_steps: int = 100,
        seed: int = 42,
    ) -> None:
        self._env_factory = env_factory
        self._num_episodes = num_episodes
        self._max_steps = max_steps
        self._seed = seed

    def evaluate_ppo(self, model_path: Path) -> EvalResult:
        """Load a trained PPO model and evaluate it."""
        from stable_baselines3 import PPO

        model = PPO.load(str(model_path))
        return self._run_episodes("PPO", model=model)

    def evaluate_sac(self, model_path: Path) -> EvalResult:
        """Load a trained SAC model and evaluate it."""
        from stable_baselines3 import SAC

        model = SAC.load(str(model_path))
        return self._run_episodes("SAC", model=model)

    def evaluate_td3(self, model_path: Path) -> EvalResult:
        """Load a trained TD3 model and evaluate it."""
        from stable_baselines3 import TD3

        model = TD3.load(str(model_path))
        return self._run_episodes("TD3", model=model)

    def evaluate_ddpg(self, model_path: Path) -> EvalResult:
        """Load a trained DDPG model and evaluate it."""
        from stable_baselines3 import DDPG

        model = DDPG.load(str(model_path))
        return self._run_episodes("DDPG", model=model)

    def evaluate_random(self) -> EvalResult:
        """Evaluate a random-action strategy."""
        return self._run_episodes("Random", agent=None)

    def evaluate_rule_based(self) -> EvalResult:
        """Evaluate the rule-based strategy."""
        from bluesky_pettingzoo.agents.rule_based_agent import RuleBasedAgent

        return self._run_episodes("RuleBased", agent=RuleBasedAgent())

    def compare_all(self) -> list[EvalResult]:
        """Evaluate Random, RuleBased, and PPO (using untrained model)."""
        return [
            self.evaluate_random(),
            self.evaluate_rule_based(),
            self._run_episodes("PPO", agent=None),
        ]

    def _run_episodes(
        self,
        strategy: str,
        model: Any = None,
        agent: Any = None,
    ) -> EvalResult:
        """Run N episodes with the given strategy and aggregate results."""
        rng = np.random.RandomState(self._seed)
        rewards: list[float] = []
        steps_list: list[int] = []
        arrivals = 0
        nmacs = 0

        for _ in range(self._num_episodes):
            env = self._env_factory()
            try:
                # Handle VecNormalize-wrapped environments (no seed argument)
                if hasattr(env, "training") and hasattr(env, "norm_reward"):
                    # VecNormalize wrapper - returns OrderedDict directly
                    obs = env.reset()
                else:
                    obs, _ = env.reset(seed=rng.randint(0, 2**31))
                total_reward = 0.0
                episode_steps = 0
                arrived = False
                nmac = False
                step_info: Any = {}

                for step in range(self._max_steps):
                    if model is not None:
                        action, _ = model.predict(obs, deterministic=True)
                        # Handle VecNormalize (4 return values: obs, reward, done, info)
                        if hasattr(env, "training") and hasattr(env, "norm_reward"):
                            step_result = env.step(action)
                            obs = step_result[0]
                            reward = step_result[1]
                            done = step_result[2]
                            terminated = done
                            truncated = False
                            step_info = step_result[3] if len(step_result) > 3 else {}
                        else:
                            obs, reward, terminated, truncated, step_info = env.step(action)
                        total_reward += float(reward)
                    elif agent is not None:
                        # Multi-agent path: obs is {agent_id: obs_dict}
                        if hasattr(env, "agents") and env.agents:
                            if not isinstance(obs, dict):
                                obs = obs[0] if isinstance(obs, tuple) else obs
                            action_spaces = {aid: env.action_space(aid) for aid in env.agents}
                            actions = agent.act(obs, action_spaces)
                            obs, step_rewards, terminations, truncations, step_info = env.step(actions)
                            total_reward += (
                                sum(step_rewards.values())
                                if isinstance(step_rewards, dict)
                                else float(step_rewards)
                            )
                            terminated = (
                                any(terminations.values())
                                if isinstance(terminations, dict)
                                else bool(terminations)
                            )
                            truncated = (
                                any(truncations.values())
                                if isinstance(truncations, dict)
                                else bool(truncations)
                            )
                        elif hasattr(env, "_env") and hasattr(env, "_ego"):
                            # SingleAgentGymWrapper path: wrap obs for multi-agent agent
                            ego = env._ego
                            action_spaces = {ego: env.action_space}
                            actions = agent.act({ego: obs}, action_spaces)
                            ego_action = actions.get(ego, env.action_space.sample())
                            obs, reward, terminated, truncated, step_info = env.step(ego_action)
                            total_reward += float(reward)
                        else:
                            break
                    else:
                        # Random action for multi-agent environment
                        if hasattr(env, "agents") and env.agents:
                            if not isinstance(obs, dict):
                                obs = obs[0] if isinstance(obs, tuple) else obs
                            actions = {aid: env.action_space(aid).sample() for aid in env.agents}
                            obs, step_rewards, terminations, truncations, step_info = env.step(actions)
                            total_reward += (
                                sum(step_rewards.values())
                                if isinstance(step_rewards, dict)
                                else float(step_rewards)
                            )
                            terminated = (
                                any(terminations.values())
                                if isinstance(terminations, dict)
                                else bool(terminations)
                            )
                            truncated = (
                                any(truncations.values())
                                if isinstance(truncations, dict)
                                else bool(truncations)
                            )
                        elif hasattr(env, "_env") and hasattr(env, "_ego"):
                            # SingleAgentGymWrapper: random action for ego agent
                            ego = env._ego
                            action = env.action_space.sample()
                            obs, reward, terminated, truncated, step_info = env.step(action)
                            total_reward += float(reward)
                        else:
                            agents = getattr(env, "agents", None)
                            action = env.action_space(agents[0]).sample() if agents else None
                            if action is not None and agents is not None:
                                obs, reward, terminated, truncated, step_info = env.step(
                                    {agents[0]: action}
                                )
                                total_reward += (
                                    sum(reward.values())
                                    if isinstance(reward, dict)
                                    else float(reward)
                                )
                                terminated = (
                                    any(terminated.values())
                                    if isinstance(terminated, dict)
                                    else bool(terminated)
                                )
                                truncated = (
                                    any(truncated.values())
                                    if isinstance(truncated, dict)
                                    else bool(truncated)
                                )

                    episode_steps = step + 1

                    if terminated:
                        arrived, nmac = self._extract_termination_reason(step_info)
                        break
                    if truncated:
                        break

                rewards.append(total_reward)
                steps_list.append(episode_steps)
                if arrived:
                    arrivals += 1
                if nmac:
                    nmacs += 1
            finally:
                env.close()

        n = len(rewards) if rewards else 1
        return EvalResult(
            strategy=strategy,
            scenario="",
            mean_reward=float(np.mean(rewards)) if rewards else 0.0,
            std_reward=float(np.std(rewards)) if rewards else 0.0,
            min_reward=float(np.min(rewards)) if rewards else 0.0,
            max_reward=float(np.max(rewards)) if rewards else 0.0,
            mean_steps=float(np.mean(steps_list)) if steps_list else 0.0,
            arrival_rate=arrivals / n,
            nmac_rate=nmacs / n,
            num_episodes=self._num_episodes,
            seed=self._seed,
        )

    @staticmethod
    def _extract_termination_reason(info: Any) -> tuple[bool, bool]:
        """Extract arrival and NMAC flags from step info.

        Handles both single-agent (flat dict) and multi-agent (dict of dicts) info.

        Args:
            info: Info dict from env.step()

        Returns:
            Tuple of (arrived, nmac) booleans.
        """
        arrived = False
        nmac = False
        if isinstance(info, dict):
            # Single-agent: info is {termination_reason: "arrival", ...}
            reason = info.get("termination_reason")
            if reason == "arrival":
                arrived = True
            elif reason == "nmac":
                nmac = True
            # Multi-agent: info is {agent_id: {termination_reason: ...}, ...}
            for v in info.values():
                if isinstance(v, dict):
                    reason = v.get("termination_reason")
                    if reason == "arrival":
                        arrived = True
                    elif reason == "nmac":
                        nmac = True
        return arrived, nmac

    @staticmethod
    def format_table(results: list[EvalResult]) -> str:
        """Format evaluation results as a comparison table."""
        header = (
            f"{'Strategy':<12} {'MeanReward':>12} {'StdReward':>10} "
            f"{'Min':>10} {'Max':>10} {'MeanSteps':>10} {'Arrival%':>10} {'NMAC%':>8}"
        )
        sep = "-" * len(header)
        lines = [sep, header, sep]
        for r in results:
            lines.append(
                f"{r.strategy:<12} {r.mean_reward:>12.2f} {r.std_reward:>10.2f} "
                f"{r.min_reward:>10.2f} {r.max_reward:>10.2f} "
                f"{r.mean_steps:>10.1f} {r.arrival_rate:>9.1%} {r.nmac_rate:>7.1%}"
            )
        lines.append(sep)
        return "\n".join(lines)
