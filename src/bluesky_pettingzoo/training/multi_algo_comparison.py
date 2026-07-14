"""Multi-algorithm comparison framework for reward tuning validation.

Trains PPO/SAC/TD3/DDPG on multiple scenarios to verify that reward
tuning parameters (reward-tune-001~003) are effective across all
algorithms, not just PPO.

Usage:
    from bluesky_pettingzoo.training.multi_algo_comparison import MultiAlgoComparison

    comparison = MultiAlgoComparison(
        scenarios=["HorizontalCR", "VerticalCR"],
        algorithms=["PPO", "SAC", "TD3", "DDPG"],
        total_timesteps=500_000,
    )
    summary = comparison.run()
    comparison.generate_report(summary)
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml


@dataclass
class AlgoScenarioResult:
    """Training result for a single algorithm-scenario pair."""

    algorithm: str
    scenario: str
    initial_reward: float
    final_reward: float
    reward_history: list[float]
    convergence_threshold: float
    converged: bool
    total_timesteps: int
    seed: int

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for JSON export."""
        return {
            "algorithm": self.algorithm,
            "scenario": self.scenario,
            "initial_reward": self.initial_reward,
            "final_reward": self.final_reward,
            "convergence_threshold": self.convergence_threshold,
            "converged": self.converged,
            "total_timesteps": self.total_timesteps,
            "seed": self.seed,
            "reward_history": self.reward_history,
        }


@dataclass
class ComparisonSummary:
    """Aggregated results across all algorithm-scenario pairs."""

    results: list[AlgoScenarioResult]
    scenarios: list[str]
    algorithms: list[str]
    timestamp: str
    total_timesteps: int
    all_converged: bool = field(init=False)

    def __post_init__(self) -> None:
        self.all_converged = all(r.converged for r in self.results)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for JSON export."""
        return {
            "scenarios": self.scenarios,
            "algorithms": self.algorithms,
            "timestamp": self.timestamp,
            "total_timesteps": self.total_timesteps,
            "all_converged": self.all_converged,
            "results": [r.to_dict() for r in self.results],
        }


# Convergence thresholds per scenario (from feature_list.json verification criteria)
CONVERGENCE_THRESHOLDS: dict[str, float] = {
    "HorizontalCR": -10.0,
    "VerticalCR": 0.0,
}

DEFAULT_ALGORITHMS: list[str] = ["PPO", "SAC", "TD3", "DDPG"]
DEFAULT_SCENARIOS: list[str] = ["HorizontalCR", "VerticalCR"]
DEFAULT_TIMESTEPS: int = 500_000


class MultiAlgoComparison:
    """Multi-algorithm comparison runner for reward tuning validation.

    Trains each algorithm on each scenario and verifies convergence
    against scenario-specific thresholds.
    """

    def __init__(
        self,
        scenarios: list[str] | None = None,
        algorithms: list[str] | None = None,
        total_timesteps: int = DEFAULT_TIMESTEPS,
        seed: int = 42,
        convergence_thresholds: dict[str, float] | None = None,
        eval_episodes: int = 5,
        log_interval: int = 10_000,
    ) -> None:
        """Initialize multi-algorithm comparison.

        Args:
            scenarios: List of scenario names to test
            algorithms: List of algorithm names to test
            total_timesteps: Training timesteps per algorithm-scenario pair
            seed: Random seed for reproducibility
            convergence_thresholds: Per-scenario final_reward thresholds
            eval_episodes: Number of episodes for reward evaluation
            log_interval: Timesteps between reward logging
        """
        self.scenarios = scenarios or DEFAULT_SCENARIOS
        self.algorithms = algorithms or DEFAULT_ALGORITHMS
        self.total_timesteps = total_timesteps
        self.seed = seed
        self.thresholds = convergence_thresholds or CONVERGENCE_THRESHOLDS
        self.eval_episodes = eval_episodes
        self.log_interval = log_interval

    def run(
        self,
        train_fn: Callable[[str, str, int, int], tuple[float, float, list[float]]] | None = None,
        save_results: bool = True,
        results_dir: str = "results/multi_algo",
    ) -> ComparisonSummary:
        """Run comparison across all algorithm-scenario pairs.

        Args:
            train_fn: Optional custom training function.
                Signature: (algorithm, scenario, timesteps, seed) -> (initial_reward, final_reward, history)
                If None, uses the default SB3-based training.
            save_results: Whether to save results to disk
            results_dir: Directory to save results

        Returns:
            ComparisonSummary with all results
        """
        all_results: list[AlgoScenarioResult] = []

        for scenario in self.scenarios:
            threshold = self.thresholds.get(scenario, -float("inf"))
            for algorithm in self.algorithms:
                if train_fn is not None:
                    initial_reward, final_reward, history = train_fn(
                        algorithm, scenario, self.total_timesteps, self.seed
                    )
                else:
                    initial_reward, final_reward, history = self._train(
                        algorithm, scenario, self.total_timesteps, self.seed
                    )

                converged = final_reward > threshold

                all_results.append(AlgoScenarioResult(
                    algorithm=algorithm,
                    scenario=scenario,
                    initial_reward=initial_reward,
                    final_reward=final_reward,
                    reward_history=history,
                    convergence_threshold=threshold,
                    converged=converged,
                    total_timesteps=self.total_timesteps,
                    seed=self.seed,
                ))

        summary = ComparisonSummary(
            results=all_results,
            scenarios=self.scenarios,
            algorithms=self.algorithms,
            timestamp=np.datetime64("now").astype(str),
            total_timesteps=self.total_timesteps,
        )

        if save_results:
            self._save_results(summary, results_dir)

        return summary

    def _train(
        self,
        algorithm: str,
        scenario_name: str,
        total_timesteps: int,
        seed: int,
    ) -> tuple[float, float, list[float]]:
        """Train a single algorithm on a single scenario using SB3.

        Args:
            algorithm: Algorithm name (PPO, SAC, TD3, DDPG)
            scenario_name: Scenario name (HorizontalCR, VerticalCR)
            total_timesteps: Total training timesteps
            seed: Random seed

        Returns:
            Tuple of (initial_reward, final_reward, reward_history)
        """
        import tempfile

        from scripts.train_ppo_scenarios import (
            PPOTrainer,
            _resolve_scenario,
            make_scenario_env_factory,
        )
        from bluesky_pettingzoo.training.algorithm_factory import AlgorithmFactory

        action_space = "continuous" if algorithm in ("SAC", "TD3", "DDPG") else "discrete"

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            scenario = _resolve_scenario(scenario_name, num_aircraft=3, seed=seed)
            if action_space == "continuous":
                scenario.action_space_type = "continuous"

            factory = make_scenario_env_factory(
                tmp_path, scenario, num_aircraft=3, max_steps=50,
            )
            env = factory()

            config_path = Path("config/algorithms.yaml")
            model = AlgorithmFactory.from_yaml(
                algorithm=algorithm,
                policy="MultiInputPolicy",
                config_path=config_path,
                env=env,
                seed=seed,
                verbose=0,
            )

            # Evaluate initial reward
            initial_reward = self._evaluate_reward(env, model, seed)

            # Train with periodic logging
            reward_history: list[float] = [initial_reward]
            num_logs = max(total_timesteps // self.log_interval, 1)
            steps_per_log = max(total_timesteps // num_logs, 1)

            for i in range(num_logs):
                remaining = min(steps_per_log, total_timesteps - i * steps_per_log)
                if remaining <= 0:
                    break
                model.learn(
                    total_timesteps=remaining,
                    reset_num_timesteps=False,
                )
                mid_reward = self._evaluate_reward(env, model, seed)
                reward_history.append(mid_reward)

            # Evaluate final reward
            final_reward = self._evaluate_reward(env, model, seed, n_episodes=self.eval_episodes)

            env.close()

        return initial_reward, final_reward, reward_history

    def _evaluate_reward(self, env: Any, model: Any, seed: int, n_episodes: int = 3) -> float:
        """Evaluate mean reward over several episodes.

        Args:
            env: Environment instance
            model: Trained model
            seed: Random seed
            n_episodes: Number of evaluation episodes

        Returns:
            Mean total reward
        """
        rewards: list[float] = []
        for _ in range(n_episodes):
            obs, _ = env.reset(seed=seed)
            total_reward = 0.0
            done = False
            steps = 0
            while not done and steps < 100:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, infos = env.step(action)
                total_reward += float(sum(reward.values())) if isinstance(reward, dict) else float(reward)
                done = all(terminated.values()) if isinstance(terminated, dict) else bool(terminated)
                if isinstance(truncated, dict) and any(truncated.values()):
                    break
                steps += 1
            rewards.append(total_reward)
        return float(np.mean(rewards)) if rewards else 0.0

    def _save_results(self, summary: ComparisonSummary, results_dir: str) -> None:
        """Save comparison results to JSON.

        Args:
            summary: Comparison summary to save
            results_dir: Directory to save results
        """
        path = Path(results_dir)
        path.mkdir(parents=True, exist_ok=True)

        file_path = path / "multi_algo_comparison.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(summary.to_dict(), f, indent=2, ensure_ascii=False)

    def generate_report(
        self,
        summary: ComparisonSummary,
        save_path: str = "results/multi_algo/comparison_report.md",
    ) -> str:
        """Generate Markdown comparison report.

        Args:
            summary: Comparison summary
            save_path: Path to save the report

        Returns:
            Report content as string
        """
        lines: list[str] = []
        lines.append("# 多算法对比验证调参效果报告")
        lines.append("")
        lines.append(f"- **训练步数**: {summary.total_timesteps:,}")
        lines.append(f"- **场景**: {', '.join(summary.scenarios)}")
        lines.append(f"- **算法**: {', '.join(summary.algorithms)}")
        lines.append(f"- **全部收敛**: {'✅ 是' if summary.all_converged else '❌ 否'}")
        lines.append(f"- **时间戳**: {summary.timestamp}")
        lines.append("")

        # Per-scenario comparison table
        for scenario in summary.scenarios:
            threshold = self.thresholds.get(scenario, -float("inf"))
            lines.append(f"## {scenario} (阈值: {threshold})")
            lines.append("")
            lines.append("| 算法 | 初始奖励 | 最终奖励 | 收敛 | 变化量 |")
            lines.append("|------|---------|---------|------|--------|")

            scenario_results = [r for r in summary.results if r.scenario == scenario]
            for r in scenario_results:
                delta = r.final_reward - r.initial_reward
                converged_str = "✅" if r.converged else "❌"
                lines.append(
                    f"| {r.algorithm} | {r.initial_reward:.2f} | {r.final_reward:.2f} | "
                    f"{converged_str} | {delta:+.2f} |"
                )
            lines.append("")

        # Overall summary
        lines.append("## 总结")
        lines.append("")
        total_pairs = len(summary.results)
        converged_pairs = sum(1 for r in summary.results if r.converged)
        lines.append(f"- **算法-场景对**: {total_pairs}")
        lines.append(f"- **收敛数**: {converged_pairs}/{total_pairs}")
        lines.append(f"- **收敛率**: {converged_pairs / total_pairs * 100:.1f}%")
        lines.append("")

        # Algorithm-wise summary
        lines.append("## 各算法平均表现")
        lines.append("")
        lines.append("| 算法 | 平均最终奖励 | 平均变化量 | 全部收敛 |")
        lines.append("|------|------------|----------|---------|")

        for algo in summary.algorithms:
            algo_results = [r for r in summary.results if r.algorithm == algo]
            avg_final = float(np.mean([r.final_reward for r in algo_results]))
            avg_delta = float(np.mean([r.final_reward - r.initial_reward for r in algo_results]))
            all_conv = all(r.converged for r in algo_results)
            lines.append(
                f"| {algo} | {avg_final:.2f} | {avg_delta:+.2f} | {'✅' if all_conv else '❌'} |"
            )

        content = "\n".join(lines)

        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(content)

        return content

    def generate_training_curve_data(
        self,
        summary: ComparisonSummary,
        save_path: str = "results/multi_algo/training_curves.json",
    ) -> None:
        """Save training curve data for visualization.

        Args:
            summary: Comparison summary
            save_path: Path to save the curve data
        """
        curves: dict[str, dict[str, list[float]]] = {}

        for scenario in summary.scenarios:
            curves[scenario] = {}
            for algo in summary.algorithms:
                result = next(
                    (r for r in summary.results if r.scenario == scenario and r.algorithm == algo),
                    None,
                )
                if result:
                    curves[scenario][algo] = result.reward_history

        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(curves, f, indent=2)
