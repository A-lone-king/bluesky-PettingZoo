"""Ablation experiment runner for action space validation."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class AblationResult:
    """Result of a single ablation experiment."""

    experiment_id: str
    name: str
    action_type: str
    action_dims: list[int] | int
    action_labels: list[str]
    total_timesteps: int
    episode_rewards: list[float] = field(default_factory=list)
    episode_lengths: list[int] = field(default_factory=list)
    conflict_rates: list[float] = field(default_factory=list)
    efficiency_scores: list[float] = field(default_factory=list)
    training_time_seconds: float = 0.0
    convergence_step: int | None = None

    @property
    def mean_reward(self) -> float:
        """Mean episode reward."""
        return float(np.mean(self.episode_rewards)) if self.episode_rewards else 0.0

    @property
    def std_reward(self) -> float:
        """Standard deviation of episode rewards."""
        return float(np.std(self.episode_rewards)) if self.episode_rewards else 0.0

    @property
    def mean_length(self) -> float:
        """Mean episode length."""
        return float(np.mean(self.episode_lengths)) if self.episode_lengths else 0.0

    @property
    def mean_conflict_rate(self) -> float:
        """Mean conflict rate."""
        return float(np.mean(self.conflict_rates)) if self.conflict_rates else 0.0

    @property
    def mean_efficiency(self) -> float:
        """Mean efficiency score."""
        return float(np.mean(self.efficiency_scores)) if self.efficiency_scores else 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "experiment_id": self.experiment_id,
            "name": self.name,
            "action_type": self.action_type,
            "action_dims": self.action_dims,
            "action_labels": self.action_labels,
            "total_timesteps": self.total_timesteps,
            "mean_reward": self.mean_reward,
            "std_reward": self.std_reward,
            "mean_length": self.mean_length,
            "mean_conflict_rate": self.mean_conflict_rate,
            "mean_efficiency": self.mean_efficiency,
            "training_time_seconds": self.training_time_seconds,
            "convergence_step": self.convergence_step,
        }


class AblationRunner:
    """Runs ablation experiments for action space validation.

    Supports discrete and continuous action spaces with different
    dimension configurations.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self._config = config
        self._experiments: dict[str, dict[str, Any]] = config.get("experiments", {})
        self._training_config = config.get("training", {})
        self._output_config = config.get("output", {})

    def get_experiment_config(self, experiment_id: str) -> dict[str, Any] | None:
        """Get configuration for a specific experiment."""
        return self._experiments.get(experiment_id)

    def list_experiments(self) -> list[str]:
        """List all available experiment IDs."""
        return list(self._experiments.keys())

    def list_discrete_experiments(self) -> list[str]:
        """List discrete action space experiments."""
        return [
            eid
            for eid, cfg in self._experiments.items()
            if cfg.get("action_type") == "discrete"
        ]

    def list_continuous_experiments(self) -> list[str]:
        """List continuous action space experiments."""
        return [
            eid
            for eid, cfg in self._experiments.items()
            if cfg.get("action_type") == "continuous"
        ]

    def create_action_space_config(
        self, experiment_id: str
    ) -> dict[str, Any]:
        """Create action space configuration for an experiment.

        Returns a config dict that can be merged with the base environment config.
        """
        exp_cfg = self._experiments.get(experiment_id)
        if exp_cfg is None:
            raise ValueError(f"Unknown experiment: {experiment_id}")

        action_type = exp_cfg["action_type"]
        action_dims = exp_cfg["action_dims"]

        config: dict[str, Any] = {
            "action": {
                "type": action_type,
            }
        }

        if action_type == "discrete":
            config["action"]["dims"] = action_dims
        else:
            config["action"]["dims"] = action_dims
            # Add scale parameters if specified
            if "heading_scale" in exp_cfg:
                config["continuous_action"] = {
                    "heading_scale": exp_cfg["heading_scale"]
                }
            if "speed_scale" in exp_cfg:
                config.setdefault("continuous_action", {})["speed_scale"] = exp_cfg[
                    "speed_scale"
                ]
            if "altitude_scale" in exp_cfg:
                config.setdefault("continuous_action", {})[
                    "altitude_scale"
                ] = exp_cfg["altitude_scale"]

        return config

    def run_experiment(
        self,
        experiment_id: str,
        env_factory: Any,  # noqa: ANN401
        agent_factory: Any,  # noqa: ANN401
        num_episodes: int = 10,
        max_steps_per_episode: int = 500,
    ) -> AblationResult:
        """Run a single ablation experiment.

        Args:
            experiment_id: Experiment identifier.
            env_factory: Callable that creates an environment with given config.
            agent_factory: Callable that creates an agent for the environment.
            num_episodes: Number of episodes to evaluate.
            max_steps_per_episode: Maximum steps per episode.

        Returns:
            AblationResult with evaluation metrics.
        """
        exp_cfg = self._experiments.get(experiment_id)
        if exp_cfg is None:
            raise ValueError(f"Unknown experiment: {experiment_id}")

        start_time = time.time()

        # Create action space config
        action_config = self.create_action_space_config(experiment_id)

        # Create environment and agent
        env = env_factory(action_config)
        agent = agent_factory(env)

        # Run evaluation episodes
        episode_rewards: list[float] = []
        episode_lengths: list[int] = []
        conflict_rates: list[float] = []
        efficiency_scores: list[float] = []

        for episode in range(num_episodes):
            obs, infos = env.reset()
            episode_reward = 0.0
            step_count = 0
            conflicts = 0

            for step in range(max_steps_per_episode):
                # Get agent action
                actions = {}
                for agent_id in env.agents:
                    agent_obs = obs[agent_id]
                    action = agent.predict(agent_obs)
                    actions[agent_id] = action

                # Step environment
                next_obs, rewards, terminations, truncations, next_infos = env.step(
                    actions
                )

                # Accumulate metrics
                for agent_id in rewards:
                    episode_reward += rewards[agent_id]

                # Check for conflicts
                for agent_id in next_infos:
                    info = next_infos[agent_id]
                    if isinstance(info, dict) and info.get("conflict_status") in (
                        "nmac",
                        "warning",
                    ):
                        conflicts += 1

                step_count += 1
                obs = next_obs

                # Check if episode is done
                if all(terminations.values()) or all(truncations.values()):
                    break

            # Compute episode metrics
            episode_rewards.append(episode_reward)
            episode_lengths.append(step_count)
            conflict_rate = conflicts / max(1, step_count * len(env.agents))
            conflict_rates.append(conflict_rate)

            # Efficiency: ratio of completed episodes without conflicts
            efficiency = 1.0 - conflict_rate
            efficiency_scores.append(efficiency)

        training_time = time.time() - start_time

        # Create result
        result = AblationResult(
            experiment_id=experiment_id,
            name=exp_cfg["name"],
            action_type=exp_cfg["action_type"],
            action_dims=exp_cfg["action_dims"],
            action_labels=exp_cfg["action_labels"],
            total_timesteps=num_episodes * max_steps_per_episode,
            episode_rewards=episode_rewards,
            episode_lengths=episode_lengths,
            conflict_rates=conflict_rates,
            efficiency_scores=efficiency_scores,
            training_time_seconds=training_time,
        )

        return result


class AblationReporter:
    """Generates comparison reports from ablation results."""

    def __init__(self, output_dir: str = "results/ablation") -> None:
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def generate_report(
        self,
        results: list[AblationResult],
        filename: str = "ablation_report.md",
    ) -> str:
        """Generate a markdown comparison report.

        Args:
            results: List of ablation results.
            filename: Output filename.

        Returns:
            Path to the generated report.
        """
        lines: list[str] = []
        lines.append("# Action Space Ablation Report")
        lines.append("")
        lines.append(
            f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}"
        )
        lines.append("")

        # Summary table
        lines.append("## Summary")
        lines.append("")
        lines.append(
            "| Experiment | Type | Dims | Mean Reward | Std Reward | Conflict Rate | Efficiency |"
        )
        lines.append(
            "|------------|------|------|-------------|------------|---------------|------------|"
        )

        for r in results:
            dims_str = str(r.action_dims) if isinstance(r.action_dims, list) else str(r.action_dims)
            lines.append(
                f"| {r.name} | {r.action_type} | {dims_str} | "
                f"{r.mean_reward:.2f} | {r.std_reward:.2f} | "
                f"{r.mean_conflict_rate:.3f} | {r.mean_efficiency:.3f} |"
            )

        lines.append("")

        # Detailed results
        lines.append("## Detailed Results")
        lines.append("")

        for r in results:
            lines.append(f"### {r.name}")
            lines.append("")
            lines.append(f"- **Action Type**: {r.action_type}")
            dims_str = str(r.action_dims) if isinstance(r.action_dims, list) else str(r.action_dims)
            lines.append(f"- **Action Dimensions**: {dims_str}")
            lines.append(f"- **Action Labels**: {', '.join(r.action_labels)}")
            lines.append(f"- **Total Timesteps**: {r.total_timesteps}")
            lines.append(f"- **Mean Reward**: {r.mean_reward:.2f} +/- {r.std_reward:.2f}")
            lines.append(f"- **Mean Episode Length**: {r.mean_length:.1f}")
            lines.append(f"- **Mean Conflict Rate**: {r.mean_conflict_rate:.3f}")
            lines.append(f"- **Mean Efficiency**: {r.mean_efficiency:.3f}")
            lines.append(f"- **Training Time**: {r.training_time_seconds:.1f}s")
            lines.append("")

        # Comparison insights
        lines.append("## Insights")
        lines.append("")

        if len(results) >= 2:
            # Find best performing
            best_reward = max(results, key=lambda x: x.mean_reward)
            best_efficiency = max(results, key=lambda x: x.mean_efficiency)
            fastest = min(results, key=lambda x: x.training_time_seconds)

            lines.append(
                f"- **Highest Reward**: {best_reward.name} "
                f"({best_reward.mean_reward:.2f})"
            )
            lines.append(
                f"- **Best Efficiency**: {best_efficiency.name} "
                f"({best_efficiency.mean_efficiency:.3f})"
            )
            lines.append(
                f"- **Fastest Training**: {fastest.name} "
                f"({fastest.training_time_seconds:.1f}s)"
            )

            # Discrete vs Continuous comparison
            discrete_results = [r for r in results if r.action_type == "discrete"]
            continuous_results = [r for r in results if r.action_type == "continuous"]

            if discrete_results and continuous_results:
                avg_discrete_reward = float(np.mean([r.mean_reward for r in discrete_results]))
                avg_continuous_reward = float(
                    np.mean([r.mean_reward for r in continuous_results])
                )
                lines.append(
                    f"- **Discrete vs Continuous**: "
                    f"Discrete avg reward = {avg_discrete_reward:.2f}, "
                    f"Continuous avg reward = {avg_continuous_reward:.2f}"
                )

        lines.append("")

        # Write report
        report_path = self._output_dir / filename
        report_path.write_text("\n".join(lines), encoding="utf-8")

        return str(report_path)

    def save_results(
        self,
        results: list[AblationResult],
        filename: str = "ablation_results.json",
    ) -> str:
        """Save results to JSON file.

        Args:
            results: List of ablation results.
            filename: Output filename.

        Returns:
            Path to the saved file.
        """
        data = [r.to_dict() for r in results]
        filepath = self._output_dir / filename
        filepath.write_text(json.dumps(data, indent=2), encoding="utf-8")

        return str(filepath)
