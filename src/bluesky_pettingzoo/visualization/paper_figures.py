"""Visualization tools for generating paper-ready figures.

Creates publication-quality plots for:
- Training curves (reward vs timesteps)
- Comparison bar charts
- Scalability curves
- Ablation study results
- Trajectory visualizations
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np


class PaperFigureGenerator:
    """Generates paper-ready figures from experiment results."""

    DEFAULT_STYLE = "seaborn-v0_8-whitegrid"
    DEFAULT_COLORS = [
        "#1f77b4",  # blue
        "#ff7f0e",  # orange
        "#2ca02c",  # green
        "#d62728",  # red
        "#9467bd",  # purple
        "#8c564b",  # brown
        "#e377c2",  # pink
        "#7f7f7f",  # gray
        "#bcbd22",  # yellow
        "#17becf",  # cyan
    ]

    def __init__(self, style: Optional[str] = None) -> None:
        """Initialize figure generator.

        Args:
            style: Matplotlib style to use
        """
        self.style = style or self.DEFAULT_STYLE
        plt.style.use(self.style)

    def plot_training_curves(
        self,
        results: Dict[str, List[Dict[str, float]]],
        save_path: str = "figures/training_curves.png",
        title: str = "Training Curves",
        xlabel: str = "Timesteps",
        ylabel: str = "Mean Reward",
    ) -> None:
        """Plot training curves for multiple algorithms.

        Args:
            results: Dictionary mapping algorithm names to list of {timestep, reward}
            save_path: Path to save the figure
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
        """
        fig, ax = plt.subplots(figsize=(8, 5))

        for i, (algorithm, data) in enumerate(results.items()):
            timesteps = [d["timestep"] for d in data]
            rewards = [d["reward"] for d in data]
            ax.plot(
                timesteps,
                rewards,
                label=algorithm,
                color=self.DEFAULT_COLORS[i % len(self.DEFAULT_COLORS)],
                linewidth=2,
            )

        ax.set_title(title, fontsize=14)
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        self._save_figure(save_path)

    def plot_comparison_bars(
        self,
        results: Dict[str, Dict[str, float]],
        metric: str = "mean_reward",
        save_path: str = "figures/comparison_bars.png",
        title: str = "Algorithm Comparison",
        ylabel: str = "Mean Reward",
    ) -> None:
        """Plot comparison bar chart.

        Args:
            results: Dictionary mapping algorithm names to metric values
            metric: Metric to plot
            save_path: Path to save the figure
            title: Plot title
            ylabel: Y-axis label
        """
        labels = list(results.keys())
        values = [results[label][metric] for label in labels]
        stds = [results[label].get(f"std_{metric.split('_')[1]}", 0) for label in labels]

        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(len(labels))
        width = 0.6

        bars = ax.bar(
            x,
            values,
            width,
            yerr=stds,
            capsize=5,
            color=self.DEFAULT_COLORS[: len(labels)],
        )

        ax.set_title(title, fontsize=14)
        ax.set_xlabel("Algorithm", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.grid(True, alpha=0.3, axis="y")

        self._save_figure(save_path)

    def plot_scalability(
        self,
        results: Dict[int, Dict[str, float]],
        save_path: str = "figures/scalability.png",
        title: str = "Scalability Analysis",
    ) -> None:
        """Plot scalability curves.

        Args:
            results: Dictionary mapping num_aircraft to metrics
            save_path: Path to save the figure
            title: Plot title
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        aircraft_counts = sorted(results.keys())

        arrival_rates = [results[ac]["mean_arrival_rate"] for ac in aircraft_counts]
        arrival_stds = [results[ac].get("std_arrival_rate", 0) for ac in aircraft_counts]

        nmac_rates = [results[ac]["mean_nmac_rate"] for ac in aircraft_counts]
        nmac_stds = [results[ac].get("std_nmac_rate", 0) for ac in aircraft_counts]

        axes[0].errorbar(
            aircraft_counts,
            arrival_rates,
            yerr=arrival_stds,
            fmt="-o",
            color=self.DEFAULT_COLORS[0],
            capsize=5,
            linewidth=2,
        )
        axes[0].set_title("Arrival Rate vs Aircraft Count", fontsize=12)
        axes[0].set_xlabel("Number of Aircraft", fontsize=10)
        axes[0].set_ylabel("Arrival Rate", fontsize=10)
        axes[0].grid(True, alpha=0.3)

        axes[1].errorbar(
            aircraft_counts,
            nmac_rates,
            yerr=nmac_stds,
            fmt="-o",
            color=self.DEFAULT_COLORS[3],
            capsize=5,
            linewidth=2,
        )
        axes[1].set_title("NMAC Rate vs Aircraft Count", fontsize=12)
        axes[1].set_xlabel("Number of Aircraft", fontsize=10)
        axes[1].set_ylabel("NMAC Rate", fontsize=10)
        axes[1].grid(True, alpha=0.3)

        fig.suptitle(title, fontsize=14)

        self._save_figure(save_path)

    def plot_ablation(
        self,
        results: Dict[str, Dict[str, float]],
        ablation_type: str = "action_space",
        save_path: str = "figures/ablation.png",
        title: str = "Ablation Study",
    ) -> None:
        """Plot ablation study results.

        Args:
            results: Dictionary mapping config names to metrics
            ablation_type: Type of ablation (action_space or reward)
            save_path: Path to save the figure
            title: Plot title
        """
        labels = list(results.keys())
        rewards = [results[label]["mean_reward"] for label in labels]
        stds = [results[label].get("std_reward", 0) for label in labels]

        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(len(labels))
        width = 0.6

        bars = ax.bar(
            x,
            rewards,
            width,
            yerr=stds,
            capsize=5,
            color=self.DEFAULT_COLORS[: len(labels)],
        )

        ax.set_title(title, fontsize=14)
        ax.set_xlabel(
            "Action Space Configuration" if ablation_type == "action_space" else "Reward Configuration",
            fontsize=12,
        )
        ax.set_ylabel("Mean Reward", fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.grid(True, alpha=0.3, axis="y")

        self._save_figure(save_path)

    def plot_extended_metrics(
        self,
        metrics: Dict[str, Dict[str, float]],
        save_path: str = "figures/extended_metrics.png",
        title: str = "Extended Metrics Comparison",
    ) -> None:
        """Plot extended metrics comparison.

        Args:
            metrics: Dictionary mapping algorithm names to extended metrics
            save_path: Path to save the figure
            title: Plot title
        """
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        axes = axes.flatten()

        metric_names = [
            ("conflict_resolution_rate", "Conflict Resolution Rate"),
            ("separation_violation_duration", "Violation Duration"),
            ("min_separation_distance_nm", "Min Separation (nm)"),
            ("trajectory_efficiency", "Trajectory Efficiency"),
            ("fuel_consumption_estimate", "Fuel Consumption"),
            ("mean_time_to_resolve", "Mean Time to Resolve"),
        ]

        algorithms = list(metrics.keys())
        x = np.arange(len(algorithms))
        width = 0.35

        for i, (metric_key, metric_label) in enumerate(metric_names):
            values = [metrics[alg].get(metric_key, 0) for alg in algorithms]

            axes[i].bar(
                x,
                values,
                width,
                color=self.DEFAULT_COLORS[: len(algorithms)],
            )
            axes[i].set_title(metric_label, fontsize=10)
            axes[i].set_xlabel("Algorithm", fontsize=8)
            axes[i].set_xticks(x)
            axes[i].set_xticklabels(algorithms, rotation=45, ha="right", fontsize=8)
            axes[i].grid(True, alpha=0.3, axis="y")

        fig.suptitle(title, fontsize=14)
        fig.tight_layout()

        self._save_figure(save_path)

    def generate_comparison_table(
        self,
        results: Dict[str, Dict[str, float]],
        save_path: str = "figures/comparison_table.md",
    ) -> None:
        """Generate comparison table in Markdown format.

        Args:
            results: Dictionary mapping algorithm names to metrics
            save_path: Path to save the table
        """
        headers = ["Algorithm", "Mean Reward", "Std Reward", "Arrival Rate", "NMAC Rate", "Mean Steps"]

        lines = ["| " + " | ".join(headers) + " |"]
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

        for algorithm, metrics in results.items():
            line = f"| {algorithm} "
            line += f"| {metrics.get('mean_reward', 0):.2f} "
            line += f"| {metrics.get('std_reward', 0):.2f} "
            line += f"| {metrics.get('mean_arrival_rate', 0):.2f} "
            line += f"| {metrics.get('mean_nmac_rate', 0):.2f} "
            line += f"| {metrics.get('mean_steps', 0):.1f} |"
            lines.append(line)

        content = "\n".join(lines)

        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(content)

    def _save_figure(self, save_path: str) -> None:
        """Save figure to file, creating directories if needed.

        Args:
            save_path: Path to save the figure
        """
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()