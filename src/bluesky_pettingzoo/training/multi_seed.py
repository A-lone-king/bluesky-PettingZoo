"""Multi-seed training infrastructure for paper-level reproducibility."""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np


@dataclass
class SeedResult:
    """Training and evaluation result for a single seed."""

    seed: int
    mean_reward: float
    std_reward: float
    mean_arrival_rate: float
    mean_nmac_rate: float
    mean_steps: float
    model_path: str
    eval_episodes: int


@dataclass
class MultiSeedSummary:
    """Aggregated results across multiple seeds."""

    scenario: str
    algorithm: str
    seeds: List[int]
    mean_reward: float
    std_reward: float
    mean_arrival_rate: float
    std_arrival_rate: float
    mean_nmac_rate: float
    std_nmac_rate: float
    mean_steps: float
    std_steps: float
    seed_results: List[Dict[str, Any]]
    timestamp: str


class MultiSeedTrainer:
    """Train an RL algorithm across multiple random seeds and aggregate results."""

    DEFAULT_SEEDS: List[int] = [42, 123, 456, 789, 1024]

    def __init__(
        self,
        scenario: str,
        algorithm: str,
        train_fn: Callable[[int, Path], SeedResult],
        seeds: Optional[List[int]] = None,
        save_dir: str = "models",
        results_dir: str = "results/multi_seed",
    ) -> None:
        """Initialize multi-seed trainer.

        Args:
            scenario: Scenario name (e.g., "HorizontalCR")
            algorithm: Algorithm name (e.g., "PPO")
            train_fn: Function that trains a single seed and returns SeedResult
                Signature: train_fn(seed: int, seed_save_dir: Path) -> SeedResult
            seeds: List of random seeds to use. Defaults to [42, 123, 456, 789, 1024]
            save_dir: Base directory for model checkpoints
            results_dir: Directory for saving summary results
        """
        self._scenario = scenario
        self._algorithm = algorithm
        self._train_fn = train_fn
        self._seeds = seeds or self.DEFAULT_SEEDS
        self._save_dir = Path(save_dir)
        self._results_dir = Path(results_dir)
        self._seed_results: List[SeedResult] = []

    @property
    def seeds(self) -> List[int]:
        """Return the list of seeds being used."""
        return self._seeds

    @property
    def seed_results(self) -> List[SeedResult]:
        """Return results from completed seed runs."""
        return self._seed_results

    def train_all(self) -> MultiSeedSummary:
        """Train on all seeds and return aggregated summary."""
        self._seed_results = []

        for seed in self._seeds:
            seed_save_dir = self._get_seed_save_dir(seed)
            seed_save_dir.mkdir(parents=True, exist_ok=True)

            result = self._train_fn(seed, seed_save_dir)
            self._seed_results.append(result)
            print(f"  Seed {seed} completed: reward={result.mean_reward:.2f} ± {result.std_reward:.2f}")

        return self._aggregate_results()

    def _get_seed_save_dir(self, seed: int) -> Path:
        """Return the save directory for a specific seed."""
        return self._save_dir / self._scenario / self._algorithm / f"seed_{seed}"

    def _aggregate_results(self) -> MultiSeedSummary:
        """Aggregate results across all seeds."""
        rewards = np.array([r.mean_reward for r in self._seed_results])
        arrival_rates = np.array([r.mean_arrival_rate for r in self._seed_results])
        nmac_rates = np.array([r.mean_nmac_rate for r in self._seed_results])
        steps = np.array([r.mean_steps for r in self._seed_results])

        summary = MultiSeedSummary(
            scenario=self._scenario,
            algorithm=self._algorithm,
            seeds=self._seeds,
            mean_reward=float(np.mean(rewards)),
            std_reward=float(np.std(rewards)),
            mean_arrival_rate=float(np.mean(arrival_rates)),
            std_arrival_rate=float(np.std(arrival_rates)),
            mean_nmac_rate=float(np.mean(nmac_rates)),
            std_nmac_rate=float(np.std(nmac_rates)),
            mean_steps=float(np.mean(steps)),
            std_steps=float(np.std(steps)),
            seed_results=[asdict(r) for r in self._seed_results],
            timestamp=self._get_timestamp(),
        )

        self._save_summary(summary)
        return summary

    def _get_timestamp(self) -> str:
        """Return current timestamp as ISO format string."""
        from datetime import datetime

        return datetime.now().isoformat()

    def _save_summary(self, summary: MultiSeedSummary) -> None:
        """Save the aggregated summary to JSON."""
        self._results_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{self._scenario}_{self._algorithm}_summary.json"
        filepath = self._results_dir / filename

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(asdict(summary), f, indent=2, ensure_ascii=False)

        print(f"\nSummary saved to {filepath}")

    @staticmethod
    def load_summary(scenario: str, algorithm: str, results_dir: str = "results/multi_seed") -> MultiSeedSummary:
        """Load a previously saved summary."""
        filepath = Path(results_dir) / f"{scenario}_{algorithm}_summary.json"
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return MultiSeedSummary(**data)

    def format_summary(self, summary: MultiSeedSummary) -> str:
        """Format the summary as a human-readable string."""
        lines = []
        lines.append("=" * 70)
        lines.append(f"Multi-Seed Summary: {summary.scenario} × {summary.algorithm}")
        lines.append(f"Seeds: {summary.seeds}")
        lines.append("=" * 70)
        lines.append(f"Mean Reward:      {summary.mean_reward:>10.2f} ± {summary.std_reward:.2f}")
        lines.append(f"Mean Arrival Rate: {summary.mean_arrival_rate:>9.1%} ± {summary.std_arrival_rate:.2%}")
        lines.append(f"Mean NMAC Rate:   {summary.mean_nmac_rate:>9.1%} ± {summary.std_nmac_rate:.2%}")
        lines.append(f"Mean Steps:       {summary.mean_steps:>10.1f} ± {summary.std_steps:.1f}")
        lines.append("-" * 70)
        lines.append("Per-seed results:")
        for i, seed_result in enumerate(summary.seed_results):
            lines.append(
                f"  Seed {summary.seeds[i]:>4}: "
                f"reward={seed_result['mean_reward']:.2f} ± {seed_result['std_reward']:.2f}, "
                f"arrival={seed_result['mean_arrival_rate']:.1%}, "
                f"nmac={seed_result['mean_nmac_rate']:.1%}"
            )
        lines.append("=" * 70)
        return "\n".join(lines)