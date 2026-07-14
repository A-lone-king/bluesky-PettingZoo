"""CLI script for multi-algorithm comparison training.

Usage:
    python scripts/train_multi_algo.py --timesteps 500000
    python scripts/train_multi_algo.py --scenarios HorizontalCR VerticalCR --algorithms PPO SAC TD3 DDPG
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from bluesky_pettingzoo.training.multi_algo_comparison import (
    DEFAULT_ALGORITHMS,
    DEFAULT_SCENARIOS,
    DEFAULT_TIMESTEPS,
    MultiAlgoComparison,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Multi-algorithm comparison for reward tuning validation"
    )
    parser.add_argument(
        "--scenarios", nargs="+", default=DEFAULT_SCENARIOS,
        help="Scenarios to train on",
    )
    parser.add_argument(
        "--algorithms", nargs="+", default=DEFAULT_ALGORITHMS,
        choices=["PPO", "SAC", "TD3", "DDPG"],
        help="Algorithms to compare",
    )
    parser.add_argument(
        "--timesteps", type=int, default=DEFAULT_TIMESTEPS,
        help="Total training timesteps per algorithm-scenario pair",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--results-dir", type=str, default="results/multi_algo",
        help="Directory to save results",
    )
    parser.add_argument(
        "--report-path", type=str, default="results/multi_algo/comparison_report.md",
        help="Path to save the Markdown report",
    )
    parser.add_argument(
        "--curves-path", type=str, default="results/multi_algo/training_curves.json",
        help="Path to save training curve data",
    )
    parser.add_argument(
        "--eval-episodes", type=int, default=5,
        help="Number of episodes for reward evaluation",
    )
    parser.add_argument(
        "--log-interval", type=int, default=10_000,
        help="Timesteps between reward logging",
    )
    return parser.parse_args(argv)


def main() -> int:
    """CLI entry point."""
    args = parse_args()

    print("=" * 60)
    print("多算法对比验证调参效果")
    print("=" * 60)
    print(f"  场景: {', '.join(args.scenarios)}")
    print(f"  算法: {', '.join(args.algorithms)}")
    print(f"  训练步数: {args.timesteps:,}")
    print(f"  种子: {args.seed}")
    print(f"  总训练次数: {len(args.scenarios) * len(args.algorithms)}")
    print("=" * 60)

    comparison = MultiAlgoComparison(
        scenarios=args.scenarios,
        algorithms=args.algorithms,
        total_timesteps=args.timesteps,
        seed=args.seed,
        eval_episodes=args.eval_episodes,
        log_interval=args.log_interval,
    )

    summary = comparison.run(
        save_results=True,
        results_dir=args.results_dir,
    )

    # Generate reports
    report = comparison.generate_report(summary, save_path=args.report_path)
    comparison.generate_training_curve_data(summary, save_path=args.curves_path)

    print("\n" + "=" * 60)
    print("训练完成！")
    print(f"  全部收敛: {'✅ 是' if summary.all_converged else '❌ 否'}")
    print(f"  结果保存到: {args.results_dir}")
    print(f"  报告保存到: {args.report_path}")
    print(f"  曲线数据: {args.curves_path}")
    print("=" * 60)

    # Print summary table
    print("\n## 结果摘要")
    print(f"{'场景':<15} {'算法':<8} {'初始奖励':>10} {'最终奖励':>10} {'收敛':>6} {'变化量':>10}")
    print("-" * 65)
    for r in summary.results:
        delta = r.final_reward - r.initial_reward
        converged_str = "✅" if r.converged else "❌"
        print(f"{r.scenario:<15} {r.algorithm:<8} {r.initial_reward:>10.2f} {r.final_reward:>10.2f} {converged_str:>6} {delta:>+10.2f}")

    return 0 if summary.all_converged else 1


if __name__ == "__main__":
    sys.exit(main())
