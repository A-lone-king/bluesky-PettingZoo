"""Run action space ablation experiments.

Usage:
    python scripts/run_ablation.py --config config/ablation_experiments.yaml
    python scripts/run_ablation.py --experiment discrete_heading_only
    python scripts/run_ablation.py --list
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bluesky_pettingzoo.training.ablation import AblationReporter, AblationRunner


def load_config(config_path: str) -> dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def list_experiments(runner: AblationRunner) -> None:
    """List all available experiments."""
    print("\nAvailable experiments:")
    print("-" * 60)

    print("\nDiscrete action space experiments:")
    for eid in runner.list_discrete_experiments():
        cfg = runner.get_experiment_config(eid)
        if cfg:
            print(f"  {eid}: {cfg['name']}")

    print("\nContinuous action space experiments:")
    for eid in runner.list_continuous_experiments():
        cfg = runner.get_experiment_config(eid)
        if cfg:
            print(f"  {eid}: {cfg['name']}")


def create_mock_env_factory(config: dict[str, Any]) -> Any:  # noqa: ANN401
    """Create a mock environment factory for testing.

    In production, this would create a real BlueSky MARL environment.
    """
    from unittest.mock import MagicMock

    import numpy as np

    def factory(action_config: dict[str, Any]) -> MagicMock:
        env = MagicMock()
        env.possible_agents = ["agent_0", "agent_1", "agent_2"]
        env.agents = ["agent_0", "agent_1", "agent_2"]

        # Set action space based on config
        action_type = action_config.get("action", {}).get("type", "discrete")
        action_dims = action_config.get("action", {}).get("dims", [5, 5, 5])

        if action_type == "discrete":
            from gymnasium import spaces

            env.action_space.return_value = spaces.MultiDiscrete(action_dims)
        else:
            from gymnasium import spaces

            env.action_space.return_value = spaces.Box(
                low=-1.0, high=1.0, shape=(action_dims,), dtype=np.float32
            )

        # Mock reset/step
        def mock_reset(**kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
            obs = {}
            infos = {}
            for agent in env.possible_agents:
                obs[agent] = np.random.randn(9).astype(np.float32)
                infos[agent] = {}
            return obs, infos

        def mock_step(
            actions: dict[str, Any],
        ) -> tuple[
            dict[str, Any],
            dict[str, Any],
            dict[str, Any],
            dict[str, Any],
            dict[str, Any],
        ]:
            obs = {}
            rewards = {}
            terminations = {}
            truncations = {}
            infos = {}
            for agent in env.possible_agents:
                obs[agent] = np.random.randn(9).astype(np.float32)
                rewards[agent] = np.random.randn()
                terminations[agent] = False
                truncations[agent] = False
                infos[agent] = {"conflict_status": "safe"}
            return obs, rewards, terminations, truncations, infos

        env.reset.side_effect = mock_reset
        env.step.side_effect = mock_step
        return env

    return factory


def create_mock_agent_factory() -> Any:  # noqa: ANN401
    """Create a mock agent factory for testing.

    In production, this would create a real RL agent.
    """
    from unittest.mock import MagicMock

    def factory(env: Any) -> MagicMock:  # noqa: ANN401
        agent = MagicMock()

        action_space = env.action_space()

        def predict(obs: Any) -> Any:  # noqa: ANN401
            return action_space.sample()

        agent.predict.side_effect = predict
        return agent

    return factory


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run action space ablation experiments"
    )
    parser.add_argument(
        "--config",
        default="config/ablation_experiments.yaml",
        help="Path to ablation experiments config",
    )
    parser.add_argument(
        "--experiment",
        help="Run a specific experiment by ID",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available experiments",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Number of episodes to evaluate (default: 5)",
    )
    parser.add_argument(
        "--output-dir",
        default="results/ablation",
        help="Output directory for results",
    )

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Create runner
    runner = AblationRunner(config)

    # List experiments
    if args.list:
        list_experiments(runner)
        return

    # Run specific experiment or all
    experiments_to_run: list[str]
    if args.experiment:
        experiments_to_run = [args.experiment]
    else:
        experiments_to_run = runner.list_experiments()

    print(f"\nRunning {len(experiments_to_run)} experiment(s)...")
    print(f"Episodes per experiment: {args.episodes}")

    # Create factories
    env_factory = create_mock_env_factory(config)
    agent_factory = create_mock_agent_factory()

    # Run experiments
    results = []
    for exp_id in experiments_to_run:
        print(f"\nRunning experiment: {exp_id}")
        try:
            result = runner.run_experiment(
                experiment_id=exp_id,
                env_factory=env_factory,
                agent_factory=agent_factory,
                num_episodes=args.episodes,
                max_steps_per_episode=100,
            )
            results.append(result)
            print(
                f"  Completed: mean_reward={result.mean_reward:.2f}, "
                f"conflict_rate={result.mean_conflict_rate:.3f}"
            )
        except ValueError as e:
            print(f"  Error: {e}")

    # Generate report
    if results:
        reporter = AblationReporter(args.output_dir)
        report_path = reporter.generate_report(results)
        results_path = reporter.save_results(results)

        print(f"\nReport generated: {report_path}")
        print(f"Results saved: {results_path}")
    else:
        print("\nNo experiments completed successfully.")


if __name__ == "__main__":
    main()
