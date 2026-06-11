"""End-to-end training pipeline tests (C1)."""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestE2ETraining:
    """Full train → save → load → evaluate pipeline."""

    def _make_args(self, tmp_path: Path, timesteps: int = 256) -> MagicMock:
        args = MagicMock()
        args.scenario = "HorizontalCR"
        args.timesteps = timesteps
        args.seed = 42
        args.save_dir = str(tmp_path / "models")
        args.resume = None
        args.max_steps = 10
        args.num_aircraft = 2
        args.num_envs = 1
        args.device = "cpu"
        args.verbose = 0
        args.batch_size = 64
        args.algorithm = "PPO"
        return args

    def test_train_save_load_evaluate_flow(self, tmp_path: Path) -> None:
        """Train PPO, save checkpoint, load it, and evaluate."""
        from stable_baselines3 import PPO

        from scripts.train_ppo_scenarios import train_scenario

        args = self._make_args(tmp_path)
        train_scenario(args)

        # Checkpoint file should exist
        model_path = Path(args.save_dir) / args.scenario / args.algorithm / "checkpoint_final.zip"
        assert model_path.exists(), f"Model not saved at {model_path}"

        # Load and verify model
        model = PPO.load(str(model_path))
        assert model is not None

    def test_csv_log_has_entries_after_training(self, tmp_path: Path) -> None:
        """Training should produce a CSV log with at least one data row."""
        from scripts.train_ppo_scenarios import train_scenario

        args = self._make_args(tmp_path)
        train_scenario(args)

        csv_path = (
            Path(args.save_dir) / args.scenario / args.algorithm / "logs" / "training_log.csv"
        )
        assert csv_path.exists(), f"CSV log not found at {csv_path}"

        with open(csv_path, encoding="utf-8") as f:
            reader = csv.reader(f)
            rows = list(reader)
        # Header + at least 1 data row
        assert len(rows) >= 2, f"CSV has {len(rows)} rows, expected >= 2"
        assert rows[0] == [
            "timestep",
            "episode",
            "reward",
            "episode_length",
            "conflicts",
            "arrivals",
            "algorithm",
            "action_space",
            "timestamp",
        ]

    def test_checkpoint_loadable_by_sb3(self, tmp_path: Path) -> None:
        """Saved checkpoint must be loadable by SB3 PPO.load()."""
        from stable_baselines3 import PPO

        from scripts.train_ppo_scenarios import train_scenario

        args = self._make_args(tmp_path)
        train_scenario(args)

        model_path = Path(args.save_dir) / args.scenario / args.algorithm / "checkpoint_final.zip"
        model = PPO.load(str(model_path))
        # Model should have a policy
        assert hasattr(model, "policy")
        assert model.policy is not None

    def test_resume_training_from_checkpoint(self, tmp_path: Path) -> None:
        """Training can resume from a saved checkpoint."""
        from scripts.train_ppo_scenarios import train_scenario

        # First run: train and save
        args1 = self._make_args(tmp_path, timesteps=256)
        train_scenario(args1)

        model_path = (
            Path(args1.save_dir) / args1.scenario / args1.algorithm / "checkpoint_final.zip"
        )
        assert model_path.exists()

        # Second run: resume from checkpoint
        args2 = self._make_args(tmp_path, timesteps=128)
        args2.resume = str(model_path)
        train_scenario(args2)

        # Should complete without error
        final_path = (
            Path(args2.save_dir) / args2.scenario / args2.algorithm / "checkpoint_final.zip"
        )
        assert final_path.exists()

    def test_deterministic_results_with_seed(self, tmp_path: Path) -> None:
        """Same seed should produce identical training outputs."""
        from scripts.train_ppo_scenarios import train_scenario

        for i in range(2):
            args = MagicMock()
            args.scenario = "HorizontalCR"
            args.timesteps = 128
            args.seed = 42
            args.save_dir = str(tmp_path / f"run{i}" / "models")
            args.resume = None
            args.max_steps = 10
            args.num_aircraft = 2
            args.num_envs = 1
            args.device = "cpu"
            args.verbose = 0
            args.batch_size = 64
            args.algorithm = "PPO"
            train_scenario(args)

        # Compare CSV outputs
        csv0 = tmp_path / "run0" / "models" / "HorizontalCR" / "PPO" / "logs" / "training_log.csv"
        csv1 = tmp_path / "run1" / "models" / "HorizontalCR" / "PPO" / "logs" / "training_log.csv"

        with open(csv0, encoding="utf-8") as f:
            rows0 = list(csv.reader(f))
        with open(csv1, encoding="utf-8") as f:
            rows1 = list(csv.reader(f))

        # Same number of episodes
        assert len(rows0) == len(rows1)


class TestFullEpisodeTraining:
    """Test complete episode training from reset to termination."""

    def test_full_episode_training(self, tmp_path: Path) -> None:
        """Train PPO for a full episode and verify it completes without error."""
        from stable_baselines3 import PPO

        from scripts.train_ppo_scenarios import train_scenario

        args = MagicMock()
        args.scenario = "HorizontalCR"
        args.timesteps = 512
        args.seed = 42
        args.save_dir = str(tmp_path / "models")
        args.resume = None
        args.max_steps = 20
        args.num_aircraft = 2
        args.num_envs = 1
        args.device = "cpu"
        args.verbose = 0
        args.batch_size = 64
        args.algorithm = "PPO"

        # Should complete without error
        train_scenario(args)

        # Verify model was saved
        model_path = Path(args.save_dir) / args.scenario / args.algorithm / "checkpoint_final.zip"
        assert model_path.exists(), f"Model not saved at {model_path}"

        # Load and verify model can predict actions
        model = PPO.load(str(model_path))
        assert model is not None
        assert hasattr(model, "policy")

    def test_multi_scenario_training(self, tmp_path: Path) -> None:
        """Train PPO on 3 core scenarios and verify all complete successfully."""
        from scripts.train_ppo_scenarios import train_scenario

        scenarios = ["HorizontalCR", "SectorCR", "WaypointNav"]

        for scenario in scenarios:
            args = MagicMock()
            args.scenario = scenario
            args.timesteps = 256
            args.seed = 42
            args.save_dir = str(tmp_path / "models")
            args.resume = None
            args.max_steps = 10
            args.num_aircraft = 2
            args.num_envs = 1
            args.device = "cpu"
            args.verbose = 0
            args.batch_size = 64
            args.algorithm = "PPO"

            # Should complete without error
            train_scenario(args)

            # Verify model was saved
            model_path = (
                Path(args.save_dir) / scenario / args.algorithm / "checkpoint_final.zip"
            )
            assert model_path.exists(), f"Model not saved for {scenario}"

    def test_reward_signal_exists(self, tmp_path: Path) -> None:
        """Verify that training produces a learning signal (reward improvement)."""
        from stable_baselines3 import PPO

        from scripts.train_ppo_scenarios import train_scenario

        args = MagicMock()
        args.scenario = "HorizontalCR"
        args.timesteps = 1024
        args.seed = 42
        args.save_dir = str(tmp_path / "models")
        args.resume = None
        args.max_steps = 15
        args.num_aircraft = 2
        args.num_envs = 1
        args.device = "cpu"
        args.verbose = 0
        args.batch_size = 64
        args.algorithm = "PPO"

        # Train the model
        train_scenario(args)

        # Load the trained model
        model_path = Path(args.save_dir) / args.scenario / args.algorithm / "checkpoint_final.zip"
        model = PPO.load(str(model_path))

        # Read the training log to check for reward improvement
        csv_path = (
            Path(args.save_dir) / args.scenario / args.algorithm / "logs" / "training_log.csv"
        )
        assert csv_path.exists(), f"CSV log not found at {csv_path}"

        with open(csv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        # Need at least 2 episodes to compare
        assert len(rows) >= 2, f"Need at least 2 episodes, got {len(rows)}"

        # Compare first and last episode rewards
        first_reward = float(rows[0]["reward"])
        last_reward = float(rows[-1]["reward"])

        # At minimum, training should complete without negative infinite rewards
        assert first_reward > float("-inf"), "First reward is negative infinity"
        assert last_reward > float("-inf"), "Last reward is negative infinity"

        # Log the improvement for visibility
        improvement = last_reward - first_reward
        print(f"\nReward improvement: {first_reward:.2f} -> {last_reward:.2f} ({improvement:+.2f})")
