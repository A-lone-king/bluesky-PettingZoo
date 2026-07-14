"""Adapter for bluesky-gym baseline comparison.

Provides a common interface to run bluesky-gym single-agent environments
and compare results with bluesky-pettingzoo multi-agent environments.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np


@dataclass
class GymEvalResult:
    """Evaluation result from bluesky-gym."""

    scenario: str
    mean_reward: float
    std_reward: float
    mean_arrival_rate: float
    mean_nmac_rate: float
    mean_steps: float
    num_episodes: int


class BlueSkyGymAdapter:
    """Adapter for running bluesky-gym as a baseline.

    Provides a unified interface to:
    1. Create bluesky-gym environments for common scenarios
    2. Train and evaluate PPO on bluesky-gym
    3. Return results in a format compatible with bluesky-pettingzoo
    """

    SCENARIO_MAP: Dict[str, str] = {
        "HorizontalCR": "HorizontalConflictResolution-v0",
        "VerticalCR": "VerticalConflictResolution-v0",
        "WaypointNav": "WaypointNavigation-v0",
        "SectorCR": "SectorConflictResolution-v0",
        "Merge": "MergeScenario-v0",
    }

    def __init__(self) -> None:
        """Initialize adapter. Check if bluesky-gym is available."""
        self._available = self._check_availability()

    @property
    def available(self) -> bool:
        """Return whether bluesky-gym is available."""
        return self._available

    def _check_availability(self) -> bool:
        """Check if bluesky-gym is installed."""
        try:
            import bluesky_gym  # type: ignore
            return True
        except ImportError:
            return False

    def create_env(self, scenario_name: str, **kwargs: Any) -> Any:
        """Create a bluesky-gym environment for the given scenario.

        Args:
            scenario_name: bluesky-pettingzoo scenario name (e.g., "HorizontalCR")
            **kwargs: Additional environment kwargs

        Returns:
            A gymnasium environment instance from bluesky-gym

        Raises:
            ImportError: If bluesky-gym is not installed
            ValueError: If scenario mapping is not found
        """
        if not self._available:
            raise ImportError(
                "bluesky-gym is not installed. Install with: pip install bluesky-gym"
            )

        import bluesky_gym  # type: ignore

        gym_id = self.SCENARIO_MAP.get(scenario_name)
        if gym_id is None:
            raise ValueError(f"No bluesky-gym mapping for scenario: {scenario_name}")

        return bluesky_gym.make(gym_id, **kwargs)

    def train_and_evaluate(
        self,
        scenario_name: str,
        total_timesteps: int = 500_000,
        num_eval_episodes: int = 20,
        seed: int = 42,
        model_save_path: Optional[str] = None,
    ) -> GymEvalResult:
        """Train PPO on bluesky-gym and return evaluation results.

        Args:
            scenario_name: bluesky-pettingzoo scenario name
            total_timesteps: Total training timesteps
            num_eval_episodes: Number of evaluation episodes
            seed: Random seed
            model_save_path: Optional path to save trained model

        Returns:
            GymEvalResult with aggregated metrics
        """
        env = self.create_env(scenario_name)
        env.seed(seed)

        from stable_baselines3 import PPO

        model = PPO(
            "MlpPolicy",
            env,
            verbose=0,
            seed=seed,
        )

        model.learn(total_timesteps=total_timesteps)

        if model_save_path:
            model.save(model_save_path)

        rewards: List[float] = []
        steps_list: List[int] = []
        arrivals = 0
        nmacs = 0

        for _ in range(num_eval_episodes):
            obs = env.reset()
            total_reward = 0.0
            episode_steps = 0
            done = False

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                total_reward += float(reward)
                episode_steps += 1

            rewards.append(total_reward)
            steps_list.append(episode_steps)

            if total_reward > 0:
                arrivals += 1
            else:
                nmacs += 1

        env.close()

        n = len(rewards) if rewards else 1
        return GymEvalResult(
            scenario=scenario_name,
            mean_reward=float(np.mean(rewards)) if rewards else 0.0,
            std_reward=float(np.std(rewards)) if rewards else 0.0,
            mean_arrival_rate=arrivals / n,
            mean_nmac_rate=nmacs / n,
            mean_steps=float(np.mean(steps_list)) if steps_list else 0.0,
            num_episodes=num_eval_episodes,
        )

    def evaluate_random(self, scenario_name: str, num_episodes: int = 20) -> GymEvalResult:
        """Evaluate random policy on bluesky-gym environment.

        Args:
            scenario_name: bluesky-pettingzoo scenario name
            num_episodes: Number of evaluation episodes

        Returns:
            GymEvalResult with random policy metrics
        """
        env = self.create_env(scenario_name)

        rewards: List[float] = []
        steps_list: List[int] = []
        arrivals = 0
        nmacs = 0

        for _ in range(num_episodes):
            obs = env.reset()
            total_reward = 0.0
            episode_steps = 0
            done = False

            while not done:
                action = env.action_space.sample()
                obs, reward, done, info = env.step(action)
                total_reward += float(reward)
                episode_steps += 1

            rewards.append(total_reward)
            steps_list.append(episode_steps)

            if total_reward > 0:
                arrivals += 1
            else:
                nmacs += 1

        env.close()

        n = len(rewards) if rewards else 1
        return GymEvalResult(
            scenario=scenario_name,
            mean_reward=float(np.mean(rewards)) if rewards else 0.0,
            std_reward=float(np.std(rewards)) if rewards else 0.0,
            mean_arrival_rate=arrivals / n,
            mean_nmac_rate=nmacs / n,
            mean_steps=float(np.mean(steps_list)) if steps_list else 0.0,
            num_episodes=num_episodes,
        )


class MockBlueSkyGymAdapter:
    """Mock adapter for bluesky-gym when it's not available.

    Provides the same interface but returns synthetic data
    for testing and development purposes.
    """

    SCENARIO_MAP: Dict[str, str] = BlueSkyGymAdapter.SCENARIO_MAP

    @property
    def available(self) -> bool:
        """Return False since this is a mock."""
        return False

    def create_env(self, scenario_name: str, **kwargs: Any) -> Any:
        """Return a mock environment."""
        raise ImportError("bluesky-gym is not installed. Using mock adapter.")

    def train_and_evaluate(
        self,
        scenario_name: str,
        total_timesteps: int = 500_000,
        num_eval_episodes: int = 20,
        seed: int = 42,
        model_save_path: Optional[str] = None,
    ) -> GymEvalResult:
        """Return synthetic evaluation results."""
        rng = np.random.RandomState(seed)
        return GymEvalResult(
            scenario=scenario_name,
            mean_reward=rng.uniform(-50, 50),
            std_reward=rng.uniform(5, 20),
            mean_arrival_rate=rng.uniform(0.3, 0.7),
            mean_nmac_rate=rng.uniform(0.1, 0.4),
            mean_steps=rng.uniform(40, 60),
            num_episodes=num_eval_episodes,
        )

    def evaluate_random(self, scenario_name: str, num_episodes: int = 20) -> GymEvalResult:
        """Return synthetic random policy results."""
        return GymEvalResult(
            scenario=scenario_name,
            mean_reward=-30.0,
            std_reward=15.0,
            mean_arrival_rate=0.2,
            mean_nmac_rate=0.5,
            mean_steps=50.0,
            num_episodes=num_episodes,
        )


def get_bluesky_gym_adapter() -> BlueSkyGymAdapter | MockBlueSkyGymAdapter:
    """Return the appropriate adapter based on bluesky-gym availability."""
    try:
        import bluesky_gym  # type: ignore
        return BlueSkyGymAdapter()
    except ImportError:
        return MockBlueSkyGymAdapter()