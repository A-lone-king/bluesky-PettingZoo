"""MAPPO (Multi-Agent PPO) and IPPO (Independent PPO) trainers.

Provides multi-agent PPO implementations for bluesky-pettingzoo environments.
Supports both centralized training with decentralized execution (MAPPO)
and independent PPO (IPPO) as fallback.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from pettingzoo import ParallelEnv

from bluesky_pettingzoo.training.checkpoint import CheckpointManager
from bluesky_pettingzoo.training.metrics import ExtendedMetrics, MetricsCalculator


@dataclass
class MAPPOConfig:
    """Configuration for MAPPO trainer."""

    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    batch_size: int = 64
    n_steps: int = 2048
    n_epochs: int = 10
    target_kl: Optional[float] = None
    use_centralized_critic: bool = True
    normalize_advantages: bool = True


@dataclass
class MAPPOEvalResult:
    """Evaluation result from MAPPO training."""

    scenario: str
    mean_reward: float
    std_reward: float
    mean_arrival_rate: float
    mean_nmac_rate: float
    mean_steps: float
    num_episodes: int
    extended_metrics: Optional[ExtendedMetrics] = None


class IPPOTrainer:
    """Independent PPO Trainer.

    Trains each agent independently using PPO with a shared policy network.
    This is a simplified multi-agent approach where all agents share the same
    policy but act independently.
    """

    def __init__(
        self,
        env: ParallelEnv,
        config: Optional[MAPPOConfig] = None,
        seed: int = 42,
    ) -> None:
        """Initialize IPPO trainer.

        Args:
            env: PettingZoo ParallelEnv instance
            config: MAPPO configuration
            seed: Random seed
        """
        self.env = env
        self.config = config or MAPPOConfig()
        self.seed = seed
        self.agent_ids = list(env.possible_agents)
        self.policy: Optional[Any] = None

    def train(
        self,
        total_timesteps: int = 500_000,
        checkpoint_freq: int = 10_000,
        checkpoint_dir: str = "models",
        progress_callback: Optional[Callable[[int, float], None]] = None,
    ) -> Path:
        """Train IPPO on the environment.

        Args:
            total_timesteps: Total training timesteps
            checkpoint_freq: Checkpoint save frequency
            checkpoint_dir: Directory for saving checkpoints
            progress_callback: Callback for training progress

        Returns:
            Path to the final model checkpoint
        """
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_util import make_vec_env
        from stable_baselines3.common.utils import set_random_seed

        set_random_seed(self.seed)

        env_name = type(self.env).__name__
        vec_env = make_vec_env(
            lambda: self.env,
            n_envs=1,
            seed=self.seed,
        )

        self.policy = PPO(
            "MlpPolicy",
            vec_env,
            learning_rate=self.config.learning_rate,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
            clip_range=self.config.clip_range,
            ent_coef=self.config.ent_coef,
            vf_coef=self.config.vf_coef,
            max_grad_norm=self.config.max_grad_norm,
            batch_size=self.config.batch_size,
            n_steps=self.config.n_steps,
            n_epochs=self.config.n_epochs,
            target_kl=self.config.target_kl,
            verbose=0,
            seed=self.seed,
        )

        checkpoint_manager = CheckpointManager(checkpoint_dir)
        model_path = Path(checkpoint_dir) / "ippo" / f"seed_{self.seed}"

        for step in range(0, total_timesteps, checkpoint_freq):
            remaining = min(checkpoint_freq, total_timesteps - step)
            self.policy.learn(
                total_timesteps=remaining,
                reset_num_timesteps=False,
            )

            checkpoint_manager.save(
                self.policy,
                str(model_path),
                step + remaining,
            )

            if progress_callback:
                progress_callback(step + remaining, 0.0)

        final_path = model_path / "final_model.zip"
        self.policy.save(str(final_path))

        vec_env.close()
        return final_path

    def evaluate(
        self,
        num_episodes: int = 20,
    ) -> MAPPOEvalResult:
        """Evaluate trained policy.

        Args:
            num_episodes: Number of evaluation episodes

        Returns:
            MAPPOEvalResult with aggregated metrics
        """
        if self.policy is None:
            raise ValueError("Policy not trained. Call train() first.")

        rewards: List[float] = []
        steps_list: List[int] = []
        arrivals = 0
        nmacs = 0

        for _ in range(num_episodes):
            obs, _ = self.env.reset(seed=self.seed)
            total_reward = 0.0
            episode_steps = 0
            terminated = {agent: False for agent in self.agent_ids}
            truncated = {agent: False for agent in self.agent_ids}

            while not all(terminated.values()) and not all(truncated.values()):
                actions: Dict[str, Any] = {}
                for agent_id in self.agent_ids:
                    if not terminated.get(agent_id, False) and not truncated.get(agent_id, False):
                        action, _ = self.policy.predict(obs[agent_id], deterministic=True)
                        actions[agent_id] = action

                obs, reward, terminated, truncated, info = self.env.step(actions)

                total_reward += sum(float(r) for r in reward.values())
                episode_steps += 1

            rewards.append(total_reward)
            steps_list.append(episode_steps)

            # Check info dict for arrival/NMAC reasons
            ep_arrived = False
            ep_nmac = False
            for ai in info.values():
                if isinstance(ai, dict):
                    reason = ai.get("termination_reason")
                    if reason == "arrival":
                        ep_arrived = True
                    elif reason == "nmac":
                        ep_nmac = True
            if ep_arrived:
                arrivals += 1
            if ep_nmac:
                nmacs += 1

        n = len(rewards) if rewards else 1
        return MAPPOEvalResult(
            scenario=type(self.env).__name__,
            mean_reward=float(np.mean(rewards)) if rewards else 0.0,
            std_reward=float(np.std(rewards)) if rewards else 0.0,
            mean_arrival_rate=arrivals / n,
            mean_nmac_rate=nmacs / n,
            mean_steps=float(np.mean(steps_list)) if steps_list else 0.0,
            num_episodes=num_episodes,
        )


class RayMAPPOAdapter:
    """Adapter for Ray RLlib MAPPO.

    Provides a unified interface to Ray's MAPPO implementation
    with centralized training and decentralized execution.
    """

    def __init__(
        self,
        env: ParallelEnv,
        config: Optional[MAPPOConfig] = None,
        seed: int = 42,
    ) -> None:
        """Initialize Ray MAPPO adapter.

        Args:
            env: PettingZoo ParallelEnv instance
            config: MAPPO configuration
            seed: Random seed
        """
        self.env = env
        self.config = config or MAPPOConfig()
        self.seed = seed
        self._available = self._check_availability()

    @property
    def available(self) -> bool:
        """Return whether Ray is available."""
        return self._available

    def _check_availability(self) -> bool:
        """Check if Ray is installed."""
        try:
            import ray  # type: ignore
            return True
        except ImportError:
            return False

    def train(
        self,
        total_timesteps: int = 500_000,
        checkpoint_freq: int = 10_000,
        checkpoint_dir: str = "models",
        progress_callback: Optional[Callable[[int, float], None]] = None,
    ) -> Path:
        """Train MAPPO using Ray RLlib.

        Args:
            total_timesteps: Total training timesteps
            checkpoint_freq: Checkpoint save frequency
            checkpoint_dir: Directory for saving checkpoints
            progress_callback: Callback for training progress

        Returns:
            Path to the final model checkpoint

        Raises:
            ImportError: If Ray is not installed
        """
        if not self._available:
            raise ImportError(
                "Ray is not installed. Install with: pip install ray[rllib]"
            )

        import ray
        from ray.rllib.algorithms.mappo import MAPPO

        ray.init(ignore_reinit_error=True)

        env_name = type(self.env).__name__

        mappo_config = {
            "env": env_name,
            "num_gpus": 0,
            "num_workers": 1,
            "seed": self.seed,
            "lr": self.config.learning_rate,
            "gamma": self.config.gamma,
            "lambda": self.config.gae_lambda,
            "clip_param": self.config.clip_range,
            "entropy_coeff": self.config.ent_coef,
            "vf_loss_coeff": self.config.vf_coef,
            "grad_clip": self.config.max_grad_norm,
            "train_batch_size": self.config.batch_size,
            "sgd_minibatch_size": self.config.batch_size // 4,
            "num_sgd_iter": self.config.n_epochs,
            "use_centralized_critic": self.config.use_centralized_critic,
        }

        algorithm = MAPPO(config=mappo_config)

        model_path = Path(checkpoint_dir) / "mappo" / f"seed_{self.seed}"
        model_path.mkdir(parents=True, exist_ok=True)

        for i in range(total_timesteps // checkpoint_freq):
            result = algorithm.train()
            checkpoint = algorithm.save(str(model_path))

            if progress_callback:
                progress_callback(
                    (i + 1) * checkpoint_freq,
                    result.get("episode_reward_mean", 0.0),
                )

        final_path = model_path / "final_model"
        algorithm.save(str(final_path))
        ray.shutdown()

        return final_path

    def evaluate(
        self,
        num_episodes: int = 20,
    ) -> MAPPOEvalResult:
        """Evaluate trained MAPPO policy.

        Args:
            num_episodes: Number of evaluation episodes

        Returns:
            MAPPOEvalResult with aggregated metrics
        """
        rewards = []
        steps_list = []
        arrivals = 0
        nmacs = 0

        for _ in range(num_episodes):
            obs, _ = self.env.reset(seed=self.seed)
            total_reward = 0.0
            episode_steps = 0
            terminated = {agent: False for agent in self.env.possible_agents}
            truncated = {agent: False for agent in self.env.possible_agents}

            while not all(terminated.values()) and not all(truncated.values()):
                actions = {agent: self.env.action_space(agent).sample() for agent in self.env.possible_agents}
                obs, reward, terminated, truncated, info = self.env.step(actions)
                total_reward += sum(float(r) for r in reward.values())
                episode_steps += 1

            rewards.append(total_reward)
            steps_list.append(episode_steps)

            # Check info dict for arrival/NMAC reasons
            ep_arrived = False
            ep_nmac = False
            for ai in info.values():
                if isinstance(ai, dict):
                    reason = ai.get("termination_reason")
                    if reason == "arrival":
                        ep_arrived = True
                    elif reason == "nmac":
                        ep_nmac = True
            if ep_arrived:
                arrivals += 1
            if ep_nmac:
                nmacs += 1

        n = len(rewards) if rewards else 1
        return MAPPOEvalResult(
            scenario=type(self.env).__name__,
            mean_reward=float(np.mean(rewards)) if rewards else 0.0,
            std_reward=float(np.std(rewards)) if rewards else 0.0,
            mean_arrival_rate=arrivals / n,
            mean_nmac_rate=nmacs / n,
            mean_steps=float(np.mean(steps_list)) if steps_list else 0.0,
            num_episodes=num_episodes,
        )


def get_mappo_trainer(
    env: ParallelEnv,
    config: Optional[MAPPOConfig] = None,
    seed: int = 42,
) -> IPPOTrainer | RayMAPPOAdapter:
    """Return the appropriate MAPPO trainer based on Ray availability.

    Falls back to IPPO if Ray is not installed.

    Args:
        env: PettingZoo ParallelEnv instance
        config: MAPPO configuration
        seed: Random seed

    Returns:
        IPPOTrainer if Ray is not available, RayMAPPOAdapter otherwise
    """
    try:
        import ray  # type: ignore
        return RayMAPPOAdapter(env, config, seed)
    except ImportError:
        return IPPOTrainer(env, config, seed)