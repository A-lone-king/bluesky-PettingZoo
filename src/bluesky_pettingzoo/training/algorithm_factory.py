"""Algorithm factory: creates SB3 algorithms from config."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from stable_baselines3 import DDPG, PPO, SAC, TD3
from stable_baselines3.common.base_class import BaseAlgorithm

_ALGORITHM_MAP: dict[str, type[BaseAlgorithm]] = {
    "PPO": PPO,
    "SAC": SAC,
    "TD3": TD3,
    "DDPG": DDPG,
}


class AlgorithmFactory:
    """Factory for creating Stable-Baselines3 algorithms."""

    @staticmethod
    def supported_algorithms() -> list[str]:
        """Return list of supported algorithm names."""
        return list(_ALGORITHM_MAP.keys())

    @staticmethod
    def create(
        algorithm: str,
        policy: str,
        env: Any = None,
        **kwargs: Any,
    ) -> BaseAlgorithm:
        """Create an algorithm instance.

        Args:
            algorithm: Algorithm name (PPO, SAC, TD3, DDPG).
            policy: Policy type (e.g. 'MlpPolicy', 'MultiInputPolicy').
            env: Optional environment.
            **kwargs: Additional algorithm parameters.

        Returns:
            Algorithm instance.

        Raises:
            ValueError: If algorithm name is not supported.
        """
        algo_cls = _ALGORITHM_MAP.get(algorithm)
        if algo_cls is None:
            raise ValueError(
                f"Unknown algorithm '{algorithm}'. Supported: {list(_ALGORITHM_MAP.keys())}"
            )
        return algo_cls(policy, env=env, **kwargs)

    @staticmethod
    def from_yaml(
        algorithm: str,
        policy: str,
        config_path: Path | str,
        env: Any = None,
        **override_kwargs: Any,
    ) -> BaseAlgorithm:
        """Create an algorithm with params loaded from YAML config.

        Args:
            algorithm: Algorithm name.
            policy: Policy type.
            config_path: Path to algorithms.yaml.
            env: Optional environment.
            **override_kwargs: Override params from config.

        Returns:
            Algorithm instance with config params applied.
        """
        config_path = Path(config_path)
        with open(config_path, encoding="utf-8") as f:
            config = yaml.safe_load(f)

        algo_config = config.get(algorithm, {})
        algo_config.update(override_kwargs)

        return AlgorithmFactory.create(algorithm, policy, env=env, **algo_config)
