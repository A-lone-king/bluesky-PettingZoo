"""Base class for reward components."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from bluesky_pettingzoo.utils.types import AircraftState, DiscreteAction


class RewardComponent(ABC):
    """Abstract base class for reward components.

    All reward components must implement:
    - compute(): Calculate the reward value for a single step
    - reset(): Reset any internal state

    Subclasses can optionally:
    - Set `component_name` class attribute for config lookup
    - Set `config_keys` class attribute for automatic config parsing
    - Override `_stateful_attrs` for automatic reset behavior
    """

    component_name: str = ""
    config_keys: dict[str, Any] = {}
    _stateful_attrs: list[str] = []

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize the reward component.

        Args:
            config: Configuration dictionary containing component settings
        """
        self.config = config
        self._parse_config()

    def _parse_config(self) -> None:
        """Parse configuration using component_name and config_keys.

        Automatically extracts config values based on config_keys mapping.
        Format: {config_key: (attribute_name, default_value)}
        """
        if not self.component_name or not self.config_keys:
            return

        comp_config = self.config.get("components", {}).get(self.component_name, {})
        for config_key, (attr_name, default_value) in self.config_keys.items():
            setattr(self, attr_name, comp_config.get(config_key, default_value))

    def get_config(self, key: str, default: Any = None) -> Any:
        """Get a configuration value for this component.

        Args:
            key: Configuration key to look up
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        if not self.component_name:
            return default
        return self.config.get("components", {}).get(self.component_name, {}).get(key, default)

    @abstractmethod
    def compute(
        self,
        agent_id: str,
        prev_state: AircraftState,
        action: DiscreteAction | list[Any] | np.ndarray,
        curr_state: AircraftState,
        all_states: dict[str, AircraftState],
        step_count: int = 0,
    ) -> float:
        """Compute the reward component value.

        Args:
            agent_id: The agent receiving the reward
            prev_state: Aircraft state before action
            action: Discrete action or continuous action array
            curr_state: Aircraft state after action
            all_states: Current states of all aircraft
            step_count: Current episode step number (0-indexed)

        Returns:
            Reward value (unweighted)
        """
        ...

    def reset(self) -> None:
        """Reset internal state for a new episode.

        Automatically clears any attributes listed in _stateful_attrs.
        Subclasses can override for custom reset logic.
        """
        for attr_name in self._stateful_attrs:
            attr = getattr(self, attr_name, None)
            if attr is not None and hasattr(attr, "clear"):
                attr.clear()
            elif attr is not None and isinstance(attr, list):
                attr.clear()
