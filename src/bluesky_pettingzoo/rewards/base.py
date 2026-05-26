"""Base class for reward components."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Union

import numpy as np

from bluesky_pettingzoo.utils.types import AircraftState, DiscreteAction


class RewardComponent(ABC):
    """Abstract base class for reward components.

    All reward components must implement:
    - compute(): Calculate the reward value for a single step
    - reset(): Reset any internal state
    """

    @abstractmethod
    def compute(
        self,
        agent_id: str,
        prev_state: AircraftState,
        action: Union[DiscreteAction, list, np.ndarray],
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

    @abstractmethod
    def reset(self) -> None:
        """Reset internal state for a new episode."""
        ...
