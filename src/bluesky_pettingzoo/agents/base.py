"""Abstract base class for agents."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from gymnasium import spaces

from bluesky_pettingzoo.utils.types import AgentID


class BaseAgent(ABC):
    """Base class for all agents in the MARL environment."""

    @abstractmethod
    def act(
        self,
        observations: dict[AgentID, Any],
        action_spaces: dict[AgentID, spaces.Space],
    ) -> dict[AgentID, Any]:
        """Select actions for all agents.

        Args:
            observations: Mapping of agent_id to observation.
            action_spaces: Mapping of agent_id to action space.

        Returns:
            Mapping of agent_id to chosen action.
        """
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reset agent state for a new episode."""
        ...
