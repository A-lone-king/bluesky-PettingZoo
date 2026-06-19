"""Random agent — samples actions uniformly from the action space."""

from __future__ import annotations

from typing import Any

from gymnasium import spaces

from bluesky_pettingzoo.agents.base import BaseAgent
from bluesky_pettingzoo.utils.types import AgentID


class RandomAgent(BaseAgent):
    """Agent that selects random actions from the action space."""

    def act(
        self,
        observations: dict[AgentID, Any],
        action_spaces: dict[AgentID, spaces.Space[Any]],
    ) -> dict[AgentID, Any]:
        """Sample a random action from each agent's action space.

        Args:
            observations: Observations keyed by agent ID (unused).
            action_spaces: Action spaces keyed by agent ID.

        Returns:
            Dictionary mapping agent IDs to sampled actions.
        """
        return {aid: space.sample() for aid, space in action_spaces.items()}

    def reset(self) -> None:
        """Reset the agent state between episodes (no-op for random agent)."""
        pass
