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
        return {aid: space.sample() for aid, space in action_spaces.items()}

    def reset(self) -> None:
        pass
