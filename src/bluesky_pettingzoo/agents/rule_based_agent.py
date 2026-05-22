"""Rule-based agent — always flies straight (no adjustment)."""

from __future__ import annotations

from typing import Any

from gymnasium import spaces

from bluesky_pettingzoo.agents.base import BaseAgent
from bluesky_pettingzoo.utils.types import AgentID

_NO_ADJUSTMENT = [2, 2, 2]


class RuleBasedAgent(BaseAgent):
    """Agent that always selects zero-adjustment actions."""

    def act(
        self,
        observations: dict[AgentID, Any],
        action_spaces: dict[AgentID, spaces.Space],
    ) -> dict[AgentID, Any]:
        return {aid: list(_NO_ADJUSTMENT) for aid in action_spaces}

    def reset(self) -> None:
        pass
