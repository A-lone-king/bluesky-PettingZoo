"""Action translator — converts discrete actions to BlueSky commands."""

from __future__ import annotations

from typing import Any

from bluesky_pettingzoo.utils.types import AircraftState, DiscreteAction


class ActionTranslator:
    """Translates discrete action indices to BlueSky text commands.

    Maps MultiDiscrete indices to adjustment values via config arrays,
    then generates BlueSky HDG/ALT/SPD commands.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        action_cfg = config.get("action", {})
        self._heading_adj: list[int] = action_cfg.get(
            "heading_adjustments", [-20, -10, 0, 10, 20],
        )
        self._altitude_adj: list[int] = action_cfg.get(
            "altitude_adjustments", [-2000, -1000, 0, 1000, 2000],
        )
        self._speed_adj: list[int] = action_cfg.get(
            "speed_adjustments", [-20, -10, 0, 10, 20],
        )

    def translate(
        self,
        agent_id: str,
        state: AircraftState,
        action: DiscreteAction,
    ) -> list[str]:
        """Translate a single agent's action to BlueSky commands.

        Args:
            agent_id: Aircraft identifier.
            state: Current aircraft state.
            action: Discrete action indices.

        Returns:
            List of BlueSky command strings. Empty if no adjustment.
        """
        commands: list[str] = []

        hdg_adj = self._heading_adj[action.heading_idx]
        if hdg_adj != 0:
            new_hdg = (state.hdg + hdg_adj) % 360
            commands.append(f"HDG {agent_id} {int(new_hdg)}")

        alt_adj = self._altitude_adj[action.altitude_idx]
        if alt_adj != 0:
            new_alt = state.alt + alt_adj
            commands.append(f"ALT {agent_id} {int(new_alt)}")

        spd_adj = self._speed_adj[action.speed_idx]
        if spd_adj != 0:
            new_spd = state.tas + spd_adj
            commands.append(f"SPD {agent_id} {int(new_spd)}")

        return commands

    def translate_batch(
        self,
        actions: dict[str, DiscreteAction],
        states: dict[str, AircraftState],
    ) -> list[str]:
        """Translate multiple agents' actions to a merged command list.

        Args:
            actions: Mapping of agent_id to DiscreteAction.
            states: Mapping of agent_id to current AircraftState.

        Returns:
            Combined list of all BlueSky commands.
        """
        commands: list[str] = []
        for agent_id, action in actions.items():
            state = states[agent_id]
            commands.extend(self.translate(agent_id, state, action))
        return commands
