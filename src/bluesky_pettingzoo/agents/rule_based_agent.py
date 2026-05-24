"""Rule-based agent — TCAS-inspired conflict avoidance with right-turn rule.

When a conflict threat is detected (closest aircraft within 10 NM),
the agent turns toward the threat's right side and climbs if at similar
altitude.  When no threat exists, the agent steers toward the goal
waypoint.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from gymnasium import spaces

from bluesky_pettingzoo.agents.base import BaseAgent
from bluesky_pettingzoo.utils.types import AgentID

# Observation array column indices (see observations/manager.py)
_DIST_IDX = 3
_BEARING_COS_IDX = 4
_BEARING_SIN_IDX = 5
_REL_ALT_IDX = 6
_PRIORITY_IDX = 9

# self_state indices
_SELF_PRIORITY_IDX = 8

# Normalization defaults (must match config/normalization section)
_DISTANCE_MAX = 20.0  # NM — observation radius
_ALT_MID = 33000.0
_ALT_RANGE = 10000.0

# Threat threshold
_THREAT_DISTANCE_NM = 10.0
_SAME_ALT_THRESHOLD_FT = 2000.0

# Action indices: 0=-20, 1=-10, 2=0, 3=+10, 4=+20
_ACTION_NEUTRAL = 2
_ACTION_RIGHT_SMALL = 3   # +10
_ACTION_RIGHT_LARGE = 4   # +20
_ACTION_CLIMB_SMALL = 3   # +1000ft
_ACTION_CLIMB_LARGE = 4   # +2000ft


def _denorm_distance(norm_dist: float) -> float:
    return norm_dist * _DISTANCE_MAX


def _denorm_altitude(norm_alt: float) -> float:
    return norm_alt * _ALT_RANGE + _ALT_MID


def _denorm_relative_alt(norm_rel_alt: float) -> float:
    return norm_rel_alt * _ALT_RANGE


def _bearing_from_cos_sin(cos_val: float, sin_val: float) -> float:
    """Reconstruct bearing in degrees [0, 360) from cos/sin components."""
    return math.degrees(math.atan2(sin_val, cos_val)) % 360


class RuleBasedAgent(BaseAgent):
    """Conflict-aware agent using TCAS-inspired right-turn rule.

    Decision logic:
    - If the closest observable aircraft is within 10 NM:
        - If own priority > threat priority → maintain course (high priority)
        - If own priority < threat priority → turn right and climb
        - If equal priority → TCAS right-turn rule
    - Otherwise:
        - Steer toward the goal waypoint using heading adjustments
    - Speed is always kept neutral (no adjustment).
    """

    def act(
        self,
        observations: dict[AgentID, Any],
        action_spaces: dict[AgentID, spaces.Space],
    ) -> dict[AgentID, Any]:
        actions: dict[AgentID, Any] = {}

        for agent_id, obs in observations.items():
            actions[agent_id] = self._decide(obs)

        return actions

    def _decide(self, obs: dict[str, Any]) -> list[int]:
        """Decide action for a single agent based on its observation."""
        other_aircraft = obs["other_aircraft"]
        mask = obs["other_aircraft_mask"]
        goal = obs["goal"]
        self_state = obs["self_state"]

        own_priority = float(self_state[_SELF_PRIORITY_IDX])

        # Find the closest threat
        closest_dist = float("inf")
        closest_idx = -1
        for i in range(len(mask)):
            if mask[i] == 0:
                continue
            dist = _denorm_distance(float(other_aircraft[i, _DIST_IDX]))
            if dist < closest_dist:
                closest_dist = dist
                closest_idx = i

        heading_idx = _ACTION_NEUTRAL
        altitude_idx = _ACTION_NEUTRAL
        speed_idx = _ACTION_NEUTRAL

        if closest_dist < _THREAT_DISTANCE_NM and closest_idx >= 0:
            threat_priority = float(other_aircraft[closest_idx, _PRIORITY_IDX])

            if own_priority > threat_priority:
                # Higher priority → maintain course
                heading_idx = self._heading_toward_goal(goal)
            else:
                # Equal or lower priority → conflict avoidance
                bearing = _bearing_from_cos_sin(
                    float(other_aircraft[closest_idx, _BEARING_COS_IDX]),
                    float(other_aircraft[closest_idx, _BEARING_SIN_IDX]),
                )

                # Threat is ahead (0-180) → turn right; behind (180-360) → turn left
                if 0 <= bearing <= 180:
                    heading_idx = (
                        _ACTION_RIGHT_LARGE if closest_dist < 5.0
                        else _ACTION_RIGHT_SMALL
                    )
                else:
                    # Threat behind — small right turn to create separation
                    heading_idx = _ACTION_RIGHT_SMALL

                # Climb if threat at similar altitude
                rel_alt_ft = _denorm_relative_alt(
                    float(other_aircraft[closest_idx, _REL_ALT_IDX])
                )
                if abs(rel_alt_ft) < _SAME_ALT_THRESHOLD_FT:
                    altitude_idx = (
                        _ACTION_CLIMB_LARGE if closest_dist < 5.0
                        else _ACTION_CLIMB_SMALL
                    )
        else:
            # No threat — steer toward goal
            heading_idx = self._heading_toward_goal(goal)

        return [heading_idx, altitude_idx, speed_idx]

    @staticmethod
    def _heading_toward_goal(goal: np.ndarray) -> int:
        """Choose heading adjustment to steer toward the goal.

        goal layout: [distance, bearing_cos, bearing_sin, alt_diff]
        """
        goal_bearing = _bearing_from_cos_sin(float(goal[1]), float(goal[2]))

        # Goal ahead (330-360 or 0-30): no adjustment
        # Goal right (30-150): turn right
        # Goal behind (150-210): turn right large
        # Goal left (210-330): turn left
        if goal_bearing <= 30 or goal_bearing >= 330:
            return _ACTION_NEUTRAL
        elif goal_bearing <= 90:
            return _ACTION_RIGHT_SMALL  # +10
        elif goal_bearing <= 150:
            return _ACTION_RIGHT_LARGE  # +20
        elif goal_bearing <= 210:
            return _ACTION_RIGHT_LARGE  # +20 (turn around)
        elif goal_bearing <= 270:
            return 1  # -10 (turn left)
        else:
            return 0  # -20 (turn left more)

    def reset(self) -> None:
        pass
