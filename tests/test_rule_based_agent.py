"""Tests for RuleBasedAgent — conflict-aware TCAS-inspired agent."""

from __future__ import annotations

import math

import numpy as np
import pytest
from gymnasium import spaces

from bluesky_pettingzoo.agents.rule_based_agent import (
    RuleBasedAgent,
    _bearing_from_cos_sin,
    _denorm_distance,
)


@pytest.fixture
def action_space() -> spaces.MultiDiscrete:
    return spaces.MultiDiscrete([5, 5, 5])


@pytest.fixture
def agent() -> RuleBasedAgent:
    return RuleBasedAgent()


def _make_obs(
    other_aircraft: list[list[float]] | None = None,
    goal_bearing_deg: float = 0.0,
    goal_distance_nm: float = 50.0,
    own_priority: float = 0.0,
    threat_priority: float = 0.0,
) -> dict:
    """Build a synthetic observation dict for testing.

    other_aircraft: list of [distance_nm, bearing_deg, rel_alt_ft]
        (raw values, will be normalized internally)
    own_priority: priority of the ownship [-1, 1]
    threat_priority: priority of the threat aircraft [-1, 1]
    """
    max_obs = 10
    obs_array = np.zeros((max_obs, 10), dtype=np.float32)
    mask = np.zeros(max_obs, dtype=np.int8)

    if other_aircraft:
        for i, entry in enumerate(other_aircraft[:max_obs]):
            dist_nm, bear_deg, rel_alt_ft = entry
            obs_array[i] = [
                0.0,  # heading
                0.0,  # altitude
                0.0,  # speed
                dist_nm / 20.0,  # normalized distance (distance_max=20)
                math.cos(math.radians(bear_deg)),  # bearing_cos
                math.sin(math.radians(bear_deg)),  # bearing_sin
                rel_alt_ft / 10000.0,  # normalized relative altitude
                0.0,  # rel_speed_x
                0.0,  # rel_speed_y
                threat_priority,  # priority
            ]
            mask[i] = 1

    self_state = np.zeros(9, dtype=np.float32)
    self_state[8] = own_priority

    goal = np.array(
        [
            goal_distance_nm / 20.0,  # normalized distance
            math.cos(math.radians(goal_bearing_deg)),  # bearing_cos
            math.sin(math.radians(goal_bearing_deg)),  # bearing_sin
            0.0,  # alt_diff
        ],
        dtype=np.float32,
    )

    return {
        "self_state": self_state,
        "other_aircraft": obs_array,
        "other_aircraft_mask": mask,
        "goal": goal,
    }


class TestActReturnsDict:
    def test_act_returns_dict(
        self,
        agent: RuleBasedAgent,
        action_space: spaces.MultiDiscrete,
    ) -> None:
        obs = {"AC001": _make_obs()}
        result = agent.act(obs, {"AC001": action_space})
        assert isinstance(result, dict)
        assert "AC001" in result

    def test_act_keys_match_agents(
        self,
        agent: RuleBasedAgent,
        action_space: spaces.MultiDiscrete,
    ) -> None:
        agents = ["A", "B", "C"]
        obs = {a: _make_obs() for a in agents}
        result = agent.act(obs, {a: action_space for a in agents})
        assert set(result.keys()) == set(agents)


class TestNoThreatBehavior:
    """When no aircraft is within 10 NM, agent steers toward goal."""

    def test_no_threat_goal_ahead(self, agent: RuleBasedAgent) -> None:
        """Goal directly ahead → no heading adjustment."""
        obs = _make_obs(goal_bearing_deg=0.0)
        action = agent._decide(obs)
        assert action[0] == 2  # neutral heading
        assert action[2] == 2  # neutral speed

    def test_no_threat_goal_right(self, agent: RuleBasedAgent) -> None:
        """Goal to the right → turn right."""
        obs = _make_obs(goal_bearing_deg=60.0)
        action = agent._decide(obs)
        assert action[0] > 2  # right turn

    def test_no_threat_goal_left(self, agent: RuleBasedAgent) -> None:
        """Goal to the left → turn left."""
        obs = _make_obs(goal_bearing_deg=300.0)
        action = agent._decide(obs)
        assert action[0] < 2  # left turn

    def test_no_threat_neutral_altitude(self, agent: RuleBasedAgent) -> None:
        """No threat → no altitude adjustment."""
        obs = _make_obs()
        action = agent._decide(obs)
        assert action[1] == 2  # neutral altitude


class TestConflictAvoidance:
    """When a threat is within 10 NM, agent avoids it."""

    def test_threat_ahead_turns_right(self, agent: RuleBasedAgent) -> None:
        """Threat directly ahead → turn right."""
        obs = _make_obs(other_aircraft=[[5.0, 45.0, 0.0]])
        action = agent._decide(obs)
        assert action[0] > 2  # right turn

    def test_threat_close_turns_right_large(self, agent: RuleBasedAgent) -> None:
        """Threat very close (< 5 NM) → large right turn."""
        obs = _make_obs(other_aircraft=[[3.0, 30.0, 0.0]])
        action = agent._decide(obs)
        assert action[0] == 4  # +20 degrees

    def test_threat_moderate_turns_right_small(self, agent: RuleBasedAgent) -> None:
        """Threat at moderate distance (5-10 NM) → small right turn."""
        obs = _make_obs(other_aircraft=[[7.0, 30.0, 0.0]])
        action = agent._decide(obs)
        assert action[0] == 3  # +10 degrees

    def test_threat_behind_turns_right_small(self, agent: RuleBasedAgent) -> None:
        """Threat behind (bearing > 180) → small right turn."""
        obs = _make_obs(other_aircraft=[[5.0, 220.0, 0.0]])
        action = agent._decide(obs)
        assert action[0] == 3  # +10 degrees

    def test_threat_same_altitude_climbs(self, agent: RuleBasedAgent) -> None:
        """Threat at similar altitude → climb."""
        obs = _make_obs(other_aircraft=[[5.0, 45.0, 500.0]])  # 500 ft diff
        action = agent._decide(obs)
        assert action[1] > 2  # climb

    def test_threat_different_altitude_no_climb(self, agent: RuleBasedAgent) -> None:
        """Threat at very different altitude → no climb."""
        obs = _make_obs(other_aircraft=[[5.0, 45.0, 5000.0]])  # 5000 ft diff
        action = agent._decide(obs)
        assert action[1] == 2  # neutral altitude

    def test_threat_close_same_alt_climbs_large(self, agent: RuleBasedAgent) -> None:
        """Close threat at same altitude → large climb."""
        obs = _make_obs(other_aircraft=[[3.0, 30.0, 0.0]])  # 0 ft diff, < 5 NM
        action = agent._decide(obs)
        assert action[1] == 4  # +2000ft

    def test_speed_always_neutral(self, agent: RuleBasedAgent) -> None:
        """Speed is never adjusted."""
        obs = _make_obs(other_aircraft=[[3.0, 30.0, 0.0]])
        action = agent._decide(obs)
        assert action[2] == 2

    def test_closest_threat_is_used(self, agent: RuleBasedAgent) -> None:
        """When multiple threats, the closest one drives the decision."""
        obs = _make_obs(
            other_aircraft=[
                [15.0, 30.0, 0.0],  # far threat
                [3.0, 120.0, 0.0],  # close threat
            ]
        )
        action = agent._decide(obs)
        # Both are ahead (bearing < 180), so right turn
        assert action[0] > 2


class TestDeterministic:
    def test_deterministic(self, agent: RuleBasedAgent) -> None:
        obs = _make_obs(other_aircraft=[[5.0, 45.0, 0.0]])
        r1 = agent._decide(obs)
        r2 = agent._decide(obs)
        assert r1 == r2


class TestReset:
    def test_reset_no_error(self, agent: RuleBasedAgent) -> None:
        agent.reset()


class TestBearingReconstruction:
    def test_bearing_north(self) -> None:
        assert abs(_bearing_from_cos_sin(1.0, 0.0) - 0.0) < 0.01

    def test_bearing_east(self) -> None:
        assert abs(_bearing_from_cos_sin(0.0, 1.0) - 90.0) < 0.01

    def test_bearing_south(self) -> None:
        assert abs(_bearing_from_cos_sin(-1.0, 0.0) - 180.0) < 0.01

    def test_bearing_west(self) -> None:
        assert abs(_bearing_from_cos_sin(0.0, -1.0) - 270.0) < 0.01


class TestDenormDistance:
    def test_denorm_zero(self) -> None:
        assert _denorm_distance(0.0) == 0.0

    def test_denorm_one(self) -> None:
        assert _denorm_distance(1.0) == 20.0


class TestPriorityAwareness:
    """Priority-based conflict avoidance behavior."""

    def test_high_priority_maintains_course(self, agent: RuleBasedAgent) -> None:
        """Agent with higher priority than threat should maintain course."""
        obs = _make_obs(
            other_aircraft=[[5.0, 45.0, 0.0]],
            own_priority=0.8,
            threat_priority=-0.5,
            goal_bearing_deg=0.0,
        )
        action = agent._decide(obs)
        # High priority agent should NOT turn right (maintain course toward goal)
        # Goal is at bearing 0 (straight ahead), so heading should be neutral
        assert action[0] == 2  # neutral heading

    def test_low_priority_avoids(self, agent: RuleBasedAgent) -> None:
        """Agent with lower priority than threat should execute avoidance."""
        obs = _make_obs(
            other_aircraft=[[5.0, 45.0, 0.0]],
            own_priority=-0.5,
            threat_priority=0.8,
        )
        action = agent._decide(obs)
        # Low priority agent should turn right
        assert action[0] > 2

    def test_equal_priority_uses_tcas_rule(self, agent: RuleBasedAgent) -> None:
        """Equal priority agents use TCAS right-turn rule."""
        obs = _make_obs(
            other_aircraft=[[5.0, 45.0, 0.0]],
            own_priority=0.0,
            threat_priority=0.0,
        )
        action = agent._decide(obs)
        # Equal priority → TCAS rule: threat ahead → turn right
        assert action[0] > 2

    def test_high_priority_still_avoids_close_threat(self, agent: RuleBasedAgent) -> None:
        """Even high priority agent should steer toward goal when no threat."""
        obs = _make_obs(
            other_aircraft=[[15.0, 45.0, 0.0]],  # beyond threat range
            own_priority=0.8,
            threat_priority=-0.5,
            goal_bearing_deg=60.0,
        )
        action = agent._decide(obs)
        # No threat → steer toward goal at 60° → right turn
        assert action[0] > 2
