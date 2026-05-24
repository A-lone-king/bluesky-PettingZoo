"""Integration tests for cross-module component协作.

Tests the data flow across observation, action, and reward subsystems
using real implementations (no mocks). BlueSky wrapper is excluded;
all scenarios are driven by synthetic AircraftState data.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from bluesky_pettingzoo.actions.translator import ActionTranslator
from bluesky_pettingzoo.observations.manager import ObservationManager
from bluesky_pettingzoo.rewards.calculator import RewardCalculator
from bluesky_pettingzoo.rewards.components.conflict import ConflictPenalty
from bluesky_pettingzoo.rewards.components.efficiency import EfficiencyReward
from bluesky_pettingzoo.rewards.components.smoothness import SmoothnessPenalty
from bluesky_pettingzoo.utils.types import AircraftState, DiscreteAction

CONFIG_DIR = Path(__file__).parent.parent.parent / "config"


@pytest.fixture
def full_config() -> dict[str, Any]:
    """Load and merge default + rewards configuration."""
    with open(CONFIG_DIR / "default.yaml", encoding="utf-8") as f:
        default = yaml.safe_load(f)
    with open(CONFIG_DIR / "rewards.yaml", encoding="utf-8") as f:
        rewards = yaml.safe_load(f)
    default["components"] = rewards["components"]
    return default


@pytest.fixture
def obs_manager(full_config: dict[str, Any]) -> ObservationManager:
    return ObservationManager(full_config)


@pytest.fixture
def action_translator(full_config: dict[str, Any]) -> ActionTranslator:
    return ActionTranslator(full_config)


@pytest.fixture
def reward_calculator(full_config: dict[str, Any]) -> RewardCalculator:
    calc = RewardCalculator()
    calc.register(ConflictPenalty(full_config), weight=1.0)
    calc.register(SmoothnessPenalty(full_config), weight=0.5)
    eff = EfficiencyReward(full_config)
    calc.register(eff, weight=0.3)
    return calc


def make_state(
    acid: str,
    lat: float,
    lon: float,
    alt: float,
    hdg: float = 90.0,
    tas: float = 450.0,
    vs: float = 0.0,
) -> AircraftState:
    return AircraftState(
        id=acid, lat=lat, lon=lon, alt=alt, hdg=hdg, tas=tas, vs=vs,
    )


# ─── Observation + Action integration ───────────────────────────────


class TestObservationToAction:
    """Verify observation → action translation data flow."""

    def test_observation_produces_valid_action_input(
        self,
        obs_manager: ObservationManager,
        action_translator: ActionTranslator,
    ) -> None:
        """Observation output can drive action translation without error."""
        own = make_state("OWN", 39.25, 116.25, 35000.0, hdg=90.0)
        other = make_state("AC001", 39.30, 116.30, 34000.0)
        goal = {"lat": 39.50, "lon": 116.50, "alt": 35000.0, "hdg": 90.0}

        pkg = obs_manager.generate(own, [other], goal)
        obs = pkg["observation"]

        # Observation has expected keys and shapes
        assert obs["self_state"].shape == (9,)
        assert obs["other_aircraft"].shape == (10, 10)
        assert obs["other_aircraft_mask"].shape == (10,)
        assert obs["goal"].shape == (4,)

        # Translate a no-op action — should produce no commands
        action = DiscreteAction(heading_idx=2, altitude_idx=2, speed_idx=2)
        cmds = action_translator.translate("OWN", own, action)
        assert cmds == []

    def test_batch_translate_matches_single(
        self,
        action_translator: ActionTranslator,
    ) -> None:
        """translate_batch produces same commands as individual translate calls."""
        s1 = make_state("A", 39.25, 116.25, 35000.0, hdg=90.0)
        s2 = make_state("B", 39.30, 116.30, 34000.0, hdg=180.0)
        a1 = DiscreteAction(heading_idx=3, altitude_idx=2, speed_idx=4)
        a2 = DiscreteAction(heading_idx=0, altitude_idx=4, speed_idx=2)

        single = (
            action_translator.translate("A", s1, a1)
            + action_translator.translate("B", s2, a2)
        )
        batch = action_translator.translate_batch(
            {"A": a1, "B": a2}, {"A": s1, "B": s2},
        )
        assert batch == single


# ─── Observation + Reward integration ───────────────────────────────


class TestObservationToReward:
    """Verify observation and reward share consistent state interpretation."""

    def test_conflict_status_matches_reward(
        self,
        obs_manager: ObservationManager,
        reward_calculator: RewardCalculator,
    ) -> None:
        """When reward detects NMAC, textual_state should reflect conflict."""
        own = make_state("OWN", 39.25, 116.25, 35000.0)
        # ~1NM away, same altitude → NMAC
        intruder = make_state("INT", 39.267, 116.25, 35000.0)
        goal = {"lat": 39.50, "lon": 116.50, "alt": 35000.0, "hdg": 90.0}
        action = DiscreteAction(heading_idx=2, altitude_idx=2, speed_idx=2)

        # Reward should be NMAC-level
        all_states = {"OWN": own, "INT": intruder}
        reward = reward_calculator.compute("OWN", own, action, own, all_states)
        assert reward <= -100.0  # NMAC penalty dominates

        # Observation textual state should contain conflict info
        pkg = obs_manager.generate(own, [intruder], goal, conflict_status="nmac")
        assert pkg["textual_state"]["conflict_status"] == "nmac"
        assert "nmac" in pkg["textual_state"]["text"].lower()

    def test_safe_scenario_reward_matches_observation(
        self,
        obs_manager: ObservationManager,
        reward_calculator: RewardCalculator,
    ) -> None:
        """Safe scenario: no conflict penalty, observation says 'safe'."""
        own = make_state("OWN", 39.25, 116.25, 35000.0)
        far = make_state("FAR", 39.75, 116.75, 35000.0)  # ~40NM away
        goal = {"lat": 39.25, "lon": 116.25, "alt": 35000.0, "hdg": 90.0}
        action = DiscreteAction(heading_idx=2, altitude_idx=2, speed_idx=2)

        # Set efficiency goal at own position → arrival reward triggers
        eff = reward_calculator._components[2][0]
        eff.set_goal("OWN", lat=39.25, lon=116.25)

        all_states = {"OWN": own, "FAR": far}
        reward = reward_calculator.compute("OWN", own, action, own, all_states)
        # step_penalty(-0.01)*0.3 + arrival(10)*0.3 = -0.003 + 3.0 = 2.997
        assert reward > 0

        pkg = obs_manager.generate(own, [far], goal, conflict_status="safe")
        assert pkg["textual_state"]["conflict_status"] == "safe"


# ─── Full pipeline: states → obs + action → reward ─────────────────


class TestFullPipeline:
    """End-to-end: generate observation, pick action, compute reward."""

    def test_safe_overtake_scenario(
        self,
        obs_manager: ObservationManager,
        action_translator: ActionTranslator,
        reward_calculator: RewardCalculator,
    ) -> None:
        """Two aircraft on parallel tracks, no conflict, straight flight."""
        own = make_state("OWN", 39.25, 116.25, 35000.0, hdg=90.0, tas=450.0)
        # ~5.4NM away, 2000ft above → no conflict (h>5, v>=1000)
        other = make_state("OTH", 39.34, 116.25, 37000.0, hdg=90.0, tas=440.0)
        goal = {"lat": 39.50, "lon": 116.50, "alt": 35000.0, "hdg": 90.0}
        all_states = {"OWN": own, "OTH": other}

        # Step 1: observe
        pkg = obs_manager.generate(own, [other], goal, conflict_status="safe")
        obs = pkg["observation"]
        assert obs["self_state"].shape == (9,)
        assert obs["other_aircraft_mask"][0] == 1  # other is observable

        # Step 2: act (straight — no adjustment)
        action = DiscreteAction(heading_idx=2, altitude_idx=2, speed_idx=2)
        cmds = action_translator.translate_batch(
            {"OWN": action, "OTH": action}, all_states,
        )
        assert cmds == []

        # Step 3: reward
        reward = reward_calculator.compute("OWN", own, action, own, all_states)
        # No conflict, just step penalty: -0.01 * 0.3 = -0.003
        # Plus efficiency step penalty: -0.01 * 0.3 = -0.003
        # Total ≈ -0.006
        assert reward < 0
        assert reward > -1.0  # well bounded

    def test_conflict_resolution_scenario(
        self,
        obs_manager: ObservationManager,
        action_translator: ActionTranslator,
        reward_calculator: RewardCalculator,
    ) -> None:
        """Aircraft in warning zone; turning away improves reward."""
        own = make_state("OWN", 39.25, 116.25, 35000.0, hdg=90.0)
        # ~8NM away, 1500ft below → warning level
        intruder = make_state("INT", 39.38, 116.25, 33500.0, hdg=270.0)
        goal = {"lat": 39.50, "lon": 116.50, "alt": 35000.0, "hdg": 90.0}
        all_states = {"OWN": own, "INT": intruder}

        # No-action: warning penalty applies
        noop = DiscreteAction(heading_idx=2, altitude_idx=2, speed_idx=2)
        r_noop = reward_calculator.compute("OWN", own, noop, own, all_states)

        # Turn action: heading +20° (index 4 = +20)
        turn = DiscreteAction(heading_idx=4, altitude_idx=2, speed_idx=2)
        cmds = action_translator.translate("OWN", own, turn)
        assert len(cmds) == 1
        assert "HDG OWN 110" in cmds[0]

        # Reward with turn: still warning (state hasn't changed yet),
        # but smoothness penalty adds -0.1*0.5 = -0.05
        r_turn = reward_calculator.compute("OWN", own, turn, own, all_states)
        # Turn has smoothness penalty, so r_turn < r_noop
        assert r_turn < r_noop

    def test_multi_aircraft_observation_filtering(
        self,
        obs_manager: ObservationManager,
        reward_calculator: RewardCalculator,
    ) -> None:
        """Mix of nearby and far aircraft; only nearby appear in observation."""
        own = make_state("OWN", 39.25, 116.25, 35000.0)
        # 3 nearby (within 20NM)
        near1 = make_state("N1", 39.30, 116.25, 35000.0)   # ~3NM
        near2 = make_state("N2", 39.40, 116.25, 35000.0)   # ~9NM
        near3 = make_state("N3", 39.25, 116.45, 35000.0)   # ~12NM
        # 2 far (outside 20NM)
        far1 = make_state("F1", 39.75, 116.25, 35000.0)    # ~30NM
        far2 = make_state("F2", 39.25, 117.00, 35000.0)    # ~45NM
        others = [near1, near2, near3, far1, far2]
        goal = {"lat": 39.50, "lon": 116.50, "alt": 35000.0, "hdg": 90.0}

        pkg = obs_manager.generate(own, others, goal)
        mask = pkg["observation"]["other_aircraft_mask"]

        # Exactly 3 observable
        assert int(mask.sum()) == 3

        # Observable aircraft are sorted by distance
        other_ac = pkg["observation"]["other_aircraft"]
        distances = []
        for i in range(3):
            if mask[i] == 1:
                distances.append(other_ac[i][3])  # normalized distance
        assert distances == sorted(distances)

    def test_efficiency_goal_tracking(
        self,
        obs_manager: ObservationManager,
        reward_calculator: RewardCalculator,
    ) -> None:
        """Observation goal vector and reward efficiency use same goal."""
        own = make_state("OWN", 39.25, 116.25, 35000.0)
        goal = {"lat": 39.25, "lon": 116.25, "alt": 35000.0, "hdg": 90.0}
        action = DiscreteAction(heading_idx=2, altitude_idx=2, speed_idx=2)

        # Set efficiency goal to same position
        eff = reward_calculator._components[2][0]  # EfficiencyReward
        eff.set_goal("OWN", lat=39.25, lon=116.25)

        # Observation goal vector
        pkg = obs_manager.generate(own, [], goal)
        goal_vec = pkg["observation"]["goal"]
        # Distance=0, bearing=irrelevant, alt=mid, hdg=mid
        assert goal_vec[0] == pytest.approx(0.0)  # normalized distance

        # Reward: arrival bonus
        reward = reward_calculator.compute("OWN", own, action, own, {"OWN": own})
        # step(-0.01)*0.3 + arrival(10)*0.3 = -0.003 + 3.0 = 2.997
        assert reward == pytest.approx(2.997, rel=1e-2)


# ─── Config consistency ─────────────────────────────────────────────


class TestConfigConsistency:
    """Verify all modules interpret the same config identically."""

    def test_shared_config_no_crash(
        self,
        full_config: dict[str, Any],
    ) -> None:
        """All modules accept the same merged config without error."""
        ObservationManager(full_config)
        ActionTranslator(full_config)
        ConflictPenalty(full_config)
        SmoothnessPenalty(full_config)
        EfficiencyReward(full_config)

    def test_action_config_values(
        self,
        action_translator: ActionTranslator,
    ) -> None:
        """Action translator uses correct config values."""
        state = make_state("T", 39.0, 116.0, 35000.0, hdg=180.0, tas=450.0)
        # Index 0 = -20 heading
        a = DiscreteAction(heading_idx=0, altitude_idx=2, speed_idx=2)
        cmds = action_translator.translate("T", state, a)
        assert cmds == ["HDG T 160"]

    def test_reward_config_values(
        self,
        full_config: dict[str, Any],
    ) -> None:
        """Reward components use correct config thresholds."""
        comp = ConflictPenalty(full_config)
        own = make_state("O", 39.25, 116.25, 35000.0)
        # Exactly at NMAC boundary: 5NM horizontal, 1000ft vertical
        boundary = make_state("B", 39.3333, 116.25, 34000.0)
        action = DiscreteAction(heading_idx=2, altitude_idx=2, speed_idx=2)
        result = comp.compute("O", own, action, own, {"O": own, "B": boundary})
        assert result < 0  # at least separation violation


# ─── Type compatibility ─────────────────────────────────────────────


class TestTypeCompatibility:
    """Verify AircraftState flows through all modules without type errors."""

    def test_aircraft_state_dual_access(
        self,
        obs_manager: ObservationManager,
    ) -> None:
        """AircraftState works with both attribute and dict access in all modules."""
        state = make_state("TEST", 39.25, 116.25, 35000.0)

        # Attribute access
        assert state.id == "TEST"
        assert state.lat == 39.25

        # Dict access
        assert state["id"] == "TEST"
        assert state["lat"] == 39.25

        # 'in' operator
        assert "id" in state
        assert "lat" in state

        # Works in observation manager
        goal = {"lat": 39.50, "lon": 116.50, "alt": 35000.0, "hdg": 90.0}
        pkg = obs_manager.generate(state, [], goal)
        assert pkg["textual_state"]["agent_id"] == "TEST"

    def test_discrete_action_tuple_unpack(
        self,
        action_translator: ActionTranslator,
    ) -> None:
        """DiscreteAction supports tuple unpacking and named access."""
        action = DiscreteAction(heading_idx=1, altitude_idx=3, speed_idx=0)
        h, a, s = action
        assert h == 1 and a == 3 and s == 0
        assert action.heading_idx == 1
        assert action.altitude_idx == 3
        assert action.speed_idx == 0

        state = make_state("X", 39.0, 116.0, 35000.0, hdg=90.0, tas=450.0)
        cmds = action_translator.translate("X", state, action)
        assert len(cmds) == 3  # all three axes adjusted


# ─── Boundary conditions ────────────────────────────────────────────


class TestBoundaryConditions:
    """Edge cases at module boundaries."""

    def test_empty_other_aircraft(
        self,
        obs_manager: ObservationManager,
        reward_calculator: RewardCalculator,
    ) -> None:
        """No other aircraft: observation and reward both handle gracefully."""
        own = make_state("OWN", 39.25, 116.25, 35000.0)
        goal = {"lat": 39.50, "lon": 116.50, "alt": 35000.0, "hdg": 90.0}
        action = DiscreteAction(heading_idx=2, altitude_idx=2, speed_idx=2)

        pkg = obs_manager.generate(own, [], goal)
        assert int(pkg["observation"]["other_aircraft_mask"].sum()) == 0

        reward = reward_calculator.compute("OWN", own, action, own, {"OWN": own})
        assert isinstance(reward, float)

    def test_all_aircraft_at_same_position(
        self,
        obs_manager: ObservationManager,
        reward_calculator: RewardCalculator,
    ) -> None:
        """All aircraft co-located: extreme conflict, observation still valid."""
        pos = make_state("A", 39.25, 116.25, 35000.0)
        others = [
            make_state("B", 39.25, 116.25, 35000.0),
            make_state("C", 39.25, 116.25, 35000.0),
        ]
        goal = {"lat": 39.50, "lon": 116.50, "alt": 35000.0, "hdg": 90.0}
        action = DiscreteAction(heading_idx=2, altitude_idx=2, speed_idx=2)

        pkg = obs_manager.generate(pos, others, goal)
        assert pkg["observation"]["self_state"].shape == (9,)

        all_states = {"A": pos, "B": others[0], "C": others[1]}
        reward = reward_calculator.compute("A", pos, action, pos, all_states)
        # NMAC penalty dominates
        assert reward <= -100.0

    def test_heading_wraparound_in_pipeline(
        self,
        action_translator: ActionTranslator,
        reward_calculator: RewardCalculator,
    ) -> None:
        """Heading 350° +20° wraps to 10°; reward unaffected."""
        own = make_state("OWN", 39.25, 116.25, 35000.0, hdg=350.0)
        # heading_idx=4 → +20° → 350+20=370%360=10
        action = DiscreteAction(heading_idx=4, altitude_idx=2, speed_idx=2)

        cmds = action_translator.translate("OWN", own, action)
        assert cmds == ["HDG OWN 10"]

        reward = reward_calculator.compute("OWN", own, action, own, {"OWN": own})
        assert isinstance(reward, float)

    def test_extreme_altitude_in_observation(
        self,
        obs_manager: ObservationManager,
    ) -> None:
        """Very high/low altitude: normalizer clips normalized values."""
        high = make_state("H", 39.25, 116.25, 50000.0)
        low = make_state("L", 39.26, 116.25, 10000.0)
        goal = {"lat": 39.50, "lon": 116.50, "alt": 35000.0, "hdg": 90.0}

        pkg = obs_manager.generate(high, [low], goal)
        self_state = pkg["observation"]["self_state"]
        # Normalized values (heading, altitude, speed) clipped to [-1, 1]
        # Indices 0=hdg, 1=alt, 2=spd are normalized; 3=lat, 4=lon are raw
        assert float(self_state[0]) >= -1.0 and float(self_state[0]) <= 1.0
        assert float(self_state[1]) >= -1.0 and float(self_state[1]) <= 1.0
        assert float(self_state[2]) >= -1.0 and float(self_state[2]) <= 1.0

    def test_reward_calculator_reset_propagation(
        self,
        full_config: dict[str, Any],
    ) -> None:
        """Reset clears internal state of all components including efficiency goals."""
        calc = RewardCalculator()
        eff = EfficiencyReward(full_config)
        calc.register(eff, weight=1.0)

        eff.set_goal("A", lat=39.0, lon=116.0)
        calc.reset()

        # After reset, goal should be cleared → only step penalty
        state = make_state("A", 39.0, 116.0, 35000.0)
        action = DiscreteAction(heading_idx=2, altitude_idx=2, speed_idx=2)
        result = calc.compute("A", state, action, state, {"A": state})
        assert result == pytest.approx(-0.01)
