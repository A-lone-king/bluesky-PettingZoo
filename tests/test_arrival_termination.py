"""Tests for arrival-based agent termination (T-V01)."""

from __future__ import annotations

from pathlib import Path

from tests.helpers.env_factory import make_config as _make_config
from tests.helpers.env_factory import make_env as _make_env

# ---------------------------------------------------------------------------
# BlueSkyWrapper (same as test_env.py, with basic movement)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class TestArrivalTriggersTermination:
    """Aircraft reaching its goal waypoint should trigger termination=True."""

    def test_arrival_triggers_termination(self, tmp_path: Path) -> None:
        config = _make_config(initial_count=2, arrival_threshold_nm=2.0)
        env = _make_env(config, tmp_path)
        env.reset(seed=42)

        eff = env._find_efficiency_component()
        assert eff is not None

        # AC000: place at goal → should terminate
        goal0 = (39.45, 116.45)
        eff.set_goal("AC000", goal0[0], goal0[1])
        env._wrapper.set_aircraft_state("AC000", lat=goal0[0], lon=goal0[1])

        # AC001: far from goal → should not terminate
        eff.set_goal("AC001", 39.40, 116.40)
        env._wrapper.set_aircraft_state("AC001", lat=39.10, lon=116.10, hdg=90.0, tas=450.0)

        actions = {a: [2, 2, 2] for a in env.agents}
        _, _, terminations, _, _ = env.step(actions)

        assert terminations["AC000"] is True


class TestArrivalRemovesFromAgents:
    """After arrival, the aircraft should be removed from env.agents."""

    def test_arrival_removes_from_agents(self, tmp_path: Path) -> None:
        config = _make_config(initial_count=2, arrival_threshold_nm=2.0)
        env = _make_env(config, tmp_path)
        env.reset(seed=42)

        eff = env._find_efficiency_component()
        assert eff is not None

        goal0 = (39.45, 116.45)
        eff.set_goal("AC000", goal0[0], goal0[1])
        env._wrapper.set_aircraft_state("AC000", lat=goal0[0], lon=goal0[1])

        eff.set_goal("AC001", 39.40, 116.40)
        env._wrapper.set_aircraft_state("AC001", lat=39.10, lon=116.10, hdg=90.0, tas=450.0)

        actions = {a: [2, 2, 2] for a in env.agents}
        env.step(actions)

        assert "AC000" not in env.agents


class TestArrivalOtherAgentsUnaffected:
    """Agents that have not reached their goal should remain active."""

    def test_arrival_other_agents_unaffected(self, tmp_path: Path) -> None:
        config = _make_config(initial_count=2, arrival_threshold_nm=2.0)
        env = _make_env(config, tmp_path)
        env.reset(seed=42)

        eff = env._find_efficiency_component()
        assert eff is not None

        goal0 = (39.45, 116.45)
        eff.set_goal("AC000", goal0[0], goal0[1])
        env._wrapper.set_aircraft_state("AC000", lat=goal0[0], lon=goal0[1])

        eff.set_goal("AC001", 39.40, 116.40)
        env._wrapper.set_aircraft_state("AC001", lat=39.10, lon=116.10, hdg=90.0, tas=450.0)

        actions = {a: [2, 2, 2] for a in env.agents}
        _, _, terminations, _, _ = env.step(actions)

        assert "AC001" in env.agents
        assert terminations["AC001"] is False


class TestArrivalThresholdConfigurable:
    """Arrival threshold should be read from config."""

    def test_arrival_threshold_configurable(self, tmp_path: Path) -> None:
        # Use a larger threshold (10 NM) so that even aircraft further away trigger arrival
        config = _make_config(initial_count=2, arrival_threshold_nm=10.0)
        env = _make_env(config, tmp_path)
        env.reset(seed=42)

        eff = env._find_efficiency_component()
        assert eff is not None

        # AC000 placed ~5 NM away from goal → within 10 NM threshold
        eff.set_goal("AC000", 39.45, 116.45)
        env._wrapper.set_aircraft_state("AC000", lat=39.40, lon=116.40)

        eff.set_goal("AC001", 39.40, 116.40)
        env._wrapper.set_aircraft_state("AC001", lat=39.10, lon=116.10, hdg=90.0, tas=450.0)

        actions = {a: [2, 2, 2] for a in env.agents}
        _, _, terminations, _, _ = env.step(actions)

        # With 10 NM threshold, ~5 NM distance should trigger arrival
        assert terminations["AC000"] is True


class TestArrivalNotReachedNoTermination:
    """Aircraft far from its goal should NOT trigger termination."""

    def test_arrival_not_reached_no_termination(self, tmp_path: Path) -> None:
        config = _make_config(initial_count=2, arrival_threshold_nm=2.0)
        env = _make_env(config, tmp_path)
        env.reset(seed=42)

        eff = env._find_efficiency_component()
        assert eff is not None

        # Both aircraft far from their goals
        eff.set_goal("AC000", 39.45, 116.45)
        env._wrapper.set_aircraft_state("AC000", lat=39.10, lon=116.10, hdg=90.0, tas=450.0)

        eff.set_goal("AC001", 39.40, 116.40)
        env._wrapper.set_aircraft_state("AC001", lat=39.20, lon=116.20, hdg=180.0, tas=450.0)

        actions = {a: [2, 2, 2] for a in env.agents}
        _, _, terminations, _, _ = env.step(actions)

        # Neither should have reached goal
        # We check: if still in agents, termination must be False
        for agent in env.agents:
            assert terminations[agent] is False
