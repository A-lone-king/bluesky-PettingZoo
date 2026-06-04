"""Tests for dynamic aircraft entry during episode (T-V05)."""

from __future__ import annotations

from tests.helpers.env_factory import make_config as _make_config
from tests.helpers.env_factory import make_env as _make_env

# ---------------------------------------------------------------------------
# BlueSkyWrapper
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class TestDynamicEntry:
    """New aircraft should enter the airspace during an episode."""

    def test_dynamic_entry_adds_agent(self) -> None:
        """After interval steps, a new agent appears in agents list."""
        config = _make_config(
            initial_count=2,
            dynamic_entry={"enabled": True, "interval": 3, "max_total": 5},
        )
        env = _make_env(config)
        env.reset(seed=42)

        initial_agents = set(env.agents)
        noop = [2, 2, 2]

        # Step enough times to trigger entry (interval=3, so after 3 steps)
        for _ in range(4):
            actions = {aid: noop for aid in env.agents}
            env.step(actions)

        # At least one new agent should have appeared
        new_agents = set(env.agents) - initial_agents
        assert len(new_agents) >= 1, (
            f"Expected new agents, but agents={env.agents}, initial={list(initial_agents)}"
        )

    def test_dynamic_entry_gets_observation(self) -> None:
        """Newly entered aircraft gets a valid observation."""
        config = _make_config(
            initial_count=2,
            dynamic_entry={"enabled": True, "interval": 3, "max_total": 5},
        )
        env = _make_env(config)
        obs, info = env.reset(seed=42)
        initial_agents = set(env.agents)
        noop = [2, 2, 2]

        # Step to trigger entry
        for _ in range(4):
            actions = {aid: noop for aid in env.agents}
            obs, rewards, terms, trunks, infos = env.step(actions)

        # Find new agents — must exist
        new_agents = set(env.agents) - initial_agents
        assert len(new_agents) >= 1, "Expected at least one new agent after dynamic entry"
        for agent_id in new_agents:
            assert agent_id in obs
            assert "self_state" in obs[agent_id]
            assert obs[agent_id]["self_state"].shape == (9,)

    def test_dynamic_entry_configurable_interval(self) -> None:
        """Entry interval is respected — no entry before interval steps."""
        config = _make_config(
            initial_count=2,
            dynamic_entry={"enabled": True, "interval": 10, "max_total": 5},
        )
        env = _make_env(config)
        env.reset(seed=42)
        initial_agents = set(env.agents)
        noop = [2, 2, 2]

        # Step fewer than interval — no entry should happen
        for _ in range(5):
            actions = {aid: noop for aid in env.agents}
            env.step(actions)

        # After fewer steps than interval, no new aircraft
        new_agents = set(env.agents) - initial_agents
        assert len(new_agents) == 0, f"Expected no new agents before interval, got {new_agents}"

    def test_dynamic_entry_max_total(self) -> None:
        """Total aircraft count never exceeds max_total."""
        config = _make_config(
            initial_count=2,
            dynamic_entry={"enabled": True, "interval": 1, "max_total": 3},
        )
        env = _make_env(config)
        env.reset(seed=42)
        noop = [2, 2, 2]

        # Step many times — should never exceed max_total
        for _ in range(20):
            actions = {aid: noop for aid in env.agents}
            env.step(actions)
            assert len(env.agents) <= 3

    def test_dynamic_entry_from_boundary(self) -> None:
        """New aircraft spawn at airspace boundary positions."""
        config = _make_config(
            initial_count=2,
            dynamic_entry={"enabled": True, "interval": 3, "max_total": 5},
        )
        env = _make_env(config)
        env.reset(seed=42)
        initial_agents = set(env.agents)
        noop = [2, 2, 2]

        for _ in range(4):
            actions = {aid: noop for aid in env.agents}
            env.step(actions)

        new_agents = set(env.agents) - initial_agents
        assert len(new_agents) >= 1, "Expected at least one new agent"
        bounds = env._airspace
        for agent_id in new_agents:
            state = env._get_aircraft_state(agent_id)
            # Aircraft should be near one of the four boundaries (within 0.1 degree)
            on_boundary = (
                abs(state.lat - bounds["lat_min"]) < 0.1
                or abs(state.lat - bounds["lat_max"]) < 0.1
                or abs(state.lon - bounds["lon_min"]) < 0.1
                or abs(state.lon - bounds["lon_max"]) < 0.1
            )
            assert on_boundary, f"{agent_id} at ({state.lat}, {state.lon}) not on boundary"

    def test_dynamic_entry_disabled_by_default(self) -> None:
        """Without dynamic_entry config, no new aircraft appear."""
        config = _make_config(
            initial_count=2,
            dynamic_entry={"enabled": False, "interval": 3, "max_total": 5},
        )
        env = _make_env(config)
        env.reset(seed=42)
        initial_agents = set(env.agents)
        noop = [2, 2, 2]

        for _ in range(5):
            actions = {aid: noop for aid in env.agents}
            env.step(actions)

        # No new aircraft should appear (agents may decrease due to removal)
        new_agents = set(env.agents) - initial_agents
        assert len(new_agents) == 0, f"Expected no new agents when disabled, got {new_agents}"
