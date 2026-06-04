"""Integration tests — no-conflict scenario (T-14).

All aircraft fly straight with no conflicts. Validates:
  1. RuleBasedAgent completes a full episode
  2. RandomAgent runs 100 steps without error
  3. No NMAC in a safe scenario
  4. Rewards remain within reasonable bounds
"""

from __future__ import annotations

from bluesky_pettingzoo.envs.parallel_env import BlueSkyMARLEnv
from tests.helpers.env_factory import make_env as _make_env

# ---------------------------------------------------------------------------
# Fake BlueSkyWrapper — in-memory, no real BlueSky dependency
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Config & env factory
# ---------------------------------------------------------------------------


def _place_aircraft(
    env: BlueSkyMARLEnv,
    positions: dict[str, dict[str, float]],
) -> None:
    """Manually place aircraft at specific positions after reset."""
    wrapper = env._wrapper
    for acid, pos in positions.items():
        if acid in wrapper.get_active_aircraft_ids():
            wrapper.set_aircraft_state(acid, **pos)


# ===========================================================================
# T-14 Test Cases
# ===========================================================================


class TestRuleBasedAgentFullEpisode:
    """RuleBasedAgent should complete a full episode without error."""

    def test_rule_based_agent_full_episode(self) -> None:
        """Run a complete episode with RuleBasedAgent (straight flight)."""
        from bluesky_pettingzoo.agents.rule_based_agent import RuleBasedAgent

        max_steps = 20
        env = _make_env(initial_count=3, max_steps=max_steps)
        env.reset(seed=42)

        # Place aircraft far apart on parallel tracks — no conflict
        _place_aircraft(
            env,
            {
                "AC000": {"lat": 39.10, "lon": 116.10, "alt": 30000, "hdg": 90, "tas": 450},
                "AC001": {"lat": 39.25, "lon": 116.25, "alt": 33000, "hdg": 90, "tas": 450},
                "AC002": {"lat": 39.40, "lon": 116.40, "alt": 36000, "hdg": 90, "tas": 450},
            },
        )

        agent = RuleBasedAgent()
        step_count = 0
        for _ in range(max_steps):
            action_spaces = {a: env.action_space(a) for a in env.agents}
            actions = agent.act({}, action_spaces)
            obs, rewards, terms, truncs, infos = env.step(actions)
            step_count += 1
            if all(truncs.values()):
                break

        assert step_count == max_steps


class TestRandomAgentRuns100Steps:
    """RandomAgent should run 100 steps without error."""

    def test_random_agent_runs_100_steps(self) -> None:
        """RandomAgent runs 100 steps; no crash, no exception."""
        from bluesky_pettingzoo.agents.random_agent import RandomAgent

        env = _make_env(initial_count=3, max_steps=200)
        env.reset(seed=42)

        # Place aircraft far apart to avoid conflicts
        _place_aircraft(
            env,
            {
                "AC000": {"lat": 39.10, "lon": 116.10, "alt": 30000, "hdg": 45, "tas": 450},
                "AC001": {"lat": 39.25, "lon": 116.25, "alt": 33000, "hdg": 135, "tas": 450},
                "AC002": {"lat": 39.40, "lon": 116.40, "alt": 36000, "hdg": 225, "tas": 450},
            },
        )

        agent = RandomAgent()
        for i in range(100):
            if not env.agents:
                break
            action_spaces = {a: env.action_space(a) for a in env.agents}
            actions = agent.act({}, action_spaces)
            obs, rewards, terms, truncs, infos = env.step(actions)

        # Reached 100 steps (or all agents left earlier)
        assert i >= 99 or len(env.agents) == 0


class TestNoNMACInSafeScenario:
    """Aircraft far apart should never trigger NMAC."""

    def test_no_nmac_in_safe_scenario(self) -> None:
        """Aircraft separated by >20NM: no NMAC penalty in any step."""
        env = _make_env(initial_count=3, max_steps=10)
        env.reset(seed=42)

        # Place aircraft very far apart — well beyond conflict range
        _place_aircraft(
            env,
            {
                "AC000": {"lat": 39.05, "lon": 116.05, "alt": 30000, "hdg": 90, "tas": 450},
                "AC001": {"lat": 39.25, "lon": 116.25, "alt": 33000, "hdg": 90, "tas": 450},
                "AC002": {"lat": 39.45, "lon": 116.45, "alt": 36000, "hdg": 90, "tas": 450},
            },
        )

        nmac_penalty = -100.0
        for _ in range(10):
            actions = {a: [2, 2, 2] for a in env.agents}
            obs, rewards, terms, truncs, infos = env.step(actions)

            for agent_id in rewards:
                assert rewards[agent_id] > nmac_penalty, (
                    f"{agent_id} got NMAC-level penalty {rewards[agent_id]}"
                )

            # Also verify textual state reports safe
            for agent_id in infos:
                assert infos[agent_id]["textual_state"]["conflict_status"] == "safe"


class TestRewardsBounded:
    """Rewards in a no-conflict scenario should stay within reasonable bounds."""

    def test_rewards_bounded(self) -> None:
        """No-conflict episode: per-step reward stays within reasonable bounds.

        Bound rationale:
        - Efficiency deviation penalty cap: -5.0 * 0.3 weight = -1.5
        - Efficiency step penalty: -0.01 * 0.3 = -0.003
        - Smoothness: 0 (no-op action)
        - Worst-case per-step: ~-1.503
        """
        env = _make_env(initial_count=3, max_steps=20)
        env.reset(seed=42)

        # Aircraft far apart, parallel tracks
        _place_aircraft(
            env,
            {
                "AC000": {"lat": 39.10, "lon": 116.10, "alt": 30000, "hdg": 90, "tas": 450},
                "AC001": {"lat": 39.25, "lon": 116.25, "alt": 33000, "hdg": 90, "tas": 450},
                "AC002": {"lat": 39.40, "lon": 116.40, "alt": 36000, "hdg": 90, "tas": 450},
            },
        )

        total_rewards: dict[str, float] = {a: 0.0 for a in env.agents}
        for _ in range(20):
            actions = {a: [2, 2, 2] for a in env.agents}
            obs, rewards, terms, truncs, infos = env.step(actions)

            for agent_id in rewards:
                # No conflict → no NMAC penalty; worst case is efficiency deviation
                assert -2.0 < rewards[agent_id] < 1.0, (
                    f"{agent_id} reward {rewards[agent_id]} out of bounds"
                )
                if agent_id in total_rewards:
                    total_rewards[agent_id] += rewards[agent_id]

        # Accumulated reward should be negative (step penalties) but bounded
        for agent_id in total_rewards:
            assert total_rewards[agent_id] < 0, "Expected negative accumulated reward"
            assert total_rewards[agent_id] > -40, "Accumulated reward too negative"
