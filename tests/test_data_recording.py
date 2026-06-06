"""Tests for the data recording module (types, recorder, wrapper)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from bluesky_pettingzoo.recording.recorder import DataRecorder
from bluesky_pettingzoo.recording.types import (
    AgentRecord,
    ConflictRecord,
    EpisodeRecord,
    RewardDecomposition,
    TrajectoryPoint,
)
from bluesky_pettingzoo.recording.wrapper import DataRecordingWrapper
from bluesky_pettingzoo.utils.types import AircraftState

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_state(
    agent_id: str = "AC001",
    lat: float = 39.0,
    lon: float = 116.0,
    alt: float = 35000.0,
) -> AircraftState:
    return AircraftState(
        id=agent_id,
        lat=lat,
        lon=lon,
        alt=alt,
        hdg=90.0,
        tas=450.0,
        vs=0.0,
    )


def _make_conflict(
    step: int = 1,
    id1: str = "AC001",
    id2: str = "AC002",
    dist: float = 5.0,
    vs: float = 1000.0,
    severity: str = "warning",
) -> ConflictRecord:
    return ConflictRecord(
        step=step,
        aircraft_ids=(id1, id2),
        distance_nm=dist,
        vertical_sep_ft=vs,
        severity=severity,
    )


def _make_mock_env(agents: list[str] | None = None) -> MagicMock:
    """Create a minimal mock ParallelEnv for wrapper tests."""
    env = MagicMock()
    agents = agents or ["AC001", "AC002"]
    env.agents = agents
    env.possible_agents = agents
    env.observation_space.return_value = MagicMock()
    env.action_space.return_value = MagicMock()

    # State for each agent
    env.unwrapped.aircraft_states = {
        aid: _make_state(aid, lat=39.0 + i * 0.1, lon=116.0 + i * 0.1)
        for i, aid in enumerate(agents)
    }

    return env


# ===================================================================
# TrajectoryPoint
# ===================================================================


class TestTrajectoryPoint:
    def test_creation(self) -> None:
        tp = TrajectoryPoint(step=0, lat=40.0, lon=116.5, alt=35000.0, hdg=90.0, tas=450.0, vs=0.0)
        assert tp.step == 0
        assert tp.lat == 40.0
        assert tp.vs == 0.0

    def test_frozen(self) -> None:
        tp = TrajectoryPoint(step=0, lat=40.0, lon=116.5, alt=35000.0, hdg=90.0, tas=450.0, vs=0.0)
        with pytest.raises(AttributeError):
            tp.step = 1  # type: ignore[misc]


# ===================================================================
# ConflictRecord
# ===================================================================


class TestConflictRecord:
    def test_creation(self) -> None:
        cr = _make_conflict()
        assert cr.step == 1
        assert cr.severity == "warning"

    def test_frozen(self) -> None:
        cr = _make_conflict()
        with pytest.raises(AttributeError):
            cr.step = 2  # type: ignore[misc]


# ===================================================================
# RewardDecomposition
# ===================================================================


class TestRewardDecomposition:
    def test_creation(self) -> None:
        rd = RewardDecomposition(
            component_name="conflict",
            weight=1.0,
            raw_value=-5.0,
            weighted_value=-5.0,
        )
        assert rd.component_name == "conflict"
        assert rd.weighted_value == -5.0


# ===================================================================
# AgentRecord
# ===================================================================


class TestAgentRecord:
    def test_defaults(self) -> None:
        ar = AgentRecord(agent_id="AC001")
        assert ar.agent_id == "AC001"
        assert ar.trajectory == []
        assert ar.rewards == []
        assert ar.terminated is False


# ===================================================================
# EpisodeRecord
# ===================================================================


class TestEpisodeRecord:
    def _build_episode(self) -> EpisodeRecord:
        ar = AgentRecord(agent_id="AC001")
        ar.trajectory = [
            TrajectoryPoint(
                step=0,
                lat=39.0,
                lon=116.0,
                alt=35000.0,
                hdg=90.0,
                tas=450.0,
                vs=0.0,
            ),
            TrajectoryPoint(
                step=1,
                lat=39.1,
                lon=116.1,
                alt=34000.0,
                hdg=95.0,
                tas=440.0,
                vs=-1000.0,
            ),
        ]
        ar.rewards = [-1.0, -2.0]
        return EpisodeRecord(
            episode_id=0,
            scenario="TestScenario",
            total_steps=2,
            agents={"AC001": ar},
            conflicts=[],
            reward_totals=[{"AC001": -1.0}, {"AC001": -2.0}],
        )

    def test_get_trajectory(self) -> None:
        ep = self._build_episode()
        traj = ep.get_trajectory("AC001")
        assert len(traj) == 2
        assert traj[0].alt == 35000.0

    def test_get_trajectory_missing_agent(self) -> None:
        ep = self._build_episode()
        assert ep.get_trajectory("NONEXIST") == []

    def test_get_total_rewards(self) -> None:
        ep = self._build_episode()
        rewards = ep.get_total_rewards("AC001")
        assert rewards == [-1.0, -2.0]

    def test_get_total_rewards_missing_agent(self) -> None:
        ep = self._build_episode()
        assert ep.get_total_rewards("NONEXIST") == []

    def test_summary(self) -> None:
        ep = self._build_episode()
        s = ep.summary()
        assert s["episode_id"] == 0
        assert s["scenario"] == "TestScenario"
        assert s["total_steps"] == 2
        assert s["num_agents"] == 1
        assert s["agents"]["AC001"]["total_reward"] == -3.0


# ===================================================================
# DataRecorder
# ===================================================================


class TestDataRecorder:
    def test_record_and_finalize(self) -> None:
        rec = DataRecorder(episode_id=1, scenario="Test")
        rec.set_agents(["AC001", "AC002"])

        states = {
            "AC001": _make_state("AC001"),
            "AC002": _make_state("AC002"),
        }
        rec.record_step(
            step=1,
            states=states,
            rewards={"AC001": -1.0, "AC002": -0.5},
            conflicts=[_make_conflict()],
        )
        rec.record_step(
            step=2,
            states=states,
            rewards={"AC001": -2.0, "AC002": -1.0},
        )

        record = rec.finalize(terminated={"AC001": True, "AC002": False})

        assert record.episode_id == 1
        assert record.scenario == "Test"
        assert record.total_steps == 2
        assert len(record.agents) == 2
        assert record.agents["AC001"].terminated is True
        assert record.agents["AC002"].terminated is False
        assert len(record.conflicts) == 1
        assert len(record.reward_totals) == 2

    def test_record_decomposition(self) -> None:
        rec = DataRecorder(episode_id=0, scenario="T")
        rec.set_agents(["AC001"])

        decomp = {"AC001": [("conflict", 1.0, -5.0, -5.0), ("efficiency", 0.5, 8.0, 4.0)]}
        rec.record_step(
            step=1,
            states={"AC001": _make_state()},
            rewards={"AC001": -1.0},
            decompositions=decomp,
        )

        record = rec.finalize()
        agent = record.agents["AC001"]
        assert len(agent.reward_decompositions) == 1
        assert len(agent.reward_decompositions[0]) == 2
        assert agent.reward_decompositions[0][0].component_name == "conflict"

    def test_add_agent(self) -> None:
        rec = DataRecorder()
        rec.set_agents(["AC001"])
        rec.add_agent("AC002")
        assert "AC002" in rec._agents

    def test_add_agent_existing(self) -> None:
        rec = DataRecorder()
        rec.set_agents(["AC001"])
        rec.add_agent("AC001")
        assert len(rec._agents) == 1

    def test_to_json(self, tmp_path: Path) -> None:
        rec = DataRecorder(episode_id=0, scenario="Test")
        rec.set_agents(["AC001"])
        rec.record_step(
            step=1,
            states={"AC001": _make_state()},
            rewards={"AC001": -1.0},
        )
        record = rec.finalize()

        out = tmp_path / "episode.json"
        rec.to_json(record, out)

        data = json.loads(out.read_text(encoding="utf-8"))
        assert data["episode_id"] == 0
        assert data["scenario"] == "Test"
        assert len(data["agents"]["AC001"]["trajectory"]) == 1
        assert data["agents"]["AC001"]["rewards"] == [-1.0]


# ===================================================================
# DataRecordingWrapper
# ===================================================================


class TestDataRecordingWrapper:
    def _make_wrapper(self, **kwargs: object) -> DataRecordingWrapper:
        env = _make_mock_env()
        env.reset.return_value = (
            {"AC001": {"self_state": [0] * 6}, "AC002": {"self_state": [0] * 6}},
            {"AC001": {}, "AC002": {}},
        )
        env.step.return_value = (
            {"AC001": {"self_state": [0] * 6}, "AC002": {"self_state": [0] * 6}},
            {"AC001": -1.0, "AC002": -0.5},
            {"AC001": False, "AC002": False},
            {"AC001": False, "AC002": False},
            {"AC001": {}, "AC002": {}},
        )
        return DataRecordingWrapper(env, **kwargs)

    def test_reset_starts_episode(self) -> None:
        w = self._make_wrapper(scenario="Test")
        obs, infos = w.reset()
        assert "AC001" in obs
        assert w._recorder is not None
        assert w._step_count == 0

    def test_step_records_trajectory(self) -> None:
        w = self._make_wrapper(scenario="Test")
        w.reset()
        w.step({"AC001": 0, "AC002": 0})
        assert w._step_count == 1
        assert len(w._recorder._agents["AC001"].trajectory) == 1

    def test_get_last_episode_before_finalize(self) -> None:
        w = self._make_wrapper()
        w.reset()
        assert w.get_last_episode() is None
        assert w.get_records() == []

    def test_get_records_after_termination(self) -> None:
        env = _make_mock_env()
        env.reset.return_value = (
            {"AC001": {}, "AC002": {}},
            {"AC001": {}, "AC002": {}},
        )
        env.step.return_value = (
            {"AC001": {}, "AC002": {}},
            {"AC001": -1.0, "AC002": -0.5},
            {"AC001": True, "AC002": True},
            {"AC001": False, "AC002": False},
            {"AC001": {}, "AC002": {}},
        )
        w = DataRecordingWrapper(env, scenario="Test")
        w.reset()
        w.step({"AC001": 0, "AC002": 0})

        records = w.get_records()
        assert len(records) == 1
        assert records[0].scenario == "Test"

    def test_multiple_episodes(self) -> None:
        env = _make_mock_env()
        env.reset.return_value = (
            {"AC001": {}, "AC002": {}},
            {"AC001": {}, "AC002": {}},
        )
        env.step.return_value = (
            {"AC001": {}, "AC002": {}},
            {"AC001": 0.0, "AC002": 0.0},
            {"AC001": True, "AC002": True},
            {"AC001": False, "AC002": False},
            {"AC001": {}, "AC002": {}},
        )
        w = DataRecordingWrapper(env, scenario="Multi")

        # Episode 0
        w.reset()
        w.step({"AC001": 0, "AC002": 0})

        # Episode 1 (reset finalizes previous)
        w.reset()
        w.step({"AC001": 0, "AC002": 0})

        records = w.get_records()
        assert len(records) == 2
        assert records[0].episode_id == 0
        assert records[1].episode_id == 1

    def test_decomposition_recorded(self) -> None:
        env = _make_mock_env()
        env.reset.return_value = (
            {"AC001": {}, "AC002": {}},
            {"AC001": {}, "AC002": {}},
        )
        env.step.return_value = (
            {"AC001": {}, "AC002": {}},
            {"AC001": -5.0, "AC002": -1.0},
            {"AC001": True, "AC002": True},
            {"AC001": False, "AC002": False},
            {
                "AC001": {"reward_decomposition": [("conflict", 1.0, -5.0, -5.0)]},
                "AC002": {},
            },
        )
        w = DataRecordingWrapper(env, scenario="Decomp")
        w.reset()
        w.step({"AC001": 0, "AC002": 0})

        record = w.get_last_episode()
        assert record is not None
        decomp = record.agents["AC001"].reward_decompositions
        assert len(decomp) == 1
        assert decomp[0][0].component_name == "conflict"

    def test_conflicts_recorded(self) -> None:
        env = _make_mock_env()
        env.reset.return_value = (
            {"AC001": {}, "AC002": {}},
            {"AC001": {}, "AC002": {}},
        )
        conflict = _make_conflict()
        env.step.return_value = (
            {"AC001": {}, "AC002": {}},
            {"AC001": 0.0, "AC002": 0.0},
            {"AC001": True, "AC002": True},
            {"AC001": False, "AC002": False},
            {"AC001": {"conflicts": [conflict]}, "AC002": {}},
        )
        w = DataRecordingWrapper(env, scenario="Conflict")
        w.reset()
        w.step({"AC001": 0, "AC002": 0})

        record = w.get_last_episode()
        assert record is not None
        assert len(record.conflicts) == 1

    def test_disable_decomposition(self) -> None:
        env = _make_mock_env()
        env.reset.return_value = ({"AC001": {}, "AC002": {}}, {"AC001": {}, "AC002": {}})
        env.step.return_value = (
            {"AC001": {}, "AC002": {}},
            {"AC001": 0.0, "AC002": 0.0},
            {"AC001": True, "AC002": True},
            {"AC001": False, "AC002": False},
            {"AC001": {"reward_decomposition": [("c", 1.0, 1.0, 1.0)]}, "AC002": {}},
        )
        w = DataRecordingWrapper(env, record_decompositions=False)
        w.reset()
        w.step({"AC001": 0, "AC002": 0})

        record = w.get_last_episode()
        assert record is not None
        assert record.agents["AC001"].reward_decompositions == []

    def test_delegates_close(self) -> None:
        env = _make_mock_env()
        w = DataRecordingWrapper(env)
        w.close()
        env.close.assert_called_once()
