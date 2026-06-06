"""DataRecorder: accumulates episode data and produces EpisodeRecord."""

from __future__ import annotations

import json
from pathlib import Path

from bluesky_pettingzoo.recording.types import (
    AgentRecord,
    ConflictRecord,
    EpisodeRecord,
    RewardDecomposition,
    TrajectoryPoint,
)
from bluesky_pettingzoo.utils.types import AircraftState


class DataRecorder:
    """Accumulates per-step data and produces an EpisodeRecord.

    Usage::

        recorder = DataRecorder(episode_id=0, scenario="HorizontalCR")
        recorder.set_agents(["AC000", "AC001"])

        # Each step:
        recorder.record_step(
            step=1,
            states=states,
            rewards={"AC000": -1.5, "AC001": -2.0},
            decompositions={"AC000": [...], "AC001": [...]},
            conflicts=[...],
        )

        # End of episode:
        record = recorder.finalize(terminated={"AC000": True}, truncated={"AC000": False})
    """

    def __init__(self, episode_id: int = 0, scenario: str = "") -> None:
        self._episode_id = episode_id
        self._scenario = scenario
        self._step_count: int = 0
        self._agents: dict[str, AgentRecord] = {}
        self._all_conflicts: list[ConflictRecord] = []
        self._reward_totals: list[dict[str, float]] = []

    def set_agents(self, agent_ids: list[str]) -> None:
        """Initialize agent records for the episode."""
        self._agents = {aid: AgentRecord(agent_id=aid) for aid in agent_ids}

    def add_agent(self, agent_id: str) -> None:
        """Add a new agent mid-episode (dynamic entry)."""
        if agent_id not in self._agents:
            self._agents[agent_id] = AgentRecord(agent_id=agent_id)

    def remove_agent(self, agent_id: str) -> None:
        """Mark an agent as removed (does not delete data)."""
        # Data is kept for post-episode analysis
        pass

    def record_step(
        self,
        step: int,
        states: dict[str, AircraftState],
        rewards: dict[str, float],
        decompositions: dict[str, list[tuple[str, float, float, float]]] | None = None,
        conflicts: list[ConflictRecord] | None = None,
    ) -> None:
        """Record data for a single timestep.

        Args:
            step: Current step number.
            states: Current aircraft states.
            rewards: Total reward per agent.
            decompositions: Per-component reward breakdown per agent.
                Each value is a list of (name, weight, raw, weighted).
            conflicts: Detected conflict events at this step.
        """
        self._step_count = step

        # Record trajectory for each agent
        for aid, state in states.items():
            rec = self._agents.get(aid)
            if rec is None:
                continue
            rec.trajectory.append(
                TrajectoryPoint(
                    step=step,
                    lat=state.lat,
                    lon=state.lon,
                    alt=state.alt,
                    hdg=state.hdg,
                    tas=state.tas,
                    vs=state.vs,
                )
            )

        # Record rewards
        for aid, reward_val in rewards.items():
            rec = self._agents.get(aid)
            if rec is not None:
                rec.rewards.append(reward_val)

        # Record reward decompositions
        if decompositions is not None:
            for aid, comp_list in decompositions.items():
                rec = self._agents.get(aid)
                if rec is None:
                    continue
                decomp = [
                    RewardDecomposition(
                        component_name=name,
                        weight=weight,
                        raw_value=raw,
                        weighted_value=weighted,
                    )
                    for name, weight, raw, weighted in comp_list
                ]
                rec.reward_decompositions.append(decomp)

        # Record conflicts
        if conflicts is not None:
            self._all_conflicts.extend(conflicts)
            # Also attach to affected agents
            for conflict in conflicts:
                for acid in conflict.aircraft_ids:
                    rec = self._agents.get(acid)
                    if rec is not None:
                        rec.conflicts.append(conflict)

        # Record per-step reward totals
        self._reward_totals.append(dict(rewards))

    def finalize(
        self,
        terminated: dict[str, bool] | None = None,
        truncated: dict[str, bool] | None = None,
    ) -> EpisodeRecord:
        """Finalize and return the episode record.

        Args:
            terminated: Per-agent termination flags.
            truncated: Per-agent truncation flags.

        Returns:
            Complete EpisodeRecord.
        """
        if terminated is not None:
            for aid, val in terminated.items():
                rec = self._agents.get(aid)
                if rec is not None:
                    rec.terminated = val

        if truncated is not None:
            for aid, val in truncated.items():
                rec = self._agents.get(aid)
                if rec is not None:
                    rec.truncated = val

        return EpisodeRecord(
            episode_id=self._episode_id,
            scenario=self._scenario,
            total_steps=self._step_count,
            agents=dict(self._agents),
            conflicts=list(self._all_conflicts),
            reward_totals=list(self._reward_totals),
        )

    def to_json(self, record: EpisodeRecord, path: str | Path) -> None:
        """Serialize an EpisodeRecord to JSON.

        Args:
            record: The episode record to save.
            path: Output file path.
        """
        agents_data: dict[str, object] = {}
        data: dict[str, object] = {
            "episode_id": record.episode_id,
            "scenario": record.scenario,
            "total_steps": record.total_steps,
            "agents": agents_data,
            "conflicts": [
                {
                    "step": c.step,
                    "aircraft_ids": list(c.aircraft_ids),
                    "distance_nm": c.distance_nm,
                    "vertical_sep_ft": c.vertical_sep_ft,
                    "severity": c.severity,
                }
                for c in record.conflicts
            ],
            "reward_totals": record.reward_totals,
        }
        for aid, rec in record.agents.items():
            agents_data[aid] = {
                "trajectory": [
                    {
                        "step": p.step,
                        "lat": p.lat,
                        "lon": p.lon,
                        "alt": p.alt,
                        "hdg": p.hdg,
                        "tas": p.tas,
                        "vs": p.vs,
                    }
                    for p in rec.trajectory
                ],
                "rewards": rec.rewards,
                "reward_decompositions": [
                    [
                        {
                            "component_name": d.component_name,
                            "weight": d.weight,
                            "raw_value": d.raw_value,
                            "weighted_value": d.weighted_value,
                        }
                        for d in step_decomp
                    ]
                    for step_decomp in rec.reward_decompositions
                ],
                "conflicts": [
                    {
                        "step": c.step,
                        "aircraft_ids": list(c.aircraft_ids),
                        "distance_nm": c.distance_nm,
                        "vertical_sep_ft": c.vertical_sep_ft,
                        "severity": c.severity,
                    }
                    for c in rec.conflicts
                ],
                "terminated": rec.terminated,
                "truncated": rec.truncated,
            }

        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
