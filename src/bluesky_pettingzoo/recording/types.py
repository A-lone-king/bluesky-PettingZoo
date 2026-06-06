"""Data types for episode recording."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class TrajectoryPoint:
    """Single trajectory snapshot for one aircraft at one timestep."""

    step: int
    lat: float
    lon: float
    alt: float
    hdg: float
    tas: float
    vs: float


@dataclass(frozen=True)
class ConflictRecord:
    """A detected conflict event between two aircraft."""

    step: int
    aircraft_ids: tuple[str, str]
    distance_nm: float
    vertical_sep_ft: float
    severity: str  # "nmac" | "warning" | "separation"


@dataclass(frozen=True)
class RewardDecomposition:
    """Per-component reward breakdown for one agent at one timestep."""

    component_name: str
    weight: float
    raw_value: float
    weighted_value: float


@dataclass
class AgentRecord:
    """Complete recording data for a single agent within an episode."""

    agent_id: str
    trajectory: list[TrajectoryPoint] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    reward_decompositions: list[list[RewardDecomposition]] = field(default_factory=list)
    conflicts: list[ConflictRecord] = field(default_factory=list)
    terminated: bool = False
    truncated: bool = False


@dataclass
class EpisodeRecord:
    """Complete recording data for one episode."""

    episode_id: int
    scenario: str
    total_steps: int
    agents: dict[str, AgentRecord] = field(default_factory=dict)
    conflicts: list[ConflictRecord] = field(default_factory=list)
    reward_totals: list[dict[str, float]] = field(default_factory=list)

    @property
    def duration_steps(self) -> int:
        """Number of steps recorded."""
        return self.total_steps

    def get_trajectory(self, agent_id: str) -> list[TrajectoryPoint]:
        """Get trajectory for a specific agent."""
        rec = self.agents.get(agent_id)
        if rec is None:
            return []
        return rec.trajectory

    def get_total_rewards(self, agent_id: str) -> list[float]:
        """Get total reward per step for a specific agent."""
        rec = self.agents.get(agent_id)
        if rec is None:
            return []
        return rec.rewards

    def summary(self) -> dict[str, object]:
        """Generate a summary statistics dict."""
        agent_summaries: dict[str, dict[str, object]] = {}
        for aid, rec in self.agents.items():
            total_reward = sum(rec.rewards) if rec.rewards else 0.0
            n_points = len(rec.trajectory)
            agent_summaries[aid] = {
                "total_reward": round(total_reward, 4),
                "trajectory_points": n_points,
                "conflicts": len(rec.conflicts),
                "terminated": rec.terminated,
                "truncated": rec.truncated,
            }
        return {
            "episode_id": self.episode_id,
            "scenario": self.scenario,
            "total_steps": self.total_steps,
            "num_agents": len(self.agents),
            "total_conflicts": len(self.conflicts),
            "agents": agent_summaries,
        }
