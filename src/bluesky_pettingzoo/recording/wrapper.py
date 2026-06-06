"""DataRecordingWrapper — transparent wrapper that records episode data."""

from __future__ import annotations

from typing import Any

from bluesky_pettingzoo.recording.recorder import DataRecorder
from bluesky_pettingzoo.recording.types import ConflictRecord, EpisodeRecord
from bluesky_pettingzoo.wrappers.base import EnvWrapperMixin


class DataRecordingWrapper(EnvWrapperMixin):
    """Wraps a ParallelEnv and transparently records episode data.

    Every ``reset()`` starts a new episode; every ``step()`` records
    trajectory, rewards, and (optionally) reward decompositions and
    conflicts.  Call ``get_records()`` or ``get_last_episode()`` after
    an episode to retrieve the :class:`EpisodeRecord`.

    Args:
        env: The PettingZoo ParallelEnv to wrap.
        episode_id: Starting episode counter.
        scenario: Scenario name (stored in the record).
        record_decompositions: If *True*, attempt to read
            ``infos[agent]["reward_decomposition"]`` and store it.
        record_conflicts: If *True*, attempt to read
            ``infos[agent]["conflicts"]`` and store them.
    """

    def __init__(
        self,
        env: Any,
        episode_id: int = 0,
        scenario: str = "",
        record_decompositions: bool = True,
        record_conflicts: bool = True,
    ) -> None:
        self._episode_id = episode_id
        self._scenario = scenario
        self._record_decompositions = record_decompositions
        self._record_conflicts = record_conflicts

        super().__init__(env)

        self._recorder: DataRecorder | None = None
        self._step_count: int = 0
        self._finished: list[EpisodeRecord] = []

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def reset(self, **kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        """Reset the wrapped env and start a new recording episode."""
        # Finalize previous episode if one was active
        if self._recorder is not None:
            self._finish_episode()

        # Start fresh recorder
        self._recorder = DataRecorder(
            episode_id=self._episode_id,
            scenario=self._scenario,
        )
        self._step_count = 0

        observations, infos = self.env.reset(**kwargs)

        # Register agents
        self._recorder.set_agents(list(observations.keys()))

        return observations, infos

    def step(
        self,
        actions: dict[str, Any],
    ) -> tuple[
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
    ]:
        """Step the wrapped env and record data."""
        observations, rewards, terminations, truncations, infos = self.env.step(actions)

        if self._recorder is not None:
            self._step_count += 1

            # Gather aircraft states from the unwrapped env
            states = self._extract_states(observations)

            # Optionally gather decomposition / conflicts from infos
            decompositions = (
                self._extract_decompositions(infos) if self._record_decompositions else None
            )
            conflicts = self._extract_conflicts(infos) if self._record_conflicts else None

            self._recorder.record_step(
                step=self._step_count,
                states=states,
                rewards=rewards,
                decompositions=decompositions,
                conflicts=conflicts,
            )

            # If every agent is terminal, finalize and store
            if all(terminations.values()) or all(truncations.values()):
                record = self._recorder.finalize(
                    terminated=terminations,
                    truncated=truncations,
                )
                self._finished.append(record)
                self._recorder = None
                self._episode_id += 1

        return observations, rewards, terminations, truncations, infos

    # ------------------------------------------------------------------
    # Query API
    # ------------------------------------------------------------------

    def get_records(self) -> list[EpisodeRecord]:
        """Return all finalized episode records."""
        return list(self._finished)

    def get_last_episode(self) -> EpisodeRecord | None:
        """Return the most recent finalized episode, or *None*."""
        return self._finished[-1] if self._finished else None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _finish_episode(self) -> None:
        """Finalize the current recorder and store the record."""
        if self._recorder is None:
            return
        record = self._recorder.finalize()
        self._finished.append(record)
        self._recorder = None

    def _extract_states(self, observations: dict[str, Any]) -> dict[str, Any]:
        """Extract aircraft states from the unwrapped environment."""
        raw_states = self.env.unwrapped.aircraft_states
        return {aid: raw_states[aid] for aid in observations if aid in raw_states}

    def _extract_decompositions(
        self, infos: dict[str, Any]
    ) -> dict[str, list[tuple[str, float, float, float]]] | None:
        """Try to read reward decomposition from infos."""
        decomps: dict[str, list[tuple[str, float, float, float]]] = {}
        for agent_id, info in infos.items():
            raw = info.get("reward_decomposition")
            if raw is not None:
                decomps[agent_id] = raw
        return decomps if decomps else None

    def _extract_conflicts(self, infos: dict[str, Any]) -> list[ConflictRecord] | None:
        """Try to read conflict events from infos."""
        for info in infos.values():
            raw = info.get("conflicts")
            if raw is not None and len(raw) > 0:
                return list(raw)
        return None
