"""Reward calculator: manages component registration and weighted sum."""

from __future__ import annotations

from bluesky_pettingzoo.rewards.base import RewardComponent
from bluesky_pettingzoo.utils.types import AircraftState, DiscreteAction


class RewardCalculator:
    """Computes total reward as a weighted sum of registered components."""

    def __init__(self) -> None:
        self._components: list[tuple[RewardComponent, float]] = []

    @property
    def components(self) -> list[tuple[RewardComponent, float]]:
        """List of (component, weight) pairs."""
        return self._components

    def register(self, component: RewardComponent, weight: float) -> None:
        """Register a reward component with a weight."""
        self._components.append((component, weight))

    def compute(
        self,
        agent_id: str,
        prev_state: AircraftState,
        action: DiscreteAction,
        curr_state: AircraftState,
        all_states: dict[str, AircraftState],
        step_count: int = 0,
    ) -> float:
        """Compute weighted sum of all component rewards."""
        total = 0.0
        for component, weight in self._components:
            total += weight * component.compute(
                agent_id, prev_state, action, curr_state, all_states,
                step_count=step_count,
            )
        return total

    def reset(self) -> None:
        """Reset all registered components."""
        for component, _ in self._components:
            component.reset()
