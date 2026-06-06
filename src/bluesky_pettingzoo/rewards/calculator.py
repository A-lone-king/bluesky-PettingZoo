"""Reward calculator: manages component registration and weighted sum."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np

from bluesky_pettingzoo.rewards.base import RewardComponent
from bluesky_pettingzoo.utils.types import AircraftState, DiscreteAction


class RewardCalculator:
    """Computes total reward as a weighted sum of registered components."""

    def __init__(self) -> None:
        self._components: list[tuple[RewardComponent, float]] = []

    @staticmethod
    def merge_reward_config(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
        """Merge scenario overrides into base reward config.

        Scenario overrides take precedence over base config values.
        Does not mutate the original base config.

        Args:
            base: Base reward configuration from rewards.yaml.
            overrides: Scenario-specific overrides from scenario YAML.

        Returns:
            Merged configuration dict.
        """
        merged = copy.deepcopy(base)
        for component_name, override_values in overrides.items():
            if component_name in merged.get("components", {}):
                merged["components"][component_name].update(override_values)
        return merged

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
        action: DiscreteAction | list[Any] | np.ndarray,
        curr_state: AircraftState,
        all_states: dict[str, AircraftState],
        step_count: int = 0,
    ) -> float:
        """Compute weighted sum of all component rewards."""
        total = 0.0
        for component, weight in self._components:
            total += weight * component.compute(
                agent_id,
                prev_state,
                action,
                curr_state,
                all_states,
                step_count=step_count,
            )
        return total

    def compute_detailed(
        self,
        agent_id: str,
        prev_state: AircraftState,
        action: DiscreteAction | list[Any] | np.ndarray,
        curr_state: AircraftState,
        all_states: dict[str, AircraftState],
        step_count: int = 0,
    ) -> tuple[float, list[tuple[str, float, float, float]]]:
        """Compute total reward and per-component breakdown.

        Returns:
            (total, breakdown) where breakdown is a list of
            (component_name, weight, raw_value, weighted_value) tuples.
        """
        total = 0.0
        breakdown: list[tuple[str, float, float, float]] = []
        for component, weight in self._components:
            raw = component.compute(
                agent_id,
                prev_state,
                action,
                curr_state,
                all_states,
                step_count=step_count,
            )
            weighted = weight * raw
            total += weighted
            breakdown.append((component.component_name, weight, raw, weighted))
        return total, breakdown

    def reset(self) -> None:
        """Reset all registered components."""
        for component, _ in self._components:
            component.reset()
