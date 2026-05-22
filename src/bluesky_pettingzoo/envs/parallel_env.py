"""BlueSky MARL environment — PettingZoo ParallelEnv integration."""

from __future__ import annotations

from typing import Any

import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

from bluesky_pettingzoo.actions.translator import ActionTranslator
from bluesky_pettingzoo.bluesky.wrapper import BlueSkyWrapper
from bluesky_pettingzoo.observations.manager import ObservationManager
from bluesky_pettingzoo.rewards.calculator import RewardCalculator
from bluesky_pettingzoo.envs.scenarios.base import BaseScenario
from bluesky_pettingzoo.utils.geometry import haversine_distance
from bluesky_pettingzoo.utils.types import AircraftState, DiscreteAction

_AIRCRAFT_TYPE = "B737"


class BlueSkyMARLEnv(ParallelEnv):
    """PettingZoo-compatible multi-agent environment wrapping BlueSky.

    Combines BlueSkyWrapper, ObservationManager, ActionTranslator,
    and RewardCalculator into a single ParallelEnv interface.
    """

    metadata: dict[str, Any] = {"name": "bluesky_marl_v0"}

    def __init__(
        self,
        config: dict[str, Any],
        wrapper: BlueSkyWrapper,
        observation_manager: ObservationManager,
        action_translator: ActionTranslator,
        reward_calculator: RewardCalculator,
        rewards_config: dict[str, Any],
        scenario: BaseScenario | None = None,
    ) -> None:
        self.config = config
        self._wrapper = wrapper
        self._obs_manager = observation_manager
        self._action_translator = action_translator
        self._reward_calculator = reward_calculator
        self._rewards_config = rewards_config
        self._scenario = scenario

        self._dt: float = config["simulation"]["dt"]
        self._action_frequency: int = config["simulation"].get("action_frequency", 1)
        self._max_steps: int = config["simulation"]["max_episode_steps"]
        self._num_aircraft: int = config["aircraft"]["initial_count"]
        self._spawn_cfg = config["aircraft"]["spawn"]

        # Airspace bounds for goal placement and boundary checks
        sectors = config.get("airspace", {}).get("sectors", [])
        lats = [s["bounds"][0][0] for s in sectors] + [s["bounds"][1][0] for s in sectors]
        lons = [s["bounds"][0][1] for s in sectors] + [s["bounds"][1][1] for s in sectors]
        self._airspace = {
            "lat_min": min(lats), "lat_max": max(lats),
            "lon_min": min(lons), "lon_max": max(lons),
        }

        # Conflict thresholds for textual state
        comp = rewards_config.get("components", {})
        thr = comp.get("conflict", {}).get("thresholds", {})
        self._nmac_h = thr.get("nmac_horizontal_nm", 5)
        self._nmac_v = thr.get("nmac_vertical_ft", 1000)
        self._warn_h = thr.get("warning_horizontal_nm", 10)
        self._warn_v = thr.get("warning_vertical_ft", 2000)

        # Arrival threshold for termination
        eff_comp = comp.get("efficiency", {})
        self._arrival_threshold: float = eff_comp.get("arrival_threshold_nm", 2)

        # PettingZoo interface
        self.agents: list[str] = []
        self.possible_agents: list[str] = [f"AC{i:03d}" for i in range(self._num_aircraft)]

        self._step_count: int = 0
        self._prev_states: dict[str, AircraftState] = {}
        self._rng = np.random.RandomState()
        self._airspace_cfg = config.get("airspace", {})

        # Dynamic entry configuration
        de = config.get("dynamic_entry", {})
        self._dynamic_entry_enabled: bool = de.get("enabled", False)
        self._dynamic_entry_interval: int = de.get("interval", 10)
        self._dynamic_entry_max_total: int = de.get("max_total", self._num_aircraft + 5)
        self._entry_step_count: int = 0
        self._next_entry_id: int = self._num_aircraft

        # Cache space objects (PettingZoo requires identity stability)
        self._obs_space: spaces.Dict = self._obs_manager.observation_space()
        self._act_space: spaces.MultiDiscrete = spaces.MultiDiscrete([5, 5, 5])

    # ------------------------------------------------------------------
    # PettingZoo ParallelEnv interface
    # ------------------------------------------------------------------

    def observation_space(self, agent_id: str) -> spaces.Dict:
        return self._obs_space

    def action_space(self, agent_id: str) -> spaces.MultiDiscrete:
        return self._act_space

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if seed is not None:
            self._rng = np.random.RandomState(seed)

        self._step_count = 0
        self._entry_step_count = 0
        self._next_entry_id = self._num_aircraft
        self._reward_calculator.reset()
        self._wrapper.init_simulation()
        self._wrapper.reset()

        if self._scenario is not None:
            self._scenario.reset()
            self.agents = self._scenario.setup(self._rng, self._airspace)
            spawn = self._scenario.get_spawn_config()
            initial_positions = (
                self._scenario.get_initial_positions()
                if hasattr(self._scenario, "get_initial_positions")
                else None
            )
            for acid in self.agents:
                if initial_positions is not None and acid in initial_positions:
                    lat, lon = initial_positions[acid]
                else:
                    lat = self._rng.uniform(self._airspace["lat_min"] + 0.05, self._airspace["lat_max"] - 0.05)
                    lon = self._rng.uniform(self._airspace["lon_min"] + 0.05, self._airspace["lon_max"] - 0.05)
                alt = self._rng.uniform(spawn.altitude_range[0], spawn.altitude_range[1])
                hdg = self._rng.uniform(spawn.heading_range[0], spawn.heading_range[1])
                spd = self._rng.uniform(spawn.speed_range[0], spawn.speed_range[1])
                self._wrapper.create_aircraft(acid, _AIRCRAFT_TYPE, lat, lon, alt, hdg, spd)

            # Set goals from scenario waypoints
            eff = self._find_efficiency_component()
            if eff is not None:
                for acid in self.agents:
                    wp = self._scenario.get_waypoint(acid)
                    eff.set_goal(acid, wp["lat"], wp["lon"])
        else:
            # V1.0 default: spawn aircraft at deterministic positions
            self.agents = []
            spawn = self._spawn_cfg
            for i in range(self._num_aircraft):
                acid = f"AC{i:03d}"
                lat = self._rng.uniform(self._airspace["lat_min"] + 0.05, self._airspace["lat_max"] - 0.05)
                lon = self._rng.uniform(self._airspace["lon_min"] + 0.05, self._airspace["lon_max"] - 0.05)
                alt = self._rng.uniform(spawn["altitude_range"][0], spawn["altitude_range"][1])
                hdg = self._rng.uniform(spawn["heading_range"][0], spawn["heading_range"][1])
                spd = self._rng.uniform(spawn["speed_range"][0], spawn["speed_range"][1])
                self._wrapper.create_aircraft(acid, _AIRCRAFT_TYPE, lat, lon, alt, hdg, spd)
                self.agents.append(acid)

            # Set efficiency goals at opposite sector corners
            for acid in self.agents:
                st = self._get_aircraft_state(acid)
                self._set_default_goal(acid, st.lat, st.lon)

        # Generate initial observations
        all_states = self._get_all_aircraft_states()
        self._prev_states = dict(all_states)
        observations, infos = self._build_obs_and_infos(all_states)
        return observations, infos

    def step(
        self,
        actions: dict[str, Any],
    ) -> tuple[dict, dict, dict, dict, dict]:
        self._step_count += 1

        # Translate actions to BlueSky commands
        all_states = self._get_all_aircraft_states()
        commands: list[str] = []
        for agent_id in self.agents:
            raw_action = actions.get(agent_id, [2, 2, 2])
            action = DiscreteAction(
                heading_idx=int(raw_action[0]),
                altitude_idx=int(raw_action[1]),
                speed_idx=int(raw_action[2]),
            )
            state = all_states.get(agent_id)
            if state is not None:
                commands.extend(self._action_translator.translate(agent_id, state, action))

        # Execute commands and advance simulation
        self._wrapper.send_commands_batch(commands)
        self._wrapper.step_n(self._action_frequency)

        # Update states
        new_states = self._get_all_aircraft_states()

        # Scenario: update and add new agents
        if self._scenario is not None:
            new_agent_ids = self._scenario.update(self._step_count, new_states)
            for new_id in new_agent_ids:
                if new_id not in self.agents:
                    spawn = self._scenario.get_spawn_config()
                    lat = self._rng.uniform(self._airspace["lat_min"] + 0.05, self._airspace["lat_max"] - 0.05)
                    lon = self._rng.uniform(self._airspace["lon_min"] + 0.05, self._airspace["lon_max"] - 0.05)
                    alt = self._rng.uniform(spawn.altitude_range[0], spawn.altitude_range[1])
                    hdg = self._rng.uniform(spawn.heading_range[0], spawn.heading_range[1])
                    spd = self._rng.uniform(spawn.speed_range[0], spawn.speed_range[1])
                    self._wrapper.create_aircraft(new_id, _AIRCRAFT_TYPE, lat, lon, alt, hdg, spd)
                    self.agents.append(new_id)
                    eff = self._find_efficiency_component()
                    if eff is not None:
                        wp = self._scenario.get_waypoint(new_id)
                        eff.set_goal(new_id, wp["lat"], wp["lon"])
            if new_agent_ids:
                new_states = self._get_all_aircraft_states()

        # Dynamic entry: add new aircraft from boundary (V1.0 only)
        if self._dynamic_entry_enabled and self._scenario is None:
            self._entry_step_count += 1
            if (
                self._entry_step_count >= self._dynamic_entry_interval
                and len(self.agents) < self._dynamic_entry_max_total
            ):
                self._entry_step_count = 0
                new_id = f"AC{self._next_entry_id:03d}"
                self._next_entry_id += 1
                lat, lon, hdg = self._generate_boundary_entry()
                spawn = self._spawn_cfg
                alt = self._rng.uniform(spawn["altitude_range"][0], spawn["altitude_range"][1])
                spd = self._rng.uniform(spawn["speed_range"][0], spawn["speed_range"][1])
                self._wrapper.create_aircraft(new_id, _AIRCRAFT_TYPE, lat, lon, alt, hdg, spd)
                self.agents.append(new_id)
                self._set_default_goal(new_id, lat, lon)
                # Refresh new_states to include the new aircraft
                new_states = self._get_all_aircraft_states()

        # Compute rewards (before removing departed aircraft)
        rewards: dict[str, float] = {}
        for agent_id in list(self.agents):
            prev_st = self._prev_states.get(agent_id)
            curr_st = new_states.get(agent_id)
            if prev_st is None or curr_st is None:
                rewards[agent_id] = 0.0
                continue
            raw_action = actions.get(agent_id, [2, 2, 2])
            action = DiscreteAction(
                heading_idx=int(raw_action[0]),
                altitude_idx=int(raw_action[1]),
                speed_idx=int(raw_action[2]),
            )
            rewards[agent_id] = self._reward_calculator.compute(
                agent_id, prev_st, action, curr_st, new_states,
            )

        # Build observations for all current agents (before removal)
        agents_snapshot = list(self.agents)
        all_agent_states = {aid: new_states[aid] for aid in agents_snapshot if aid in new_states}
        observations, infos = self._build_obs_and_infos(all_agent_states)

        # Remove NMAC, arrived, and departed aircraft
        eff = self._find_efficiency_component()
        scenario_truncated: set[str] = set()
        for agent_id in agents_snapshot:
            if agent_id not in self.agents:
                continue
            own = new_states.get(agent_id)
            if own is not None:
                others = [s for aid, s in new_states.items() if aid != agent_id]
                conflict = self._compute_conflict_status(own, others)
                if conflict == "nmac":
                    self.agents.remove(agent_id)
                    self._wrapper.remove_aircraft(agent_id)
                    continue
                # Arrival termination
                if eff is not None and hasattr(eff, "_goals"):
                    goal = eff._goals.get(agent_id)
                    if goal is not None:
                        dist = haversine_distance(own.lat, own.lon, goal[0], goal[1])
                        if dist < self._arrival_threshold:
                            self.agents.remove(agent_id)
                            self._wrapper.remove_aircraft(agent_id)
                            continue
            if self._scenario is not None and own is not None:
                if self._scenario.should_truncate(agent_id, own, self._airspace):
                    scenario_truncated.add(agent_id)
                    self.agents.remove(agent_id)
                    self._wrapper.remove_aircraft(agent_id)
            elif not self._wrapper.is_aircraft_in_airspace(agent_id):
                self.agents.remove(agent_id)
                self._wrapper.remove_aircraft(agent_id)

        # Terminations — NMAC, arrived, or left airspace (but not scenario-truncated)
        terminations: dict[str, bool] = {}
        for agent_id in agents_snapshot:
            terminations[agent_id] = (
                agent_id not in self.agents and agent_id not in scenario_truncated
            )

        # Truncations — max steps reached or scenario truncated
        truncated = self._step_count >= self._max_steps
        truncations: dict[str, bool] = {
            aid: truncated or aid in scenario_truncated for aid in agents_snapshot
        }

        # Fill in rewards/terms/truncs for removed agents that have no observation
        for agent_id in agents_snapshot:
            if agent_id not in observations:
                observations[agent_id] = self._default_observation()
            if agent_id not in infos:
                infos[agent_id] = {}

        self._prev_states = new_states

        return observations, rewards, terminations, truncations, infos

    def close(self) -> None:
        self._wrapper.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_aircraft_state(self, acid: str) -> AircraftState:
        raw = self._wrapper.get_aircraft_state(acid)
        return AircraftState(
            id=raw["id"], lat=raw["lat"], lon=raw["lon"],
            alt=raw["alt"], hdg=raw["hdg"], tas=raw["tas"], vs=raw["vs"],
        )

    def _get_all_aircraft_states(self) -> dict[str, AircraftState]:
        raw = self._wrapper.get_all_aircraft_states()
        return {
            acid: AircraftState(
                id=v["id"], lat=v["lat"], lon=v["lon"],
                alt=v["alt"], hdg=v["hdg"], tas=v["tas"], vs=v["vs"],
            )
            for acid, v in raw.items()
        }

    def _build_obs_and_infos(
        self,
        all_states: dict[str, AircraftState],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        observations: dict[str, Any] = {}
        infos: dict[str, Any] = {}

        for agent_id in self.agents:
            own = all_states.get(agent_id)
            if own is None:
                continue

            others = [s for aid, s in all_states.items() if aid != agent_id]
            conflict_status = self._compute_conflict_status(own, others)

            goal = self._make_goal(agent_id, own)
            result = self._obs_manager.generate(
                own_state=own,
                other_states=others,
                goal=goal,
                conflict_status=conflict_status,
                airspace=self._airspace_cfg,
            )
            observations[agent_id] = result["observation"]
            infos[agent_id] = {
                "textual_state": result["textual_state"],
                "airspace_snapshot": result["airspace_snapshot"],
            }

        return observations, infos

    def _make_goal(self, agent_id: str, own: AircraftState) -> dict[str, float]:
        eff = self._find_efficiency_component()
        if eff is not None and hasattr(eff, "_goals"):
            goal_tuple = eff._goals.get(agent_id)
            if goal_tuple is not None:
                return {"lat": goal_tuple[0], "lon": goal_tuple[1], "alt": own.alt, "hdg": own.hdg}
        mid_lat = (self._airspace["lat_min"] + self._airspace["lat_max"]) / 2
        mid_lon = (self._airspace["lon_min"] + self._airspace["lon_max"]) / 2
        return {
            "lat": self._airspace["lat_max"] if own.lat < mid_lat else self._airspace["lat_min"],
            "lon": self._airspace["lon_max"] if own.lon < mid_lon else self._airspace["lon_min"],
            "alt": own.alt,
            "hdg": own.hdg,
        }

    def _find_efficiency_component(self) -> Any:
        for comp, _ in self._reward_calculator.components:
            if hasattr(comp, "set_goal"):
                return comp
        return None

    def _compute_conflict_status(
        self,
        own: AircraftState,
        others: list[AircraftState],
    ) -> str:
        for other in others:
            h_dist = haversine_distance(own.lat, own.lon, other.lat, other.lon)
            v_dist = abs(own.alt - other.alt)
            if h_dist < self._nmac_h and v_dist < self._nmac_v:
                return "nmac"
            if h_dist < self._warn_h and v_dist < self._warn_v:
                return "warning"
        return "safe"

    def _set_default_goal(self, agent_id: str, lat: float, lon: float) -> None:
        """Set efficiency goal to the opposite corner from the given position."""
        eff = self._find_efficiency_component()
        if eff is None:
            return
        mid_lat = (self._airspace["lat_min"] + self._airspace["lat_max"]) / 2
        mid_lon = (self._airspace["lon_min"] + self._airspace["lon_max"]) / 2
        goal_lat = self._airspace["lat_max"] if lat < mid_lat else self._airspace["lat_min"]
        goal_lon = self._airspace["lon_max"] if lon < mid_lon else self._airspace["lon_min"]
        eff.set_goal(agent_id, goal_lat, goal_lon)

    def _generate_boundary_entry(self) -> tuple[float, float, float]:
        """Generate a random position near the airspace boundary with inward heading.

        Returns:
            (lat, lon, hdg) — position just inside boundary, heading toward center.
        """
        bounds = self._airspace
        margin = 0.02  # degrees inside boundary to avoid immediate exit
        side = self._rng.randint(0, 4)

        if side == 0:  # south edge
            lat = bounds["lat_min"] + margin
            lon = self._rng.uniform(bounds["lon_min"] + margin, bounds["lon_max"] - margin)
            hdg = self._rng.uniform(20, 160)  # northward
        elif side == 1:  # north edge
            lat = bounds["lat_max"] - margin
            lon = self._rng.uniform(bounds["lon_min"] + margin, bounds["lon_max"] - margin)
            hdg = self._rng.uniform(200, 340)  # southward
        elif side == 2:  # west edge
            lat = self._rng.uniform(bounds["lat_min"] + margin, bounds["lat_max"] - margin)
            lon = bounds["lon_min"] + margin
            hdg = self._rng.uniform(20, 160)  # eastward
        else:  # east edge
            lat = self._rng.uniform(bounds["lat_min"] + margin, bounds["lat_max"] - margin)
            lon = bounds["lon_max"] - margin
            hdg = self._rng.uniform(200, 340)  # westward

        return lat, lon, hdg

    def _default_observation(self) -> dict[str, Any]:
        obs_space = self._obs_manager.observation_space()
        return obs_space.sample()
