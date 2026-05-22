"""Tests for scenario configuration data models (T-V06)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import yaml

from bluesky_pettingzoo.utils.types import (
    AircraftConfig,
    AirspaceConfig,
    ConflictConfig,
    DynamicEntryConfig,
    ScenarioConfig,
    SectorConfig,
    SimulationConfig,
    SpawnConfig,
    WaypointConfig,
)


# ===========================================================================
# T-V06: ScenarioConfig
# ===========================================================================


class TestScenarioConfig:
    """ScenarioConfig should hold all sub-configs."""

    def test_scenario_config_creation(self) -> None:
        """Create ScenarioConfig with all required fields."""
        cfg = ScenarioConfig(
            name="test_scenario",
            simulation=SimulationConfig(dt=5.0, max_episode_steps=360, headless=True),
            airspace=AirspaceConfig(
                name="test_airspace",
                sectors=[SectorConfig(id="s1", bounds=[[39.0, 116.0], [39.5, 116.5]])],
            ),
            aircraft=AircraftConfig(
                initial_count=5,
                spawn=SpawnConfig(
                    altitude_range=(29000, 37000),
                    speed_range=(400, 500),
                    heading_range=(0, 360),
                ),
            ),
        )
        assert cfg.name == "test_scenario"
        assert cfg.simulation.dt == 5.0
        assert cfg.aircraft.initial_count == 5

    def test_scenario_config_dict_access(self) -> None:
        """ScenarioConfig supports dict-style access."""
        cfg = ScenarioConfig(
            name="test",
            simulation=SimulationConfig(dt=5.0, max_episode_steps=360, headless=True),
            airspace=AirspaceConfig(
                name="a",
                sectors=[SectorConfig(id="s1", bounds=[[0, 0], [1, 1]])],
            ),
            aircraft=AircraftConfig(
                initial_count=3,
                spawn=SpawnConfig(
                    altitude_range=(29000, 37000),
                    speed_range=(400, 500),
                    heading_range=(0, 360),
                ),
            ),
        )
        assert cfg["name"] == "test"
        assert cfg["simulation"]["dt"] == 5.0
        assert "name" in cfg


# ===========================================================================
# T-V06: AirspaceConfig
# ===========================================================================


class TestAirspaceConfig:
    """AirspaceConfig should hold sectors and optional waypoints."""

    def test_airspace_config_rectangular(self) -> None:
        """Create AirspaceConfig with rectangular sectors."""
        cfg = AirspaceConfig(
            name="test",
            sectors=[SectorConfig(id="s1", bounds=[[39.0, 116.0], [39.5, 116.5]])],
        )
        assert len(cfg.sectors) == 1
        assert cfg.sectors[0].id == "s1"

    def test_airspace_config_with_waypoints(self) -> None:
        """AirspaceConfig can hold waypoints."""
        cfg = AirspaceConfig(
            name="test",
            sectors=[SectorConfig(id="s1", bounds=[[0, 0], [1, 1]])],
            waypoints=[WaypointConfig(id="WP1", lat=39.5, lon=116.5, alt=35000)],
        )
        assert len(cfg.waypoints) == 1
        assert cfg.waypoints[0].id == "WP1"

    def test_airspace_config_dict_access(self) -> None:
        """AirspaceConfig supports dict-style access."""
        cfg = AirspaceConfig(
            name="test",
            sectors=[SectorConfig(id="s1", bounds=[[0, 0], [1, 1]])],
        )
        assert cfg["name"] == "test"
        assert "sectors" in cfg


# ===========================================================================
# T-V06: SectorConfig
# ===========================================================================


class TestSectorConfig:
    """SectorConfig should hold sector definition."""

    def test_sector_config_creation(self) -> None:
        """Create SectorConfig with id and bounds."""
        cfg = SectorConfig(id="s1", bounds=[[39.0, 116.0], [39.5, 116.5]])
        assert cfg.id == "s1"
        assert cfg.bounds == [[39.0, 116.0], [39.5, 116.5]]

    def test_sector_config_dict_access(self) -> None:
        """SectorConfig supports dict-style access."""
        cfg = SectorConfig(id="s1", bounds=[[0, 0], [1, 1]])
        assert cfg["id"] == "s1"
        assert "bounds" in cfg


# ===========================================================================
# T-V06: WaypointConfig
# ===========================================================================


class TestWaypointConfig:
    """WaypointConfig should hold waypoint definition."""

    def test_waypoint_config_creation(self) -> None:
        """Create WaypointConfig with coordinates."""
        cfg = WaypointConfig(id="WP1", lat=39.5, lon=116.5, alt=35000)
        assert cfg.id == "WP1"
        assert cfg.lat == 39.5
        assert cfg.lon == 116.5
        assert cfg.alt == 35000

    def test_waypoint_config_dict_access(self) -> None:
        """WaypointConfig supports dict-style access."""
        cfg = WaypointConfig(id="WP1", lat=39.5, lon=116.5, alt=35000)
        assert cfg["lat"] == 39.5
        assert "id" in cfg


# ===========================================================================
# T-V06: AircraftConfig
# ===========================================================================


class TestAircraftConfig:
    """AircraftConfig should hold aircraft count and spawn config."""

    def test_aircraft_config_creation(self) -> None:
        """Create AircraftConfig with initial_count and spawn."""
        cfg = AircraftConfig(
            initial_count=5,
            spawn=SpawnConfig(
                altitude_range=(29000, 37000),
                speed_range=(400, 500),
                heading_range=(0, 360),
            ),
        )
        assert cfg.initial_count == 5
        assert cfg.spawn.altitude_range == (29000, 37000)

    def test_aircraft_config_dict_access(self) -> None:
        """AircraftConfig supports dict-style access."""
        cfg = AircraftConfig(
            initial_count=3,
            spawn=SpawnConfig(
                altitude_range=(29000, 37000),
                speed_range=(400, 500),
                heading_range=(0, 360),
            ),
        )
        assert cfg["initial_count"] == 3
        assert "spawn" in cfg


# ===========================================================================
# T-V06: SpawnConfig
# ===========================================================================


class TestSpawnConfig:
    """SpawnConfig should hold spawn range parameters."""

    def test_spawn_config_creation(self) -> None:
        """Create SpawnConfig with ranges."""
        cfg = SpawnConfig(
            altitude_range=(29000, 37000),
            speed_range=(400, 500),
            heading_range=(0, 360),
        )
        assert cfg.altitude_range == (29000, 37000)
        assert cfg.speed_range == (400, 500)
        assert cfg.heading_range == (0, 360)

    def test_spawn_config_dict_access(self) -> None:
        """SpawnConfig supports dict-style access."""
        cfg = SpawnConfig(
            altitude_range=(29000, 37000),
            speed_range=(400, 500),
            heading_range=(0, 360),
        )
        assert cfg["speed_range"] == (400, 500)


# ===========================================================================
# T-V06: DynamicEntryConfig
# ===========================================================================


class TestDynamicEntryConfig:
    """DynamicEntryConfig should hold dynamic entry parameters."""

    def test_dynamic_entry_config_creation(self) -> None:
        """Create DynamicEntryConfig with all fields."""
        cfg = DynamicEntryConfig(enabled=True, interval=10, max_total=20)
        assert cfg.enabled is True
        assert cfg.interval == 10
        assert cfg.max_total == 20

    def test_dynamic_entry_config_disabled(self) -> None:
        """DynamicEntryConfig can be disabled."""
        cfg = DynamicEntryConfig(enabled=False, interval=10, max_total=20)
        assert cfg.enabled is False

    def test_dynamic_entry_config_dict_access(self) -> None:
        """DynamicEntryConfig supports dict-style access."""
        cfg = DynamicEntryConfig(enabled=True, interval=5, max_total=10)
        assert cfg["interval"] == 5
        assert "enabled" in cfg


# ===========================================================================
# T-V06: ConflictConfig
# ===========================================================================


class TestConflictConfig:
    """ConflictConfig should hold conflict thresholds."""

    def test_conflict_config_creation(self) -> None:
        """Create ConflictConfig with threshold values."""
        cfg = ConflictConfig(
            nmac_horizontal_nm=5.0,
            nmac_vertical_ft=1000.0,
            warning_horizontal_nm=10.0,
            warning_vertical_ft=2000.0,
        )
        assert cfg.nmac_horizontal_nm == 5.0
        assert cfg.nmac_vertical_ft == 1000.0

    def test_conflict_config_dict_access(self) -> None:
        """ConflictConfig supports dict-style access."""
        cfg = ConflictConfig(
            nmac_horizontal_nm=5.0,
            nmac_vertical_ft=1000.0,
            warning_horizontal_nm=10.0,
            warning_vertical_ft=2000.0,
        )
        assert cfg["warning_horizontal_nm"] == 10.0


# ===========================================================================
# T-V06: SimulationConfig
# ===========================================================================


class TestSimulationConfig:
    """SimulationConfig should hold simulation parameters."""

    def test_simulation_config_creation(self) -> None:
        """Create SimulationConfig with required fields."""
        cfg = SimulationConfig(dt=5.0, max_episode_steps=360, headless=True)
        assert cfg.dt == 5.0
        assert cfg.max_episode_steps == 360
        assert cfg.headless is True

    def test_simulation_config_with_frequency(self) -> None:
        """SimulationConfig supports optional action_frequency."""
        cfg = SimulationConfig(dt=5.0, max_episode_steps=360, headless=True, action_frequency=3)
        assert cfg.action_frequency == 3

    def test_simulation_config_dict_access(self) -> None:
        """SimulationConfig supports dict-style access."""
        cfg = SimulationConfig(dt=5.0, max_episode_steps=360, headless=True)
        assert cfg["dt"] == 5.0
        assert "headless" in cfg


# ===========================================================================
# T-V06: YAML loading
# ===========================================================================


class TestConfigFromYaml:
    """ScenarioConfig should be constructible from a YAML file."""

    def test_config_from_yaml(self) -> None:
        """Load ScenarioConfig from a YAML file."""
        yaml_content = {
            "scenario": {
                "name": "from_yaml",
                "simulation": {"dt": 10.0, "max_episode_steps": 100, "headless": True},
                "airspace": {
                    "name": "yaml_airspace",
                    "sectors": [{"id": "s1", "bounds": [[39.0, 116.0], [39.5, 116.5]]}],
                },
                "aircraft": {
                    "initial_count": 3,
                    "spawn": {
                        "altitude_range": [29000, 37000],
                        "speed_range": [400, 500],
                        "heading_range": [0, 360],
                    },
                },
            }
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(yaml_content, f)
            tmp_path = Path(f.name)

        try:
            with open(tmp_path, encoding="utf-8") as f:
                data = yaml.safe_load(f)

            cfg = ScenarioConfig.from_dict(data["scenario"])
            assert cfg.name == "from_yaml"
            assert cfg.simulation.dt == 10.0
            assert cfg.aircraft.initial_count == 3
            assert cfg.airspace.sectors[0].id == "s1"
        finally:
            tmp_path.unlink()
