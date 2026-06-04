"""Tests for scenario YAML control_mode and conflict_generation fields."""

from __future__ import annotations

from pathlib import Path

import yaml

SCENARIOS_DIR = Path(__file__).parent.parent / "config" / "scenarios"


class TestScenarioConfigMode:
    """Verify scenario YAML files contain control_mode and conflict_generation."""

    def _load_yaml(self, name: str) -> dict:
        path = SCENARIOS_DIR / f"{name}.yaml"
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f)

    def test_horizontal_cr_has_control_mode(self):
        data = self._load_yaml("horizontal_cr")
        assert "control_mode" in data

    def test_horizontal_cr_control_mode_value(self):
        data = self._load_yaml("horizontal_cr")
        assert data["control_mode"] in ("MULTI_RL", "SINGLE_RL")

    def test_horizontal_cr_has_conflict_generation(self):
        data = self._load_yaml("horizontal_cr")
        assert "conflict_generation" in data

    def test_vertical_cr_has_control_mode(self):
        data = self._load_yaml("vertical_cr")
        assert "control_mode" in data

    def test_sector_cr_has_control_mode(self):
        data = self._load_yaml("sector_cr")
        assert "control_mode" in data

    def test_conflict_generation_values(self):
        data = self._load_yaml("horizontal_cr")
        cg = data.get("conflict_generation", {})
        assert isinstance(cg, str) or isinstance(cg, dict)
