"""Factory functions for creating test environments and configs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from bluesky_pettingzoo.actions.translator import ActionTranslator
from bluesky_pettingzoo.envs.parallel_env import BlueSkyMARLEnv
from bluesky_pettingzoo.envs.scenarios.base import BaseScenario
from bluesky_pettingzoo.observations.manager import ObservationManager
from bluesky_pettingzoo.rewards.calculator import RewardCalculator
from bluesky_pettingzoo.rewards.components.conflict import ConflictPenalty
from bluesky_pettingzoo.rewards.components.efficiency import EfficiencyReward
from bluesky_pettingzoo.rewards.components.smoothness import SmoothnessPenalty

from .fake_wrapper import FakeBlueSkyWrapper

# Default test config template
_DEFAULT_CONFIG: dict[str, Any] = {
    "simulation": {"dt": 5.0, "max_episode_steps": 50, "headless": True},
    "airspace": {
        "name": "test",
        "sectors": [
            {"id": "s1", "bounds": [[39.0, 116.0], [41.0, 118.0]]},
        ],
    },
    "aircraft": {
        "initial_count": 5,
        "spawn": {
            "altitude_range": [29000, 37000],
            "speed_range": [400, 500],
            "heading_range": [0, 360],
        },
    },
    "observation": {
        "perception_radius_nm": 40,
        "perception_alt_diff_ft": 5000,
        "max_observable_aircraft": 5,
    },
    "action": {
        "heading_adjustments": [-20, -10, 0, 10, 20],
        "altitude_adjustments": [-2000, -1000, 0, 1000, 2000],
        "speed_adjustments": [-20, -10, 0, 10, 20],
    },
    "normalization": {
        "heading": {"mid": 180, "range": 180},
        "altitude": {"mid": 33000, "range": 10000},
        "speed": {"mid": 450, "range": 100},
        "distance": {"max": 40},
    },
}

_DEFAULT_REWARDS: dict[str, Any] = {
    "components": {
        "conflict": {
            "enabled": True,
            "weight": 1.0,
            "nmac_penalty": -100,
            "warning_penalty": -10,
            "separation_penalty": -5,
            "thresholds": {
                "nmac_horizontal_nm": 5,
                "nmac_vertical_ft": 1000,
                "warning_horizontal_nm": 10,
                "warning_vertical_ft": 2000,
            },
        },
        "smoothness": {"enabled": True, "weight": 0.5, "action_penalty": -0.1},
        "efficiency": {
            "enabled": True,
            "weight": 0.3,
            "max_deviation_nm": 200,
            "deviation_penalty_scale": 5,
            "arrival_reward": 10,
            "step_penalty": -0.01,
            "arrival_threshold_nm": 2,
        },
    }
}


def make_config(**overrides: Any) -> dict[str, Any]:
    """Build a test config dict with optional overrides.

    Supports flat overrides like ``initial_count=3`` or nested overrides
    like ``simulation={"dt": 10}``.
    """
    import copy
    config = copy.deepcopy(_DEFAULT_CONFIG)

    # Apply flat overrides
    if "initial_count" in overrides:
        config["aircraft"]["initial_count"] = overrides["initial_count"]
    if "max_steps" in overrides:
        config["simulation"]["max_episode_steps"] = overrides["max_steps"]
    if "arrival_threshold_nm" in overrides:
        config.setdefault("components", {}).setdefault("efficiency", {})["arrival_threshold_nm"] = overrides["arrival_threshold_nm"]

    # Apply nested dict overrides (known keys merge into existing dicts,
    # unknown keys are added directly — e.g. dynamic_entry, components)
    _known_nested = {"simulation", "airspace", "aircraft", "observation", "action", "normalization"}
    for key, val in overrides.items():
        if key in _known_nested and isinstance(val, dict):
            config[key].update(val)
        elif key not in _known_nested and key not in {"initial_count", "max_steps", "arrival_threshold_nm"}:
            config[key] = val

    return config


def write_rewards_yaml(tmp_path: Path, rewards: dict[str, Any] | None = None) -> Path:
    """Write rewards YAML to tmp_path and return the path."""
    p = tmp_path / "rewards.yaml"
    p.write_text(yaml.dump(rewards or _DEFAULT_REWARDS), encoding="utf-8")
    return p


def make_env(
    tmp_path: Path | None = None,
    config: dict[str, Any] | None = None,
    scenario: BaseScenario | None = None,
    rewards: dict[str, Any] | None = None,
    **config_overrides: Any,
) -> BlueSkyMARLEnv:
    """Create a fully wired BlueSkyMARLEnv with FakeBlueSkyWrapper.

    Args:
        tmp_path: pytest tmp_path fixture for writing rewards YAML.
            If None, uses a temporary directory.
        config: Override config (uses defaults if None).
        scenario: Optional scenario instance.
        rewards: Override rewards config (uses defaults if None).
        **config_overrides: Passed to make_config() when config is None.
    """
    import tempfile
    # Handle swapped args: _make_env(config, tmp_path) pattern
    if isinstance(tmp_path, dict) and isinstance(config, Path):
        tmp_path, config = config, tmp_path  # type: ignore[assignment]
    if isinstance(tmp_path, dict) and config is None:
        config, tmp_path = tmp_path, None  # type: ignore[assignment]
    if tmp_path is None:
        tmp_path = Path(tempfile.mkdtemp())
    if config is None:
        config = make_config(**config_overrides)

    rewards_yaml = rewards or _DEFAULT_REWARDS
    rewards_path = write_rewards_yaml(tmp_path, rewards_yaml)
    config["_rewards_yaml"] = str(rewards_path)

    wrapper = FakeBlueSkyWrapper(config)
    obs_manager = ObservationManager(config)
    action_translator = ActionTranslator(config)

    with open(rewards_path, encoding="utf-8") as f:
        rewards_cfg = yaml.safe_load(f)
    # Propagate config component overrides into rewards before merge
    # so that overrides like arrival_threshold_nm survive the shallow merge.
    cfg_components = config.get("components", {})
    for comp_name, comp_vals in cfg_components.items():
        if isinstance(comp_vals, dict) and comp_name in rewards_cfg.get("components", {}):
            rewards_cfg["components"][comp_name].update(comp_vals)
    merged = {**config, **rewards_cfg}

    calc = RewardCalculator()
    calc.register(ConflictPenalty(merged), weight=1.0)
    calc.register(SmoothnessPenalty(merged), weight=0.5)
    eff = EfficiencyReward(merged)
    calc.register(eff, weight=0.3)

    return BlueSkyMARLEnv(
        config=config,
        wrapper=wrapper,
        observation_manager=obs_manager,
        action_translator=action_translator,
        reward_calculator=calc,
        rewards_config=rewards_cfg,
        scenario=scenario,
    )
