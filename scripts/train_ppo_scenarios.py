"""PPO multi-scenario training and baseline comparison.

Train PPO on all scenarios and compare against Random and RuleBased baselines.

Usage:
    python scripts/train_ppo_scenarios.py --scenario HorizontalCR --timesteps 50000
    python scripts/train_ppo_scenarios.py --scenario HorizontalCR --resume models/HorizontalCR/checkpoint_20000.zip
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from bluesky_pettingzoo.actions.translator import ActionTranslator
from bluesky_pettingzoo.envs.parallel_env import BlueSkyMARLEnv
from bluesky_pettingzoo.envs.scenarios.base import BaseScenario
from bluesky_pettingzoo.observations.manager import ObservationManager
from bluesky_pettingzoo.rewards.calculator import RewardCalculator
from bluesky_pettingzoo.rewards.components.capacity import CapacityPenalty
from bluesky_pettingzoo.rewards.components.conflict import ConflictPenalty
from bluesky_pettingzoo.rewards.components.efficiency import EfficiencyReward
from bluesky_pettingzoo.rewards.components.fairness import FairnessReward
from bluesky_pettingzoo.rewards.components.flow_efficiency import FlowEfficiencyReward
from bluesky_pettingzoo.rewards.components.obstacle_intrusion import ObstacleIntrusion
from bluesky_pettingzoo.rewards.components.smoothness import SmoothnessPenalty
from bluesky_pettingzoo.training.checkpoint import CheckpointManager
from bluesky_pettingzoo.training.logger import CSVLoggerCallback
from bluesky_pettingzoo.training.progress import ProgressCallback
from bluesky_pettingzoo.wrappers.single_agent import SingleAgentGymWrapper
from stable_baselines3.common.callbacks import BaseCallback

from bluesky_pettingzoo.bluesky.wrapper import BlueSkyWrapper
from tests.helpers.env_factory import make_config, write_rewards_yaml


SCENARIO_MAP: dict[str, str] = {
    "HorizontalCR": "HorizontalCRScenario",
    "VerticalCR": "VerticalCRScenario",
    "SectorCR": "SectorCRScenario",
    "WaypointNav": "WaypointNavScenario",
    "Merge": "MergeScenario",
    "Descent": "DescentScenario",
    "StaticObstacle": "StaticObstacleScenario",
    "SectorCapacity": "SectorCapacityScenario",
    "RouteNav": "RouteNavScenario",
    "PlanWaypoint": "PlanWaypointScenario",
}


def _resolve_scenario(name: str, num_aircraft: int, seed: int) -> BaseScenario:
    """Import and instantiate a scenario class by name."""
    cls_name = SCENARIO_MAP[name]
    from bluesky_pettingzoo.envs.scenarios import horizontal_cr, vertical_cr, sector_cr
    from bluesky_pettingzoo.envs.scenarios import waypoint_nav, merge, descent
    from bluesky_pettingzoo.envs.scenarios import static_obstacle, sector_capacity, route_nav
    from bluesky_pettingzoo.envs.scenarios import plan_waypoint

    module_map = {
        "HorizontalCRScenario": horizontal_cr,
        "VerticalCRScenario": vertical_cr,
        "SectorCRScenario": sector_cr,
        "WaypointNavScenario": waypoint_nav,
        "MergeScenario": merge,
        "DescentScenario": descent,
        "StaticObstacleScenario": static_obstacle,
        "SectorCapacityScenario": sector_capacity,
        "RouteNavScenario": route_nav,
        "PlanWaypointScenario": plan_waypoint,
    }
    mod = module_map[cls_name]
    cls = getattr(mod, cls_name)
    return cls(num_aircraft=num_aircraft, seed=seed)


def make_scenario_env_factory(
    tmp_path: Path,
    scenario: BaseScenario,
    num_aircraft: int,
    max_steps: int,
    wrapper_cls: type | None = None,
    render_mode: str | None = None,
) -> Callable[[], SingleAgentGymWrapper]:
    """Return a callable that creates a SingleAgentGymWrapper env."""

    def factory() -> SingleAgentGymWrapper:
        config = make_config(initial_count=num_aircraft, max_steps=max_steps)
        if render_mode:
            config["render_mode"] = render_mode
        rewards_path = write_rewards_yaml(tmp_path)
        config["_rewards_yaml"] = str(rewards_path)

        with open(rewards_path, encoding="utf-8") as f:
            rewards_cfg = yaml.safe_load(f)
        merged = {**config, **rewards_cfg}

        if wrapper_cls is not None:
            wrapper = wrapper_cls(config)
        else:
            wrapper = BlueSkyWrapper(config)
        obs_manager = ObservationManager(config)
        action_translator = ActionTranslator(config)
        calc = RewardCalculator()
        calc.register(ConflictPenalty(merged), weight=1.0)
        calc.register(SmoothnessPenalty(merged), weight=0.5)
        calc.register(EfficiencyReward(merged), weight=0.3)
        if hasattr(scenario, "get_obstacles"):
            obs_comp = ObstacleIntrusion()
            obs_comp.set_obstacles(scenario.get_obstacles())
            calc.register(obs_comp, weight=1.0)
        if hasattr(scenario, "get_sectors"):
            cap_comp = CapacityPenalty(merged)
            calc.register(cap_comp, weight=1.0)
        if hasattr(scenario, "get_sectors"):
            calc.register(FlowEfficiencyReward(merged), weight=0.2)
            calc.register(FairnessReward(merged), weight=0.1)

        env = BlueSkyMARLEnv(
            config=config,
            wrapper=wrapper,
            observation_manager=obs_manager,
            action_translator=action_translator,
            reward_calculator=calc,
            rewards_config=rewards_cfg,
            scenario=scenario,
        )
        return SingleAgentGymWrapper(env, ego_agent="AC000")

    return factory


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="PPO training on BlueSky scenarios")
    # Scenario and algorithm
    parser.add_argument("--scenario", type=str, default="HorizontalCR",
                        choices=list(SCENARIO_MAP.keys()),
                        help="Scenario to train on")
    parser.add_argument("--algorithm", type=str, default="PPO",
                        choices=["PPO", "SAC", "TD3", "DDPG"],
                        help="RL algorithm to use")
    parser.add_argument("--action-space", type=str, default="discrete",
                        choices=["discrete", "continuous"],
                        help="Action space type")
    # Training scale
    parser.add_argument("--timesteps", type=int, default=500_000,
                        help="Total training timesteps")
    parser.add_argument("--num-aircraft", type=int, default=3,
                        help="Number of aircraft in scenario")
    parser.add_argument("--max-steps", type=int, default=50,
                        help="Max steps per episode")
    parser.add_argument("--num-envs", type=int, default=1,
                        help="Number of parallel environments (1=single env)")
    # PPO hyperparameters
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate")
    parser.add_argument("--n-steps", type=int, default=2048,
                        help="Number of steps per rollout (per env)")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Mini-batch size for PPO updates")
    parser.add_argument("--n-epochs", type=int, default=4,
                        help="Number of PPO epochs per update")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
                        help="GAE lambda")
    # Normalization
    parser.add_argument("--norm-reward", action="store_true", default=False,
                        help="Enable VecNormalize reward normalization")
    # Infrastructure
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--save-dir", type=str, default="models",
                        help="Directory to save models and logs")
    parser.add_argument("--render", action="store_true", default=False,
                        help="Enable Pygame rendering during training (slow!)")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: auto, cpu, cuda, cuda:0, etc.")
    parser.add_argument("--verbose", type=int, default=0,
                        help="Verbosity level (0=silent, 1=info, 2=debug)")
    return parser.parse_args(argv)


def _get_algo_class(algorithm: str):
    """Return the SB3 model class for the given algorithm name."""
    if algorithm == "PPO":
        from stable_baselines3 import PPO
        return PPO
    elif algorithm == "SAC":
        from stable_baselines3 import SAC
        return SAC
    elif algorithm == "TD3":
        from stable_baselines3 import TD3
        return TD3
    elif algorithm == "DDPG":
        from stable_baselines3 import DDPG
        return DDPG
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")


def _resolve_device(device_str: str) -> str:
    """Resolve device string, auto-detect GPU if not specified."""
    if device_str != "auto":
        return device_str
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def _make_model(algo_cls, env, args):
    """Create a new model instance for the given algorithm."""
    device = _resolve_device(args.device)
    print(f"  Using device: {device}")

    # Determine number of environments for batch size scaling
    num_envs = getattr(args, "num_envs", 1)

    common_kwargs = dict(
        policy="MultiInputPolicy",
        env=env,
        learning_rate=getattr(args, "lr", 3e-4),
        verbose=getattr(args, "verbose", 0),
        device=device,
        seed=args.seed,
    )

    if args.algorithm == "PPO":
        # Scale n_steps with num_envs to maintain same collection ratio
        n_steps = getattr(args, "n_steps", 2048)
        batch_size = getattr(args, "batch_size", 256)
        n_epochs = getattr(args, "n_epochs", 4)

        # Auto-scale for multi-env
        if num_envs > 1:
            n_steps = max(n_steps // num_envs, 128)

        return algo_cls(
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=getattr(args, "gamma", 0.99),
            gae_lambda=getattr(args, "gae_lambda", 0.95),
            **common_kwargs,
        )
    else:
        # SAC/TD3/DDPG use batch_size and buffer_size
        return algo_cls(batch_size=256, buffer_size=100_000, **common_kwargs)


def train_scenario(args: argparse.Namespace) -> dict[str, float]:
    """Train an RL algorithm on a single scenario with logging and checkpoints."""
    algo_cls = _get_algo_class(args.algorithm)
    # SAC/TD3/DDPG require continuous action space
    action_space = getattr(args, "action_space", "discrete")
    if args.algorithm in ("SAC", "TD3", "DDPG"):
        action_space = "continuous"

    scenario = _resolve_scenario(args.scenario, args.num_aircraft, args.seed)
    # Set action space type on scenario so env creates the right space
    if action_space == "continuous":
        scenario.action_space_type = "continuous"
    save_dir = Path(args.save_dir)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        # Create environments
        num_envs = getattr(args, "num_envs", 1)
        render_mode = "human" if getattr(args, "render", False) else None

        if num_envs == 1:
            # Single environment (original behavior)
            factory = make_scenario_env_factory(
                tmp_path, scenario, args.num_aircraft, args.max_steps,
                render_mode=render_mode,
            )
            env = factory()
        else:
            # Multiple environments using DummyVecEnv (same process, sequential)
            # Note: BlueSky is a global singleton, so we cannot use SubprocVecEnv
            from stable_baselines3.common.vec_env import DummyVecEnv

            def make_env_fn(idx: int):
                def _init():
                    # Each env needs its own tmp_path to avoid YAML conflicts
                    env_tmp = Path(tempfile.mkdtemp())
                    env_factory = make_scenario_env_factory(
                        env_tmp, scenario, args.num_aircraft, args.max_steps,
                        render_mode=render_mode if idx == 0 else None,  # Only render first env
                    )
                    return env_factory()
                return _init

            env = DummyVecEnv([make_env_fn(i) for i in range(num_envs)])
            print(f"  Using {num_envs} parallel environments (DummyVecEnv)")

        # Wrap with VecNormalize for reward normalization if requested
        # VecNormalize requires a VecEnv, so wrap single env in DummyVecEnv first
        norm_reward = getattr(args, "norm_reward", False)
        if norm_reward:
            from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
            if num_envs == 1:
                # Wrap single env in DummyVecEnv for VecNormalize compatibility
                env = DummyVecEnv([lambda: env])
            env = VecNormalize(env, norm_obs=False, norm_reward=True, clip_reward=10.0)
            print("  VecNormalize: reward normalization enabled")

        # Setup checkpoint manager (saves to save_dir/{scenario}/{algorithm}/)
        ckpt_mgr = CheckpointManager(
            save_dir=save_dir,
            scenario=args.scenario,
            save_interval=max(args.timesteps // 5, 1),
            max_checkpoints=5,
            seed=args.seed,
            algorithm=args.algorithm,
        )

        # Setup CSV logger
        log_dir = save_dir / args.scenario / args.algorithm / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        csv_callback = CSVLoggerCallback(
            csv_path=log_dir / "training_log.csv",
            algorithm=args.algorithm,
            action_space=action_space,
        )

        # Create or resume model
        if args.resume:
            model = algo_cls.load(args.resume, env=env)
        else:
            model = _make_model(algo_cls, env, args)

        # Train
        print(f"\nTraining {args.algorithm} on {args.scenario} ({args.timesteps:,} timesteps)...")
        model.learn(
            total_timesteps=args.timesteps,
            callback=[csv_callback, _CheckpointCallback(ckpt_mgr), ProgressCallback()],
        )

        # Save final
        ckpt_mgr.save_final(model, timestep=model.num_timesteps, episode=csv_callback._episode)
        csv_callback._on_training_end()

        # Save VecNormalize stats if used
        if norm_reward and hasattr(env, "save"):
            vec_norm_path = save_dir / args.scenario / args.algorithm / "vec_normalize.pkl"
            env.save(str(vec_norm_path))
            print(f"  VecNormalize stats saved to {vec_norm_path}")

        env.close()

    return {"timesteps": model.num_timesteps, "algorithm": args.algorithm}


class _CheckpointCallback(BaseCallback):
    """Callback that delegates checkpoint saving to CheckpointManager."""

    def __init__(self, mgr: CheckpointManager, verbose: int = 0) -> None:
        super().__init__(verbose=verbose)
        self._mgr = mgr

    def _on_step(self) -> bool:
        self._mgr.maybe_save(self.model, timestep=self.num_timesteps, episode=self.n_calls)
        return True


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    train_scenario(args)


if __name__ == "__main__":
    main()
