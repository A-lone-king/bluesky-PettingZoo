"""Performance benchmark for bluesky-pettingzoo.

Measures:
1. Single step() time for different aircraft counts
2. Full episode time for different max_steps
3. Memory usage per episode

Usage:
    python scripts/benchmark_performance.py
"""

from __future__ import annotations

import gc
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from bluesky_pettingzoo.actions.translator import ActionTranslator
from bluesky_pettingzoo.envs.parallel_env import BlueSkyMARLEnv
from bluesky_pettingzoo.envs.scenarios.waypoint_nav import WaypointNavScenario
from bluesky_pettingzoo.observations.manager import ObservationManager
from bluesky_pettingzoo.rewards.calculator import RewardCalculator
from bluesky_pettingzoo.rewards.components.conflict import ConflictPenalty
from bluesky_pettingzoo.rewards.components.efficiency import EfficiencyReward
from bluesky_pettingzoo.rewards.components.smoothness import SmoothnessPenalty
from tests.helpers.fake_wrapper import FakeBlueSkyWrapper
from tests.helpers.env_factory import make_config, write_rewards_yaml


def _try_get_memory_mb() -> float | None:
    """Return current process RSS in MB, or None if unavailable."""
    try:
        import psutil
        return psutil.Process().memory_info().rss / (1024 * 1024)
    except ImportError:
        pass
    try:
        import resource
        # macOS/Linux: ru_maxrss is in KB on Linux, bytes on macOS
        usage = resource.getrusage(resource.RUSAGE_SELF)
        kb = usage.ru_maxrss
        if sys.platform == "darwin":
            kb = kb / 1024
        return kb / 1024
    except (ImportError, AttributeError):
        return None


def make_env(num_aircraft: int, max_steps: int) -> BlueSkyMARLEnv:
    """Create a BlueSkyMARLEnv with FakeBlueSkyWrapper."""
    config = make_config(initial_count=num_aircraft, max_steps=max_steps)
    with tempfile.TemporaryDirectory() as tmp:
        rewards_path = write_rewards_yaml(Path(tmp))
        config["_rewards_yaml"] = str(rewards_path)
        with open(rewards_path, encoding="utf-8") as f:
            rewards_cfg = yaml.safe_load(f)
    merged = {**config, **rewards_cfg}

    wrapper = FakeBlueSkyWrapper(config)
    obs_manager = ObservationManager(config)
    action_translator = ActionTranslator(config)
    calc = RewardCalculator()
    calc.register(ConflictPenalty(merged), weight=1.0)
    calc.register(SmoothnessPenalty(merged), weight=0.5)
    calc.register(EfficiencyReward(merged), weight=0.3)

    scenario = WaypointNavScenario(num_aircraft=num_aircraft, seed=42)
    return BlueSkyMARLEnv(
        config=config,
        wrapper=wrapper,
        observation_manager=obs_manager,
        action_translator=action_translator,
        reward_calculator=calc,
        rewards_config=rewards_cfg,
        scenario=scenario,
    )


def benchmark_step_time(num_aircraft: int, n_steps: int = 50) -> float:
    """Measure average step() time in milliseconds."""
    env = make_env(num_aircraft, max_steps=n_steps + 10)
    env.reset(seed=42)

    times = []
    for _ in range(n_steps):
        actions = {a: env.action_space(a).sample() for a in env.agents}
        gc.disable()
        t0 = time.perf_counter()
        env.step(actions)
        t1 = time.perf_counter()
        gc.enable()
        times.append((t1 - t0) * 1000)
        if not env.agents:
            break

    env.close()
    return float(np.mean(times))


def benchmark_episode_time(num_aircraft: int, max_steps: int) -> tuple[float, float]:
    """Measure full episode time (seconds) and memory delta (MB)."""
    gc.collect()
    mem_before = _try_get_memory_mb()

    env = make_env(num_aircraft, max_steps=max_steps)
    env.reset(seed=42)

    t0 = time.perf_counter()
    for _ in range(max_steps):
        actions = {a: env.action_space(a).sample() for a in env.agents}
        env.step(actions)
        if not env.agents:
            break
    t1 = time.perf_counter()

    env.close()
    gc.collect()
    mem_after = _try_get_memory_mb()

    elapsed = t1 - t0
    mem_delta = (mem_after - mem_before) if (mem_before is not None and mem_after is not None) else 0.0
    return elapsed, max(0.0, mem_delta)


def main() -> None:
    """Run all benchmarks and print results table."""
    aircraft_counts = [1, 5, 10, 20]
    step_counts = [50, 100, 200]

    print("=" * 70)
    print("bluesky-pettingzoo Performance Benchmark")
    print("=" * 70)

    # Step time benchmark
    print("\n--- Step Time Benchmark ---")
    print(f"{'Aircraft':>8}  {'Mean(ms)':>10}  {'Std(ms)':>10}  {'Steps':>6}")
    print("-" * 40)
    for n in aircraft_counts:
        env = make_env(n, max_steps=60)
        env.reset(seed=42)
        times = []
        for _ in range(50):
            actions = {a: env.action_space(a).sample() for a in env.agents}
            gc.disable()
            t0 = time.perf_counter()
            env.step(actions)
            t1 = time.perf_counter()
            gc.enable()
            times.append((t1 - t0) * 1000)
            if not env.agents:
                break
        env.close()
        mean_t = float(np.mean(times))
        std_t = float(np.std(times))
        print(f"{n:>8}  {mean_t:>10.3f}  {std_t:>10.3f}  {len(times):>6}")

    # Episode time + memory benchmark
    print(f"\n--- Episode Benchmark ---")
    print(f"{'Aircraft':>8}  {'MaxSteps':>8}  {'Time(s)':>10}  {'Mem(MB)':>10}")
    print("-" * 42)
    for n in aircraft_counts:
        for ms in step_counts:
            elapsed, mem = benchmark_episode_time(n, ms)
            print(f"{n:>8}  {ms:>8}  {elapsed:>10.3f}  {mem:>10.2f}")

    print("=" * 70)
    print("Benchmark complete.")


if __name__ == "__main__":
    main()
