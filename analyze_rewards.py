"""Analyze reward component magnitudes for a 30-step episode."""
import tempfile, yaml
from pathlib import Path
from tests.helpers.env_factory import write_rewards_yaml

with tempfile.TemporaryDirectory() as tmp:
    rewards_path = write_rewards_yaml(Path(tmp))
    with open(rewards_path) as f:
        rewards = yaml.safe_load(f)
    
    comp = rewards['components']
    print('=== Reward Component Analysis ===')
    for name, cfg in comp.items():
        w = cfg.get('weight', 1.0)
        print(f'  {name:25s} weight={w}')
    print()
    
    # Estimate reward for a 30-step episode with no conflicts
    step_penalty = comp['efficiency']['step_penalty'] * 30
    smoothness = comp['smoothness']['action_penalty'] * 30 * 0.3
    drift = comp['drift_penalty']['scale'] * 30 * 0.5
    delay = comp['delay']['delay_penalty_per_step'] * 30
    arrival = comp['efficiency']['arrival_reward']
    print(f'Estimated penalty for 30-step no-conflict episode:')
    print(f'  step_penalty:    {step_penalty:.3f}')
    print(f'  smoothness:      {smoothness:.3f}')
    print(f'  drift:           {drift:.3f}')
    print(f'  delay:           {delay:.3f}')
    total_penalty = step_penalty + smoothness + drift + delay
    print(f'  TOTAL baseline:  {total_penalty:.3f}')
    print(f'  with arrival:    +{arrival:.1f}')
    print(f'  net with arrival: {total_penalty + arrival:.3f}')
    print(f'  net no arrival:   {total_penalty:.3f}')
