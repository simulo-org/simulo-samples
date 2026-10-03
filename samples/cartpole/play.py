"""Play a trained Cartpole policy and record it."""

from __future__ import annotations

import simulo

from task import CartpoleTask, app


@app.job(type="play", system=simulo.SystemType.TIER_1, timeout=15 * 60)
def play(policy: simulo.PolicyCheckpoint, episodes: int = 3) -> None:
    """Play a policy: simulo run play.py --policy policy_x:best."""
    env = simulo.LearningEnv(
        task=CartpoleTask(with_camera=True),
        num_envs=1,
        device="cuda",
        dt=1.0 / 120.0,
        physics_steps_per_action=2,
        env_spacing=4.0,
        headless=True,
        seed=42,
        enable_cameras=True,
    )
    player = simulo.RLPlayer(env=env, policy=policy, device="cuda")
    try:
        player.play(num_episodes=episodes, record=simulo.RecordConfig(include_video=True))
    finally:
        player.close()
        env.close()
